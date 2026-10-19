# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experiment: can kernel microbenchmarks pick the ROCm attention backend?

For each scenario in ``configs/rocm_autoselect_e2e.yaml`` (one model and spec
config that was served end to end with each backend), this script:

1. Builds every candidate backend the way the serving run did: its own block
   size and KV cache layout, a KV pool as large as the served one, and full
   CUDA graph metadata for uniform decode batches (``max_seq_len`` set to
   ``max_model_len``, as at capture time).
2. Checks correctness against an fp32 reference on a mixed batch whose blocks
   sit at the top of the pool. A backend that fails is not a candidate.
3. Times the attention work of each serving point: the prefill chunks of one
   request and one decode (or spec verify) step of the whole batch.
4. Turns the per-layer times into a per-request attention cost and predicts
   the end-to-end throughput ratio between two backends, assuming everything
   except attention costs the same. It compares this with the measured ratio.

Results and findings are in ``ROCM_BACKEND_AUTOSELECT.md``.

Usage (inside a ROCm container, from this directory):
    python rocm_backend_autoselect.py --scenarios qwen3_8b_nospec \\
        --output-json /tmp/autoselect.json
"""

import argparse
import json
import math
import random
import statistics
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from common import BenchmarkConfig, run_do_bench
from runner import (
    _create_backend_impl,
    _create_kv_cache,
    _create_metadata_builder,
    _get_backend_config,
    log_warnings_and_errors_only,
)

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    get_supported_kv_cache_layouts,
    resolve_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "rocm_autoselect_e2e.yaml"
# Only used for its tokenizer-free config; the attention shape is mocked.
MOCK_MODEL = "Qwen/Qwen3-8B"


# ============================================================================
# Backend setup
# ============================================================================


@dataclass
class BackendSetup:
    name: str
    bench_cfg: BenchmarkConfig
    vllm_config: VllmConfig
    impl: object
    layer: object
    builder: object
    cache_list: list
    num_blocks: int
    layout: str
    dtype: torch.dtype


def _create_vllm_config(
    bench_cfg: BenchmarkConfig,
    num_blocks: int,
    layout: str | None,
    max_num_batched_tokens: int,
) -> VllmConfig:
    model_config = ModelConfig(
        model=MOCK_MODEL,
        tokenizer=MOCK_MODEL,
        dtype="bfloat16",
        seed=0,
        max_model_len=1024,
    )
    cache_config = CacheConfig(block_size=bench_cfg.block_size, cache_dtype="auto")
    cache_config.num_gpu_blocks = num_blocks
    cache_config.num_cpu_blocks = 0
    if layout is not None:
        cache_config.kv_cache_layout = layout
    scheduler_config = SchedulerConfig(
        max_num_seqs=256,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_num_batched_tokens,
        is_encoder_decoder=False,
        enable_chunked_prefill=True,
    )

    def mock(fn):
        return types.MethodType(fn, model_config)

    model_config.get_num_layers = mock(lambda self: bench_cfg.num_layers)
    model_config.get_sliding_window_for_layer = mock(lambda self, i: None)
    model_config.get_logits_soft_cap_for_layer = mock(lambda self, i: 0.0)
    model_config.get_sm_scale_for_layer = mock(
        lambda self, i: 1.0 / bench_cfg.head_dim**0.5
    )
    model_config.get_num_attention_heads = mock(
        lambda self, parallel_config=None: bench_cfg.num_q_heads
    )
    model_config.get_num_kv_heads = mock(
        lambda self, parallel_config=None: bench_cfg.num_kv_heads
    )
    model_config.get_head_size = mock(lambda self: bench_cfg.head_dim)
    model_config.get_sliding_window = mock(lambda self: None)

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        scheduler_config=scheduler_config,
        device_config=DeviceConfig(),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
    )


def create_backend(
    backend: str,
    model: dict,
    block_size: int,
    layout: str | None,
    kv_cache_tokens: int,
    num_layers: int,
    max_num_batched_tokens: int,
    device: torch.device,
) -> BackendSetup:
    bench_cfg = BenchmarkConfig(
        backend=backend,
        batch_spec="",
        num_layers=num_layers,
        head_dim=model["head_dim"],
        num_q_heads=model["num_q_heads"],
        num_kv_heads=model["num_kv_heads"],
        block_size=block_size,
        device=str(device),
    )
    num_blocks = kv_cache_tokens // block_size
    backend_cfg = _get_backend_config(backend)
    with log_warnings_and_errors_only():
        vllm_config = _create_vllm_config(
            bench_cfg, num_blocks, layout, max_num_batched_tokens
        )
        dtype = vllm_config.model_config.dtype
        with set_current_vllm_config(vllm_config):
            backend_class, impl, layer = _create_backend_impl(
                backend_cfg, bench_cfg, device, dtype
            )
            supported = get_supported_kv_cache_layouts([backend_class])
            resolved = resolve_kv_cache_layout(
                vllm_config, [[m.name for m in supported]]
            )
            spec = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=model["num_kv_heads"],
                head_size=model["head_dim"],
                dtype=dtype,
            )
            builder = _create_metadata_builder(
                backend_class, spec, vllm_config, device, backend
            )
            cache_list = _create_kv_cache(
                bench_cfg, num_blocks, backend_class, device, dtype, resolved
            )
    return BackendSetup(
        name=backend,
        bench_cfg=bench_cfg,
        vllm_config=vllm_config,
        impl=impl,
        layer=layer,
        builder=builder,
        cache_list=cache_list,
        num_blocks=num_blocks,
        layout=resolved.name,
        dtype=dtype,
    )


# ============================================================================
# Metadata and timing
# ============================================================================


@dataclass
class Batch:
    q_lens: list[int]
    kv_lens: list[int]
    # Graph-replayed uniform decode: kernels see max_seq_len = max_model_len.
    graph_max_seq_len: int | None = None
    block_ids: list[list[int]] = field(default_factory=list)


def assign_blocks(
    batch: Batch, block_size: int, num_blocks: int, rng: random.Random, top: bool
) -> None:
    """Scatter each request's blocks over the pool, as a busy server would.

    With ``top`` the blocks come from the upper half of the pool, so kernels
    that address the cache with 32-bit offsets get tested where it matters.
    """
    needed = [math.ceil(kv / block_size) for kv in batch.kv_lens]
    total = sum(needed)
    lo = num_blocks // 2 if top else 0
    assert total <= num_blocks - lo, (
        f"Batch needs {total} blocks, pool has {num_blocks - lo}."
    )
    ids = rng.sample(range(lo, num_blocks), total)
    batch.block_ids = []
    for n in needed:
        batch.block_ids.append(ids[:n])
        ids = ids[n:]


def build_metadata(setup: BackendSetup, batch: Batch, device: torch.device):
    block_size = setup.bench_cfg.block_size
    q_lens, kv_lens = batch.q_lens, batch.kv_lens
    num_reqs = len(q_lens)
    max_blocks = max(len(b) for b in batch.block_ids)
    block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32)
    slots = []
    for i, (q, kv, ids) in enumerate(zip(q_lens, kv_lens, batch.block_ids)):
        block_table[i, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        for pos in range(kv - q, kv):
            slots.append(ids[pos // block_size] * block_size + pos % block_size)
    query_start_loc_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc_cpu[1:] = torch.tensor(q_lens, dtype=torch.int32).cumsum(0)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu.to(device),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        num_reqs=num_reqs,
        num_actual_tokens=sum(q_lens),
        max_query_len=max(q_lens),
        max_seq_len=max(kv_lens),
        block_table_tensor=block_table.to(device),
        slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
        causal=True,
    )
    with set_current_vllm_config(setup.vllm_config):
        md = setup.builder.build(common_prefix_len=0, common_attn_metadata=common)
    if batch.graph_max_seq_len is not None:
        md.max_seq_len = batch.graph_max_seq_len
    return md


def time_batch(
    setup: BackendSetup, batch: Batch, device: torch.device, rng: random.Random
) -> float:
    """Median time per layer in ms."""
    cfg = setup.bench_cfg
    assign_blocks(batch, cfg.block_size, setup.num_blocks, rng, top=False)
    md = build_metadata(setup, batch, device)
    total_q = sum(batch.q_lens)
    hq, hkv, d = cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim
    qs = [torch.randn(total_q, hq, d, device=device, dtype=setup.dtype)]
    ks = [torch.randn(total_q, hkv, d, device=device, dtype=setup.dtype)]
    vs = [torch.randn(total_q, hkv, d, device=device, dtype=setup.dtype)]
    out = torch.empty(total_q, hq, d, device=device, dtype=setup.dtype)

    def fn():
        for i in range(cfg.num_layers):
            setup.impl.forward(
                setup.layer, qs[0], ks[0], vs[0], setup.cache_list[i], md, output=out
            )

    with set_current_vllm_config(setup.vllm_config):
        # Graph replay for uniform decode, eager (piecewise graphs) otherwise.
        ms = run_do_bench(fn, use_cuda_graphs=batch.graph_max_seq_len is not None)
    return statistics.median(ms) / cfg.num_layers


# ============================================================================
# Correctness
# ============================================================================


def reference_attention(q, k, v, q_lens, kv_lens, scale):
    """fp32 causal attention; k/v hold every token of every request."""
    outs = []
    q_off = kv_off = 0
    group = q.shape[1] // k.shape[1]
    for ql, kvl in zip(q_lens, kv_lens):
        qi = q[q_off : q_off + ql].float()
        ki = k[kv_off : kv_off + kvl].float().repeat_interleave(group, dim=1)
        vi = v[kv_off : kv_off + kvl].float().repeat_interleave(group, dim=1)
        s = torch.einsum("qhd,khd->hqk", qi, ki) * scale
        q_pos = torch.arange(kvl - ql, kvl, device=q.device)[:, None]
        k_pos = torch.arange(kvl, device=q.device)[None, :]
        s.masked_fill_(k_pos > q_pos, float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", s.softmax(-1), vi))
        q_off += ql
        kv_off += kvl
    return torch.cat(outs)


CHECK_BATCH = ([1, 1, 4, 37, 200], [1500, 700, 900, 300, 200])


def check_correctness(
    setup: BackendSetup,
    device: torch.device,
    rng: random.Random,
    max_model_len: int,
    query_len: int,
) -> float:
    """Max abs error against fp32, worst of a mixed batch and a graph-replayed
    uniform decode batch. NaN if the output is not finite."""
    cfg = setup.bench_cfg
    hq, hkv, d = cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim
    scale = 1.0 / d**0.5
    batches = [
        Batch(*CHECK_BATCH),
        Batch([query_len] * 4, [1500, 700, 900, 300], max_model_len),
    ]
    worst = 0.0
    for batch in batches:
        assign_blocks(batch, cfg.block_size, setup.num_blocks, rng, top=True)
        md = build_metadata(setup, batch, device)
        total_kv = sum(batch.kv_lens)
        k_all = torch.randn(total_kv, hkv, d, device=device, dtype=setup.dtype)
        v_all = torch.randn(total_kv, hkv, d, device=device, dtype=setup.dtype)
        slots = []
        for kv, ids in zip(batch.kv_lens, batch.block_ids):
            slots += [
                ids[p // cfg.block_size] * cfg.block_size + p % cfg.block_size
                for p in range(kv)
            ]
        slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)
        new_idx = torch.cat(
            [
                torch.arange(off + kv - q, off + kv)
                for q, kv, off in zip(
                    batch.q_lens,
                    batch.kv_lens,
                    [0] + list(torch.tensor(batch.kv_lens).cumsum(0)[:-1].tolist()),
                )
            ]
        ).to(device)
        q = torch.randn(len(new_idx), hq, d, device=device, dtype=setup.dtype)
        out = torch.empty_like(q)
        cache = setup.cache_list[0]
        with set_current_vllm_config(setup.vllm_config):
            setup.impl.do_kv_cache_update(
                setup.layer, k_all, v_all, cache, slot_mapping
            )
            setup.impl.forward(
                setup.layer, q, k_all[new_idx], v_all[new_idx], cache, md, output=out
            )
        torch.accelerator.synchronize()
        if not torch.isfinite(out).all():
            return float("nan")
        ref = reference_attention(q, k_all, v_all, batch.q_lens, batch.kv_lens, scale)
        worst = max(worst, (out.float() - ref).abs().max().item())
    return worst


# ============================================================================
# Serving-point workloads and the cost model
# ============================================================================


def prefill_batches(
    isl: int,
    concurrency: int,
    chunk: int,
    riders: int = 0,
    rider: tuple[int, int] = (1, 1),
) -> tuple[list, int]:
    """Prefill batches for one request slot and how many requests they cover.

    Short prompts are packed into one chunk; long ones are split into chunks.
    ``riders`` running requests, each a ``rider=(query_len, kv_len)`` decode,
    share every prefill step, as in the mixed steps of a serving run.
    """
    if isl <= chunk:
        n = max(1, min(concurrency, chunk // isl))
        chunks = [(isl, isl)]
    else:
        n = 1
        starts = range(0, isl, chunk)
        chunks = [(min(chunk, isl - s), min(isl, s + chunk)) for s in starts]
    dq, dkv = rider
    batches = [
        Batch([dq] * riders + [q] * n, [dkv] * riders + [kv] * n) for q, kv in chunks
    ]
    return batches, n


def decode_batch(isl: int, concurrency: int, osl: int, query_len: int, mml: int):
    # Mid-generation context length stands for the whole decode phase.
    return Batch([query_len] * concurrency, [isl + osl // 2] * concurrency, mml)


def running_requests(concurrency: int, isl: int, osl: int, kv_tokens: int) -> int:
    """Requests that fit in the KV cache at once (the scheduler caps at this)."""
    return max(1, min(concurrency, kv_tokens // (isl + osl)))


def rider_mix(running: int, n: int, num_prompts: int) -> list[tuple[int, float]]:
    """Decodes sharing a prefill step in a bench run, as (count, weight) pairs.

    ``vllm bench serve`` starts ``running`` requests at once. In the first wave
    the k-th group of ``n`` prefills shares its steps with the k - 1 groups
    before it; in later waves a request is admitted when another finishes, so
    all the others are decoding. The count matters on its own, not just its
    mean: some kernels slow down sharply at particular batch sizes.
    """
    groups = max(1, running // n)
    waves = max(1.0, num_prompts / running)
    mix = [(k * n, 1 / waves / groups) for k in range(groups)]
    if waves > 1:
        mix.append((running - n, 1 - 1 / waves))
    return mix


@dataclass
class PointCost:
    prefill_ms: float  # attention time of the prefill steps of one request
    prefill_steps: float  # steps spent prefilling one request
    riders: float  # mean number of decodes sharing a prefill step
    step_ms: float  # attention time of one decode/verify step, all layers
    running: int  # requests running at once

    def per_request_ms(self, osl: int, accept: float) -> float:
        # A request needs osl / accept decode steps. Riders get theirs from
        # the prefill steps; pure decode steps serve all running requests.
        tokens = osl / accept - self.prefill_steps * self.riders
        return self.prefill_ms + max(tokens, 0.0) / self.running * self.step_ms


def time_points(
    setup: BackendSetup,
    scen: dict,
    points: list,
    osl: int,
    chunk: int,
    device: torch.device,
    rng: random.Random,
) -> dict:
    """Attention time of each serving point, scaled to all attention layers."""
    layers = scen["model"]["attn_layers"]
    qlen, mml = scen["query_len"], scen["max_model_len"]
    kv_tokens = setup.num_blocks * setup.bench_cfg.block_size
    memo: dict[tuple, float] = {}

    def timed(batch: Batch) -> float:
        key = (tuple(batch.q_lens), tuple(batch.kv_lens), batch.graph_max_seq_len)
        if key not in memo:
            memo[key] = time_batch(setup, batch, device, rng) * layers
        return memo[key]

    times = {}
    for isl, c, num_prompts in points:
        r = running_requests(c, isl, osl, kv_tokens)
        dec = decode_batch(isl, r, osl, qlen, mml)
        alone, n = prefill_batches(isl, r, chunk)
        mix = rider_mix(r, n, num_prompts)
        rider = (qlen, dec.kv_lens[0])
        mixed_ms = sum(
            w * sum(timed(b) for b in prefill_batches(isl, r, chunk, k, rider)[0])
            for k, w in mix
        )
        t = {
            "running": r,
            "riders": sum(k * w for k, w in mix),
            "prefill_alone_ms": sum(timed(b) for b in alone) / n,
            "prefill_mixed_ms": mixed_ms / n,
            "prefill_steps": len(alone) / n,
            "prefill_one_ms": sum(timed(b) for b in prefill_batches(isl, 1, chunk)[0]),
            "step_ms": timed(dec),
        }
        times[f"{isl}x{c}"] = t
        print(
            f"  isl={isl:>6} c={c:>3} (running {r}, riders {t['riders']:.1f})"
            f"  prefill/req alone {t['prefill_alone_ms']:9.2f}"
            f" mixed {t['prefill_mixed_ms']:9.2f} ms  step {t['step_ms']:8.3f} ms",
            flush=True,
        )
    return times


# A grid small enough to time at engine startup: (label, q_lens, kv_lens, graph).
def quick_grid(query_len: int, chunk: int, mml: int) -> list[tuple]:
    grid = []
    for bs in (1, 16, 64):
        for ctx in (4096, 32768):
            q = [query_len] * bs
            grid.append((f"decode bs={bs} ctx={ctx}", q, [ctx] * bs, mml))
    for pre in (0, 32768):
        grid.append((f"prefill {chunk} +{pre}", [chunk], [pre + chunk], None))
    q, kv = [query_len] * 7 + [chunk], [32768] * 7 + [32768 + chunk]
    grid.append((f"mixed {chunk} +32768, 7 decodes", q, kv, None))
    return grid


def time_quick_grid(
    setup: BackendSetup, scen: dict, chunk: int, device: torch.device, rng
) -> dict:
    layers = scen["model"]["attn_layers"]
    pool = setup.num_blocks * setup.bench_cfg.block_size
    table = {}
    for label, q, kv, graph in quick_grid(
        scen["query_len"], chunk, scen["max_model_len"]
    ):
        if sum(kv) > pool:
            continue
        ms = time_batch(setup, Batch(q, kv, graph), device, rng) * layers
        table[label] = ms
        print(f"  {label:34s} {ms:10.3f} ms", flush=True)
    return table


def run_scenario(name: str, scen: dict, args, osl: int, chunk: int) -> dict:
    device = torch.device(args.device)
    torch.accelerator.set_device_index(device)
    model = scen["model"]
    mml = scen["max_model_len"]
    qlen = scen["query_len"]
    runs = scen["runs"]
    backends = args.backends or list(runs)
    ref_run = next(iter(runs.values()))
    points = sorted(
        {tuple(r[:2] + r[6:]) for run in runs.values() for r in run["results"]}
    )

    result = {"backends": {}}
    for be in backends:
        if be in runs:
            run = runs[be]
            block_size, layout = run["block_size"], run["kv_cache_layout"]
            kv_tokens = run["kv_cache_tokens"]
        else:
            # Not served: hybrid models share the mamba-aligned block size and
            # LBHNC; dense models use the default block size and native layout.
            block_size = ref_run["block_size"] if scen["hybrid"] else 16
            layout = "LBHNC" if scen["hybrid"] else None
            kv_tokens = ref_run["kv_cache_tokens"]
        rng = random.Random(0)
        start = time.perf_counter()
        try:
            setup = create_backend(
                be, model, block_size, layout, kv_tokens, args.num_layers, chunk, device
            )
            err = check_correctness(setup, device, rng, mml, qlen)
        except Exception as e:  # noqa: BLE001 - report and skip the backend
            print(f"[{name}] {be}: setup failed: {e!r}")
            result["backends"][be] = {"error": repr(e)}
            continue
        ok = err == err and err < args.atol
        info = {
            "block_size": block_size,
            "layout": setup.layout,
            "kv_cache_tokens": kv_tokens,
            "max_abs_err": err,
            "correct": ok,
        }
        result["backends"][be] = info
        print(f"[{name}] {be}: block {block_size} {setup.layout} max_abs_err={err:.3g}")
        if args.quick:
            info["quick"] = time_quick_grid(setup, scen, chunk, device, rng)
        else:
            info["times"] = time_points(setup, scen, points, osl, chunk, device, rng)
        info["seconds"] = time.perf_counter() - start
        del setup
        torch.accelerator.empty_cache()
    return result


# ============================================================================
# Report
# ============================================================================


def point_cost(t: dict, mixed: bool) -> PointCost:
    pre = t["prefill_mixed_ms"] if mixed else t["prefill_alone_ms"]
    steps, riders = (t["prefill_steps"], t["riders"]) if mixed else (0.0, 0.0)
    return PointCost(pre, steps, riders, t["step_ms"], t["running"])


def report(name: str, scen: dict, res: dict, osl: int, base: str, other: str) -> list:
    """Compare predicted and measured throughput ratios of ``other`` over ``base``.

    The prediction starts from the measured run of ``other``, keeps its
    non-attention time per request and swaps in the attention time of ``base``.
    "alone" times prefill chunks on their own; "mixed" times them together with
    the decodes that share their steps in the bench run.
    """
    runs = scen["runs"]
    if base not in runs or other not in runs:
        return []
    tb, to = res["backends"].get(base, {}), res["backends"].get(other, {})
    if "times" not in tb or "times" not in to:
        return []
    e2e = {b: {(r[0], r[1]): r for r in runs[b]["results"]} for b in (base, other)}
    rows = []
    lines = [
        f"\n### {name}: {scen['description']}",
        "",
        f"| ISL | conc | attn ms/req {base} (alone / mixed) | {other} (alone / mixed)"
        " | e2e tok/s ratio | predicted alone / mixed | winner |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in sorted(e2e[base], key=lambda k: (k[0], k[1])):
        if key not in e2e[other]:
            continue
        rb, ro = e2e[base][key], e2e[other][key]
        isl, c = key
        ab, ao = rb[5] or 1.0, ro[5] or 1.0
        same_accept = abs(ab - ao) / max(ab, ao) < 0.05
        a = (ab + ao) / 2
        t = f"{isl}x{c}"
        wall_o = osl / ro[2] * 1000  # ms of serving time per request
        e2e_ratio = ro[2] / rb[2]
        row = dict(isl=isl, c=c, e2e_ratio=e2e_ratio, same_accept=same_accept)
        cells = {}
        for model in ("alone", "mixed"):
            attn_b = point_cost(tb["times"][t], model == "mixed").per_request_ms(osl, a)
            attn_o = point_cost(to["times"][t], model == "mixed").per_request_ms(osl, a)
            row[f"pred_{model}"] = (max(wall_o - attn_o, 0.0) + attn_b) / wall_o
            row[f"win_{model}"] = other if attn_o < attn_b else base
            cells[model] = (attn_b, attn_o)
        row["win_e2e"] = other if e2e_ratio > 1 else base
        rows.append(row)
        note = "" if same_accept else f" (accept {ab:.2f} vs {ao:.2f})"
        agree = "agree" if row["win_e2e"] == row["win_mixed"] else "DIFFER"
        lines.append(
            f"| {isl} | {c} | {cells['alone'][0]:.0f} / {cells['mixed'][0]:.0f}"
            f" | {cells['alone'][1]:.0f} / {cells['mixed'][1]:.0f}"
            f" | {e2e_ratio:.2f} | {row['pred_alone']:.2f} / {row['pred_mixed']:.2f}"
            f" | {agree}{note} |"
        )
    print("\n".join(lines))
    return rows


def print_lookup_table(name: str, res: dict, atol: float) -> None:
    """Per-batch winners of the quick grid among correct backends."""
    bes = {
        b: v["quick"]
        for b, v in res["backends"].items()
        if v.get("max_abs_err", math.inf) < atol
    }
    if not bes:
        return
    print(f"\n### {name}\n")
    print("| batch | " + " | ".join(bes) + " | best |")
    print("|---" * (len(bes) + 2) + "|")
    labels = list(next(iter(bes.values())))
    for label in labels:
        ms = {b: t[label] for b, t in bes.items() if label in t}
        best = min(ms, key=ms.get)
        cells = " | ".join(f"{ms[b]:.3f}" if b in ms else "-" for b in bes)
        print(f"| {label} | {cells} | {best} |")
    secs = ", ".join(f"{b} {v['seconds']:.0f}s" for b, v in res["backends"].items())
    print(f"\nSetup + correctness + timing: {secs}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scenarios", nargs="*", help="default: all")
    parser.add_argument(
        "--backends",
        nargs="*",
        help="candidate backends (default: the backends served in the scenario)",
    )
    parser.add_argument("--num-layers", type=int, default=2)
    # bf16 output against fp32 is off by up to ~2e-2 (an ulp or so at |o| ~ 2-4);
    # the indexing bugs this guards against give errors of 0.4 and up.
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base", default="ROCM_ATTN")
    parser.add_argument("--other", default="ROCM_AITER_UNIFIED_ATTN")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="time a small startup-sized grid and print a per-batch lookup table",
    )
    parser.add_argument("--output-json")
    parser.add_argument(
        "--report-only", help="skip benchmarking and report from this JSON"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    osl = cfg["osl"]
    chunk = cfg["max_num_batched_tokens"]
    names = args.scenarios or list(cfg["scenarios"])
    if args.report_only:
        with open(args.report_only) as f:
            results = json.load(f)
    else:
        results = {}
        for name in names:
            results[name] = run_scenario(name, cfg["scenarios"][name], args, osl, chunk)
            if args.output_json:
                with open(args.output_json, "w") as f:
                    json.dump(results, f, indent=1)
    if args.quick:
        for name in names:
            print_lookup_table(name, results[name], args.atol)
        return
    all_rows = []
    for name in names:
        if name in results:
            all_rows += report(
                name, cfg["scenarios"][name], results[name], osl, args.base, args.other
            )
    rows = [r for r in all_rows if r["same_accept"]]
    if rows:
        print(f"\n{len(rows)} points with equal acceptance:")
        for model in ("alone", "mixed"):
            agree = sum(r["win_e2e"] == r[f"win_{model}"] for r in rows)
            err = [abs(math.log(r[f"pred_{model}"] / r["e2e_ratio"])) for r in rows]
            print(
                f"  {model:5s}: winner agrees on {agree}/{len(rows)},"
                f" median |log(pred/e2e)| = {statistics.median(err):.3f}"
            )
        inside = sum(
            min(r["pred_alone"], r["pred_mixed"]) * 0.9
            <= r["e2e_ratio"]
            <= max(r["pred_alone"], r["pred_mixed"]) * 1.1
            for r in rows
        )
        print(f"  e2e ratio within [alone, mixed] +-10%: {inside}/{len(rows)}")


if __name__ == "__main__":
    main()
