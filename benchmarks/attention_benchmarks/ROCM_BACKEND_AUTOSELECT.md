# Experiment: picking the ROCm attention backend with microbenchmarks

This note records an experiment for an `AUTO` attention backend on ROCm. The
idea: at engine startup, build each attention backend that supports the model,
check that it computes the right answer on the real KV pool, time a few batch
shapes, and keep the fastest correct one in a lookup table.

The question was whether kernel microbenchmarks predict end-to-end serving
results well enough to make that choice. They do. The experiment also found two
kernel bugs that made one backend look much worse than it is.

Script: [`rocm_backend_autoselect.py`](rocm_backend_autoselect.py). Measured
end-to-end runs: [`configs/rocm_autoselect_e2e.yaml`](configs/rocm_autoselect_e2e.yaml).

## Setup

- MI355X, 1 GPU, vLLM 0.30.1rc1.dev558+g0cbac6cd1, aiter 0.1.23, model runner V2.
- Candidates: `ROCM_ATTN`, `ROCM_AITER_UNIFIED_ATTN` (UA below), and `TRITON_ATTN`.
- End to end: `vllm bench serve` with OSL 1024 (`--ignore-eos`), greedy decoding,
  `--max-num-batched-tokens 16384`, and `--gpu-memory-utilization 0.85`. Each run
  forces one backend.
- Scenarios:

  | scenario | model | kind | spec decode |
  | --- | --- | --- | --- |
  | `qwen3_8b_nospec` | Qwen3-8B, YaRN x4 | dense | none |
  | `qwen3_32b_eagle3` | Qwen3-32B | dense | EAGLE3, 3 tokens |
  | `qwen3_next_nospec` | Qwen3-Next-80B-A3B | hybrid | none |
  | `qwen3_next_mtp3` | Qwen3-Next-80B-A3B | hybrid | MTP, 3 tokens |
  | `qwen3_8_27b_nospec` | Qwen3.8-27B | hybrid | none |
  | `qwen3_6_35b_a3b_nospec` | Qwen3.6-35B-A3B | hybrid MoE | none |

## Method

For each backend the script follows what the serving run did:

- **Same setup as serving.** It uses the backend's own block size and KV cache
  layout, and a KV pool of the served size. Hybrid models use mamba-aligned
  blocks of 544–1088 tokens. Uniform decode batches replay a full CUDA graph
  whose metadata has `max_seq_len = max_model_len`, as at capture time.
- **Correctness gate.** Each backend runs a mixed batch and a graph-replayed
  decode batch, with blocks drawn from the upper half of the pool, and is
  compared with an fp32 reference. A backend whose max abs error is above
  `--atol` (default `5e-2`) is dropped. Correct kernels give 0.008–0.021 in bf16;
  the bug below gives 0.5–1.1.
- **Cost per request.** At each serving point (ISL, concurrency), one request
  costs:
    - the attention time of its prefill steps, plus
    - its share of the pure decode steps: `osl / accept_len` steps, less the
      decode tokens already produced during prefill steps, spread over all
      running requests.

  Running requests are capped by KV capacity: `kv_tokens // (isl + osl)`.
- **Mixed steps.** Prefill chunks share steps with running decodes ("riders").
  The rider count follows how `vllm bench serve` admits requests. In the first
  wave, the k-th group of prefills shares its steps with `k * n` decodes. In
  later waves, every other running request is decoding. The script times the
  real count, not its mean, because some kernels slow down sharply at particular
  batch sizes (see below). "alone" times prefill chunks without riders; "mixed"
  times them with riders.
- **Prediction.** The measured UA run gives the wall time per request, `W`.
  The predicted throughput ratio UA/ROCM_ATTN is
  `(max(W - attn_UA, 0) + attn_ROCM) / W`. This assumes the non-attention time
  is the same for both backends.

```bash
# Full comparison against the measured runs (about 10 min per scenario).
python rocm_backend_autoselect.py --scenarios qwen3_8b_nospec \
    --backends ROCM_ATTN ROCM_AITER_UNIFIED_ATTN TRITON_ATTN --output-json out.json
python rocm_backend_autoselect.py --report-only out.json

# Startup-sized grid and per-batch lookup table.
python rocm_backend_autoselect.py --quick --backends ROCM_ATTN ROCM_AITER_UNIFIED_ATTN TRITON_ATTN
```

## Results: the microbenchmark predicts the end-to-end winner

The test set is the 65 points where both runs have the same acceptance length:

| model | winner agrees with e2e | median \|log(pred / e2e)\| |
| --- | --- | --- |
| alone (prefill timed without riders) | 63/65 | 0.160 |
| mixed (prefill timed with riders) | 63/65 | 0.050 |

The two misses are both 1k, concurrency 1, where attention is a small share of
the time and the gap is small:

- Qwen3-8B: e2e ratio 0.95, predicted 1.04.
- Qwen3-Next + MTP: e2e ratio 0.85, predicted 1.16.

At those points, effects outside attention (graph capture, block-size-dependent
kernels) decide the winner.

Selected rows. The ratio is UA throughput over ROCM_ATTN throughput. The full
tables are in the script's output.

| scenario | ISL | conc | attn ms/req ROCM_ATTN (alone / mixed) | UA (alone / mixed) | e2e ratio | predicted (alone / mixed) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 8k | 64 | 319 / 995 | 205 / 206 | 1.96 | 1.30 / 3.05 |
| Qwen3-8B | 64k | 8 | 6569 / 27089 | 2479 / 2634 | 4.78 | 1.96 / 6.76 |
| Qwen3-8B | 128k | 8 | 24610 / 117413 | 7015 / 7637 | 6.17 | 2.63 / 11.18 |
| Qwen3-32B + EAGLE3 | 64k | 8 | 79106 / 312093 | 11365 / 11446 | 17.15 | 4.74 / 17.62 |
| Qwen3-Next | 64k | 16 | 6351 / 19465 | 693 / 804 | 9.40 | 3.29 / 8.54 |
| Qwen3-Next + MTP | 64k | 16 | 4491 / 50056 | 1201 / 1233 | 17.97 | 2.12 / 17.63 |
| Qwen3.8-27B | 128k | 8 | 42094 / 103722 | 4942 / 5229 | 8.96 | 3.89 / 8.67 |
| Qwen3.6-35B-A3B | 64k | 8 | 9248 / 14886 | 590 / 683 | 7.38 | 4.90 / 7.39 |

Timing prefill alone underestimates ROCM_ATTN's cost by up to 11×. Almost all
of the gap comes from mixed steps, which led to the first bug below.

## Finding 1: `context_attention_fwd` runs mixed batches on part of the GPU

`ROCM_ATTN` runs prefills and spec-decode verifies through
`context_attention_fwd` in `vllm/v1/attention/ops/prefix_prefill.py`.
It has two problems that only appear when short requests share a batch with a
long one.

1. **Grid order.** The grid was `(batch, head, query_blocks)`, with the batch on
   the fastest axis. MI300/MI355 deal workgroups round-robin over 8 XCDs, so
   the programs of one request have linear ids `i, i + b, i + 2b, ...` for
   batch size `b`, and with even `b` they reach only some of the XCDs.
   Decodes (`query_len == 1`) return at once, so a prefill batched with decodes
   runs on part of the GPU. The slowdown follows the power of two in `b`
   (8k chunk on a 16k prefix plus `b - 1` decodes):

   | b | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 12 | 16 |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | ms | 9.1 | 18.4 | 10.4 | 36.8 | 10.3 | 20.8 | 10.3 | 80.2 | 40.3 | 143.9 |

   Putting the batch on the slowest axis, `(head, query_blocks, batch)`, fixes
   this.
2. **Empty query blocks.** The grid has `cdiv(max_query_len, BLOCK_M)` query
   blocks for every request. A 4-token verify batched with a 16k-token chunk
   gets 128 blocks; 127 of them have no rows but still loop over the whole
   context. Returning early when `BLOCK_M * start_m >= query_len` fixes this.

Kernel time for an 8k chunk on a 16k prefix, Qwen3-8B shapes, with 7 other
requests in the batch:

| batch | before | grid fix | grid fix + early return |
| --- | --- | --- | --- |
| chunk alone | 9.1 ms | 9.1 ms | 9.1 ms |
| + 1 decode | 18.6 ms | 9.1 ms | 9.2 ms |
| + 7 decodes | 73.5 ms | 9.1 ms | 9.2 ms |
| + 7 verifies (q=2, kv 1k) | 73.7 ms | 12.2 ms | 9.2 ms |
| + 7 verifies (q=4, kv 32k) | 148.7 ms | 117.0 ms | 11.6 ms |

The outputs are bit-identical to the original kernel. With the fix,
`tests/kernels/attention/test_prefix_prefill.py` gives the same results as
before.

Startup-grid entry "16k chunk on a 32k prefix with 7 decodes", in ms for all layers:

| scenario | before | grid fix | grid fix + early return | UA |
| --- | --- | --- | --- | --- |
| Qwen3-8B | 11763 | 1440 | – | 341 |
| Qwen3-32B + EAGLE3 | 47128 | 34326 | 5320 | (fails gate) |
| Qwen3-Next | 4064 | 533 | – | 150 |

(–: not measured; the early return does not matter without spec decode.)

End to end (Qwen3-8B, `--max-model-len 131072`, output tok/s). Without spec
decode only the grid fix matters, because decodes return at once. With EAGLE3
both fixes are applied.

| spec | ISL | conc | ROCM_ATTN before | ROCM_ATTN fixed | UA |
| --- | --- | --- | --- | --- | --- |
| none | 8k | 16 | 1052 | 1339 | 1630 |
| none | 8k | 64 | 1357 | 1949 | 2666 |
| none | 64k | 8 | 50.4 | 108.8 | 241.0 |
| none | 64k | 16 | 45.2 | 110.0 | 261.3 |
| EAGLE3 | 8k | 16 | 219 | 485 | 1780 |
| EAGLE3 | 8k | 64 | 106 | 525 | 3288 |
| EAGLE3 | 64k | 8 | 11.4 | 38.5 | 122.9 |
| EAGLE3 | 64k | 16 | 6.4 | 40.1 | 188.4 |

Concurrency-1 points do not change (for example, 64k c1: 91.9 vs 92.2
without spec, and 6.5 vs 6.4 with EAGLE3).

The fixes speed up ROCM_ATTN by 1.3–6×, and its mixed steps now cost about
as much as its standalone prefill. UA is still 1.2–6× faster, for two reasons:

- ROCM_ATTN's prefill kernel is 2–4× slower than UA's.
- ROCM_ATTN runs spec-decode verifies through `context_attention_fwd`. With
  `BLOCK_M = 128` and one program per query head, a 4-token verify wastes most
  of each block. This path is 3–14× slower than TRITON_ATTN's decode; the
  TPOT at 64k c1 with EAGLE3 is 150 ms.

## Finding 2: UA computes wrong results with some KV pool sizes

With the KV pool served for Qwen3-32B (657472 tokens, about 2.5 GiB per layer
for K and V), UA's output is off by 0.5–1.1. This holds for 32 or 64 query
heads and for 1 or 4 query tokens. With a 1.6M-token pool (6.2 GiB per layer)
the same batches pass. This matches the aiter buffer-load bug fixed in
[ROCm/aiter#6661](https://github.com/ROCm/aiter/pull/6661), which affects
per-layer caches of 2–4 GiB.

The end-to-end UA run of this scenario is affected too: its EAGLE3 acceptance
length drops from 2.30 to 1.64 at 1k c64, and from 2.10 to 1.66 at 8k c16. So
a startup check on the real pool catches a silent correctness bug that no
throughput benchmark would.

## Startup cost and the lookup table

`--quick` times 8–9 batch shapes. It takes 4–16 s per backend, including setup
and the correctness check. A real `AUTO` would also share the work with
warmup and graph capture.

Per-shape winners among correct backends, with the current kernels:

- Hybrid models: UA wins prefill, mixed batches, and long-context decode. Triton
  wins short-context decode at batch 1–16. For example, Qwen3-Next at bs 1,
  ctx 4k: 0.21 ms vs 0.47 ms.
- Dense Qwen3-8B: UA wins every shape.
- Dense Qwen3-32B + EAGLE3: UA fails the gate. ROCM_ATTN wins prefill, and
  Triton wins decode by 3–14×.

## Limitations and next steps

- **Per-shape switching is limited.** Backends differ in block size (hybrid:
  UA 576–1088, others 544–1056) and KV layout. One engine can only switch
  between backends that share both. In practice the table picks one backend
  per model, and per-shape switching is limited to compatible pairs such as
  ROCM_ATTN and TRITON_ATTN.
- **Only attention is timed.** Where attention is a small share of the time
  (short prompts, low concurrency), the predicted ratio is within noise and
  other effects decide.
- **The serving-point model is tied to `vllm bench serve`.** The rider mix
  assumes its admission pattern. A real `AUTO` would weight shapes by the
  scheduler's own batch statistics, or simply by the startup grid.
- **Coverage.** The experiment used MI355X only, bf16 KV only, head size 128,
  and no sliding window.

Recommendation: ship the kernel fixes (Finding 1) and the aiter fix (Finding 2)
first. With them, the global backend order can be decided from data like the
above. A startup correctness check on the real KV pool is cheap and catches
pool-size bugs. An `AUTO` mode that times kernels is worth adding once there
is more than one competitive backend per shape.
