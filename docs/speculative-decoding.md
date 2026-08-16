# Qwen3.6 MTP speculative decoding

Qwen3.6-35B-A3B includes one auxiliary multi-token-prediction (MTP) transformer
block in the same GGUF as the target model. Inferq can use it as an opt-in
greedy draft predictor with `--speculative-mtp N`. Speculation remains disabled
by default: the optimized draft=1 path is now closer to break-even, but it is
still slower than target-only decoding on the qualified host.

## Try it

Build host-native binaries first:

```bash
CARGO_TARGET_DIR=target-native \
RUSTFLAGS='-C target-cpu=native' \
cargo build --release \
  --bin gguf_infer --bin gguf_bench --bin gguf_verify_bench
```

Then run the fully resident Q4 model with one draft token per verification
boundary:

```bash
MODEL_ROOT=/data/projects/localllm/models/Qwen3.6-35B-A3B

CANDLE_NUM_THREADS=4 \
RAYON_NUM_THREADS=4 \
./target-native/release/gguf_infer \
  --model "${MODEL_ROOT}/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  --tokenizer-model "${MODEL_ROOT}" \
  --chat \
  --prompt 'Write a detailed Rust implementation of a thread-safe LRU cache with tests.' \
  --max-new-tokens 128 \
  --expert-cache-mib 46000 \
  --warmup-all-experts \
  --speculative-mtp 1
```

Use `--speculative-mtp 0`, or omit it, for target-only generation. The optional
`--speculative-mtp-min-margin 0.3` gate falls back to a one-row target pass when
the MTP top-1/top-2 raw-logit margin is below 0.3. It improved this particular
workload, but remains experimental and is not enabled automatically.

Speculation requires greedy sampling and does not support routing traces or
censuses. The final report includes draft acceptance, gated proposals,
verification rows, rollback/replay counts, and draft, verification,
checkpoint, restore, replay, and MTP-resynchronization time. `gguf_bench`
records the same data in JSONL schema version 6.

## Bounded thinking

Chat generation preserves the model's normal thinking behavior when neither
of these options is supplied:

- `--no-thinking` renders the assistant prefix as
  `<think>\n\n</think>\n\n`, matching the Qwen template's non-thinking form.
- `--thinking-budget N` starts in normal thinking mode. If the tokenizer's
  complete `</think>` token sequence has not been committed after `N` generated
  thinking tokens, Inferq injects the tokenizer's complete
  `</think>\n\n` sequence into the output and evaluates every injected token
  through the target model before answer generation continues.

The budget is per assistant turn, including in `--interactive` mode. Only
authoritative output tokens count: rejected MTP drafts are neither emitted nor
charged to the budget. A real two-turn Q4 smoke test with
`--thinking-budget 2 --speculative-mtp 1` force-closed both turns independently,
kept 100% of the exercised drafts accepted, and continued with `Hi!` and
`Bye!` after the evaluated closures.

Example:

```bash
CANDLE_NUM_THREADS=4 RAYON_NUM_THREADS=4 \
./target-native/release/gguf_infer \
  --model "${MODEL_ROOT}/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  --tokenizer-model "${MODEL_ROOT}" \
  --interactive --chat \
  --thinking-budget 64 \
  --max-new-tokens 256 \
  --expert-cache-mib 46000 --warmup-all-experts \
  --speculative-mtp 1
```

`--no-thinking` and `--thinking-budget` are mutually exclusive and require a
Qwen chat template that declares thinking support.

## Execution and state semantics

The MTP predictor matches the architecture encoded by the Qwen config and
GGUF:

```text
current token embedding -- RMSNorm --+
                                      +-- concatenate -- eh_proj -- block 40
previous target hidden -- RMSNorm ---+                         |
                                                               +-- shared final norm/head -- draft logits
```

At a speculation boundary the runtime drafts up to `N` tokens, evaluates the
pending target token plus the drafts in one target pass, accepts the longest
greedy-matching prefix, and restores/replays target recurrent state after a
rejection. Output is transactional: rejected draft bytes are never emitted.
Stop tokens come only from the authoritative target result.

After every verification, MTP state is truncated to the pre-draft boundary and
rebuilt from the committed tokens and their authoritative target hidden rows.
Retaining predictor state made from approximated draft hidden rows can preserve
the target token sequence while silently changing later MTP predictions, so it
is deliberately not treated as synchronized state. Forced thinking-closure
tokens use the same authoritative target-plus-MTP path, keeping target state,
MTP position, and the interactive pending token aligned.

## Reproducible verifier benchmark

`gguf_verify_bench` holds one deterministic prefetched context, keeps every
expert resident, and evaluates the same fixed tokens at K=1,2,4,8. It reports
total and per-row stage time, checkpoint/restore/rejection replay, and all five
expert-reuse metrics for every layer.

```bash
MODEL_ROOT=/data/projects/localllm/models/Qwen3.6-35B-A3B

CANDLE_NUM_THREADS=4 \
RAYON_NUM_THREADS=4 \
./target-native/release/gguf_verify_bench \
  --model "${MODEL_ROOT}/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  --tokenizer-model "${MODEL_ROOT}" \
  --batch-sizes 1,2,4,8 \
  --repetitions 3 \
  --expert-cache-mib 46000 \
  --output verifier-q4.json
```

The default deterministic verification IDs are
`[8160,579,264,7047,1817,25,271,16]`. The JSON contains, per layer and K,
token-to-expert assignments, unique selected experts, duplicate assignment
rate, average rows per selected expert, and the maximum rows assigned to one
expert.

## What changed

The routed MoE small-batch path is now expert-major. It computes all routes,
groups `(row, route weight)` records by expert, gathers each expert's input
rows, executes gate/up and down once for that group, stores each route result,
then performs the final weighted accumulation in original token/route order.
K=1 retains the original token-major path, and experts stay in their resident
compressed representation.

Measurement also isolated a separate dense-matrix problem. Candle's existing
quantized CPU matmul traversed a large quantized matrix once per input row on
this workload: the LM head nearly doubled from 17.4 ms at K=1 to 35.8 ms at
K=2. Inferq therefore adds a measured small-M path for Q4_K, Q5_K, Q6_K, and
Q8_0 matrices of at least 4 MiB. It quantizes the M input rows once, traverses
each compressed weight row once, applies that row to every input while it is
cache-hot, and transposes the output. The size threshold is important: using
the same path for the much smaller expert matrices regressed their stages by
18-42%, so routed experts continue to use Candle's grouped multi-row path.

## K=1/2/4/8 verification scaling

Qualified host: Intel i7-6700, four physical cores, four Candle/Rayon threads,
native release build, fully resident Q4_K_M, three repetitions.

| K | Before total | Before / row | After total | After / row | Per-row change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 122.09 ms | 122.09 ms | 121.20 ms | 121.20 ms | -0.7% |
| 2 | 227.23 ms | 113.61 ms | 189.88 ms | 94.94 ms | -16.4% |
| 4 | 428.74 ms | 107.18 ms | 335.20 ms | 83.80 ms | -21.8% |
| 8 | 807.39 ms | 100.92 ms | 595.30 ms | 74.41 ms | -26.3% |

After optimization, total stage times were:

| Stage | K=1 | K=2 | K=4 | K=8 |
| --- | ---: | ---: | ---: | ---: |
| DeltaNet projections | 25.04 ms | 36.41 ms | 60.91 ms | 106.45 ms |
| DeltaNet recurrence | 10.31 ms | 17.29 ms | 29.51 ms | 52.88 ms |
| Full attention | 10.00 ms | 15.87 ms | 27.56 ms | 51.40 ms |
| MoE router | 3.15 ms | 5.44 ms | 7.14 ms | 9.88 ms |
| MoE top-k | 0.48 ms | 0.86 ms | 1.62 ms | 3.30 ms |
| Routed expert gate/up | 18.99 ms | 33.81 ms | 66.07 ms | 119.40 ms |
| Expert activation | 1.36 ms | 2.51 ms | 5.08 ms | 8.74 ms |
| Routed expert down | 11.22 ms | 20.06 ms | 39.23 ms | 69.27 ms |
| Routed accumulation | 0.70 ms | 0.97 ms | 2.09 ms | 4.47 ms |
| Shared expert | 5.98 ms | 9.03 ms | 15.55 ms | 26.74 ms |
| Dense projections outside MoE | 43.29 ms | 62.84 ms | 105.25 ms | 186.80 ms |
| Final norm | 0.007 ms | 0.010 ms | 0.015 ms | 0.021 ms |
| LM head | 17.75 ms | 21.23 ms | 34.52 ms | 60.47 ms |

Checkpoint averaged 41.5-45.9 ms, restore 6.7-7.3 ms, and one-row rejection
replay 122.0-122.9 ms in the standalone harness. End-to-end draft=1 measured
0.52 seconds of checkpoint time over the whole 128-token run, so allocator and
process context materially affect the standalone checkpoint number.

Aggregating the benchmark's per-layer reuse records gives:

| K | Assignments | Unique experts | Duplicate rate | Rows / selected expert | Maximum rows / expert |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 320 | 320 | 0.0% | 1.000 | 1 |
| 2 | 640 | 506 | 20.9% | 1.265 | 2 |
| 4 | 1,280 | 867 | 32.3% | 1.476 | 4 |
| 8 | 2,560 | 1,261 | 50.7% | 2.030 | 8 |

The JSON report retains all 40 individual layer records rather than hiding
layer-to-layer variation behind these aggregates.

## End-to-end result

The workload is the same 25-token rendered chat prompt followed by 128 greedy
output tokens. All runs emitted the exact same complete target token sequence.

| Mode | Before tok/s | After tok/s | Decode time | Accepted / drafted | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target only | 8.10 | **8.09** | 15.699 s | 0 / 0 | n/a |
| MTP draft=1 | 6.31 | 7.17 | 17.722 s | 59 / 67 | 88.1% |
| MTP draft=2 | 5.14 | 6.09 | 20.844 s | 76 / 100 | 76.0% |
| MTP draft=3 | 4.63 | 5.64 | 22.511 s | 85 / 126 | 67.5% |
| MTP draft=1, margin 0.3 | n/a | 7.51 | 16.905 s | 58 / 68; 6 gated | 85.3% |

Draft=1 is substantially faster than the earlier implementation, but has not
crossed break-even: the ungated result is 11.4% slower than target-only, and
the measured margin gate is still 7.1% slower. Consequently speculation stays
opt-in, and there is no reason yet to tune draft lengths 2 or 3.

For ungated draft=1, target verification is the largest measured component at
14.15 seconds. The avoidable work separating it from target-only includes
1.49 seconds in the MTP block, 1.01 seconds replaying rejected prefixes, 0.52
seconds checkpointing, 0.44 seconds resynchronizing, and 0.06 seconds restoring.
At K=2 the largest verifier stages are dense non-MoE projections (62.84 ms),
routed gate/up plus down (53.88 ms), LM head (21.23 ms), DeltaNet recurrence
(17.29 ms), and full attention (15.87 ms). These measurements isolate several
contributors; they do not support declaring any single remaining kernel the
sole bottleneck.

Architecture references:

- [Qwen3.6-35B-A3B config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)
- [llama.cpp Qwen3.5/3.6 MoE graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35moe.cpp)
- [llama.cpp speculative-decoding documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
