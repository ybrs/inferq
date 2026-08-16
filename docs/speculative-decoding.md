# Qwen3.6 MTP speculative decoding

Qwen3.6-35B-A3B includes one auxiliary multi-token-prediction (MTP) transformer
block in the same GGUF as the target model. Inferq can use that block as an
in-model draft predictor with `--speculative-mtp N`. The implementation is a
correct, observable greedy baseline. It is not faster than ordinary decoding
yet because the existing quantized target verifier does not efficiently reuse
weights across its small token batch.

## Try it

Build the host-native binaries first:

```bash
CARGO_TARGET_DIR=target-native \
RUSTFLAGS='-C target-cpu=native' \
cargo build --release --bin gguf_infer --bin gguf_bench
```

Then run the Q4 model with one draft token per verification boundary:

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

Use `--speculative-mtp 0`, or omit the option, for ordinary target-only
generation. `gguf_infer` is greedy; callers of the Rust runtime must likewise
use temperature zero for speculation. The mode does not support
`--routing-trace`, `--routing-census`, or
`--routing-census-resume`. A draft length of 1 is the least costly current
setting; increasing it is useful for development experiments but reduced
throughput in the qualified benchmark.

The final report includes drafted and accepted counts, acceptance rate,
verification passes and rows, rollback/replay counts, and time spent drafting,
verifying, and resynchronizing the MTP state. `gguf_bench` records the same
fields in the JSONL `speculative` object; its output schema is version 4.

## Execution and state semantics

The predictor matches the architecture encoded by the Qwen config and GGUF:

```text
current token embedding -- RMSNorm --+
                                      +-- concatenate -- eh_proj -- block 40
previous target hidden -- RMSNorm ---+                         |
                                                               +-- shared final norm/head -- draft logits
```

Block 40 is a full-attention plus MoE layer. It shares the target token
embedding, final RMSNorm, and LM head; it has its own input norms, `eh_proj`,
attention KV state, and experts. `mtp_use_dedicated_embeddings = false` is
required. Inferq currently supports exactly one MTP layer.

At a speculation boundary the runtime:

1. checkpoints the target model's DeltaNet recurrent/conv state and full
   attention positions;
2. generates up to `N` draft tokens sequentially with the MTP block;
3. evaluates the pending target token plus the draft tokens in one target
   forward pass;
4. accepts the longest prefix whose greedy target tokens match the drafts;
5. on rejection, restores the target checkpoint and replays only the
   authoritative accepted/replacement tokens; and
6. resynchronizes the MTP cache from the target model's normalized hidden
   states.

Output callbacks are transactional: rejected draft bytes are never emitted.
Stop tokens are accepted only from the authoritative target result. This keeps
the externally visible token stream identical to ordinary greedy generation.

## Correctness evidence

The qualified Q4 artifact produced the llama.cpp-matched eight-token prefix
`[8160, 579, 264, 7047, 1817, 25, 271, 16]` in both ordinary and speculative
mode. A separate 32-token run accepted 19 of 22 drafts and exercised two
rollback paths while still matching every target-only token.

The sustained 128-token benchmark below produced exactly the same complete
token-ID sequence at draft lengths 0, 1, 2, and 3. This covers both the
all-accepted path and rejection/restore/replay behavior.

## Fully resident Q4 result

The benchmark host was an Intel i7-6700 with four physical cores, 62.6 GiB
RAM, four Candle and Rayon threads, and a native release build. The workload
was a 25-token rendered chat prompt followed by 128 greedy output tokens.
All 41 blocks' experts were warm and resident, with no evictions or physical
inference reads.

| Max drafts | Decode tok/s | Decode time | Accepted / drafted | Acceptance | Verify passes / rows | Replay tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | **8.10** | 15.685 s | 0 / 0 | n/a | 0 / 0 | 0 |
| 1 | 6.31 | 20.126 s | 59 / 67 | 88.1% | 68 / 135 | 8 |
| 2 | 5.14 | 24.703 s | 76 / 100 | 76.0% | 51 / 151 | 30 |
| 3 | 4.63 | 27.444 s | 85 / 126 | 67.5% | 42 / 168 | 39 |

Draft length 1 spent 1.477 seconds in the MTP block, 16.454 seconds in target
verification, and 0.504 seconds resynchronizing MTP state. Its final RSS was
about 21.13 GiB versus 21.07 GiB for the control. The small runtime delta is
mostly the approximately 60 MiB target-state checkpoint; the comparison keeps
the 0.84 GiB auxiliary Q4 expert block resident in every run.

## Why it is slower and what comes next

The target verifier receives two to four rows, but the current quantized
matrix and MoE implementation remains effectively row-oriented. In particular,
routed experts are evaluated token by token. The verifier therefore rereads or
revisits large quantized weights without getting a proportional small-batch
GEMM benefit. Its measured per-row gain is only about 6%, while rejected target
rows, rollback replay, MTP compute, and synchronization add work.

The critical next step is a true small-batch quantized verification kernel:

- group routed work by expert across all verification rows;
- dequantize or stream each selected expert tile once and apply it to every row
  assigned to that expert;
- add multi-row Q4/Q8 matrix kernels for the dense projections and LM head;
  and
- benchmark draft confidence gating after verification is materially cheaper.

Only after this change can fewer target passes translate into fewer target
weight reads. Confidence thresholds and cheaper recurrent checkpoints can
reduce waste afterward, but they cannot fix the present verifier bottleneck on
their own.

Architecture references:

- [Qwen3.6-35B-A3B config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)
- [llama.cpp Qwen3.5/3.6 MoE graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35moe.cpp)
- [llama.cpp speculative-decoding documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
