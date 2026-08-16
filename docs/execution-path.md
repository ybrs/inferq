# Phase 1 execution path

The engine consumes a local Hugging Face Qwen3-Coder-Next directory containing
`config.json`, `tokenizer.json`, and either `model.safetensors` or a sharded
`model.safetensors.index.json`. The loader memory-maps every shard, enumerates
the tensors, and validates every required name, shape, and floating-point dtype
before inference. The official checkpoint stores each routed expert separately
as `experts.<id>.{gate,up,down}_proj.weight`; inference loads only the selected
experts for a token.

Inference is CPU-only and batch-size one:

```text
token IDs -> embedding -> 48 decoder layers -> final RMSNorm -> LM head -> logits
                              |
                              +-- 3 linear-attention layers: causal depthwise conv
                              |   and recurrent Gated Delta Rule state
                              +-- 1 full-attention layer: Q/K norm, partial RoPE,
                              |   gated GQA, and persistent KV state
                              +-- every layer: top-10 routed experts plus gated
                                  shared expert
```

`Runtime::prefill` resets and fills KV, convolution, and DeltaNet recurrent
state. `Runtime::decode` accepts exactly one previously sampled token and
updates that state. Greedy decoding is selected with temperature zero; seeded
temperature, top-k, top-p, and min-p sampling are also available.

For Qwen3.6 models that declare one auxiliary MTP layer, the quantized runtime
also has an opt-in speculative path. It feeds the previous target hidden state
and current token embedding through the auxiliary predictor, drafts up to the
requested number of greedy tokens, verifies them in one target-model batch,
accepts the matching prefix, and restores/replays target recurrent state after
a rejection. The MTP KV cache is then synchronized from authoritative target
hidden states. Ordinary generation does not execute this layer. See
[speculative-decoding.md](speculative-decoding.md) for the exact data flow,
current restrictions, correctness evidence, and measured performance.

The scalar Delta Rule, causal convolution, attention loop, and expert selection
are intentionally readable reference implementations. Candle supplies tensor
storage, dtype conversion, and matrix multiplication. Because Candle CPU does
not provide BF16 matmul, projections promote inputs and weights to F32 for the
operation and cast outputs back to the checkpoint dtype. Routing
traces are optional JSONL and contain the absolute token index, token ID, layer,
expert IDs, normalized route weights, and optionally full router logits.
In the fully resident GGUF path, compatible routed and shared expert gate/up
matrices are row-concatenated without dequantization. Their row results remain
identical while one compressed kernel launch produces both projections.

## End-to-end validation

The official BF16 checkpoint at revision
`a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb` was exercised on 2026-08-14 on an
Intel i7-6700 (4 cores/8 threads, 62 GiB RAM). The loader validated all 74,391
BF16 tensors. With raw prompt `a` (token ID 64), greedy Rust inference produced
token 284 (` =`). The independently quantized
`Qwen3-Coder-Next-UD-Q4_K_M.gguf` produced the same token with llama.cpp. A
two-token Rust run produced `[284, 526]` (` = int`), exercising the persistent
state decode path for the second token.

These are correctness smoke-test timings, not benchmarks: the first cold Rust
prefill took 57.8 seconds. After llama.cpp evicted the SafeTensor pages, the
two-token run took 117.0 seconds for prefill and 63.9 seconds for one decode
pass. llama.cpp spent 277.7 seconds loading/repacking the Q4 model and 49.6
seconds evaluating its token. Commands used the default thread settings with
eight logical CPUs visible.

## Commands

```bash
cargo run --release --bin inspect -- --model /models/Qwen3-Coder-Next \
  --tensors

cargo run --release --bin infer -- --model /models/Qwen3-Coder-Next \
  --chat --prompt "Write a Rust parser" --max-new-tokens 128

cargo run --release --bin compare -- --model /models/Qwen3-Coder-Next \
  --tokens reference-artifacts/tokens.json \
  --reference-logits reference-artifacts/logits.json \
  --dump-logits rust-logits.bin

cargo run --release --bin routing_trace -- --model /models/Qwen3-Coder-Next \
  --prompt "Fix this test" --output routing.jsonl

cargo run --release --bin bench -- --model /models/Qwen3-Coder-Next
```

The profiling artifact schema, stable micro-cases, cache-state contract, and
`perf stat` wrapper are documented in [profiling.md](profiling.md).

Generate reference artifacts with `python/reference_logits.py`. Set
`QWEN_MODEL_DIR` and `QWEN_REFERENCE_DIR` to include a full-model differential
comparison in `scripts/validate.sh`.

## Current format boundary

Phase 1 uses the permitted SafeTensors path so the architecture can be made
correct and observable first. GGUF v3 metadata and tensor inventories are also
parsed and validated; this has been exercised against the local 46 GB
`Qwen3-Coder-Next-UD-Q4_K_M.gguf` (F32/Q4K/Q5K/Q6K/Q8_0, 843 tensors).
Direct GGUF matrix execution and selected-expert range loading are now
implemented as the first Stage 2B boundary; see
[quantized-execution.md](quantized-execution.md). Direct routed MoE and one
complete layer of each type now match their BF16 comparison gates, including a
two-step full-attention KV-cache comparison. The first complete GGUF runtime is
also operational and matches the two-token BF16 regression. It currently uses
the Hugging Face directory for config/tokenizer assets. Its interactive mode
keeps loaded weights and sequence state alive across turns and can write a
resumable per-layer expert census. On the 62 GiB target, its explicit full
expert residency mode pins all compressed expert matrices at 47.2 GiB process
RSS and measured 1.61 decode token/s on an unseen prompt. A subsequent
128-token chat generation sustained 1.55 token/s over 127 decode passes while
streaming byte-safe text and remaining at 47.263 GiB RSS. Lower-memory partial
residency remains workload-dependent and experimental on the rotational disk.
