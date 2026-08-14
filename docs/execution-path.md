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

The scalar Delta Rule, causal convolution, attention loop, and expert selection
are intentionally readable reference implementations. Candle supplies tensor
storage, dtype conversion, and matrix multiplication. Because Candle CPU does
not provide BF16 matmul, projections promote inputs and weights to F32 for the
operation and cast outputs back to the checkpoint dtype. Routing
traces are optional JSONL and contain the absolute token index, token ID, layer,
expert IDs, normalized route weights, and optionally full router logits.

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
complete quantized linear-attention layer now match their BF16 comparison
gates. A complete GGUF model runtime and tokenizer construction from GGUF
metadata are not yet implemented. Full-attention layers, embedding, LM head,
and whole-model state assembly remain before end-to-end GGUF generation.
