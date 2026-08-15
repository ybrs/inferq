# Repository Agent Guide

## Mission and scope

This repository builds a narrow, inspectable CPU inference engine for
Qwen3-Coder-Next. It targets x86-64 Linux, batch size one, and one active
sequence. Prefer model-specific code over general framework abstractions.

Read `phase-1-qwen-rust-inference.md` before changing the Phase 1 runtime and
`qwen-cpu-inference-target-architecture.md` before making architectural or
performance decisions.

## Correctness rules

- Correctness and observability come before performance. Do not optimize an
  operation until it has a readable reference implementation and tests.
- Match the published Qwen3-Next inference semantics. Record the source and
  rationale when behavior intentionally differs from the reference.
- Validate configuration, tensor presence, tensor shapes, and supported dtypes
  at load time. Fail with a tensor name and expected/actual shape; never defer a
  malformed checkpoint failure until a later forward pass.
- Keep prefill and one-token decode as explicit entry points even when they
  initially share scalar primitives.
- Preserve deterministic greedy decoding and numerical-comparison paths.
- Routing traces and detailed timings must be opt-in and must not change model
  results.
- Keep exact inference separate from approximate routing, pruning, or other
  workload specialization. Approximate behavior must never become the default.
- Add a regression test with every bug fix when a small deterministic case can
  reproduce it.

## Rust conventions

- Use stable Rust and `rustfmt`; keep the default path safe Rust.
- Restrict `unsafe` to memory mapping, SIMD, packed weights, or FFI. Every unsafe
  block needs a local safety comment explaining its invariant.
- Return structured errors with context rather than panicking in library code.
- Avoid hidden global state. Runtime caches, scratch space, sampler state, and
  trace sinks belong to the owning runtime/session.
- Avoid allocation in steady-state decode when practical, but do not obscure the
  reference implementation merely to remove an allocation in Phase 1.
- Derive dimensions and layer types from validated model configuration. Constants
  specific to Qwen3-Coder-Next are acceptable only when named and documented.
- Keep model execution readable in `src/qwen/`; generic I/O and orchestration
  belong outside it.

## Repository layout

- `src/qwen/`: Qwen3-Next primitives and layer execution.
- `src/config.rs`: checkpoint configuration and invariant validation.
- `src/loader.rs`: checkpoint discovery, tensor enumeration, and shape checks.
- `src/runtime.rs`: prefill/decode session state, generation, timings, and traces.
- `src/bin/`: thin command-line adapters; business logic stays in the library.
- `tests/`: integration and regression tests.
- `python/`: reference comparison and trace-analysis utilities only.
- `scripts/validate.sh`: the canonical local validation entry point.

## Validation

Run `./scripts/validate.sh` before handing off a change. During iteration, run the
narrowest relevant test first. The canonical sequence is:

```bash
cargo fmt --check
cargo check --all-targets
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo build --release --bins
```

Tests requiring the full checkpoint or llama.cpp reference artifacts must be
opt-in and skip with a clear message when their inputs are absent. Unit tests
must remain small and offline.

## Change discipline

- Do not commit model weights, generated logits, routing traces, benchmark
  output, or other large artifacts.
- Do not add GPU, server, batching, custom SIMD, or broad architecture support
  during Phase 1 unless the task explicitly changes scope.
- Benchmark results are meaningful only when the command, model revision,
  quantization/dtype, thread count, and host are recorded.
- Update the execution-path documentation when model semantics, tensor names,
  checkpoint support, CLI contracts, or validation commands change.
