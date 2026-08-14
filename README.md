# Qwen3-Coder-Next CPU Inference Engine

A highly experimental, custom inference engine built specifically for **Qwen3-Coder-Next** on CPU.

Model reference: https://huggingface.co/Qwen/Qwen3-Coder-Next

This repo is for learning. The goal is to understand and build a model-specific CPU inference path for Qwen3-Coder-Next, with a focus on correctness, observability, and later specialization for coding-agent workloads.

It is not a general-purpose serving framework. It is intentionally narrow: CPU-only, batch size 1, single active sequence, x86-64 Linux first.

## Purpose

* Establish a correct, inspectable Rust inference path for Qwen3-Coder-Next
* Compare numerically against llama.cpp
* Instrument MoE routing and DeltaNet state
* Build a benchmark harness for real coding-agent workloads
* Explore workload-specific specialization later

Performance is not the primary objective in Phase 1. Correctness and observability are.

## Repo contents

* `phase-1-qwen-rust-inference.md` – Phase 1 plan: milestones from loading weights to autoregressive decode, routing instrumentation, and benchmark harness.
* `docs/profiling.md` – versioned profiling artifacts, stable micro-cases, and hardware-counter capture.
* `docs/quantized-execution.md` – direct compressed GGUF projections and selected-expert range loading.
* `docs/usable-performance-roadmap.md` – measured baseline and the critical path from the reference engine to usable CPU performance.
* `qwen-cpu-inference-target-architecture.md` – Long-term target architecture, design principles, language split Rust/Python/C++, and the 14-phase roadmap toward workload-aware specialization.

## Development focus

* Rust for model loading, runtime, and architecture execution
* Python for analysis, routing telemetry, and experiment orchestration
* Specialization over generality: separate prefill vs decode, MoE as first-class operation, bytes-per-token mindset

## Status

Phase 1 reference implementation is present. It loads and validates sharded
SafeTensors checkpoints, uses the model tokenizer, executes full-attention,
Gated DeltaNet, and MoE layers, maintains decode state, supports greedy/basic
sampling, emits routing traces, compares logits, and runs a coding-workload
benchmark. It is intentionally scalar and slow.

Start with [the execution-path guide](docs/execution-path.md). Run all offline
checks with `./scripts/validate.sh`. Full-checkpoint differential validation is
opt-in because the model weights and Transformers reference artifacts are not
stored in this repository.

The measured post-Phase-1 optimization sequence is documented in the
[roadmap to usable performance](docs/usable-performance-roadmap.md).
