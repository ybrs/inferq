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
* `qwen-cpu-inference-target-architecture.md` – Long-term target architecture, design principles, language split Rust/Python/C++, and the 14-phase roadmap toward workload-aware specialization.

## Development focus

* Rust for model loading, runtime, and architecture execution
* Python for analysis, routing telemetry, and experiment orchestration
* Specialization over generality: separate prefill vs decode, MoE as first-class operation, bytes-per-token mindset

## Status

Experimental. No code yet — documentation and architecture planning only.

See the two docs above for milestones and design.
