# Qwen3-Coder-Next CPU Inference Engine — Target Architecture and Development Roadmap

## Purpose

Build a highly specialized CPU inference engine for Qwen3-Coder-Next and related structurally compatible models.

The engine is intended for a narrow, predictable coding-agent workload rather than general-purpose model serving.

Primary workload examples:

- writing and modifying small software projects;
- reading source code;
- writing tests;
- running tests;
- interpreting failures;
- fixing code;
- shell interaction;
- git interaction;
- task-manager MCP interaction;
- repeated work on a relatively small set of languages and tools.

Latency is not the primary objective.

We care about:

- generation throughput;
- CPU efficiency;
- memory traffic;
- model specialization opportunities;
- correctness;
- observability;
- experimentation;
- learning how inference engines work.

The project is also deliberately structured so that Qwen3-Coder-Next can participate in developing and optimizing the engine that eventually runs Qwen3-Coder-Next itself.

---

# End State

The intended end state is not a Rust rewrite of llama.cpp.

It is a model-specific execution system roughly shaped like this:

```text
                    Agent / CLI / MCP harness
                              |
                              v
                       Qwen runtime
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
          Prefill                            Decode
        execution                         one-token path
             |                                 |
             +---------------+-----------------+
                             |
                             v
                    Qwen execution plan
                             |
        +--------------------+--------------------+
        |                    |                    |
        v                    v                    v
    Attention            Gated DeltaNet           MoE
                                                   |
                              +--------------------+------------------+
                              |                    |                  |
                              v                    v                  v
                           Router           Routed experts      Shared expert
                                                   |
                                                   v
                                      Specialized quantized GEMV
                                                   |
                         +-------------------------+-------------------------+
                         |                         |                         |
                         v                         v                         v
                       AVX2                 AVX-512 / VNNI            FFI fallback
                         |                         |                         |
                         +-------------------------+-------------------------+
                                                   |
                                                   v
                                            x86-64 CPU / DRAM
```

The runtime should eventually be specialized around:

- one architecture family;
- CPU-only execution;
- batch size 1;
- one-token decode;
- our actual target CPU;
- a small set of quantization formats;
- fixed tensor dimensions where useful;
- model-specific MoE behavior;
- workload-specific statistics.

General-purpose abstractions should exist only where they provide real value.

---

# Language Split

## Rust

Rust is the primary implementation language.

Rust owns:

- model loading;
- runtime state;
- architecture execution;
- attention;
- DeltaNet orchestration;
- MoE routing;
- scheduling;
- threading;
- memory ownership;
- profiling hooks;
- correctness harness integration;
- CLI;
- agent-facing interfaces.

Most Rust code should remain safe.

`unsafe` should be concentrated around:

- SIMD;
- raw packed-weight access;
- memory mapping where necessary;
- low-level kernel interfaces;
- explicit prefetch operations;
- FFI.

## Python

Python is the research and analysis environment.

Python owns:

- benchmark analysis;
- routing analysis;
- expert-frequency analysis;
- routing transition statistics;
- pruning experiments;
- quality evaluation;
- corpus construction;
- plotting;
- statistical modeling;
- experiment orchestration;
- model-output comparison;
- regression analysis.

Python should consume traces and benchmark results emitted by the Rust engine.

## C/C++

C and C++ are escape hatches, references, and possible kernel sources.

Use them where they provide clear benefit:

- adapting a strong existing SIMD kernel;
- comparing compiler output;
- using an implementation from llama.cpp or another engine as a performance oracle;
- architecture-specific assembly or intrinsics that are awkward in Rust.

FFI must remain narrow.

The project should not gradually become a C++ engine wrapped in Rust.

---

# Core Design Principles

## Specialization Over Generality

The engine does not need to support arbitrary transformers.

A function such as:

```text
qwen3_next_moe_decode
```

is preferable to a large generic graph abstraction if it provides better clarity or performance.

## Separate Prefill and Decode

Prompt processing and one-token generation are different computational workloads.

They should have different execution paths.

Prefill is dominated by larger matrix operations and sequence processing.

Decode is dominated by small-batch matrix-vector operations, recurrent state updates, expert routing, memory traffic, and synchronization.

Trying to force both through one generic kernel strategy is not a goal.

## Minimize Bytes per Generated Token

The eventual decode engine should be analyzed primarily in terms of:

```text
bytes read per generated token
```

and:

```text
achieved memory bandwidth
```

rather than FLOPS alone.

CPU token generation is expected to become memory-bandwidth dominated once kernels are sufficiently optimized.

## Measure Before Specializing

Workload-specific behavior must be measured.

Do not assume:

- that only a small set of experts matters;
- that expert routing has temporal locality;
- that token IDs predict expert routing;
- that speculative prefetch improves performance;
- that fewer active experts preserve quality.

Instrument first.

## Keep a Correct Scalar Reference

Optimized kernels should have straightforward reference implementations.

The reference implementation is not dead code.

It is the correctness oracle for:

- SIMD rewrites;
- quantization changes;
- weight packing;
- threading changes;
- FFI kernels.

## Make Agent Feedback Mechanical

Every code change should be evaluated through objective tooling.

The eventual agent loop should resemble:

```text
edit
  ↓
cargo check
  ↓
tests
  ↓
numerical regression
  ↓
benchmark
  ↓
perf counters
  ↓
accept or reject
```

---

# Target Runtime Architecture

## Model Loader

Responsibilities:

- parse model metadata;
- locate tensors;
- validate tensor shapes;
- expose weight storage;
- support memory mapping;
- support later repacked weight files;
- distinguish disk representation from execution representation.

GGUF may remain the external model format indefinitely.

The runtime is not required to execute directly from GGUF layout.

---

## Execution Representation

Eventually model loading may become:

```text
GGUF
  ↓
mmap
  ↓
validate
  ↓
repack once
  ↓
CPU-native model representation
  ↓
mmap optimized representation on future runs
```

Potential internal representations may be specific to:

- target quantization;
- target ISA;
- expert dimensions;
- matrix access pattern;
- cache-line behavior.

An internal format does not need to be portable if portability costs meaningful performance.

---

## Model State

The runtime maintains only the state required for a single active sequence initially.

Possible state includes:

- current position;
- KV cache;
- DeltaNet recurrent state;
- temporary hidden buffers;
- routing buffers;
- per-layer scratch;
- thread-local scratch;
- sampler state.

Avoid unnecessary dynamic allocation during decode.

All steady-state buffers should eventually be allocated before generation starts.

---

## Layer Execution

Each layer should execute through explicit model-specific code.

Conceptually:

```text
input hidden state
      ↓
normalization
      ↓
attention or DeltaNet path
      ↓
residual
      ↓
normalization
      ↓
MoE
      ↓
residual
      ↓
next layer
```

The implementation should make the real model structure obvious when reading the source.

---

# Attention Path

The attention implementation should eventually have:

- dedicated prefill path;
- dedicated one-token decode path;
- compact KV state;
- predictable scratch usage;
- no graph construction during decode;
- no per-token heap allocation.

If attention layers represent a minority of the architecture, optimize them according to actual profile weight rather than familiarity.

---

# Gated DeltaNet Path

Qwen3-Next's recurrent/linear-attention-style component deserves a model-specific implementation.

The decode path should operate directly on the recurrent state required for one token.

It should not emulate a generic sequence-oriented implementation if that performs unnecessary work.

This is a likely source of model-specific gains.

The state layout should eventually be optimized for:

- repeated one-token updates;
- cache behavior;
- minimal copies;
- fixed dimensions;
- vectorized operations.

---

# MoE as a First-Class Runtime Operation

MoE is central to the project.

Do not represent it merely as a sequence of generic tensor operations.

The runtime should understand:

```text
hidden state
    ↓
router
    ↓
top-k expert IDs and weights
    ↓
routed expert execution
    ↓
weighted accumulation
    ↓
shared expert
    ↓
output
```

This enables optimization across operation boundaries.

---

# Router

The router should eventually have a dedicated implementation specialized for known dimensions.

Responsibilities:

- compute router scores;
- select top-k experts;
- normalize or transform routing weights according to model semantics;
- expose routing trace information;
- support experimental policies without contaminating the exact path.

Keep the exact model router available permanently.

Experimental routing should be selectable.

---

# Expert Execution

A routed expert should eventually be treated as one fused computational object.

Conceptually:

```text
gate = W_gate × x
up   = W_up × x
tmp  = activation(gate) × up
out  = W_down × tmp
```

Potential optimization opportunities:

- fused gate/up traversal;
- fixed-size intermediate buffers;
- per-thread scratch;
- cache-aware weight layout;
- direct quantized dot products;
- fused activation;
- fused weighted accumulation;
- processing several selected experts in a layout-aware order.

The engine should avoid materializing unnecessary intermediate tensors.

---

# Quantized Kernels

The final decode performance will depend heavily on quantized matrix-vector kernels.

Target structure:

```text
safe Rust kernel API
        ↓
runtime ISA dispatch
        ↓
scalar reference
AVX2 implementation
AVX-512/VNNI implementation
optional C/C++ implementation
```

Possible long-term internal quantization formats may differ from GGUF quantization layouts.

Important metrics:

- GB/s achieved;
- cycles per output element;
- instructions per byte;
- cache misses;
- bytes touched per token;
- numerical error.

---

# Threading

The final runtime should use persistent workers rather than creating work dynamically for every tensor operation.

Likely structure:

```text
main inference thread
      |
      +---- worker 0
      +---- worker 1
      +---- worker 2
      +---- ...
```

Workers live for the lifetime of the runtime.

Later concerns include:

- CPU affinity;
- physical-core versus SMT behavior;
- spin versus sleep;
- barriers;
- expert work partitioning;
- matrix-row partitioning;
- shared versus thread-local scratch;
- NUMA placement.

Thread count should be benchmarked rather than equated with logical CPU count.

---

# Memory Management

The mature runtime should aim for:

- mmap-backed model storage;
- no allocation during steady-state decode;
- aligned scratch buffers;
- predictable buffer reuse;
- explicit weight packing;
- controlled page behavior;
- optional huge pages if demonstrated useful;
- NUMA-aware placement where relevant;
- model representation optimized for actual access patterns.

---

# Profiling

Performance work should be driven by measurements.

Use tools such as:

```text
perf stat
perf record
perf report
objdump
cargo asm or equivalent
llvm-mca where useful
```

Track at least:

- tokens/sec;
- cycles/token;
- instructions/token;
- IPC;
- cache misses;
- LLC misses;
- branch misses;
- memory bandwidth;
- CPU utilization;
- bytes read/token where measurable.

Keep benchmark history under version control or in a structured experiment log.

---

# Routing Telemetry

Routing telemetry is a first-class research output.

For each layer-expert instance, collect statistics such as:

```text
activation count
router-weight mass
average router weight
rank distribution
coactivation
token-position distribution
workload-category distribution
```

Also analyze:

```text
P(expert at layer L+1 | routing at layer L)
P(expert at token t+1 | routing at token t)
expert pair frequency
expert-set similarity
expert entropy
routing stability across repositories
routing stability across languages
routing stability across task types
```

Statistics must be keyed per layer.

An expert ID is not globally equivalent across layers.

---

# Workload-Specific Optimization Track

Once exact inference is fast and measurable, exploit the predictability of the real workload.

This work must remain separate from baseline engine optimization because it changes model behavior.

Possible experiments include the following.

## Expert Frequency Optimization

Keep model semantics exact but place frequently selected experts more favorably in memory.

Potential techniques:

- expert packing order;
- page placement;
- cache-aware grouping;
- NUMA placement;
- high-frequency expert replication if worthwhile.

## Dynamic Active Expert Count

Instead of always executing the model's full selected set, experiment with executing fewer experts based on router-weight mass.

Examples:

```text
fixed top-8
fixed top-6
minimum experts covering 99% router mass
minimum experts covering 97% router mass
minimum experts covering 95% router mass
```

This changes model semantics and therefore requires quality evaluation.

## Expert Masking

Build workload-derived allowlists per layer.

The native router still produces scores, but excluded experts cannot be selected.

Evaluate progressively smaller pools.

Examples:

```text
512 allowed
384 allowed
256 allowed
192 allowed
128 allowed
96 allowed
64 allowed
```

This is workload-specific pruning and must be evaluated against real tasks.

## Expert Removal

If masking experiments show that certain experts are unnecessary for the target workload, produce smaller weight files that omit them.

This reduces:

- stored model size;
- mapped memory;
- page pressure.

It does not automatically reduce per-token traffic unless routing or active expert count changes.

## Expert Merging or Distillation

Later work may train or derive smaller expert sets specialized for the workload.

This moves beyond pure inference optimization into model compression.

Keep this as a distinct research track.

---

# Speculative Expert Preparation

The engine may experiment with predicting future expert use.

The exact native router remains authoritative.

A predictor may use information such as:

- previous layer routing;
- previous token routing;
- current hidden-state summary;
- token class;
- task class;
- recent routing history.

Possible uses:

- software prefetch;
- cache preparation;
- likely-expert scheduling.

Prediction failure must never change exact inference unless running an explicitly approximate mode.

Because CPU decode may be DRAM-bandwidth bound, bad predictions can reduce performance by consuming additional memory bandwidth.

Therefore optimize precision, not just recall.

Measure before keeping this feature.

---

# Quality Evaluation

Tokens/sec alone is insufficient once model behavior is changed.

Maintain a workload-specific evaluation corpus.

Evaluate:

- task completion;
- tests passing after agent modification;
- patch correctness;
- compile success;
- regression rate;
- number of agent iterations;
- generated-token count;
- tool-use correctness;
- task-manager behavior.

Exact inference optimizations should preserve numerical behavior within expected tolerance.

Approximate/model-specialization experiments should be judged primarily by actual agent outcomes.

---

# Self-Hosting Goal

A major project milestone is running Qwen3-Coder-Next under the engine it helped build.

Development progression:

```text
Qwen under llama.cpp
        ↓
writes and debugs Rust engine
        ↓
Rust engine reaches correctness
        ↓
Rust engine reaches usable speed
        ↓
Qwen runs under Rust engine
        ↓
Qwen profiles and improves its own runtime
```

At that point the optimization agent can receive objective inputs such as:

```text
source code
benchmark history
perf stat
perf report
assembly
routing traces
correctness failures
```

and optimize against objective outputs:

```text
correctness
tokens/sec
cycles/token
bytes/token
task success
```

---

# Development Phases

## Phase 1 — Correct Running Baseline

Goal:

> Execute Qwen3-Coder-Next correctly in Rust.

Use any libraries necessary.

Deliver:

- model loading;
- tokenization;
- forward pass;
- prefill;
- autoregressive decode;
- greedy generation;
- basic sampling;
- correctness comparison;
- benchmark harness;
- routing traces.

Performance does not matter yet.

---

## Phase 2 — Understand the Runtime

Goal:

> Know exactly where time and memory traffic go.

Add detailed profiling and establish stable baselines.

Measure:

- per-component timing;
- per-layer timing;
- router time;
- expert time;
- attention time;
- DeltaNet time;
- LM-head time;
- allocations;
- memory bandwidth;
- cache misses;
- thread scaling.

Build benchmark history.

At the end of Phase 2, we should be able to explain why the engine has its current tokens/sec.

---

## Phase 3 — Separate Prefill and Decode

Goal:

> Stop treating prompt processing and token generation as one execution problem.

Create distinct optimized paths.

Decode becomes explicitly specialized for:

- batch 1;
- one token;
- persistent state;
- fixed tensor dimensions;
- no graph construction;
- no steady-state allocation.

Prefill may continue using more generic tensor/GEMM libraries initially.

This phase should simplify the decode hot path before hand-written kernels arrive.

---

## Phase 4 — Specialized Rust Decode Kernels

Goal:

> Replace generic tensor execution in the decode hot path.

Implement scalar reference kernels and then SIMD kernels for the important operations.

Priority determined by profiling.

Likely targets:

- quantized GEMV;
- RMSNorm;
- router projection;
- top-k;
- expert gate/up/down;
- DeltaNet operations;
- attention projections.

Use AVX2 first if appropriate for the target CPU.

Add AVX-512/VNNI paths only if hardware supports them and measurements justify them.

---

## Phase 5 — Specialized MoE Engine

Goal:

> Treat MoE as one model-specific operation rather than generic tensor composition.

Implement:

- dedicated router;
- top-k routing;
- selected-expert scheduling;
- shared expert;
- fused expert execution where useful;
- fixed scratch buffers;
- fused weighted accumulation;
- routing-aware threading.

This phase is expected to be one of the most important for Qwen3-Coder-Next.

---

## Phase 6 — Weight Repacking and Memory Layout

Goal:

> Make the model representation match the CPU execution pattern.

Separate external GGUF layout from runtime layout.

Experiment with:

- row interleaving;
- block ordering;
- expert-local packing;
- fused gate/up layout;
- ISA-specific packing;
- alignment;
- page locality;
- expert ordering.

Create a repack tool so expensive conversion happens once.

Potential output:

```text
model.gguf
    ↓
qwen-repack
    ↓
model.qcpu
```

The internal file may be tied to model version, quantization, and CPU ISA.

---

## Phase 7 — Threading, NUMA, and Memory Bandwidth

Goal:

> Approach the machine's useful memory-bandwidth ceiling.

Replace generic threading with persistent workers if not already done.

Benchmark:

- physical cores;
- SMT;
- affinity;
- work partitioning;
- barriers;
- NUMA placement;
- huge pages;
- prefetch distance;
- expert scheduling.

Track achieved versus theoretical memory bandwidth.

At this phase, bytes/token becomes a primary design metric.

---

## Phase 8 — Routing Analysis of the Real Workload

Goal:

> Understand whether the intended coding-agent workload uses the model differently from general workloads.

Run large quantities of actual representative tasks.

Analyze per-layer expert behavior.

Questions include:

- How many experts account for 50%, 90%, 95%, 99% of routing mass?
- Are some experts effectively unused?
- Does routing differ significantly between Python, Rust, shell, and task-manager interactions?
- Does debugging use different experts from code generation?
- Does expert distribution vary significantly between repositories?
- How predictable is routing from previous layers or tokens?
- Are there stable high-frequency expert subsets?

No approximate optimization is adopted until this analysis exists.

---

## Phase 9 — Exact Workload-Aware Optimizations

Goal:

> Exploit measured workload behavior without changing model output.

Possible optimizations:

- expert placement;
- memory ordering;
- hot-expert grouping;
- page placement;
- NUMA placement;
- selective cache strategies;
- high-confidence expert prefetch;
- routing-informed scheduling.

Every change must preserve exact routing and outputs within normal numerical tolerance.

---

## Phase 10 — Approximate Expert Execution

Goal:

> Trade unused generality for more decode speed while preserving coding-agent quality.

Experiments:

- top-8;
- top-6;
- dynamic router-mass thresholds;
- expert allowlists;
- low-frequency expert masking.

Run quality evaluations after every change.

Record both:

```text
tokens/sec
```

and:

```text
agent task success
```

Do not optimize synthetic perplexity while destroying the actual coding workflow.

---

## Phase 11 — Physical Expert Pruning

Goal:

> Reduce stored model size after masking experiments prove that some experts are unnecessary.

Produce workload-specific model variants.

Examples:

```text
full
qwen-coder-workload-384
qwen-coder-workload-256
qwen-coder-workload-128
```

Names are illustrative.

Measure:

- model size;
- memory mapping size;
- startup behavior;
- page faults;
- tokens/sec;
- task success.

---

## Phase 12 — Expert Prediction and Speculative Preparation

Goal:

> Determine whether routing predictability can improve CPU memory behavior.

Train or derive lightweight predictors from routing traces.

Potential predictors:

- transition tables;
- linear models;
- tiny neural networks;
- hidden-state sketches;
- layer-specific statistical predictors.

Use prediction initially only for prefetch/preparation.

Compare:

```text
prediction precision
coverage
extra bytes read
cache effect
tokens/sec
```

Discard the idea if it does not outperform the exact baseline.

---

## Phase 13 — Model Specialization

Goal:

> Explore whether a smaller model specialized for our agent workload can retain useful capability.

Possible directions:

- expert merging;
- distillation;
- continued training;
- router retraining;
- reduced expert counts;
- lower precision for selected components.

This phase is no longer only inference-engine engineering.

Treat model changes as a separate artifact with reproducible training/evaluation procedures.

---

## Phase 14 — Self-Hosted Optimization Agent

Goal:

> Run the coding agent on the engine and let it optimize the engine through the same harness.

Agent receives:

- source;
- benchmark results;
- profiler output;
- hardware counters;
- failed tests;
- numerical diffs.

Agent actions are evaluated automatically.

A candidate optimization is accepted only when it satisfies configured gates.

Example:

```text
cargo check: pass
cargo test: pass
numerical regression: pass
workload smoke test: pass
decode throughput: +4.8%
```

This provides a constrained environment for exploring the limits of Qwen3-Coder-Next as a systems-programming agent.

---

# Benchmark Strategy

Keep two benchmark classes.

## Engine Benchmarks

These isolate runtime performance.

Examples:

- short prefill;
- long prefill;
- 128-token decode;
- 1024-token decode;
- isolated expert kernel;
- isolated router;
- isolated DeltaNet update.

## Agent Workload Benchmarks

These measure useful end-to-end behavior.

Examples:

- implement a small feature;
- repair failing tests;
- add test coverage;
- modify a CLI;
- inspect a compiler error;
- perform a task-manager MCP update;
- work through a small repository issue.

Engine benchmarks identify performance changes.

Agent benchmarks identify whether approximations remain useful.

---

# Experiment Logging

Every meaningful performance result should capture:

```text
git commit
model
quantization
CPU
RAM configuration
thread count
compiler version
compiler flags
prompt set
prefill tok/s
decode tok/s
cycles/token
memory bandwidth
correctness result
quality result if applicable
```

Avoid relying on remembered numbers from terminal output.

---

# Success Criteria

There are several independent definitions of success.

## Learning Success

We can explain the complete inference path and its hardware bottlenecks.

## Engine Success

The Rust engine correctly runs Qwen3-Coder-Next and reaches performance comparable to or better than llama.cpp for the target workload and hardware.

## Specialization Success

Workload-specific changes reduce memory traffic or active computation without materially harming agent task success.

## Agent-Harness Success

Qwen3-Coder-Next can make useful, measurable improvements to the codebase using compiler, tests, numerical comparison, and profiling as feedback.

## Research Success

Even failed ideas produce useful measurements about:

- MoE routing;
- expert locality;
- CPU cache behavior;
- quantized GEMV;
- workload specialization;
- the practical limits of coding agents on low-level systems work.

---

# Non-Goals

Unless requirements change, the final engine does not need to become:

- a general transformer framework;
- a llama.cpp replacement;
- a production multi-tenant serving platform;
- a GPU runtime;
- a distributed inference system;
- a cross-platform abstraction layer;
- a universal GGUF implementation;
- a library supporting every quantization.

Supporting another model is worthwhile only when it teaches us something useful or shares enough architecture to justify it.

---

# Final Mental Model

The project should evolve through three distinct layers.

First:

```text
make it correct
```

Then:

```text
make the exact model fast
```

Only then:

```text
change the model/runtime contract to exploit our workload
```

Keeping these stages separate is important.

Otherwise it becomes impossible to distinguish:

- inference-engine improvements;
- quantization improvements;
- architectural approximations;
- workload-specific model degradation.

The intended final system is therefore both an inference engine and an experimental platform for asking:

> How much of a very large sparse coding model do we actually need for this narrow agent workload, and how efficiently can a CPU execute exactly that subset?
