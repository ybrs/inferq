# Phase 1 — Get Qwen3-Coder-Next Running in Rust

## Objective

Build the smallest useful Rust inference implementation for Qwen3-Coder-Next on CPU.

Phase 1 is not about performance. It is not about avoiding dependencies. It is not about writing our own tokenizer, GGUF parser, SIMD kernels, thread pool, quantization library, or HTTP server.

The goal is to establish a correct executable inference path that we fully control and can compare against llama.cpp.

At the end of Phase 1 we should be able to:

1. Load Qwen3-Coder-Next weights.
2. Tokenize a prompt.
3. Run prompt prefill.
4. Decode one token at a time.
5. Produce logits.
6. Generate text.
7. Compare output numerically against a known implementation.
8. Measure prompt-processing and token-generation speed.
9. Record expert-routing decisions for later analysis.

Correctness and observability matter more than speed.

---

## Scope

We are targeting one model family:

- Qwen3-Coder-Next
- CPU execution
- Linux first
- x86-64 first
- batch size 1
- one active conversation/sequence
- greedy decoding first
- a single known model checkpoint and quantization initially

We can broaden this later if doing so becomes useful.

We explicitly do not need to design a general-purpose inference framework.

---

## Guiding Principle

Use existing libraries wherever they reduce irrelevant work.

Phase 1 should avoid spending time rebuilding infrastructure whose implementation teaches us little about the actual inference path.

It is acceptable to use libraries for:

- GGUF parsing
- memory mapping
- tokenization
- tensor representations
- half-precision types
- quantization decoding
- CLI parsing
- error handling
- serialization
- logging
- test utilities
- benchmarking

We can replace any dependency later.

The important thing is that the Qwen execution path is visible and understandable.

---

## Initial Technology Choices

### Rust

Use current stable Rust unless a specific compiler feature requires nightly.

Suggested project layout:

```text
qwen-engine/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model.rs
│   ├── config.rs
│   ├── loader.rs
│   ├── tokenizer.rs
│   ├── runtime.rs
│   ├── qwen/
│   │   ├── mod.rs
│   │   ├── attention.rs
│   │   ├── deltanet.rs
│   │   ├── moe.rs
│   │   ├── norm.rs
│   │   └── layer.rs
│   └── bin/
│       ├── infer.rs
│       ├── compare.rs
│       └── routing_trace.rs
├── tests/
└── python/
```

The exact module boundaries can change once the model implementation becomes clearer.

### Libraries

Use libraries pragmatically.

Possible starting choices include:

- `candle-core`
- `candle-nn`
- `candle-transformers`
- `tokenizers`
- `memmap2`
- `half`
- `bytemuck`
- `safetensors`
- a GGUF crate if one is suitable
- `clap`
- `anyhow`
- `serde`
- `serde_json`
- `tracing`
- `criterion`

We should not force these particular crates if another implementation gives us a shorter path.

If an existing Rust implementation of part of Qwen3-Next exists, it is acceptable to study it, reuse compatible pieces, or temporarily depend on it.

The goal is not originality. The goal is to obtain a working, inspectable baseline.

---

## Model Format

Prefer loading the same GGUF file currently used by llama.cpp if practical.

This makes comparison easier because both engines consume the same quantized weights.

If Rust GGUF support for this architecture or quantization is inconvenient, Phase 1 may temporarily use another representation such as:

- safetensors
- dequantized weights
- converted tensors
- a smaller quantization format that is easier to support

That is acceptable if it gets the execution path running sooner.

However, before Phase 1 is considered fully complete, we should have a path toward consuming the same model weights used by llama.cpp.

---

## Reference Implementation Strategy

Phase 1 should favor obviously correct implementations over clever implementations.

For example:

```text
quantized weight
    ↓
dequantize if necessary
    ↓
ordinary Rust tensor operation
    ↓
correct output
```

is acceptable.

We should not begin by implementing a hand-written AVX Q4 kernel.

Likewise, the first MoE implementation may allocate temporary buffers and perform operations separately.

That is fine.

We need a baseline before optimization.

---

## Required Model Components

Implement or reuse enough functionality for the Qwen3-Coder-Next forward pass.

The execution path will include the model-specific components used by Qwen3-Coder-Next, including:

- token embedding
- RMS normalization
- residual connections
- attention layers
- Qwen3-Next linear-attention / Gated DeltaNet layers
- KV state where required
- DeltaNet recurrent state
- MoE routing
- top-k routed expert selection
- shared expert execution
- routed expert execution
- output normalization
- LM head
- logits
- sampling

We should derive exact dimensions and layer structure from the model configuration rather than hard-coding assumptions unnecessarily.

Hard-coding the target architecture is acceptable. Hard-coding values that make debugging harder is not.

---

## Milestone 1 — Load Configuration and Weights

The program should be able to open the model and print a validated summary of what it found.

Example:

```text
architecture: qwen3-coder-next
layers: 48
hidden_size: ...
experts_per_layer: ...
experts_selected: ...
vocab_size: ...
quantization: ...
```

The loader should verify that every tensor expected by the implementation exists and has the expected shape.

Failure should be immediate and explicit.

Do not defer malformed model discovery until inference.

### Exit Criteria

- Model configuration loads.
- Tensor names can be enumerated.
- Expected tensors are found.
- Tensor dimensions are validated.
- Quantization metadata is readable.
- No inference is required yet.

---

## Milestone 2 — Tokenization

Use an existing tokenizer implementation.

We need:

```text
string → token IDs
token IDs → string
```

Test this against the tokenizer used by the reference model.

### Exit Criteria

For a fixed set of prompts:

```text
Rust token IDs == reference token IDs
```

and decoding those token IDs reproduces the same text.

---

## Milestone 3 — Single-Layer Primitive Validation

Before assembling the full model, validate important operations independently.

At minimum:

- RMSNorm
- matrix-vector or matrix-matrix multiplication
- activation functions
- rotary position handling if applicable
- attention
- router projection
- top-k selection
- routed expert output combination
- shared expert
- DeltaNet state update

Use small synthetic inputs where practical.

Where possible, compare each primitive against Python or another implementation.

The purpose is to avoid debugging an incorrect primitive through a 48-layer forward pass.

---

## Milestone 4 — One Full Forward Pass

Run a short token sequence through the entire model and produce logits.

The first target does not need to generate text.

Example interface:

```bash
cargo run --release --bin compare -- \
  --model /data/model.gguf \
  --tokens tokens.json \
  --dump-logits logits.bin
```

The exact CLI can change.

### Exit Criteria

For a small input:

- all layers execute;
- no NaNs appear;
- logits are produced;
- output shape is correct.

---

## Milestone 5 — Numerical Comparison Against llama.cpp

Create a repeatable reference procedure.

For a fixed prompt, collect from llama.cpp:

- token IDs
- final logits if accessible
- selected logits if full logit export is inconvenient
- generated greedy token sequence
- optionally intermediate layer outputs

Then compare Rust output.

Useful metrics:

- maximum absolute error
- mean absolute error
- cosine similarity
- top-1 token agreement
- top-N token overlap

Quantized inference does not require bit-identical floating-point output, but differences must be understood.

If final logits diverge significantly, add optional intermediate dumps and identify the first layer where the implementations separate.

### Exit Criteria

The Rust implementation produces numerically compatible logits for a short prompt.

---

## Milestone 6 — Autoregressive Decode

Add persistent inference state.

The runtime should support:

```text
prompt
    ↓
prefill
    ↓
state
    ↓
decode token
    ↓
update state
    ↓
decode next token
```

Persist whatever the architecture requires:

- KV cache
- DeltaNet recurrent state
- position
- sequence metadata

Implement greedy decoding first:

```text
next_token = argmax(logits)
```

Do not add complicated sampling until correctness is established.

### Exit Criteria

Given a deterministic prompt and greedy decoding, Rust and the reference implementation generate the same or numerically equivalent token sequence for a useful number of tokens.

---

## Milestone 7 — Basic Sampling and Chat Execution

Add enough generation functionality for normal use:

- temperature
- top-p
- top-k if needed
- min-p if needed
- stop tokens

Apply the Qwen chat template using an existing implementation or a simple known template.

A local CLI is sufficient.

Example:

```bash
cargo run --release --bin infer -- \
  --model /data/model.gguf \
  --prompt "Write a Rust function that parses..."
```

No HTTP server is necessary in Phase 1.

---

## Milestone 8 — Routing Instrumentation

This is important enough to include before performance work.

For every generated token and every MoE layer, optionally record:

```text
token_index
token_id
layer
selected_expert_ids
router_weights
router_logits if practical
```

An efficient binary trace format is preferable eventually, but JSONL is acceptable initially.

Example conceptual record:

```json
{
  "token": 421,
  "layer": 17,
  "experts": [14, 57, 82, 103, 201, 299, 331, 402, 477, 501],
  "weights": [0.18, 0.15, 0.13, 0.11, 0.1, 0.09, 0.08, 0.07, 0.05, 0.04]
}
```

Routing instrumentation must be optional so normal benchmarks do not pay its cost.

### Why This Is Included in Phase 1

Our eventual optimization strategy may depend heavily on observed expert behavior under our actual workload.

We should start collecting this data as soon as the engine works.

---

## Milestone 9 — Benchmark Harness

Create a benchmark executable that reports at least:

```text
prompt tokens
generated tokens
prefill wall time
prefill tokens/sec
decode wall time
decode tokens/sec
peak RSS if convenient
threads
model
quantization
```

Use fixed benchmark prompts stored in the repository.

Do not optimize based on one prompt.

Include several categories:

- short code generation
- source editing
- test writing
- compiler/test failure diagnosis
- shell interaction
- MCP/task-management-like text
- long context
- short context

The harness should make repeated runs easy.

---

## Milestone 10 — Workload Corpus

Create an initial corpus representing the actual intended agent workload.

It does not need to be large.

A few hundred realistic trajectories are more valuable than a generic benchmark.

Examples:

- inspect a small repository and implement one function;
- add tests for an existing module;
- run a failing test and repair the implementation;
- update a task through MCP-like messages;
- inspect git diff and correct a regression;
- explain a test failure and patch it;
- refactor a small project;
- write shell commands;
- modify a Rust project;
- modify a Python project.

The corpus will later support:

- routing analysis;
- expert-frequency analysis;
- quality regression testing;
- pruning experiments;
- performance measurements.

---

## Correctness Harness

Every optimization phase will depend on a strong correctness harness, so build it now.

Use several layers of validation.

### Unit Tests

Test mathematical primitives independently.

### Differential Tests

Compare Rust against a reference implementation.

### Randomized Tests

Generate random dimensions and inputs for operations where this is meaningful.

### Model-Level Tests

Run fixed prompts and compare logits or generated tokens.

### Regression Tests

Whenever a bug is found, preserve a case that reproduces it.

---

## Agent Development Harness

The repository should be friendly to coding agents from the beginning.

The standard validation sequence should be one command.

For example:

```bash
cargo fmt --check
cargo check
cargo test
cargo build --release
./scripts/correctness.sh
./scripts/bench-smoke.sh
```

This can later become:

```bash
./scripts/validate.sh
```

The important property is that an agent receives objective feedback quickly.

Do not require manual interpretation for routine correctness.

---

## Observability

Add structured timing around major execution regions.

At minimum:

```text
embedding
layer total
attention
DeltaNet
router
routed experts
shared expert
LM head
sampling
```

It is acceptable for the initial timing instrumentation to be relatively expensive because it can be disabled.

We want to know where time goes before attempting optimization.

---

## Performance Expectations

Phase 1 performance is irrelevant except insofar as the program must be usable.

It may be significantly slower than llama.cpp.

That is expected.

We should resist optimizing until:

1. the model works;
2. correctness comparisons exist;
3. routing traces exist;
4. benchmarks exist.

Without those, performance changes are difficult to evaluate safely.

---

## Things We Explicitly Do Not Do in Phase 1

Do not spend serious time on:

- hand-written AVX kernels;
- custom quantization formats;
- custom expert packing;
- custom allocator design;
- NUMA placement;
- speculative expert prefetch;
- expert pruning;
- dynamic top-k;
- expert merging;
- distillation;
- custom tokenizer implementation;
- OpenAI-compatible HTTP API;
- multi-user batching;
- GPU support;
- Windows support;
- macOS support;
- support for arbitrary model architectures;
- support for every GGUF quantization.

These belong later.

---

## Phase 1 Deliverables

Phase 1 is complete when the repository contains:

1. A Rust executable that loads Qwen3-Coder-Next.
2. A working tokenizer path.
3. A correct forward pass.
4. Stateful autoregressive decoding.
5. Greedy generation.
6. Basic configurable sampling.
7. Numerical comparison against a known implementation.
8. A repeatable benchmark harness.
9. Optional expert-routing traces.
10. An initial realistic coding-agent workload corpus.
11. One-command correctness validation.
12. Documentation describing the current execution path.

---

## Definition of Done

The strongest Phase 1 completion test is:

```text
Given the same model and deterministic prompt:

llama.cpp
    ↓
reference generated tokens

our Rust runtime
    ↓
same generated tokens or understood numerical-equivalent behavior
```

and:

```text
our Rust runtime
    ↓
routing trace
    ↓
benchmark results
```

At that point we have a real inference engine.

It can be slow.

Phase 2 starts when we can profile it and make changes without guessing.
