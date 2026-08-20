# Qwen3-Coder-Next CPU Inference Engine

A highly experimental, custom inference engine built for **Qwen3-Coder-Next**
and **Qwen3.6-35B-A3B** on CPU. Qwen3.6-35B-A3B is the configuration the
OpenAI-compatible server and the agent-workload measurements are qualified
against; Qwen3-Coder-Next remains the original reference path.

Model references: https://huggingface.co/Qwen/Qwen3-Coder-Next, https://huggingface.co/Qwen/Qwen3.6-35B-A3B

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
* `docs/qwen36-35b-a3b.md` – Qwen3.6-35B-A3B compatibility, Q4/Q8 reproduction commands, and measured architecture/performance comparison.
* `docs/speculative-decoding.md` – Qwen3.6 auxiliary MTP execution, correctness gates, benchmark results, and the optimization path required for a speedup.
* `docs/openai-server.md` – the OpenAI-compatible HTTP server: supported request surface, the single inference slot behind it, and authentication.
* `docs/prompt-cache.md` – persistent prefix caching: what a cached state contains, how boundaries are chosen, and what makes a restored prefix exact.
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
benchmark. The readable SafeTensors path remains a reference implementation;
the optimized GGUF path adds fully resident expert caching, reusable DeltaNet
state, and direct compressed-weight kernels for usable sustained decoding.

Start with [the execution-path guide](docs/execution-path.md). Run all offline
checks with `./scripts/validate.sh`. Full-checkpoint differential validation is
opt-in because the model weights and Transformers reference artifacts are not
stored in this repository.

The measured post-Phase-1 optimization sequence is documented in the
[roadmap to usable performance](docs/usable-performance-roadmap.md).

The optimized GGUF path also supports text-only Qwen3.6-35B-A3B, which is the
configuration the server and agent-workload measurements are qualified
against. On the current i7-8700 host at six threads with the experts
resident, a wide prefill pass reaches 36.2 tok/s at width 256 (36.6 at width
512); decode measures 7.9 tok/s at 1024 context tokens and 6.8 at 3072
target-only, and 8–10 tok/s on agentic turns with drafting; and the seven-turn
pi reference task completes in 175 s with every turn after the first
continuing the live session. On the older i7-6700 host, the fully resident
Q4_K_M artifact reached 8.12 decode tok/s at 20.26 GiB RSS; Q8_0 reached
6.27 tok/s at 34.72 GiB RSS. See the
[Qwen3.6 Q4/Q8 comparison](docs/qwen36-35b-a3b.md) for exact artifacts,
commands, compatibility details, and methodology, and
[the OpenAI server guide](docs/openai-server.md) for the current
end-to-end measurements.

Greedy speculative decoding is opt-in through `--speculative`, and every mode
emits the exact token sequence ordinary greedy decoding would: proposals are
verified by the same multi-row target pass and committed only where the
target's own choice matches. There are two draft sources — prompt lookup over
the tokens already in context, and Qwen3.6's bundled MTP predictor — and
`--speculative auto` runs both behind one adaptive policy that picks per decode
step: free literal evidence where the index has it, an MTP draft where that arm
is currently earning its cost, and otherwise exactly the pass an unspeculated
run would make. `--speculative ngram` and `--speculative mtp` restrict it to one
arm.

The CLI keeps speculation off by default; the server defaults to `auto` for
greedy requests, and a sampled request decodes plainly because verification
is defined against the target's argmax. With the MTP arm gated on the draft
head's own confidence and drafting against a vocabulary prefix of the LM
head, the policy measured 1.09–1.31x across all four qualification
workloads — copy-heavy, prose, self-repetitive and mixed — where each single
arm still loses somewhere. Use
`--thinking-budget N` to retain reasoning with a per-turn hard limit, or
`--no-thinking` to render Qwen's closed thinking prefix. See
[Speculative decoding](docs/speculative-decoding.md) for the policy, the
controllers, commands and restrictions, and
`policy-report-702d043633e0.md` for the measurements.

## Installation on a new server

### Hardware and operating-system requirements

The current runtime targets x86-64 Linux and must be compiled on the server
where it will run. AVX2 is strongly recommended and FMA is required: opening
a checkpoint refuses a build without it, because the one-row and multi-row
kernels only decode identically under fused multiply-add. The qualified
high-memory configuration needs:

- at least 64 GiB RAM; sustained inference pins about 43.5 GiB of experts and
  the complete process uses about 47.3 GiB RSS;
- at least 55 GiB free disk for the repository, build products, one 46 GiB
  GGUF, and tokenizer/config files;
- four or more physical CPU cores; and
- a 64-bit Linux distribution. The commands below assume Debian or Ubuntu.

A smaller-memory machine can execute the model with no expert cache, but it
will repeatedly read routed expert ranges and is not the usable-performance
configuration. The full 160 GiB SafeTensors checkpoint is not required for
GGUF inference.

### 1. Install system packages and Rust

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates clang cmake curl git pkg-config \
  python3 python3-venv

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal
source "${HOME}/.cargo/env"

rustc --version
cargo --version
```

`Cargo.toml` requires Rust 1.88 or newer. Stable Rust is sufficient.

### 2. Clone and validate the repository

```bash
git clone https://github.com/ybrs/inferq.git
cd inferq

./scripts/validate.sh
```

The validation script builds a portable release binary. Build a second,
host-native copy for inference so validation cannot overwrite it:

```bash
CARGO_TARGET_DIR=target-native \
RUSTFLAGS='-C target-cpu=native' \
cargo build --release \
  --bin gguf_infer --bin gguf_bench --bin gguf_verify_bench
```

Do not copy `target-native` from a different server: `target-cpu=native` may
emit instructions unavailable on the destination CPU.

### 3. Download the model artifacts

Choose a model directory with at least 50 GiB free. These commands install the
Hugging Face CLI in a small virtual environment, then fetch the exact revisions
used by the qualified benchmark:

```bash
INFERQ_MODEL_ROOT=/data/models/qwen3-coder-next
INFERQ_TOKENIZER_DIR="${INFERQ_MODEL_ROOT}/tokenizer"
INFERQ_GGUF="${INFERQ_MODEL_ROOT}/Qwen3-Coder-Next-UD-Q4_K_M.gguf"

mkdir -p "${INFERQ_TOKENIZER_DIR}"
python3 -m venv "${INFERQ_MODEL_ROOT}/hf-venv"
"${INFERQ_MODEL_ROOT}/hf-venv/bin/pip" install --upgrade huggingface_hub

"${INFERQ_MODEL_ROOT}/hf-venv/bin/hf" download \
  Qwen/Qwen3-Coder-Next \
  config.json tokenizer.json tokenizer_config.json \
  --revision a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb \
  --local-dir "${INFERQ_TOKENIZER_DIR}"

"${INFERQ_MODEL_ROOT}/hf-venv/bin/hf" download \
  unsloth/Qwen3-Coder-Next-GGUF \
  Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --revision ce09c67b53bc8739eef83fe67b2f5d293c270632 \
  --local-dir "${INFERQ_MODEL_ROOT}"

ls -lh "${INFERQ_GGUF}" \
  "${INFERQ_TOKENIZER_DIR}/config.json" \
  "${INFERQ_TOKENIZER_DIR}/tokenizer.json" \
  "${INFERQ_TOKENIZER_DIR}/tokenizer_config.json"
```

The GGUF is 49,301,055,488 bytes (about 45.9 GiB). If Hugging Face requires
authentication in your environment, run `hf auth login` with the virtual
environment's `hf` executable before downloading.

### 4. Run the correctness smoke test

Run from the repository root. The first run may be slow while model and expert
pages are cold:

```bash
INFERQ_MODEL_ROOT=/data/models/qwen3-coder-next
INFERQ_TOKENIZER_DIR="${INFERQ_MODEL_ROOT}/tokenizer"
INFERQ_GGUF="${INFERQ_MODEL_ROOT}/Qwen3-Coder-Next-UD-Q4_K_M.gguf"

INFERQ_NUM_THREADS=4 \
./target-native/release/gguf_infer \
  --model "${INFERQ_GGUF}" \
  --tokenizer-model "${INFERQ_TOKENIZER_DIR}" \
  --prompt a \
  --max-new-tokens 2
```

The expected greedy token IDs are `[284, 526]`, decoded as `" = int"`. A
different result means the model artifact, tokenizer/config revision, build, or
runtime behavior does not match the qualified setup.

### 5. Start the usable fully resident session

This is the recommended mode for a server with at least 64 GiB RAM. Startup
streams and pins all 43.5 GiB of expert matrices; four to five minutes is normal
on the original HDD host. Once warm, inference performs no model-file reads:

```bash
INFERQ_MODEL_ROOT=/data/models/qwen3-coder-next
INFERQ_TOKENIZER_DIR="${INFERQ_MODEL_ROOT}/tokenizer"
INFERQ_GGUF="${INFERQ_MODEL_ROOT}/Qwen3-Coder-Next-UD-Q4_K_M.gguf"

INFERQ_NUM_THREADS=4 \
./target-native/release/gguf_infer \
  --model "${INFERQ_GGUF}" \
  --tokenizer-model "${INFERQ_TOKENIZER_DIR}" \
  --interactive \
  --chat \
  --max-new-tokens 256 \
  --expert-cache-mib 46000 \
  --warmup-all-experts
```

Enter one user message per line. `/reset` clears the conversation state and
`/quit` exits. On the original Intel i7-6700 host, the qualified 23-token prompt
plus 128 generated tokens reached 5.99 decode token/s with 100% expert-cache
hits, zero physical inference reads, and about 47,328 MiB RSS.

The four-thread setting reproduces the current benchmark host. On a different
CPU, compare it with the number of physical cores using `gguf_bench`; more
threads are not automatically faster because MoE and DeltaNet contend for
memory bandwidth. `INFERQ_NUM_THREADS` sizes every CPU thread pool the engine
touches (candle's own quantized-matvec and general-op pools, and inferq's
multi-row dense-path rayon pool) consistently in one place; set it once
instead of pairing `CANDLE_NUM_THREADS`/`RAYON_NUM_THREADS` by hand. The old
pair still works if both are set to the same value, and still takes effect
when `INFERQ_NUM_THREADS` is unset. See [Reproducible profiling](docs/profiling.md)
for the full benchmark command and JSONL output contract.

### 6. Serve the OpenAI-compatible API

Anything that speaks the OpenAI chat-completions API can drive the same
runtime through `serve`, with the same warmup and thread settings:

```bash
INFERQ_NUM_THREADS=4 \
./target-native/release/serve \
  --model "${INFERQ_GGUF}" \
  --tokenizer-model "${INFERQ_TOKENIZER_DIR}" \
  --host 127.0.0.1 \
  --port 8080 \
  --api-key "$(openssl rand -hex 24)" \
  --max-new-tokens 512 \
  --expert-cache-mib 46000 \
  --warmup-all-experts
```

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer ${INFERQ_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name three primary colors."}],
       "max_tokens":64}'
```

The model loads before the port is bound, so a bad checkpoint fails at startup.
Requests are stateless and served one at a time, first in, first out; the model
runs on its own thread and the engine never decodes two requests at once. See
[the OpenAI server guide](docs/openai-server.md) for the supported request
fields, streaming, authentication, and what is deliberately not implemented.

Add `--prompt-cache-dir` when an agent will open every task with the same long
preamble. Prefill cost grows with the square of the prompt, so the preamble that
takes minutes on the first task is restored from disk on every later one,
including after a restart:

```bash
  --prompt-cache-dir ~/.cache/inferq/prompts \
  --prompt-cache-mib 20480
```

Entries hold the token ids of cached prompts, so nothing is written without that
flag. See [the prompt cache guide](docs/prompt-cache.md).

The server also implements function calling in this checkpoint's own
`<tool_call><function=…>` format, so coding agents that speak the OpenAI API
can drive it. Its prompt rendering is checked byte for byte against the
checkpoint's `chat_template`, and a tool call survives the API round trip byte
for byte as well — an assistant turn sent back is written into the next prompt
as the bytes the model generated, even by a client that re-serialised the call's
arguments on the way, which is what keeps a multi-turn agent on the live session
instead of re-prefilling it.

Thinking is bounded per request. OpenAI's API expresses this as
`reasoning_effort` rather than a token count, so the server maps each level to
a budget the operator sets (`--thinking-budget`, `--max-thinking-budget`,
`--reasoning-budget high=8192`) and reports what it cost in
`usage.completion_tokens_details.reasoning_tokens`.

### Troubleshooting

- `Illegal instruction`: delete `target-native` and rebuild it on that server.
- Out-of-memory termination during warmup: do not use full pinning; retry
  without `--warmup-all-experts` and with a smaller or zero
  `--expert-cache-mib`, accepting much lower and storage-dependent throughput.
- `tokenizer_config.json does not define a chat_template`: download all three
  tokenizer/config files at the pinned revision above.
- Very slow cold inference with cache misses: this is expected, especially on
  HDD storage. The usable mode requires the completed full-expert warmup and a
  report showing `fully resident: true` with no evictions.
