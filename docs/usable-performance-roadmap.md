# Roadmap to usable performance

## Outcome we are targeting

Phase 1 established a correct, inspectable BF16 reference. It is suitable as a
correctness oracle, but not as an interactive engine. The next objective is a
warm, persistent CPU runtime that can complete real coding prompts without
minute-long pauses between tokens.

For the current host, define three performance gates:

| Gate | Warm decode | Warm TTFT for a 32-token prompt | Meaning |
| --- | ---: | ---: | --- |
| Research-usable | at least 0.5 token/s | at most 60 s | Routing and workload experiments stop being prohibitively slow. |
| Agent-usable | at least 1 token/s | at most 30 s | Slow, but practical for unattended coding-agent tasks. |
| Stretch | at least 2 tokens/s | at most 15 s | Interactive enough to inspect and steer regularly. |

These are project gates, not promises. Every number must come from the
repeatable benchmark harness and retain the exact greedy-token correctness
gate. A persistent session is the primary operating mode; cold startup from a
rotational disk is reported separately.

## Measured starting point

The current reference path runs the official BF16 SafeTensors checkpoint and
promotes every matrix multiplication to F32 because Candle CPU does not provide
BF16 matmul.

On 2026-08-14, prompt `a` produced the same first greedy token as the local
llama.cpp Q4_K_M checkpoint. The relevant measurements were:

| Path | Result |
| --- | ---: |
| Rust BF16 cold one-token prefill | 57.8 s |
| Rust BF16 prefill after cache eviction | 117.0 s |
| Rust BF16 cached decode pass | 63.9 s/token |
| llama.cpp Q4 model load/repack | 277.7 s |
| llama.cpp Q4 token evaluation | 49.6 s/token |

The 12-token raw prompt tested later took 239.3 seconds to prefill, followed by
29.7 seconds for its one actual decode pass. These measurements are
page-cache-sensitive and are smoke-test baselines rather than stable
benchmarks.

## Current hardware constraints

The development host is materially different from the high-bandwidth systems
used by several reference projects:

- Intel Core i7-6700: 4 physical cores, 8 threads, AVX2/FMA, no AVX-512;
- 62 GiB RAM and 32 GiB swap;
- 8 MiB shared L3 cache;
- model storage on a rotational SATA disk, exposed as Btrfs at `/data`;
- official BF16 checkpoint: approximately 160 GB;
- local Q4_K_M GGUF: 45.9 GiB.

The 45.9 GiB GGUF can plausibly coexist with runtime state inside 62 GiB RAM if
we execute it directly. It cannot coexist comfortably with a second 27+ GiB
repacked copy, which is what the observed llama.cpp CPU-repack path attempted;
that run filled swap. Avoiding expanded or duplicate weight representations is
therefore both a speed and a correctness-of-operation requirement.

The rotational disk also changes the expected storage design. Randomly reading
ten experts per layer from HDD cannot be the steady-state decode path. The
initial exact strategy is:

1. keep the compact quantized model mmap-backed;
2. avoid whole-weight dequantization and duplicate repack buffers;
3. use the OS page cache first;
4. keep the inference process alive between requests;
5. measure residency and page faults before adding an explicit cache.

NVMe-oriented O_DIRECT, io_uring, and asynchronous expert streaming remain a
separate hardware-gated track.

## Lessons from related projects

### Flash-MoE

[Flash-MoE](https://github.com/danveloper/flash-moe) streams only selected
experts from SSD, uses per-expert packed files, fuses dequantization with GEMV,
uses BLAS for the Gated DeltaNet recurrence, and reports that trusting the OS
page cache beat its custom caches. Its negative experiments are equally useful:
compression, speculative expert prediction, explicit prefetch, and mmap of cold
expert files all lost in its tested pipeline.

Adopt now:

- selected-expert execution rather than materializing all experts;
- a one-time expert-local packing format;
- fused dequantization and dot product;
- BLAS or vectorized DeltaNet operations;
- OS page cache as the first cache implementation;
- explicit experiment logs including discarded approaches.

Do not extrapolate its throughput. Its M3 Max has approximately 400 GB/s unified
memory bandwidth, a 17.5 GB/s SSD, and GPU compute. This host has neither that
memory system nor an SSD.

### Pulsar

[Pulsar](https://github.com/giannisanni/pulsar) keeps routing and other
decision-making weights resident, streams routed experts, records a warm expert
census, and uses the census to populate resident tiers on later runs. It also
separates prefill GEMM from memory-bound one-token decode and treats
teacher-forced/reference agreement as a release gate.

Adopt now:

- separate resident decision weights from routed expert storage;
- preserve distinct prefill and decode kernels;
- record per-layer expert popularity in a versioned sidecar;
- distinguish cold-census and sustained warm measurements;
- keep deterministic and teacher-forced comparison tools.

Its reported throughput depends on CUDA GPUs and PCIe placement. The tiering
concept transfers; the absolute numbers and kernels do not.

### Micro-Expert-Router

[Micro-Expert-Router](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER)
is the closest CPU-oriented reference. It uses layer-qualified expert blobs,
page-aligned buffers, optional O_DIRECT/io_uring, a bounded shared Rayon pool,
real learned routing, and separate synthetic versus full-transformer benchmark
claims. Its documented Qwen run shows why measured cache behavior matters: a
25% expert cache achieved about 72% hits and 0.55 token/s, while halving cache
capacity reduced learned-routing hits to about 53% and decode to 0.33 token/s.
Its synthetic routing had predicted a much higher hit rate and did not transfer.

Adopt now:

- layer-qualified expert identities;
- real model routing in every performance qualification;
- a bounded, persistent worker pool with explicit thread-count experiments;
- cache hit/miss, bytes-read, and I/O-stall telemetry;
- fail-closed execution modes and explicit fallback reporting;
- page-aligned expert blobs as an optional internal format.

Defer O_DIRECT and io_uring until the model is on actual NVMe. On this HDD they
would bypass the page cache that makes warm execution possible.

### GdsLLM

[GdsLLM](https://github.com/rscunha13/gdsllm) demonstrates direct NVMe-to-VRAM
DMA, selective expert loading, and fused dequantization/GEMV. Its Qwen path
loads ten of 512 experts rather than an entire layer, which is the same central
data-reduction opportunity present here.

Adopt now:

- make selected expert ranges directly addressable;
- fuse dequantization with compute;
- represent weight residency and transfer as scheduler-visible events.

GPUDirect Storage itself is out of scope for the current CPU-only host. It
becomes relevant only if the project later gains a suitable NVIDIA GPU and
NVMe device.

## Critical path

### Stage 2A: reproducible profiling

Before changing kernels, make the current cost visible.

Deliverables:

- structured JSON benchmark output with model revision, host, thread count,
  cold/warm status, prompt tokens, generated tokens, and correctness result;
- timings split into weight load/conversion, GEMV/GEMM, normalization, router,
  top-k, routed experts, shared expert, DeltaNet, full attention, and LM head;
- allocation counts and peak RSS;
- major/minor faults and bytes read where the OS exposes them;
- `perf stat` capture for cycles, instructions, cache misses, faults, and context
  switches;
- stable one-token, 12-token prefill, and 16-token decode benchmark cases.

Exit gate: component timings explain at least 95% of wall time and repeated warm
runs have a documented variance band.

### Stage 2B: direct quantized execution

This is the highest-leverage stage. Stop expanding BF16 weights to F32.

Deliverables:

- executable tensor views for the existing GGUF Q4_K, Q5_K, Q6_K, Q8_0, and
  F32 tensors used by this exact checkpoint;
- a `QuantizedMatrix`/GEMV boundary that cannot accidentally dequantize a whole
  matrix;
- scalar reference dequant-dot kernels with block-level tests;
- a narrow llama.cpp/ggml kernel adapter as the initial fast implementation and
  performance oracle, if its integration is shorter than a correct native AVX2
  implementation;
- direct execution of embedding, router, shared expert, selected experts,
  attention/DeltaNet projections, and LM head from quantized weights;
- per-layer and final-logit comparisons against the BF16 reference.

The initial FFI bridge is allowed to get a working quantized path quickly. The
Rust API owns scheduling and state; FFI should expose only well-bounded
quantized dot/GEMV operations. Native Rust AVX2 can replace it incrementally.

Exit gate: identical greedy IDs on the regression prompts, no full-matrix F32
temporary, peak RSS below 55 GiB, and at least a 4x warm decode improvement over
the BF16 baseline.

### Stage 3: specialized one-token decode

Make decode a fixed-shape execution plan instead of a series of generic tensor
operations.

Deliverables:

- preallocated hidden, projection, expert, routing, attention, and DeltaNet
  scratch buffers;
- no steady-state heap allocation;
- persistent worker pool, benchmarking 4 physical workers against 8 SMT
  workers and smaller counts;
- AVX2/FMA quantized GEMV for the dominant checkpoint formats;
- fused gate/up traversal, SwiGLU, down projection, route weighting, and
  accumulation where measurement supports it;
- specialized router top-k and final LM-head scan;
- vectorized or BLAS-backed DeltaNet state scale/GEMV/rank-one update;
- incremental text output after every generated token.

Exit gate: research-usable warm decode of at least 0.5 token/s while preserving
the greedy-token regression suite.

### Stage 4: prefill path

The current implementation repeats selected-expert loading and computation per
prompt token. Prefill needs its own batched plan.

Deliverables:

- batch GEMM kernels or a proven BLAS/ggml path for dense projections;
- group prompt tokens by routed expert so one expert view serves all assigned
  tokens;
- chunked causal DeltaNet prefill with equivalence against the recurrent
  reference;
- batched full attention and LM-head work only for required output positions;
- tokenizer/chat-template timing separated from model prefill.

Exit gate: a warm 32-token prompt reaches TTFT below 60 seconds. Continue toward
30 seconds after decode has crossed 0.5 token/s.

### Stage 5: residency, persistence, and packaging

Usability requires keeping expensive state alive.

Deliverables:

- an interactive process that loads once and retains model mappings and session
  state across turns (complete);
- optional census and full-expert warmup with progress and Ctrl-C cancellation
  (complete);
- RSS, fault, physical-read, and expert-cache telemetry (complete);
- an atomically resumable per-layer routing census sidecar keyed by local model
  identity and quantization (complete);
- optional expert-local `qcpu` repack files with contiguous gate/up/down blocks,
  alignment, checksums, and source-model identity;
- page-cache-first hot expert ordering or prefaulting, evaluated without
  duplicating the entire 46 GiB model in heap memory;
- proper tokenizer chat-template application and newline-safe streaming CLI.
  The plain-message, no-tools official template and byte-safe incremental
  decoder are complete; tool rendering remains future work.

Exit gate: agent-usable warm decode of at least 1 token/s and warm TTFT below 30
seconds, with the process remaining below the memory budget during a 128-token
generation.

### Stage 6: storage experiments, gated by hardware

On the current HDD, benchmark only buffered/page-cache-backed execution. If the
model moves to NVMe, add controlled alternatives:

- buffered `pread` versus mmap faults versus O_DIRECT;
- io_uring registered fixed buffers and queue-depth sweeps;
- explicit RAM expert cache sizes;
- hot expert pinning from real routing census data;
- I/O/compute overlap only when it reduces end-to-end token time;
- per-expert contiguous packing versus fused GGUF tensor slicing.

Reject any storage change that improves synthetic I/O while reducing full-model
decode throughput. Record cold and sustained-warm results separately.

### Stage 7: exact workload-aware placement

After the exact quantized engine is usable, collect routing traces from actual
coding-agent work. Use them only for output-preserving changes first:

- expert disk/layout ordering;
- hot-page warming;
- cache sizing;
- thread scheduling;
- high-precision prefetch with extra bytes measured.

Speculative routing, fewer active experts, masking, pruning, and distillation
remain later approximate tracks. Flash-MoE and MER both provide evidence that
naive predictors or synthetic locality assumptions can lose end-to-end.

## Immediate implementation sequence

Implementation status on 2026-08-14: the versioned profiling artifact and
stable cases are in place. Direct Q4_K/Q5_K/Q6_K/Q8_0/F32 projection execution,
selected-expert byte-range loading, and a complete routed MoE sublayer are also
working. Layer-0 MoE selected the same ten experts as the BF16 oracle and its
output RMSE was 0.00077 on the deterministic comparison vector. Full
DeltaNet integration is now complete as well: the whole quantized linear layer
matched BF16 routing, produced output RMSE 0.00193, and took 11.0 ms warm. Full
attention is now complete too: layer 3 matched BF16 routing, produced output
RMSE 0.0107, took 7.79 ms warm, and passed a two-step persistent-KV comparison.
Whole-model assembly is now operational: prompt `a` produced the expected
two-token sequence `[284, 526]`. Warm prefill reached 2.34 token/s, but the next
decode encountered new cold expert pages and took 20.3 seconds. The runtime now
has an interactive mode that loads once, preserves sequence state correctly
across turns, and writes either detailed routes or a compact per-layer census.
An opt-in global byte-bounded expert LRU now provides per-turn hit/miss/read/
resident/eviction telemetry while the default remains the page-cache-only
baseline. Census resumption and census/full warmup are operational.

The first controlled warm-route result clarifies the opportunity. A
zero-capacity three-token run decoded at 0.19 token/s while loading 2610 MiB of
expert ranges; the identical repeat reached 1.28 token/s once those pages were
resident. A 1024 MiB explicit LRU achieved 21.5% hits on a two-token
continuation but was slower than the zero-capacity warm control and thrashed
with 2008 evictions. These are narrow smoke measurements, but they support
page-cache-first census warming rather than enabling the explicit LRU by
default.

Broader measurements changed the operating recommendation. A five-prompt
census covered 41.9% of an unseen route; a resumed 17-token census and 6 GiB
cache reached 76.9% hits, but the remaining 400 MiB of random physical reads
still limited decode to 0.19 token/s. Warming all experts into page cache also
failed because eviction order left route holes. Pinning the entire compressed
expert set succeeded: 43.5 GiB loaded in 276.5 seconds, 47,194.7 MiB process
RSS, 100% expert hits, zero inference reads, and 1.61 decode token/s on an
unseen prompt. This crosses the agent-usable decode gate within the memory
budget.

Sustained qualification also passes. A 17-token chat prompt plus 16 generated
tokens prefetched at 1.66 token/s and decoded at 1.54 token/s. A 23-token chat
prompt plus the full 128 generated tokens prefetched at 1.55 token/s and held
1.55 token/s across 127 decode passes. The latter reached 151 context tokens,
216,000/216,000 expert hits, zero evictions, and 47,263 MiB RSS. Tokenizer-aware
incremental output streamed throughout the 81.75-second decode.

The persistent multi-case benchmark is now implemented. It validates all
rendered token counts before model load, performs one shared full-expert
warmup, resets sequence state per repetition, and emits source, host, model,
warmup, correctness, timing, cache, RSS, fault, and physical-I/O data in one
JSONL artifact. Its manifest includes the greedy regression, 16-token decode,
an exactly 32-token templated coding prompt, and sustained 128-token decode.

The first complete `target-cpu=native` suite measured the exact 32-token TTFT
at 6.20 seconds. Structured nested timing then identified the strided DeltaNet
recurrence as 35% of sustained decode. Contiguous row traversal reduced that
operation from 11.36 to 1.42 seconds over 127 passes. Together with resolved
expert handles and disabled LRU bookkeeping after proven full residency, the
same 128 generated IDs improved from 3.74 to 5.62 token/s and TTFT from 4.51
to 3.15 seconds. The run retained 216,000/216,000 expert hits, zero physical
inference reads, and 47,263.1 MiB RSS. This crosses both stretch gates on the
current host; remaining qualification is variance and conversational/tool
correctness rather than basic throughput.

Layer-owned flat DeltaNet scratch subsequently removed the nested projection
readback and transient Q/K/V, convolution, repeat-head, beta/decay, and output
vectors from steady-state decode. The exact 128-token stream remained
unchanged and decode measured 5.77 token/s (22.01 seconds for 127 passes), with
DeltaNet accounting for 9.29 seconds. Its input and output quantized
projections now dominate that stage, while MoE remains the other major
end-to-end cost.

The next three bounded changes should be:

1. Record a warm repeated variance band for the short cases in the same
   persistent process.
2. Validate multiple templated user/assistant turns, then add the official tool
   description/call subset needed by an agent harness.
3. Use schema-2 timing to evaluate fused expert compute and batched long-prompt
   prefill only where they improve real agent workloads or memory headroom.

Expert I/O is removed from the recommended pinned decode path. The critical
path now moves to variance qualification and the tool-capable
conversation boundary. io_uring, speculative routing, and approximate expert
counts remain outside the current path.

## Benchmark and correctness contract

Every accepted performance change must report:

- Git commit and dirty status;
- exact model file and revision/hash;
- quantization and internal packing version;
- CPU, RAM, storage type, filesystem, compiler, and thread count;
- whether the run is cold, warm, or persistent-session steady state;
- prompt/decode token counts and raw durations;
- TTFT, prefill tokens/s, decode tokens/s, RSS, faults, and bytes read;
- greedy token IDs and numerical comparison result;
- routing-trace equivalence for changes touching MoE.

The BF16 implementation remains the readable oracle even after it stops being
the default execution path.
