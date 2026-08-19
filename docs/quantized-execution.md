# Direct GGUF execution boundary

Stage 2B now has an initial executable boundary. `GgufCheckpoint` parses the
GGUF once and loads supported matrices in their stored Q4_K, Q5_K, Q6_K, Q8_0,
or F32 representation. `QuantizedMatrix::forward` accepts F32 activations and
calls the block matmul directly; its public API intentionally has no
whole-matrix dequantization method.

## Multi-row kernels

`QuantizedMatrix::forward` routes two or more input rows on CPU, for matrices of
at least 4 MiB, to a fused multi-row kernel in `src/qgemm.rs`. One row keeps
Candle's matvec: no weight byte is reused there, so there is nothing to fuse.
A pass wider than 16 rows is tiled into 16-row groups rather than handed over
whole; the kernel's per-row cost is lowest around that width.

`forward_spread(xs, RowSpread)` and `forward_rows(values, rows, RowSpread)` say
where the output rows are computed. `RowSpread::Pool` splits them across the
global rayon pool and is what `forward` uses. `RowSpread::Caller` computes them
all on the calling thread, for a caller that is already inside a parallel
iterator and holds one of a batch of independent matrices — the MoE's expert
loop, and nothing else today. Every output row reads the whole input and writes
only its own slice of the result, so the choice cannot change a value;
`row_spread_does_not_change_the_result` asserts that rather than arguing it.

Under `RowSpread::Caller` the size threshold and the two-row minimum are both
skipped and the fused kernels always run. That is not a tuning preference. The
Candle fallback is `apply_op1_no_bwd`, whose CPU quantized matmul runs on
candle's `BarrierPool` — a process-wide singleton (see `src/threading.rs`), so
six experts entering it at once queue for the same workers instead of getting
six matmuls. The fused kernels touch only the calling thread and have no such
shared resource.

The fused kernel exists because a per-row dot product re-runs the *block
decode* — nibble unpacking and scale extraction — once per input row, and that
decode is roughly half the per-row work. Measured on the qualified host, the
existing per-row path was compute-bound from M=4 upward: per-row time stayed
flat as M grew (a bandwidth-bound kernel's would fall ~8x from M=2 to M=16) and
each pass ran 3-4x above its weight-traffic floor. The fused kernel decodes each
weight block once and applies it to every row of a register tile, measuring
1.58x at M=8 on a Q4K dense projection.

Tile width is 8 rows, chosen by measurement across 2/4/8/12/16. Beyond 8 the
per-row accumulators stop fitting the 16 architectural ymm registers alongside
the decoded weight state, and the spill costs more than the saved decode. A
16-row pass therefore decodes each block twice and runs 8-10% worse per row than
an 8-row pass.

Candle keeps its block fields `pub(crate)`, so `qgemm` re-declares the GGUF
on-disk block layouts as `repr(C)` mirrors, each pinned by a compile-time size
assertion against Candle's own type. Accumulation order matches Candle's
`vec_dot`: the per-block integer sums are associative and the f32 accumulators
advance once per block in the same sequence, so results are bit-identical to
Candle's AVX2 kernel. Candle compiles that kernel only when the build sets
`target_feature = "avx2"`; against its scalar fallback the kernels agree to
about 1e-6, which is reordering noise. Build with `-C target-cpu=native`.

Q4K, Q6K and Q8_0 have fused kernels. Q5K stays on the per-row path: it carries
0.3% of Qwen3.6-35B-A3B's bytes.

Routed expert matrices are 576-840 KiB, below the 4 MiB threshold, so a decode
pass still meets them on Candle's loop. A *prefill* pass does not: its MoE runs
the experts in parallel and therefore under `RowSpread::Caller`, where the
threshold does not apply.

The threshold's original justification — that at the 2-3 row groups MoE routing
produces, both multi-row paths are slower than Candle's loop, because the fixed
per-call cost cannot be amortized by a matrix that small
(`multirow-report-702d043633e0.md`) — was measured on one expert at a time.
Measured on 64 experts as one batch, which is what a layer meets, it holds only
while each matmul splits its own output rows across the pool. Timing one
expert repeatedly leaves it resident in L3 and charges the fused kernel for
fork/join latency it would not pay if the experts were the parallel unit;
`gguf_matmul_bench --expert-batch N` times the batch instead. See the table in
the commit that introduced the flag.

`QuantizedMatrix::forward_via(xs, MultiRowPath, RowSpread)` pins one
implementation regardless of dispatch; it exists for `gguf_matmul_bench` and the
differential tests, and production code should call `forward`, `forward_spread`
or `forward_rows`. `MultiRowPath::Fused` is one wide call and
`MultiRowPath::FusedTiled` is what dispatch runs above 16 rows; they are not the
same cost.

Fused expert tensors use the GGUF shape `[experts, rows, columns]`.
`load_expert_matrix` seeks directly to one expert and reads only that matrix's
compressed range. On the local Q4_K_M checkpoint this reduces a gate/up expert
read from the full 288 MiB tensor to 576 KiB, and a down expert read from
352 MiB to 704 KiB.

Compile for the deployment CPU so Candle includes its AVX2 ggml kernels. The
resulting binary is host-specific:

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin gguf_projection

./target/release/gguf_projection \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --tensor blk.0.attn_qkv.weight \
  --repetitions 20

./target/release/gguf_projection \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --tensor blk.0.ffn_gate_exps.weight \
  --expert 0 \
  --repetitions 100
```

Warm measurements on the i7-6700 were about 5.8 ms for the 8,192×2,048 Q8_0
projection and 0.18 ms for one 512×2,048 Q4_K expert projection. These timings
exercise the direct kernel boundary only. They do not include routing, the rest
of a layer, or all 48 layers and are not full-model throughput claims.

The current loader owns a compact copy of each loaded matrix. It does not yet
mmap the tensor bytes, cache resident non-expert matrices, or reuse selected
expert buffers. The next integration step is one complete quantized layer:

1. keep router, norms, shared expert, attention/DeltaNet projections resident;
2. route with the real F32 gate and load only the ten selected expert ranges;
3. handle GGUF DeltaNet's split `attn_qkv`/`attn_gate` representation;
4. compare layer output and selected expert IDs with the BF16 oracle;
5. promote the path to a complete runtime only after that comparison passes.

## Real routed MoE checkpoint

`QuantizedMoeLayer` is the next integrated slice. It keeps the real F32 router
and quantized shared-expert weights resident, runs learned top-k routing, and
loads only the selected gate/up/down expert matrices. Compare it with the BF16
oracle using the same deterministic hidden vector:

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin gguf_moe

./target/release/gguf_moe \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --reference-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --layer 0 \
  --top-k 10
```

On the local checkpoints, all ten selected expert IDs matched. Quantized versus
BF16 MoE output had maximum absolute error 0.00303 and RMSE 0.00077. With the
selected expert pages warm, the quantized MoE sublayer took 6.4 ms; a prior
cold-page run took roughly 250 ms, of which 244 ms was expert loading and only
3.2 ms expert compute. This confirms both the direct-kernel path and the need
for a persistent page-cache/residency strategy.

## Complete linear-attention layer

The quantized DeltaNet path handles GGUF's optimized global Q/K/V projection,
separate Z projection, converted `-exp(A_log)` state scale, convolution state,
recurrent Gated Delta Rule state, gated normalization, and output projection.
The complete layer adds converted GGUF RMSNorm weights, residuals, and the
routed MoE:

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin gguf_layer

./target/release/gguf_layer \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --reference-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --layer 0
```

For the deterministic comparison vector, the DeltaNet mixer alone had maximum
absolute error `1.78e-5` and RMSE `2.90e-6`. The complete layer selected the
same ten experts as BF16 and had maximum error `0.0180`, RMSE `0.00193`, and
reference output L2 norm `25.56`. A warm complete layer took `11.0 ms`: about
`5.97 ms` in DeltaNet and `4.99 ms` in MoE. The DeltaNet scalar recurrent update
was about `2.01 ms`, making it a later vectorization target.

This closes the one-linear-layer correctness gate.

## Complete full-attention layer

The second layer implementation covers the joint query/gate projection, Q/K
norms, partial RoPE, persistent grouped-query KV state, causal attention,
sigmoid output gate, output projection, residuals, and routed MoE. Run its
complete comparison with:

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin gguf_full_layer

./target/release/gguf_full_layer \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --reference-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --layer 3
```

The complete layer selected the same ten experts as BF16 and produced maximum
absolute error `0.0801`, RMSE `0.0107`, and reference output L2 norm `40.59`.
Warm time was `7.79 ms`: `1.85 ms` attention and `5.87 ms` MoE. A cold-page run
took `586 ms`, dominated by `575 ms` of selected-expert reads.

Persistent KV decode is compared independently with two sequential steps:

```bash
./target/release/gguf_attention \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --reference-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --layer 3 \
  --steps 2
```

At the second cached position the attention mixer RMSE was `0.00439`; the
quantized and BF16 sessions both reused their first-step keys and values. Both
decoder layer types have now crossed their isolated correctness gate. The next
work was the 48-layer runtime described below.

## End-to-end GGUF inference

The first complete runtime now connects the Q8_0 token embedding, all 48 mixed
decoder layers and their persistent states, the final norm, and Q6_K LM head.
It uses the Hugging Face directory only for `config.json` and `tokenizer.json`:

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin gguf_infer

./target/release/gguf_infer \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --tokenizer-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --prompt a \
  --max-new-tokens 2
```

Prompt `a` produced `[284, 526]` (`" = int"`), exactly matching the existing
BF16 regression. With pages left warm by the first run, a new process loaded
resident weights in `1.89 s` and prefetched the first token in `0.427 s`
(`2.34 token/s`). The next decode pass took `20.3 s` because it routed to a new
set of cold expert pages. This is end-to-end correct but not yet sustained
usable performance without one of the residency modes described below.

### Persistent session and routing census

`--interactive` loads the model once and retains attention, DeltaNet, and
pending-token state across input lines. The input is raw continuation text; use
`/reset` before starting an independent prompt and `/quit` to exit:

```bash
./target/release/gguf_infer \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --tokenizer-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --interactive \
  --max-new-tokens 2 \
  --routing-census routing-census.json
```

Add `--routing-trace routing.jsonl` for individual layer-qualified decisions.
Add `--resume-routing-census` to update an existing census atomically across
processes instead of replacing it. Resumption rejects a sidecar from a
different checkpoint.
Full router logits are omitted unless `--trace-router-logits` is supplied. The
census is cumulative for the process and contains per-layer expert counts plus
the checkpoint size, quantization types, modification time, and a stable GGUF
layout fingerprint. The fingerprint avoids a 46 GiB startup hash and is a
local checkpoint identity, not a cryptographic content digest.

The default expert cache capacity is zero, leaving the OS page cache as the
baseline. `--expert-cache-mib N` enables a global byte-bounded LRU of compressed
expert matrices. Each turn reports requests, hits, GGUF range bytes loaded,
resident bytes, entries, and evictions. Range bytes are application-level
copies and may come from the OS page cache; they are not physical disk bytes.
The cache never changes routing or numeric
formats; it only retains already loaded `gate`, `up`, and `down` matrix ranges.
Start capacity experiments conservatively (for example, 1024 MiB) because the
46 GiB model and recurrent state must remain below the 55 GiB RSS gate.

On 2026-08-14, the first three-token `a` run with the zero-capacity baseline
loaded 2610 MiB of expert ranges across prefill and two decode passes; decode
was `0.19 token/s`. Repeating the identical command after those routes were in
the OS page cache produced the same IDs `[284, 526, 5384]` and reached `1.28
token/s` decode. This crosses the agent-usable gate for that narrow warm trace,
not yet for varied prompts.

A 1024 MiB in-process LRU recorded 618 hits from 2880 matrix requests (21.5%)
on the second turn of the `a`, `a` continuation, but its 2.38-second input pass
was slower than the 1.57-second zero-capacity warm control and caused 2008
evictions. The default therefore remains zero: this first result agrees with
Flash-MoE's observation that the OS page cache can beat an explicit cache. More
capacity and prompt-diversity sweeps are required before retaining the LRU as a
recommended operating mode.

### Expert warmup and the usable high-memory mode

`--warmup-census PATH --warmup-experts-per-layer N` validates a prior census
against the loaded GGUF and warms its hottest experts independently per layer.
With a nonzero expert cache these matrices remain pinned; with the default
zero-capacity cache only the OS page cache retains them. A five-prompt census
covered 41.9% of an unseen `struct` route, and a 17-token census plus 6 GiB
cache reached 76.9% hits on unseen `await`. Both were still about `0.19
token/s` when the uncovered route caused 358--400 MiB of random HDD reads.
Partial residency is therefore experimental on this host: high hit rate is not
enough when misses seek on rotational storage.

Page-cache-only `--warmup-all-experts` streamed all 43.5 GiB in 220.8 seconds,
but an unseen prompt still decoded at `0.19 token/s`; the kernel did not retain
the entire set in a useful eviction order. Repeating that route immediately
reached `1.15 token/s`. This mode is retained as an explicit negative
experiment, not a recommendation.

The current usable configuration pins every compressed expert matrix in the
process-owned cache:

```bash
./target/release/gguf_infer \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --tokenizer-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --interactive \
  --chat \
  --max-new-tokens 16 \
  --expert-cache-mib 46000 \
  --warmup-all-experts
```

The command refuses to start full pinning unless the configured cache can hold
all expert bytes. On this host it loaded 73,728 matrices (43.5 GiB) in 276.5
seconds. An unseen `match` prompt then had 4,320/4,320 cache hits, zero physical
reads or evictions, 1.38 input tok/s, and 1.61 decode tok/s. RSS was 47,194.7
MiB, below the 55 GiB project gate. This is the recommended persistent mode on
the 62 GiB machine, but it leaves limited headroom for concurrent memory-heavy
builds and should not be used on a smaller host.

### Chat and sustained streaming qualification

`--chat` applies the official tokenizer configuration's plain-message,
no-tools Qwen template. `--system-prompt TEXT` adds a first-turn system message.
Interactive continuation preserves the exact evaluated assistant state: if the
model emitted `<|im_end|>`, the next turn adds only the required newline; if a
token limit stopped generation first, the runtime closes the assistant message
before appending the next user turn. `/reset` starts a new templated
conversation.

Generated text is decoded through `tokenizers::DecodeStream` and flushed after
every complete byte-safe chunk. This avoids both the silence of whole-response
decoding and the broken whitespace/UTF-8 behavior of decoding tokens
individually.

Pinned sustained measurements on 2026-08-14 were:

| Case | Prefill | Decode | Expert hits | Physical reads | RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17-token chat + 16 generated | 10.24 s (1.66 tok/s) | 15 passes in 9.73 s (1.54 tok/s) | 46,080/46,080 | 0 MiB | 47,240.8 MiB |
| 23-token chat + 128 generated | 14.89 s (1.55 tok/s) | 127 passes in 81.75 s (1.55 tok/s) | 216,000/216,000 | 3.8 MiB | 47,263.0 MiB |

The long run reached a 151-token context with zero expert evictions and no
throughput degradation. Its 22 MiB RSS increase over the short case leaves the
full process below the 55 GiB gate. The response stopped at the requested token
cap, so an unfinished final code block is expected and is not a decoder error.

Use `gguf_bench` for qualification rather than running these cases as separate
processes. It amortizes the full expert pinning across the regression, short
decode, exact 32-token TTFT, and sustained 128-token cases while resetting
sequence state between measurements. The complete command and JSONL schema are
documented in [profiling.md](profiling.md#persistent-gguf-qualification).

The first native-CPU suite measured the exact 32-token TTFT at `6.20 s` and
held `3.74 token/s` over 127 sustained decode passes. The next profiled
optimization changed DeltaNet recurrence to traverse contiguous state rows and
removed fully-resident LRU bookkeeping. The identical 128 generated token IDs
then decoded at `5.62 token/s`, with TTFT reduced from `4.51 s` to `3.15 s`.
The DeltaNet sequence state now also owns reusable flat projection and
recurrence work buffers. That allocation-neutral decode path preserved all 128
IDs and measured `5.77 token/s` in a subsequent sustained run. All 216,000 expert
requests in the long case hit the pinned cache, measured inference reads were
zero, and final RSS was `47,263.1 MiB`. This crosses both stretch gates while
remaining below the 55 GiB memory gate. The earlier sustained numbers above
came from a different release build configuration and remain useful historical
measurements, not the current native-build qualification.

The causal depthwise convolution consumes the flat Q/K/V projection in place
and stores its small F32 weight table flat as well. This removes redundant
projection concatenation and row-pointer traversal. Its measured sustained
cost fell from `0.856 s` to `0.604 s` across 127 decode passes while preserving
the exact 128-token output.

At load time, the Q8_0 QKV and gate matrices are concatenated along their output
rows. This retains the compressed representation and identical per-row dot
products while sharing one quantized kernel launch. Combined with the flat
convolution, sustained decode measured `5.87 token/s` with the same 128 IDs,
zero inference reads, and `47,260 MiB` RSS.

The same exact row fusion now applies to routed and shared MoE gate/up
projections. Full pinning preserves sequential GGUF reads, then converts the
resident gate/up cache entries in memory. The cache still owns `43.5 GiB`, but
contains 49,152 combined-gate/up and down entries instead of 73,728 separate
matrices. The exact sustained run made 144,000/144,000 cache hits, reduced MoE
from `8.46 s` to `8.00 s`, and decoded at `5.99 token/s` with zero physical
inference reads and `47,328 MiB` RSS.
