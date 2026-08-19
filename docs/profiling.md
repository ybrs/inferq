# Reproducible profiling

The `bench` binary emits one JSON object per measured generation. Schema
version 1 records source revision and dirty state, model revision and format,
host and storage metadata, the explicit cache-state label, token IDs,
correctness, wall-clock rates, model timing attribution, RSS, faults, and
process I/O bytes.

Top-level model stages are disjoint and the artifact reports their accounted
fraction of `Model::forward` wall time. `nested_operations` is a second,
overlapping view: weight load, dtype conversion, matmul, router, top-k, routed
experts, and shared expert. Do not add nested values to their parent stage.
Decode model timings cover the forward passes after the first sampled token;
sampling and text decoding remain visible in the enclosing decode wall time.

The benchmark does not evict or warm the Linux page cache. `--cache-state` is a
mandatory experimental claim made by the operator, with `unknown` as the safe
default. A cold result must come from a separately controlled cold-cache run;
subsequent repetitions in one persistent process are not cold.

Build once, then capture a stable case plus hardware counters:

```bash
cargo build --release --bin bench

./scripts/profile-case.sh \
  /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  one-token-smoke cold artifacts/one-token-cold

./scripts/profile-case.sh \
  /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  twelve-token-prefill warm artifacts/twelve-token-warm

./scripts/profile-case.sh \
  /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  sixteen-token-decode persistent artifacts/decode-persistent
```

The wrapper refuses to overwrite either its `.jsonl` record or `.perf.csv`
counter file. `perf_event_paranoid` may prevent unprivileged hardware counters;
that is a host policy failure and does not invalidate the JSON telemetry.

For a repeated warm variance band without `perf`:

```bash
./target/release/bench \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --prompts benchmarks/profile-prompts.json \
  --only one-token-smoke \
  --warmup-repetitions 1 \
  --repetitions 5 \
  --cache-state persistent \
  --output artifacts/one-token-warm-5.jsonl
```

Exact allocator call counts are not yet in schema version 1. RSS and fault/I/O
counters are session-owned observations; an allocator-count implementation
must also be explicitly owned and opt-in rather than a hidden process-global
counter.

## Persistent GGUF qualification

`gguf_bench` is the quantized counterpart. It opens the GGUF once, pins all
expert matrices once, resets sequence state before each measured generation,
and writes one JSONL record per case. The default manifest contains the exact
greedy regression, a 16-token generation, an exactly 32-token templated prompt
for the TTFT gate, and a 128-token sustained generation. Prompt counts are
validated before the GGUF is opened.

Build and run the complete pinned suite with:

```bash
CARGO_TARGET_DIR=target-native \
RUSTFLAGS='-C target-cpu=native' \
cargo build --release --bin gguf_bench

INFERQ_NUM_THREADS=4 \
./target-native/release/gguf_bench \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --tokenizer-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --prompts benchmarks/gguf-prompts.json \
  --expert-cache-mib 46000 \
  --warmup-all-experts true \
  --output artifacts/gguf-pinned-suite.jsonl
```

`INFERQ_NUM_THREADS` sizes candle's own CPU thread pools and inferq's
multi-row dense-path rayon pool consistently from one value; setting
`CANDLE_NUM_THREADS`/`RAYON_NUM_THREADS` directly still works (equal values
take effect as before; unequal values log a warning and `CANDLE_NUM_THREADS`
wins), and either pair still applies when `INFERQ_NUM_THREADS` is unset.

The output path is created with no-overwrite semantics. Quantized schema
version 3 embeds the source revision, compile-time AVX2/FMA status, effective
Candle/Rayon thread counts, host and storage metadata, local GGUF identity,
combined load telemetry, the shared warmup report, rendered prompt and token
IDs, greedy correctness status, TTFT/decode rates, expert-cache activity, RSS,
faults, and physical I/O counters. Timing attribution includes disjoint model
stages plus nested attention, DeltaNet, and MoE operations and all 48 layers.
Routed-expert compute is further split into fused gate/up projection,
activation, down projection, and route-weighted accumulation. These four
suboperations overlap the enclosing routed-expert compute value and must not be
added to it.
The separate target directory prevents the canonical generic validation build
from overwriting the native benchmark binary.

Check or edit prompt cases without loading the 46 GiB GGUF:

```bash
./target/release/gguf_bench \
  --tokenizer-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --validate-prompts-only
```

For a bounded rerun, add `--only chat-prefill-32`. The full expert pinning is
still performed because the measured case must have the same residency state
as the complete suite. `--warmup-all-experts false --expert-cache-mib 0`
retains the cold/page-cache experimental path, and the artifact labels that
state explicitly.

The first complete native-CPU suite on 2026-08-14 produced four valid records:

| Workload | Input | Output | TTFT | Decode | Expert hits | Physical reads | RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `greedy-regression` | 1 | 2 | 0.35 s | 3.72 tok/s | 2,880/2,880 | 0 MiB | 47,259 MiB |
| `chat-decode-16` | 17 | 16 | 3.35 s | 3.87 tok/s | 46,080/46,080 | 0 MiB | 47,306 MiB |
| `chat-prefill-32` | 32 | 1 | 6.20 s | n/a | 46,080/46,080 | 0 MiB | 47,309 MiB |
| `chat-sustained-128` | 23 | 128 | 4.51 s | 3.74 tok/s | 216,000/216,000 | 0 MiB | 47,319 MiB |

The shared warmup loaded 43.5 GiB into 73,728 cache entries in 227.3 seconds.
The exact greedy IDs were `[284, 526]`; the 32-token TTFT and sustained decode
both exceed the project's stretch gates. These numbers came from a
`target-cpu=native` build on the host recorded in the artifact. Do not compare
them to a generic x86-64 release binary as though the build configuration were
the same.

### Native decode optimization result

The first schema-2 profile exposed DeltaNet recurrence as 11.36 seconds of a
32.49-second sustained decode. Its scalar loop traversed row-major
`[key,value]` state with a 128-float stride. Reordering the mathematically
identical work over contiguous value rows reduced recurrence to 1.42 seconds.
Pre-resolved numeric expert handles and a fully-resident cache state also
remove per-token tensor-name allocation and unnecessary LRU maintenance.

On the same 23-input/128-output workload with the same four native threads:

| Revision | TTFT | Decode | Cache hits | Physical reads | RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial native suite | 4.51 s | 3.74 tok/s | 216,000 | 0 MiB | 47,319 MiB |
| Contiguous DeltaNet + resident cache | 3.15 s | 5.62 tok/s | 216,000 | 0 MiB | 47,263 MiB |
| Reusable DeltaNet scratch | 3.25 s | 5.77 tok/s | 216,000 | 0 MiB | 47,271 MiB |
| Flat convolution + fused QKV/gate | 3.30 s | 5.87 tok/s | 216,000 | 0 MiB | 47,260 MiB |
| Fused routed/shared expert gate-up | 3.20 s | 5.99 tok/s | 144,000 | 0 MiB | 47,328 MiB |

All 128 generated token IDs matched the initial artifact exactly. The accepted
change improves sustained decode by 50.2%; an algebraically equivalent version
that reassociated floating-point multiplication was rejected after diverging
at generated token 28. Replacing nested projection readback and per-token
DeltaNet work-vector allocation with flat, layer-owned scratch preserved all
128 IDs and reduced sustained decode from 22.59 to 22.01 seconds. DeltaNet
gated normalization fell from 188 to 132 ms over the run; its quantized
projections remain the largest DeltaNet operation at 5.06 seconds.

A follow-up flattened the depthwise-convolution weights and consumed the
already-contiguous Q/K/V projection directly instead of copying it through four
work vectors. All 128 IDs still matched and the convolution substage fell from
856 to 604 ms over 4,572 DeltaNet layer calls (29.4%). That run's total decode
was 22.54 seconds because every unrelated major stage was 2--4% slower, so it
is recorded as a targeted kernel improvement rather than a new end-to-end
headline. The final same-channel history-update fold also retained the canonical
two-token `[284, 526]` smoke result.

QKV and gate are both Q8_0 matrices with 2,048 input columns. Concatenating
their compressed rows once during load preserves each row byte-for-byte and
reduces the two DeltaNet input projections to one kernel launch. The resulting
run held 5.87 token/s over 127 decode passes; all 128 IDs matched, all 216,000
expert requests hit cache, and inference performed no physical reads. Relative
to the flat-convolution control, DeltaNet projections fell from 5.20 to 5.03
seconds and total decode from 22.54 to 21.65 seconds. Unrelated stages improved
by 3--4% in the same run, so only part of the projection delta can be assigned
confidently to fusion.

A separate 8-Candle/4-Rayon-thread run was rejected as the sustained default.
It improved DeltaNet projections by 3.7% and LM head by 6.1%, but slowed MoE by
2.6% and total decode by 0.7% versus the 4/4 control. Four Candle threads remain
the documented sustained setting on this four-core/eight-thread host.

Routed and shared expert gate/up matrices have the same shape and quantization.
Full warmup still reads all 144 GGUF tensors sequentially, then replaces each
resident gate/up pair with one byte-preserving row-concatenated entry. Resident
expert bytes stay at 43.5 GiB while entries fall from 73,728 to 49,152. Runtime
cache requests fall from 216,000 to 144,000 and each expert uses one gate/up
kernel launch instead of two. All 128 IDs matched; MoE fell from 8.46 to 8.00
seconds, routed compute from 6.30 to 5.87 seconds, and total decode from 21.65
to 21.21 seconds (5.99 token/s).

### Routed-expert projection profile

A three-repetition fully resident control established a `6.0206 token/s` mean
with a `5.9978`--`6.0409 token/s` range and `0.294%` coefficient of variation.
Mean decode wall time was `21.094 s`; all three 128-token sequences matched the
accepted artifact exactly. Use this variance band when judging small changes on
the i7-6700 host.

Schema 3 split the `5.835 s` routed-expert compute region as follows:

| Routed-expert operation | Decode time | Share of routed compute |
| --- | ---: | ---: |
| Fused gate/up quantized projection | 3.454 s | 59.2% |
| Down quantized projection | 2.029 s | 34.8% |
| SiLU and gate/up product | 0.231 s | 4.0% |
| Route-weighted accumulation | 0.112 s | 1.9% |
| Timer remainder | 0.009 s | 0.2% |

The two compressed projections therefore account for 94.0% of routed-expert
compute. Fusing only route-weighted accumulation has an end-to-end ceiling near
0.5% and is below the threshold for invasive work.

An exact selected-expert prototype reused Candle's Q4_K/Q5_K dot products but
scheduled all ten experts through Rayon so gate/up could share one input
quantization and both projections could share a dispatch. Its outputs and all
128 generated IDs were bit-identical. Fine-grained scheduling decoded at
`5.754 token/s`; a coarser 40-task layout reached only `5.827 token/s`. In the
latter run routed-expert compute rose to `6.168 s` and total MoE time to
`8.389 s`. Both versions were rejected and removed.

The next credible exact MoE improvement must work inside Candle's persistent
quantized-matmul worker pool (or a model-specific replacement), not wrap the
existing kernels in a second scheduler. A selected-matrix API there could
share input quantization and barriers without giving up the worker pool's
lower dispatch cost. Larger gains still require fewer weight bytes per token,
which means a different execution quantization or an explicitly approximate
reduction in active experts.

That paragraph turned out to name the missing piece exactly. The rejected
prototype scheduled the experts through Rayon while each expert's matmul was
still Candle's, so it *was* a second scheduler wrapped around candle's own
`BarrierPool` and it lost. Doing the same scheduling with a model-specific
replacement for the inner kernel — `src/qgemm.rs`, which runs entirely on the
calling thread — wins: MoE compute in a 256-row prefill pass fell from 12.21 to
6.12 ms/token and prefill from 26.4 to 32.9 tok/s. Decode is untouched, still
one row and still token-major.

That change also moved what the MoE's timing fields mean, in the grouped
(multi-row) path only. `expert_compute` is the wall time of the whole parallel
region, loads included, which is what the pass actually spent. `expert_load`,
`expert_gate_up`, `expert_activation` and `expert_down` are summed across the
workers and are therefore thread time: they add up to roughly the thread count
times `expert_compute`, and must not be read as a share of it. The token-major
path is unchanged and all six remain serial wall time there.
