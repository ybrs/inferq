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

CANDLE_NUM_THREADS=4 \
RAYON_NUM_THREADS=4 \
./target-native/release/gguf_bench \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --tokenizer-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --prompts benchmarks/gguf-prompts.json \
  --expert-cache-mib 46000 \
  --warmup-all-experts true \
  --output artifacts/gguf-pinned-suite.jsonl
```

The output path is created with no-overwrite semantics. Quantized schema
version 2 embeds the source revision, compile-time AVX2/FMA status, effective
Candle/Rayon thread counts, host and storage metadata, local GGUF identity,
combined load telemetry, the shared warmup report, rendered prompt and token
IDs, greedy correctness status, TTFT/decode rates, expert-cache activity, RSS,
faults, and physical I/O counters. Timing attribution includes disjoint model
stages plus nested attention, DeltaNet, and MoE operations and all 48 layers.
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

On the same 23-input/128-output workload, with the same four native threads and
all 216,000 expert requests hitting cache:

| Revision | TTFT | Decode | Physical reads | RSS |
| --- | ---: | ---: | ---: | ---: |
| Initial native suite | 4.51 s | 3.74 tok/s | 0 MiB | 47,319 MiB |
| Contiguous DeltaNet + resident cache | 3.15 s | 5.62 tok/s | 0 MiB | 47,263 MiB |
| Reusable DeltaNet scratch | 3.25 s | 5.77 tok/s | 0 MiB | 47,271 MiB |
| Flat convolution + fused QKV/gate | 3.30 s | 5.87 tok/s | 0 MiB | 47,260 MiB |

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
