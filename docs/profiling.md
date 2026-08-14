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
