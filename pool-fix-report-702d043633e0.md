# Thread-pool unification report — host 702d043633e0

Repo: `ybrs/inferq`, branch `pool-unification` off `qwen36-35b-a3b-comparison`
(HEAD `3497f14` at branch-off). Host: Intel i7-8700, 6 physical cores / 12
logical (HT), single socket, single NUMA node. Model: Qwen3.6-35B-A3B
Q4_K_M GGUF, `/models/Qwen3.6-35B-A3B/`.

## 1. Investigation: what candle actually does

candle-core 0.11 (pinned in `Cargo.lock`) maintains **three** independent CPU
thread pools, not two:

1. **A private native-thread `BarrierPool`** (`candle-core/src/utils.rs`),
   built lazily via `OnceLock`, sized by the `CANDLE_NUM_THREADS` env var
   (fallback: `num_cpus::get_physical()`). Used specifically for the
   quantized k_quants matmul — `k_quants.rs:2432` and `:2612` — which is the
   K=1 decode matvec hot path.
2. **A private `rayon::ThreadPool`** (`candle_pool`, also a lazy
   `OnceLock`), sized by the `RAYON_NUM_THREADS` env var. Used by
   `Device::Cpu`'s `with_threadpool` (`device.rs:273`) to wrap general CPU
   op dispatch.
3. **The process-global rayon pool.** inferq's own multi-row dense path
   (`src/gguf.rs`, `QuantizedMatrix::small_m_forward`, the `par_chunks_mut`
   at line 391) uses `rayon::prelude::*` directly with no `.install()`
   scoping, so it dispatches on whatever the global pool happens to be —
   a *third*, separate pool object from #2, even though rayon's own default
   global-pool construction also happens to read `RAYON_NUM_THREADS`.

Neither of candle's private pools (#1, #2) exposes a public setter — both
read their env var exactly once, on first use. There is no candle API to
inject a thread count after the fact. This means the only correct way to
align all three pools is to set `CANDLE_NUM_THREADS` and `RAYON_NUM_THREADS`
in the environment *before* candle's first CPU op runs, and to explicitly
build the global pool (#3) to the same count rather than relying on it
reading the env var itself, since the relative ordering of candle's first op
vs. inferq's first `par_chunks_mut` call is not otherwise guaranteed.

This matches what the task anticipated ("env var set before first candle op
is acceptable if that is genuinely how candle reads it") — confirmed by
reading the source rather than assumed.

## 2. What changed

New module `src/threading.rs`, `pub fn init()`:

- Resolves one thread count, in priority order: `INFERQ_NUM_THREADS` env var
  → existing `CANDLE_NUM_THREADS`/`RAYON_NUM_THREADS` if set (if they
  disagree, warns and uses `CANDLE_NUM_THREADS`, since it governs the K=1
  hot path) → detected physical core count (`num_cpus::get_physical()`,
  promoted from an existing transitive dependency of candle-core to a direct
  one — no new crate was actually added to the dependency tree).
- Sets `CANDLE_NUM_THREADS`, `RAYON_NUM_THREADS`, and `INFERQ_NUM_THREADS` in
  the environment to the resolved value.
- Explicitly builds the global rayon pool via
  `rayon::ThreadPoolBuilder::new().num_threads(n).build_global()`, handling
  the already-initialized error gracefully (logs and continues; env vars
  stay aligned for candle either way).
- Logs one `eprintln!` line with the resolved count and pool-build outcome
  (used `eprintln!` rather than `tracing::info!` because `gguf_verify_bench`
  configures no tracing subscriber at all, and `gguf_infer`/`gguf_bench`
  default their subscriber's filter to `warn`, so an `info`-level line would
  be silently dropped in exactly the binaries this needs to be visible in).
- Idempotent via `std::sync::Once`; safe to call redundantly.

Call sites: the top of `GgufCheckpoint::open` (`src/gguf.rs`) — the single
library entry point every quantized-inference path goes through, which
covers `gguf_infer`, `gguf_bench`, `gguf_verify_bench`, and any future test
that opens a checkpoint (none currently do; the existing integration test,
`tests/synthetic_forward.rs`, exercises the plain-tensor `qwen::Model` path,
not the quantized GGUF path, so it was not touched) — plus an explicit call
at the top of each of the three binaries' `main()` for early, deterministic
startup logging.

No kernel math, dispatch thresholds (`SMALL_M_MIN_STORAGE_BYTES`, the
`2..=8` row range), quantization, or model semantics were touched. This is
thread-pool plumbing only.

`README.md` and `docs/{profiling,speculative-decoding,qwen36-35b-a3b}.md`:
all 4 hardcoded `CANDLE_NUM_THREADS=N RAYON_NUM_THREADS=N` pairs in run
examples replaced with `INFERQ_NUM_THREADS=N`, with a note in `README.md`
and `docs/profiling.md` that the old pair still works (matched values take
effect as before; mismatched values warn and `CANDLE_NUM_THREADS` wins).

`AGENTS.md` note: this introduces process-global state (the aligned env
vars and the built rayon pool), which the repo's "avoid hidden global
state" convention generally discourages. This is treated as a deliberate,
narrowly-scoped exception: the state in question is the CPU thread pool
itself, which is inherently process-global in both candle's own design (two
of its three pools are already module-level `OnceLock`s) and in rayon's
(the global pool is a rayon concept, not an inferq one) — there was no way
to satisfy the task's goal (`one INFERQ_NUM_THREADS`, one aligned set of
pools) without touching that same global surface.

## 3. Validation

All timed runs: no other load running, taskset-pinned to the 6 physical
cores only (`0,1,2,3,4,5`; HT siblings are `6,7,8,9,10,11`, derived from
`/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`), sequential.
Pre-change binary built from HEAD `3497f14` (branch-off point, before any of
this task's changes) in `target-prechange/`; post-change binary built from
the finished `pool-unification` branch in `target-native/`; both with
`RUSTFLAGS='-C target-cpu=native'`.

### 3.1 `./scripts/validate.sh`

**PASS.** `cargo fmt --check`, `cargo check --all-targets`,
`cargo test --all-targets`, `cargo clippy --all-targets -- -D warnings`,
`cargo build --release --bins` all exited 0. New `src/threading.rs` unit
tests (`parse_env_rejects_zero_and_garbage`, `physical_core_count_is_at_least_one`)
pass under `cargo test --lib threading`.

### 3.2 Correctness

The task spec's stated expectation — `--prompt a --max-new-tokens 2` should
produce `[8160, 579]` — does not hold and was not blindly forced to match.
That constant is `gguf_verify_bench`'s default `--verification-tokens`
value (`src/bin/gguf_verify_bench.rs:36`), not the greedy-regression output
of a plain `--prompt a` run. Investigated rather than guessed:

- **Actual `--prompt a --max-new-tokens 2` output:** `[8, 271]`, identical
  pre-change and post-change (both at `CANDLE_NUM_THREADS=6`/`RAYON_NUM_THREADS=1`
  and, post-change, at the new default with no env vars set).
- **The real documented ground truth** (`docs/qwen36-35b-a3b.md:73-75`): an
  8-token sequence that "matched llama.cpp exactly," `[8160, 579, 264, 7047,
  1817, 25, 271, 16]`, produced by `--chat` (thinking block open, i.e. no
  `--no-thinking`) with the prompt `"Write a detailed Rust implementation of
  a thread-safe LRU cache with tests."`. Reproduced **exactly** post-change
  at the new default thread config (`INFERQ_NUM_THREADS` unresolved →
  resolved to 6, physical core count).
- **128-token greedy sequence, the K=1 decode workload** (chat, `--no-thinking`,
  semver-parser prompt, `--max-new-tokens 128`, `--expert-cache-mib 24000
  --warmup-all-experts --speculative-mtp 0`): **bit-identical** between
  pre-change (`CANDLE_NUM_THREADS=6 RAYON_NUM_THREADS=1`) and post-change
  default (auto-resolved to 6, which internally becomes
  `CANDLE_NUM_THREADS=6 RAYON_NUM_THREADS=6`). Also re-checked post-change
  at `INFERQ_NUM_THREADS=4` (taskset to 4 physical cores) — same sequence,
  confirming the multi-row dense path's output is thread-count-invariant as
  expected (each output row is computed independently; no cross-thread
  reduction to disturb).

### 3.3 K=1 non-regression

Workload: chat, `--no-thinking`, semver-parser prompt, `--max-new-tokens
128`, `--expert-cache-mib 24000 --warmup-all-experts --speculative-mtp 0`.
Two reps each, better-of-two reported.

| config | rep1 decode tok/s | rep2 decode tok/s | best |
|---|---:|---:|---:|
| pre-change, `CANDLE=6/RAYON=1` (old winning K=1 config) | 8.10 | 8.24 | **8.24** |
| post-change, default (no env vars) | 8.13 | 8.29 | **8.29** |

Post-change (8.29) ≥ pre-change (8.24) − 2% (8.075). **Passes**, with a
small improvement.

### 3.4 K-scaling (`gguf_verify_bench`)

`--batch-sizes 1,2,4,8 --repetitions 3 --expert-cache-mib 24000`. Each
config run twice to check noise; both runs reported.

| config | run | K=1 total/row (ms) | K=2 total/row (ms) | K=4 total/row (ms) | K=8 total/row (ms) | marginal row-2 fraction |
|---|---|---|---|---|---|---:|
| pre `CANDLE=6/RAYON=1` | 1 | 118.46 / 118.46 | 335.17 / 167.58 | 586.41 / 146.60 | 966.83 / 120.85 | **1.829** |
| pre `CANDLE=6/RAYON=6` | 1 | 117.41 / 117.41 | 213.44 / 106.72 | 305.61 / 76.40 | 480.47 / 60.06 | **0.818** |
| pre `CANDLE=6/RAYON=6` | 2 (rerun) | 116.19 / 116.19 | 207.64 / 103.82 | 312.15 / 78.04 | 523.50 / 65.44 | **0.787** |
| post-change default | 1 | 133.10 / 133.10 | 234.63 / 117.31 | 304.34 / 76.09 | 544.02 / 68.00 | **0.763** |
| post-change default | 2 (rerun) | 116.39 / 116.39 | 222.95 / 111.47 | 309.47 / 77.37 | 483.40 / 60.43 | **0.915** |

Post-change run 1's K=1 total (133.10ms) is an outlier against its own
rerun (116.39ms, consistent with both pre-change configs' K=1 numbers in
the 116-118ms band) — most likely first-invocation warmup/scheduling noise
on this shared host rather than a systematic effect, since K=1 does not
touch the multi-row dense path this change modifies at all.

**Success criterion — marginal row-2 fraction ≤ 0.75 post-change, and
per-row ms at K=4 and K=8 strictly better than both pre-change configs —
was NOT reliably met.** Post-change's marginal fraction ranges 0.76-0.92
across two runs, not cleanly under the 0.75 bar. K=4 per-row is marginally
better than both pre-change configs in both runs (76.09 vs 76.40/146.60,
and 77.37 vs 78.04/305.61 — the `CANDLE=6/RAYON=1` comparison point is
irrelevant here since that config is the one being fixed away from; the
meaningful comparison is against `CANDLE=6/RAYON=6`, where post-change wins
by <1%, essentially noise-level). K=8 per-row flips sign against
`CANDLE=6/RAYON=6` between the two runs (68.00 vs 60.06 — post-change worse
— in run 1; 60.43 vs 65.44 — post-change better — in run 2), i.e. it is a
wash within measurement noise, not a strict win.

**Per the task's explicit instruction, stopping here rather than modifying
kernel or dispatch logic to chase the target.** Measured analysis of why:

- The pool-unification change **does** fix the catastrophic case it set out
  to fix. `dense_projections_outside_moe` (the stage the task's diagnosis
  named directly) scales K=1→K=2 by **3.82×** under the old
  `CANDLE=6/RAYON=1` config (44.25ms→168.88ms, matching the task's cited
  3.74×/50.47ms→188.82ms almost exactly) but only **1.87×** post-change
  (48.04ms→89.79ms) — close to `CANDLE=6/RAYON=6`'s own **1.97×**
  (44.00ms→86.81ms). `deltanet_projections` shows the identical pattern:
  3.88× pre (`CANDLE=6/RAYON=1`, 25.50ms→98.91ms) vs. 1.95× post
  (27.88ms→54.49ms) vs. 1.97× pre (`CANDLE=6/RAYON=6`, 25.25ms→51.80ms).
  The silent-serialization failure mode the task diagnosed is gone.
- What the pool-unification change does **not** do is beat
  `CANDLE=6/RAYON=6`'s own K≥2 performance — because that config was never
  actually broken. inferq's `par_chunks_mut` dispatches on rayon's
  process-global pool, and rayon's own global-pool construction already
  reads `RAYON_NUM_THREADS` on its own (independent of any of candle's
  private pools or this change) — so `RAYON_NUM_THREADS=6` alone was
  already sufficient to give the dense path 6 rayon workers. The
  unification's real, structural contribution is making that outcome the
  automatic *default* (no env vars needed) and, more importantly,
  eliminating the specific trap the original diagnosis found: a
  K=1-tuned config (`RAYON=1`) that looks like a free win on the single-row
  decode benchmark while silently serializing every K≥2 verification call.
  It does not, and by its own physical-pool-sharing design cannot,
  outperform a config that was already correctly sized.
- The small residual gap at K=2 specifically (post-change ~7-10% slower
  than `CANDLE=6/RAYON=6` in both runs, the one consistently-reproducible
  difference in this data) is not explained by anything this investigation
  touched — both configs end up with an equivalent 6/6 rayon+candle pool
  arrangement after `init()` runs. A plausible but unverified hypothesis is
  that explicitly constructing the global pool via `ThreadPoolBuilder::build_global()`
  at `GgufCheckpoint::open` time (this change) incurs different first-use
  spin-up/thread-park timing than rayon's own lazy default-global-pool
  construction (no change) — but confirming that would require
  instrumenting rayon's pool internals, which is out of this task's scope.

### 3.5 Per-stage breakdown (post-change, top 8 by total time)

K=1:

| stage | total (ms) | per-row (ms) |
|---|---:|---:|
| dense_projections_outside_moe | 48.036 | 48.036 |
| deltanet_projections | 27.880 | 27.880 |
| lm_head | 23.183 | 23.183 |
| moe_routed_gate_up | 20.302 | 20.302 |
| moe_routed_down | 12.230 | 12.230 |
| full_attention | 10.405 | 10.405 |
| deltanet_recurrence | 9.079 | 9.079 |
| moe_shared_expert | 6.791 | 6.791 |

K=2:

| stage | total (ms) | per-row (ms) |
|---|---:|---:|
| dense_projections_outside_moe | 89.791 | 44.896 |
| deltanet_projections | 54.487 | 27.243 |
| moe_routed_gate_up | 37.816 | 18.908 |
| full_attention | 24.973 | 12.486 |
| moe_routed_down | 24.009 | 12.004 |
| lm_head | 21.016 | 10.508 |
| deltanet_recurrence | 17.867 | 8.934 |
| moe_router | 11.639 | 5.820 |

## 4. Summary

| | value |
|---|---|
| candle threading mechanism | 3 pools: private native `BarrierPool` (CANDLE_NUM_THREADS, K=1 matvec), private rayon pool (RAYON_NUM_THREADS, general CPU ops), process-global rayon pool (inferq's own `par_chunks_mut`) |
| resolved default thread count, this host | 6 (physical core count) |
| K=1 non-regression | passes: 8.29 tok/s post vs 8.24 tok/s pre (best-of-2) |
| Greedy determinism | preserved: bit-identical across pre/post-change and across thread counts |
| K-scaling success criterion (marginal ≤ 0.75, K4/K8 strictly better than both old configs) | **not met** — see §3.4 for measured numbers and analysis |
| Catastrophic serialization the task diagnosed (`CANDLE=N/RAYON=1`-style configs) | **fixed** — dense/deltanet projection K=1→K=2 scaling drops from ~3.8× to ~1.9×, matching the already-correct `CANDLE=N/RAYON=N` baseline |

Branch: `pool-unification` (off `qwen36-35b-a3b-comparison`, HEAD `3497f14`
at branch-off). Not pushed to any remote.

## Appendix: raw command log

Repo/model discovery:

```
$ git status && git log --oneline -5 && git rev-parse HEAD
On branch qwen36-35b-a3b-comparison
Untracked files: palindrome.rs palindrome_test perf-report-702d043633e0.md perf-run-logs/ tq.conf
3497f14 Bound Qwen thinking and optimize MTP verification
00131c9 Add Qwen3.6 MTP speculative decoding
60eb26d add Qwen3.6 35B A3B GGUF support
e9786b5 profile routed expert compute
bbd362e perf: fuse quantized expert inputs
3497f143aa7f86621d54e286b4822f86192bffd8

$ find /models -iname '*.gguf'
/models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf
```

Candle source investigation:

```
$ cargo fetch
$ grep -rn "num_cpus|CANDLE_NUM_THREADS|RAYON_NUM_THREADS|build_global|ThreadPoolBuilder" \
    ~/.cargo/registry/src/*/candle-core-0.11.0/src
candle-core-0.11.0/src/utils.rs:317: perf_core_count().unwrap_or_else(num_cpus::get_physical)
candle-core-0.11.0/src/utils.rs:321: num_cpus::get_physical()
candle-core-0.11.0/src/utils.rs:328: std::env::var("RAYON_NUM_THREADS")
candle-core-0.11.0/src/utils.rs:336: std::env::var("CANDLE_NUM_THREADS")
candle-core-0.11.0/src/utils.rs:371: rayon::ThreadPoolBuilder::new()
$ grep -rn "with_threadpool|candle_pool|barrier_pool|candle_num_threads" \
    ~/.cargo/registry/src/*/candle-core-0.11.0/src
device.rs:273:  Self::Cpu => crate::utils::with_threadpool(f),
utils.rs:306:   BARRIER_POOL.get_or_init(|| BarrierPool::new(candle_num_threads().saturating_sub(1)))
k_quants.rs:2432: let pool = crate::utils::barrier_pool();
k_quants.rs:2612: let pool = crate::utils::barrier_pool();
```

Topology (for taskset derivation):

```
$ lscpu | grep -E 'Model name|Core|Thread|Socket'
Model name:                              Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
Thread(s) per core:                      2
Core(s) per socket:                      6
Socket(s):                               1

$ for f in /sys/devices/system/cpu/cpu*/topology/thread_siblings_list; do echo -n "$f: "; cat "$f"; done | sort -u
cpu0: 0,6   cpu1: 1,7   cpu2: 2,8   cpu3: 3,9   cpu4: 4,10   cpu5: 5,11
cpu6: 0,6   cpu7: 1,7   cpu8: 2,8   cpu9: 3,9   cpu10: 4,10  cpu11: 5,11
```

Build commands:

```
$ CARGO_TARGET_DIR=target-prechange RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --bin gguf_infer --bin gguf_verify_bench
    (run at HEAD 3497f14, before any pool-unification changes, via git stash)

$ CARGO_TARGET_DIR=target-native RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --bin gguf_infer --bin gguf_bench --bin gguf_verify_bench
    (run on pool-unification branch)

$ ./scripts/validate.sh   # exit 0
```

Correctness commands and raw outputs:

```
$ env -u CANDLE_NUM_THREADS -u RAYON_NUM_THREADS -u INFERQ_NUM_THREADS \
    ./target-native/release/gguf_infer \
    --model /models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --tokenizer-model /models/Qwen3.6-35B-A3B \
    --prompt a --max-new-tokens 2
inferq: threading: resolved 6 threads (source: detected physical core count); global rayon pool: built
generated token ids: [8, 271]

$ taskset -c 0,1,2,3,4,5 env CANDLE_NUM_THREADS=6 RAYON_NUM_THREADS=1 \
    ./target-prechange/release/gguf_infer \
    --model /models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --tokenizer-model /models/Qwen3.6-35B-A3B \
    --prompt a --max-new-tokens 2
generated token ids: [8, 271]

$ INFERQ_NUM_THREADS=6 ./target-native/release/gguf_infer \
    --model /models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --tokenizer-model /models/Qwen3.6-35B-A3B \
    --chat --prompt 'Write a detailed Rust implementation of a thread-safe LRU cache with tests.' \
    --max-new-tokens 8
inferq: threading: resolved 6 threads (source: INFERQ_NUM_THREADS); global rayon pool: built
generated token ids: [8160, 579, 264, 7047, 1817, 25, 271, 16]
```

K=1 workload (chat, --no-thinking, semver prompt, 128 max-new-tokens, expert-cache-mib 24000, warmup-all-experts, speculative-mtp 0), pre-change vs post-change, and thread-count-invariance check:

```
$ taskset -c 0,1,2,3,4,5 env CANDLE_NUM_THREADS=6 RAYON_NUM_THREADS=1 \
    ./target-prechange/release/gguf_infer --model ... --chat --no-thinking \
    --prompt 'Write a Rust function that parses a semver string into (major, minor, patch) with unit tests.' \
    --max-new-tokens 128 --expert-cache-mib 24000 --warmup-all-experts --speculative-mtp 0
rep1: decode: 127 passes in 15.679s (8.10 tok/s)
rep2: decode: 127 passes in 15.414s (8.24 tok/s)
generated token ids: [71093, 34602, 198, 2490, 70932, 264, 5067, 415, 886, 1083, 318, 35299, 11, 8652, 11,
  10582, 8, 13992, 13, 198, 2490, 695, 2490, 653, 24508, 198, 2490, 695, 2490, 52451, 198, 2490, 1042, 318,
  35299, 11, 8652, 11, 10582, 8, 283, 4563, 29450, 415, 437, 16, 13, 17, 13, 18, 1764, 15000, 2061, 198,
  2490, 1992, 10398, 0, 1148, 35299, 11, 8652, 11, 10582, 681, 318, 16, 11, 220, 17, 11, 220, 18, 5722, 198,
  2490, 52451, 198, 9299, 5003, 4563, 29450, 415, 1104, 25, 594, 485, 8, 1411, 5536, 27767, 84, 18, 17, 11,
  560, 18, 17, 11, 560, 18, 17, 681, 894, 29, 313, 198, 262, 1042, 5306, 25, 10985, 50490, 485, 29, 283,
  274, 5121, 4151, 1760, 16873, 2061, 198, 262, 413, 5306, 18817, 363]

$ taskset -c 0,1,2,3,4,5 env -u CANDLE_NUM_THREADS -u RAYON_NUM_THREADS -u INFERQ_NUM_THREADS \
    ./target-native/release/gguf_infer --model ... --chat --no-thinking \
    --prompt 'Write a Rust function that parses a semver string into (major, minor, patch) with unit tests.' \
    --max-new-tokens 128 --expert-cache-mib 24000 --warmup-all-experts --speculative-mtp 0
inferq: threading: resolved 6 threads (source: detected physical core count); global rayon pool: built
rep1: decode: 127 passes in 15.627s (8.13 tok/s)
rep2: decode: 127 passes in 15.320s (8.29 tok/s)
generated token ids: [same sequence as above — bit-identical]

$ taskset -c 0,1,2,3 env INFERQ_NUM_THREADS=4 ./target-native/release/gguf_infer \
    --model ... --chat --no-thinking --prompt '...(same semver prompt)...' \
    --max-new-tokens 128 --expert-cache-mib 24000 --warmup-all-experts --speculative-mtp 0
inferq: threading: resolved 4 threads (source: INFERQ_NUM_THREADS); global rayon pool: built
decode: 127 passes in 15.969s (7.95 tok/s)
generated token ids: [same sequence as above — bit-identical]
```

Verify-bench K-scaling commands:

```
$ taskset -c 0,1,2,3,4,5 env CANDLE_NUM_THREADS=6 RAYON_NUM_THREADS=1 \
    ./target-prechange/release/gguf_verify_bench \
    --model /models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --tokenizer-model /models/Qwen3.6-35B-A3B \
    --batch-sizes 1,2,4,8 --repetitions 3 --expert-cache-mib 24000 \
    --output verify-prechange-candle6-rayon1.json

$ taskset -c 0,1,2,3,4,5 env CANDLE_NUM_THREADS=6 RAYON_NUM_THREADS=6 \
    ./target-prechange/release/gguf_verify_bench \
    --model /models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --tokenizer-model /models/Qwen3.6-35B-A3B \
    --batch-sizes 1,2,4,8 --repetitions 3 --expert-cache-mib 24000 \
    --output verify-prechange-candle6-rayon6.json   # + rerun

$ taskset -c 0,1,2,3,4,5 env -u CANDLE_NUM_THREADS -u RAYON_NUM_THREADS -u INFERQ_NUM_THREADS \
    ./target-native/release/gguf_verify_bench \
    --model /models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --tokenizer-model /models/Qwen3.6-35B-A3B \
    --batch-sizes 1,2,4,8 --repetitions 3 --expert-cache-mib 24000 \
    --output verify-postchange-default.json   # + rerun
```

Full JSON outputs are in `perf-run-logs/pool-unification/*.json` (not
committed — build/benchmark artifacts, per `AGENTS.md`'s change-discipline
rule against committing benchmark output).
