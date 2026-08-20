# inferq performance diagnosis — host `702d043633e0`

Repo: `ybrs/inferq`, branch `qwen36-35b-a3b-comparison`, HEAD `3497f14` at time of testing.
Model: `/models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` (22,285,080,192 bytes), tokenizer/config in `/models/Qwen3.6-35B-A3B/`.
Date: 2026-08-17.

No source files were modified. Only `git checkout` was used to move between commits for the Step 6 bisect; the working tree was restored to `qwen36-35b-a3b-comparison` (HEAD `3497f14`) at the end. Pre-existing untracked files in the repo root (`palindrome.rs`, `palindrome_test`, `tq.conf`) are unrelated scratch files left over from other work and were not touched.

**Important topology note up front:** the host is a 6-physical-core / 12-thread desktop CPU (Intel i7-8700), single socket, single NUMA node — not the 12-physical-core machine the task's example thread-sweep table assumes. All thread-sweep configurations below were adapted to this topology as instructed (see Section 2).

---

## 1. Host summary

| Item | Value |
|---|---|
| CPU | Intel(R) Core(TM) i7-8700 @ 3.20GHz (Coffee Lake) |
| Physical cores | 6 |
| Threads/core | 2 (HT) → 12 logical CPUs |
| Sockets | 1 |
| NUMA nodes | 1 (`numactl` not installed; confirmed via `lscpu`: "NUMA node(s): 1", "NUMA node0 CPU(s): 0-11") |
| L1d / L1i | 192 KiB / 192 KiB (6 instances, i.e. 32 KiB each per core) |
| L2 | 1.5 MiB total (6 instances → 256 KiB/core) |
| L3 | 12 MiB (1 shared instance) |
| ISA flags of interest | `avx2` ✅, `fma` ✅, `avx512f` ❌, `avx512_vnni` ❌, `avx_vnni` ❌ (Coffee Lake predates both AVX-512 client support and AVX-VNNI; those need Alder Lake+ or Skylake-SP+) |
| Governor | `powersave` on all 12 logical CPUs |
| Max turbo (advertised) | 4600 MHz |
| Observed frequency under load | ~4295–4300 MHz sampled every 2s during a live decode run (one idle-state sample at 1813 MHz was caught before the run's compute phase started); no throttling or core-parking observed |
| MemTotal | 131,813,152 kB (≈125.7 GiB) |
| HugePages | `Hugepagesize` 2048 kB, `HugePages_Total` 0 (no static hugepages configured); `AnonHugePages` 34,816 kB (~34 MB, incidental THP) |
| Memory channels | Unknown — `dmidecode` requires sudo, which is unavailable in this environment; skipped as instructed |

### Measured memory bandwidth (custom STREAM Triad, `-O3 -fopenmp`, `STREAM_ARRAY_SIZE=80000000`, best-of-10 excluding the first iteration)

| OMP_NUM_THREADS | Triad MB/s |
|---|---|
| 1 | 15,445.9 |
| 3 (half of 6 physical) | 18,041.4 |
| 6 (all physical) | 18,280.0 |

Bandwidth is essentially saturated already at 3 threads — consistent with a dual-channel desktop memory controller. **Best measured Triad bandwidth: 18.28 GB/s.**

### Roofline

Per the task's stated per-token weight traffic estimate of 1.8–2.0 GB for this model:

```
ceiling_tok_s ≈ 18.28 GB/s / 1.9 GB ≈ 9.62 tok/s
```

**Roofline ceiling ≈ 9.6 tok/s** for target-only (K=1) decode on this host.

---

## 2. Thread sweep (Step 4) — target-only decode

Fixed workload for every run (greedy, identical prompt/length):
```
--model /models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
--tokenizer-model /models/Qwen3.6-35B-A3B \
--chat --no-thinking \
--prompt 'Write a Rust function that parses a semver string into (major, minor, patch) with unit tests.' \
--max-new-tokens 128 \
--expert-cache-mib 24000 --warmup-all-experts \
--speculative-mtp 0
```

Physical-core list derived from `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`: physical cores are logical CPUs **0,1,2,3,4,5**; their HT siblings are **6,7,8,9,10,11** (pairs 0↔6, 1↔7, 2↔8, 3↔9, 4↔10, 5↔11).

Since this host has only 6 physical cores (not 12 as the example table assumed), the table was adapted:

| run | CANDLE_NUM_THREADS | RAYON_NUM_THREADS | taskset | notes |
|---|---|---|---|---|
| a | 4 | 4 | `0,1,2,3` (4 physical) | |
| b | 6 | 6 | `0,1,2,3,4,5` (all 6 physical) | |
| c | 8 | 8 | `0,1,2,3,4,5,6,7` (6 physical + 2 HT siblings, since only 6 physical cores exist) | |
| d | 12 | 12 | `0-11` (all logical, full HT) | matches spec's "12 = logical count" fallback |
| e | 6 | 6 | *(none)* | best-so-far config, no taskset, for comparison |
| f | 6 | 1 | `0,1,2,3,4,5` | tests rayon/candle pool double-subscription at the best cores |

Each configuration was run twice (page cache warm after run 1); better-of-two decode tok/s reported. Every run also does a ~12.5–13.0s in-process expert-cache warmup (not counted in decode timing) — RSS was consistently ~21.5 GiB peak across all configurations, so it carries no signal for thread choice here.

| run | rep1 decode tok/s | rep2 decode tok/s | best decode tok/s | decode wall (best) | prefill tok/s (best) | RSS peak (best) |
|---|---|---|---|---|---|---|
| a (4/4, taskset 4 phys) | 8.07 | 8.09 | **8.09** | 15.707s | 12.06 | 21,574.8 MiB |
| b (6/6, taskset 6 phys) | 8.14 | 8.21 | **8.21** | 15.469s | 11.87 | 21,574.7 MiB |
| c (8/8, taskset 6 phys+2 HT) | 8.14 | 8.21 | **8.21** | 15.468s | 13.08 | 21,576.8 MiB |
| d (12/12, taskset all logical/HT) | 5.67 | 6.06 | **6.06** | 20.945s | 11.43 | 21,572.1 MiB |
| e (6/6, no taskset) | 8.18 | 8.14 | **8.18** | 15.534s | 14.00 | 21,577.0 MiB |
| f (6 CANDLE / 1 RAYON, taskset 6 phys) | 8.29 | 8.30 | **8.30** | 15.308s | 14.35 | 21,577.1 MiB |

### Observations

- **12 threads is NOT the winner.** Config d (all 12 logical CPUs, i.e. full hyperthreading) is 26–34% *slower* than every other configuration tested (6.06 vs 8.09–8.30 tok/s). This is the single biggest thread-configuration effect observed and would explain a large fraction of any "just use all cores" regression.
- Configs a/b/c/e/f cluster tightly between 8.07 and 8.30 tok/s (≤2.8% spread) — once you're at 4+ physical cores and off full HT, throughput is essentially flat. This is consistent with the workload being memory-bandwidth-bound (bandwidth itself saturates by 3 threads, see Section 1) rather than compute-thread-bound.
- Adding 2 HT-sibling threads on top of all 6 physical cores (config c, 8 threads) neither helped nor hurt vs. 6 threads alone (config b) — both landed at 8.21 tok/s. Oversubscribing only 2 of 6 physical cores did not measurably contribute or regress.
- `taskset` pinning vs. no pinning (config b vs. e, both CANDLE=6/RAYON=6) made no measurable difference (8.21 vs 8.18, within noise) — the OS scheduler already keeps this workload off HT siblings reasonably well without explicit pinning, at least when only this process is running.
- Rayon/candle pool double-subscription (config f: CANDLE=6 threads doing the compute, RAYON=1 i.e. no separate rayon-level fan-out) was the *best* observed run at 8.30 tok/s, marginally ahead of the CANDLE=6/RAYON=6 configs (8.14–8.21). The gap is small (~1–2%) and close to run-to-run noise, but there is no evidence that a fully-subscribed rayon pool on top of candle's own thread pool helps on this host; if anything it trends slightly worse.

**Winning configuration for Steps 5 and 6: `CANDLE_NUM_THREADS=6 RAYON_NUM_THREADS=1`, `taskset -c 0,1,2,3,4,5` (config f, 8.30 tok/s decode).** Configs b/c/e are statistically indistinguishable from this and would be reasonable alternate choices.

---

## 3. Verification batch scaling (Step 5)

Run at the winning config (`CANDLE_NUM_THREADS=6 RAYON_NUM_THREADS=1`, `taskset -c 0,1,2,3,4,5`):

```
gguf_verify_bench --model ... --tokenizer-model ... \
  --batch-sizes 1,2,4,8 --repetitions 3 --expert-cache-mib 24000 \
  --output verify-702d043633e0.json
```

| K (rows) | wall total (ms) | wall per-row (ms) |
|---|---|---|
| 1 | 132.40 | 132.40 |
| 2 | 381.19 | 190.60 |
| 4 | 547.16 | 136.79 |
| 8 | 989.00 | 123.62 |

**Marginal cost of row 2** = (total_K2 − total_K1) / total_K1 = (381.19 − 132.40) / 132.40 = **1.879** (row 2 costs ~188% *on top of* the full K=1 cost — i.e. total_K2 is nearly 3× total_K1, not ~1.57× as on the reference 4-core host).

Per the task's stated interpretation guide (reference host fraction 0.57, speculation viable only well below ~0.8): **this host's marginal row-2 fraction of 1.88 is far above the 0.8 threshold.** Batched verification is markedly more expensive per row at K=2 than at K=1 on this host, measured via `gguf_verify_bench`'s wall-clock instrumentation.

Note for interpretation: `wall` total does not equal the sum of the individual per-stage timers below (e.g. at K=1, stage sum is 165.56ms vs wall 132.40ms; at K=8, stage sum is 1328.71ms vs wall 989.00ms) — the per-stage instrumentation and the wall clock are evidently not simple sums (likely overlapping/threaded stage timers). Numbers below are reported as emitted by the tool, without further interpretation.

### Top-5 stages by total time, K=1

| stage | total (ms) | per-row (ms) |
|---|---|---|
| dense_projections_outside_moe | 50.47 | 50.47 |
| deltanet_projections | 29.12 | 29.12 |
| moe_routed_gate_up | 20.87 | 20.87 |
| lm_head | 19.09 | 19.09 |
| moe_routed_down | 12.75 | 12.75 |

### Top-5 stages by total time, K=2

| stage | total (ms) | per-row (ms) |
|---|---|---|
| dense_projections_outside_moe | 188.82 | 94.41 |
| deltanet_projections | 112.92 | 56.46 |
| lm_head | 64.52 | 32.26 |
| full_attention | 44.05 | 22.02 |
| moe_routed_gate_up | 43.39 | 21.70 |

(`state_operations` for both K=1 and K=2 report `rejection_replay_seconds` ≈130ms, `checkpoint_seconds` ≈31–33ms, `restore_seconds` ≈8ms — these are the bench harness's row-replay/reset bookkeeping between repetitions, tracked separately from the `stages` timers above.)

Full JSON output: `perf-run-logs/verify-702d043633e0.json` (57,612 bytes; not embedded in full here — see Appendix for the command and a condensed excerpt).

---

## 4. Regression bisect (Step 6)

Rebuilt `gguf_infer` at each commit in a distinct `CARGO_TARGET_DIR`, ran the Step-4 workload at the winning thread config (`CANDLE_NUM_THREADS=6 RAYON_NUM_THREADS=1`, `taskset -c 0,1,2,3,4,5`), twice each, better-of-two reported.

Flag availability differed across commits:
- `--no-thinking` does not exist at `60eb26d` or `00131c9` (it was added in the HEAD commit `3497f14`, "Bound Qwen thinking and optimize MTP verification"). **Dropped for all three commits, including HEAD**, so the comparison is apples-to-apples (HEAD was re-run without `--no-thinking` specifically for this bisect table; its Section 2 number of 8.30 tok/s used `--no-thinking` and is not directly comparable to this table).
- `--speculative-mtp 0` does not exist at `60eb26d` (added in `00131c9`). Dropped only for that commit; kept (`0`, i.e. disabled) for `00131c9` and HEAD.
- Because `--no-thinking` is absent at the older commits, the model emits an open thinking block, which very slightly changes the rendered chat-template token count (160 context tokens vs. 162 with `--no-thinking`) — noted, not corrected for, since it's an unavoidable side effect of matching flags across commits rather than matching template output.

| commit | decode tok/s (best of 2) | delta vs HEAD |
|---|---|---|
| `60eb26d` (add Qwen3.6 35B A3B GGUF support) | 8.23 | +0.02 (+0.24%) |
| `00131c9` (Add Qwen3.6 MTP speculative decoding) | 8.25 | +0.04 (+0.49%) |
| `3497f14` (HEAD — Bound Qwen thinking and optimize MTP verification) | 8.21 | baseline |

Raw reps: `60eb26d` 8.23/8.20; `00131c9` 8.25/8.14; `3497f14` 8.21/8.19.

**Conclusion: no significant regression.** The spread across all three commits (8.21–8.25 tok/s, 0.5%) is smaller than the run-to-run noise observed elsewhere in this report (e.g. config b in Section 2 varied 8.14→8.21 between two back-to-back reps of the identical binary/config, a 0.9% swing; config f varied 8.29→8.30). HEAD's single-token (K=1) decode throughput is statistically indistinguishable from both ancestor commits on this host.

---

## 5. Gap analysis

- Winning thread config decode throughput (Section 2, config f, with `--no-thinking`): **8.30 tok/s**.
- Bisect-comparable throughput at HEAD without `--no-thinking` (Section 4): **8.21 tok/s**.
- Roofline ceiling from measured memory bandwidth (Section 1): **9.62 tok/s**.
- Gap: 8.30 / 9.62 = **86.3%** of roofline achieved (8.21 / 9.62 = 85.3% for the non-`--no-thinking` figure).

This host is already running close to its memory-bandwidth roofline for target-only decode — there is limited headroom left from thread-count tuning alone once you're at 4+ physical cores and avoiding full HT oversubscription (Section 2's a/b/c/e/f cluster is flat within noise, and bandwidth itself saturates at 3 OpenMP threads per Section 1).

The remaining ~14% gap to roofline is consistent with the per-stage K=1 breakdown in Section 3: `dense_projections_outside_moe` (50.47ms) and `deltanet_projections` (29.12ms) together account for the largest share of per-token stage time, ahead of the routed-MoE stages (`moe_routed_gate_up` 20.87ms, `moe_routed_down` 12.75ms) and `lm_head` (19.09ms). These are compute stages whose cost does not perfectly overlap with weight-streaming from memory, which is the expected source of any roofline shortfall on a pure bandwidth model.

---

## 6. Appendix — raw command log

All raw logs are preserved under `perf-run-logs/` in the repo working directory (not committed): `topology.log`, `stream.c`/`stream` (STREAM source/binary), `stream_results.log`, `sweep_ab_cd.log` + `sweep_ab_cd.sh`, `sweep_ef.log` + `sweep_ef.sh`, `verify_bench.log`, `verify-702d043633e0.json`, `bisect_60eb26d.log`, `bisect_00131c9.log`, `bisect_3497f14.log`, `freq_sample.log`, `freq_concurrent_run.log`, `build-head.log`, `build-60eb26d.log`, `build-00131c9.log`, `sample-run.log`, `paths.env`. Condensed (warmup-progress-line-stripped) copies are also under `perf-run-logs/trimmed_*.log`. Below are the commands run and key excerpts of their output.

### Step 1 — topology

```
$ lscpu
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Model name:                              Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
CPU(s):                                  12
Thread(s) per core:                      2
Core(s) per socket:                      6
Socket(s):                               1
CPU max MHz:                             4600.0000
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-11
L1d cache:                               192 KiB (6 instances)
L1i cache:                               192 KiB (6 instances)
L2 cache:                                1.5 MiB (6 instances)
L3 cache:                                12 MiB (1 instance)
Flags: ... fma cx16 ... sse4_1 sse4_2 ... avx f16c ... avx2 ... bmi2 ...
  [no avx512* flags present; no avx_vnni]

$ numactl --hardware
command not found: numactl

$ cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c
     12 powersave

$ grep -E 'MemTotal|Hugepagesize|HugePages_Total|AnonHugePages' /proc/meminfo
MemTotal:       131813152 kB
AnonHugePages:     34816 kB
HugePages_Total:       0
Hugepagesize:       2048 kB

$ sudo -n dmidecode -t memory | grep -E 'Speed|Size|Locator'
sudo dmidecode unavailable (no sudo access in this environment)

$ for f in /sys/devices/system/cpu/cpu*/topology/thread_siblings_list; do echo -n "$f: "; cat $f; done
cpu0: 0,6   cpu1: 1,7   cpu2: 2,8   cpu3: 3,9   cpu4: 4,10   cpu5: 5,11
cpu6: 0,6   cpu7: 1,7   cpu8: 2,8   cpu9: 3,9   cpu10: 4,10  cpu11: 5,11
```

CPU frequency sample taken concurrently with a live decode run (`for i in 1 2 3 4 5; do grep MHz /proc/cpuinfo | sort -n | head -1; sleep 2; done`):
```
cpu MHz		: 1813.880   (caught before compute ramped up)
cpu MHz		: 4295.913
cpu MHz		: 4299.944
cpu MHz		: 4299.309
cpu MHz		: 4298.620
```
Concurrent run's decode line: `input: 34 tokens evaluated in 2.520s (13.49 tok/s); decode: 127 passes in 16.047s (7.91 tok/s); context: 162 tokens` — no other load was running on the host during this or any other timed window.

### Step 2 — memory bandwidth

```
$ gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=80000000 -o stream stream.c
$ OMP_NUM_THREADS=1 OMP_PROC_BIND=true ./stream
Threads: 1, Array size: 80000000 elements (610.4 MB each)
Function    Best Rate MB/s   Avg time     Min time     Max time
Copy             25748.8     0.050746     0.049711     0.052748
Scale            14347.2     0.090642     0.089216     0.093365
Add              15599.2     0.125319     0.123083     0.129006
Triad            15445.9     0.126421     0.124305     0.130553

$ OMP_NUM_THREADS=3 OMP_PROC_BIND=true ./stream
Triad            18041.4     0.106992     0.106422     0.107809

$ OMP_NUM_THREADS=6 OMP_PROC_BIND=true ./stream
Triad            18280.0     0.105748     0.105033     0.106958
```
(`sysbench` was not installed; STREAM was compiled instead, per the task's preference order.)

### Step 3 — build (HEAD)

```
$ cd /workspace
$ CARGO_TARGET_DIR=target-native RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --bin gguf_infer --bin gguf_bench --bin gguf_verify_bench
   Compiling qwen-engine v0.1.0 (/workspace)
    Finished `release` profile [optimized] target(s) in 1m 40s
```

### Step 4 — thread sweep (excerpt; full logs in `sweep_ab_cd.log` / `sweep_ef.log`)

```
$ bash sweep_ab_cd.sh   # runs configs a,b,c,d, 2 reps each
=== run a rep1 CANDLE=4 RAYON=4 cores=[0,1,2,3] ===
input: 34 tokens evaluated in 2.644s (12.86 tok/s); decode: 127 passes in 15.733s (8.07 tok/s); context: 162 tokens
=== run a rep2 ===
input: 34 tokens evaluated in 2.819s (12.06 tok/s); decode: 127 passes in 15.707s (8.09 tok/s); context: 162 tokens
=== run b rep1 CANDLE=6 RAYON=6 cores=[0,1,2,3,4,5] ===
input: 34 tokens evaluated in 2.446s (13.90 tok/s); decode: 127 passes in 15.596s (8.14 tok/s); context: 162 tokens
=== run b rep2 ===
input: 34 tokens evaluated in 2.865s (11.87 tok/s); decode: 127 passes in 15.469s (8.21 tok/s); context: 162 tokens
=== run c rep1 CANDLE=8 RAYON=8 cores=[0,1,2,3,4,5,6,7] ===
input: 34 tokens evaluated in 2.485s (13.68 tok/s); decode: 127 passes in 15.607s (8.14 tok/s); context: 162 tokens
=== run c rep2 ===
input: 34 tokens evaluated in 2.600s (13.08 tok/s); decode: 127 passes in 15.468s (8.21 tok/s); context: 162 tokens
=== run d rep1 CANDLE=12 RAYON=12 cores=[0-11] ===
input: 34 tokens evaluated in 3.473s (9.79 tok/s); decode: 127 passes in 22.396s (5.67 tok/s); context: 162 tokens
=== run d rep2 ===
input: 34 tokens evaluated in 2.976s (11.43 tok/s); decode: 127 passes in 20.945s (6.06 tok/s); context: 162 tokens

$ bash sweep_ef.sh   # runs configs e,f, 2 reps each
=== run e rep1 CANDLE=6 RAYON=6 cores=[] (no taskset) ===
input: 34 tokens evaluated in 2.429s (14.00 tok/s); decode: 127 passes in 15.534s (8.18 tok/s); context: 162 tokens
=== run e rep2 ===
input: 34 tokens evaluated in 2.545s (13.36 tok/s); decode: 127 passes in 15.595s (8.14 tok/s); context: 162 tokens
=== run f rep1 CANDLE=6 RAYON=1 cores=[0,1,2,3,4,5] ===
input: 34 tokens evaluated in 2.345s (14.50 tok/s); decode: 127 passes in 15.322s (8.29 tok/s); context: 162 tokens
=== run f rep2 ===
input: 34 tokens evaluated in 2.369s (14.35 tok/s); decode: 127 passes in 15.308s (8.30 tok/s); context: 162 tokens
```
Every run also printed: `expert cache: 88038/88038 hits (100.0%); ... resident 19416.0/24000.0 MiB in 20992 entries (fully resident: true); 0 evictions`, and `process: physical reads 0.0 MiB; faults ~46000–49000 minor / 0 major; RSS ~21545–21548 MiB, peak ~21572–21577 MiB` — expert-cache behavior and RSS were stable across all six configurations.

### Step 5 — verify-bench

```
$ taskset -c 0,1,2,3,4,5 env CANDLE_NUM_THREADS=6 RAYON_NUM_THREADS=1 \
    ./target-native/release/gguf_verify_bench \
    --model /models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --tokenizer-model /models/Qwen3.6-35B-A3B \
    --batch-sizes 1,2,4,8 --repetitions 3 \
    --expert-cache-mib 24000 \
    --output verify-702d043633e0.json
expert warmup: 123/123 tensors (19.0/19.0 GiB)
```
stdout otherwise silent beyond warmup progress; full results are in the output JSON (see Section 3 for extracted tables). JSON header:
```json
{
  "schema_version": 1,
  "source": {"git_commit": "3497f143aa7f86621d54e286b4822f86192bffd8", "git_dirty": true},
  "build": {"target_arch": "x86_64", "avx2": true, "fma": true, "candle_threads": 6, "rayon_threads": 1},
  "host": {"cpu_model": "Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz", "logical_cpus": 6, "physical_cores": 6, ...},
  "model": {"path": "/models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf", "size_bytes": 22285080192, ...},
  "batch_sizes": [1, 2, 4, 8], "expert_cache_mib": 24000
}
```
(`host.logical_cpus`/`physical_cores` read 6 because the process was taskset-confined to 6 CPUs at launch.)

### Step 6 — bisect

```
$ git checkout 60eb26d
$ CARGO_TARGET_DIR=target-60eb26d RUSTFLAGS='-C target-cpu=native' cargo build --release --bin gguf_infer
    Finished `release` profile [optimized] target(s) in 1m 27s
$ ./target-60eb26d/release/gguf_infer --help   # confirmed: no --no-thinking, no --speculative-mtp at this commit
$ taskset -c 0,1,2,3,4,5 env CANDLE_NUM_THREADS=6 RAYON_NUM_THREADS=1 ./target-60eb26d/release/gguf_infer \
    --model ... --tokenizer-model ... --chat \
    --prompt 'Write a Rust function that parses a semver string into (major, minor, patch) with unit tests.' \
    --max-new-tokens 128 --expert-cache-mib 24000 --warmup-all-experts
rep1: input: 32 tokens evaluated in 2.707s (11.82 tok/s); decode: 127 passes in 15.431s (8.23 tok/s); context: 160 tokens
rep2: input: 32 tokens evaluated in 2.680s (11.94 tok/s); decode: 127 passes in 15.480s (8.20 tok/s); context: 160 tokens

$ git checkout 00131c9
$ CARGO_TARGET_DIR=target-00131c9 RUSTFLAGS='-C target-cpu=native' cargo build --release --bin gguf_infer
    Finished `release` profile [optimized] target(s) in 1m 27s
$ ./target-00131c9/release/gguf_infer --help   # confirmed: no --no-thinking; --speculative-mtp present
$ taskset -c 0,1,2,3,4,5 env CANDLE_NUM_THREADS=6 RAYON_NUM_THREADS=1 ./target-00131c9/release/gguf_infer \
    --model ... --tokenizer-model ... --chat \
    --prompt '...' --max-new-tokens 128 --expert-cache-mib 24000 --warmup-all-experts --speculative-mtp 0
rep1: input: 32 tokens evaluated in 2.685s (11.92 tok/s); decode: 127 passes in 15.387s (8.25 tok/s); context: 160 tokens
rep2: input: 32 tokens evaluated in 2.803s (11.42 tok/s); decode: 127 passes in 15.597s (8.14 tok/s); context: 160 tokens

$ git checkout qwen36-35b-a3b-comparison   # back to HEAD (3497f14)
$ taskset -c 0,1,2,3,4,5 env CANDLE_NUM_THREADS=6 RAYON_NUM_THREADS=1 ./target-native/release/gguf_infer \
    --model ... --tokenizer-model ... --chat \
    --prompt '...' --max-new-tokens 128 --expert-cache-mib 24000 --warmup-all-experts --speculative-mtp 0
rep1: input: 32 tokens evaluated in 2.297s (13.93 tok/s); decode: 127 passes in 15.460s (8.21 tok/s); context: 160 tokens
rep2: input: 32 tokens evaluated in 2.289s (13.98 tok/s); decode: 127 passes in 15.514s (8.19 tok/s); context: 160 tokens
```

Final state confirmed: `git log --oneline -1` → `3497f14 Bound Qwen thinking and optimize MTP verification`, matching pre-task HEAD.
