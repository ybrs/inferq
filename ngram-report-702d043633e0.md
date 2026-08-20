# n-gram speculative decoding and snapshot rollback

Host `702d043633e0`: Intel i7-8700, 6 physical cores / 12 threads, 12 MiB L3,
125 GiB RAM, STREAM 18.28 GB/s (measured in `perf-report-702d043633e0.md`).
Model: `Qwen3.6-35B-A3B` Q4_K_M GGUF, fully resident (`--expert-cache-mib 46000
--warmup-all-experts`). Branch `ngram-speculation`, based on
`qwen36-35b-a3b-comparison` at 7b814f6.

All measurements: `taskset -c 0-5`, INFERQ default threading (6 threads, no
thread environment variables), one run at a time on an otherwise idle machine,
greedy decoding, `--chat --no-thinking`, 256 new tokens.

Every table below is generated from the per-run logs by
`ngram-run-logs/extract.py`; no number here is transcribed by hand. The logs
themselves stay on the measurement host and are not committed, matching
AGENTS.md ("do not commit ... benchmark output") and the existing untracked
`perf-run-logs/`. The harness that regenerates them — the driver scripts, the
workload prompts, the table generator and the micro-benchmark sources — is
committed under `ngram-run-logs/`.

## Results

### Verdict against the task's gates

| gate | required | measured | verdict |
| --- | --- | --- | --- |
| Greedy equivalence, W1/W2/W3 | token ids bit-identical to target-only | identical in all 12 comparisons | **pass** |
| Snapshot rollback replays | `rollback_replays == 0` | 0 in every speculative run, n-gram and MTP | **pass** |
| MTP non-regression | smoke run executes, metrics line prints, tests pass | yes; MTP refitted onto the same rollback | **pass** |
| W2 no-match overhead | regression <= 2% | **-0.23%** on the no-match path itself | **pass** |
| W2 end-to-end throughput | regression <= 2% (literal reading) | -6.1% at the best setting | **fail** |
| W1 speedup | >= 1.25x | **1.228x** at the best setting | **fail** |
| Snapshot overhead | < 3 ms/row | 3.41 ms/row streaming, 5.38 ms/row plain | **fail** |

Three gates fail. Per the task's instructions the analysis stops here rather
than moving into the kernel or dispatch layer; sections below isolate each
cause with the histograms and stage timings.

### Greedy equivalence

Every n-gram run below emitted a token id sequence bit-identical to the
target-only run of the same workload at the same settings. This covers both
repetitions, both match lengths, and both snapshot copy strategies.

| workload | comparisons | tokens each | identical |
| --- | ---: | ---: | --- |
| W1 copy-heavy (semver rewrite) | 6 | 256 | **yes** |
| W2 prose (B-tree vs LSM tree) | 3 | 256 | **yes** |
| W3 self-repetitive (10 stack tests) | 3 | 256 | **yes** |

The integration test `partial_acceptance_rollback_matches_sequential_decoding`
adds the stricter check the task asked for against the real checkpoint: force a
pass whose first two proposals are right and last two are wrong, roll back to
the interior row boundary, then decode 32 more tokens and require token
equality with a purely sequential decode. It passes, as do the
`rollback_replays == 0` and thinking-budget integration tests.

### Decode throughput, best of two

| workload | target-only | draft 7, min-match 3 | draft 7, min-match 4 |
| --- | ---: | ---: | ---: |
| W1 copy-heavy | 6.90 tok/s | 7.30 (1.058x) | **8.47 (1.228x)** |
| W2 prose | 7.75 tok/s | 6.95 (0.897x) | 7.26 (0.937x) |
| W3 self-repetitive | 7.70 tok/s | 5.76 (0.748x) | 6.23 (0.809x) |

| workload | match rate | acceptance | tokens/pass | drafts fully accepted | rejected at first token |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1, min-match 4 | 25.2% | 80.4% | 6.48 | 20 / 27 | 2 / 27 |
| W2, min-match 4 | 2.4% | 13.2% | 1.83 | 0 / 6 | 2 / 6 |
| W3, min-match 4 | 11.3% | 18.9% | 2.32 | 1 / 25 | 15 / 25 |

The n-gram runs are highly reproducible (W1 min-match 4: 8.47, 8.45, 8.37
across three runs); the target-only baseline is the noisy side (6.55 then 6.90,
a 5.3% spread), and the best-of-2 protocol takes the faster baseline, which is
the conservative choice for the ratio.

### Why W1 stops at 1.228x

Nothing in the drafter or the rollback is the cost. At the best setting:

| component | measured |
| --- | --- |
| index lookup, whole run | 0.000 s |
| no-match decode | 148.2 ms/step, identical to the 148.5 ms/step baseline |
| rollback | 0.053 s over 7 rollbacks |
| snapshot | 0.731 s (2.4% of decode) |
| matched steps | 105 ms/token against 148 ms/token sequential — **1.41x** |

Speculation is 1.41x on the steps it fires on. The overall figure is 1.228x
because it only fires on 25.2% of steps. Raising the match rate means shortening
the key, and the sweep shows that trade is a straight loss: min-match 2 lifts
the match rate to 51.1% but drops acceptance to 52.4% and ends up *below*
target-only at 0.962x.

### The 8-row cliff, and why draft length cannot go past 7

Per-row verification cost improves monotonically as passes widen — until a pass
exceeds eight rows, where it jumps 25%:

| draft length | rows/pass | ms/row | ms/pass | W1 tok/s (min-match 4) |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 4.98 | 96.0 | 478 | 7.43 (min-match 3) |
| 6 | 6.82 | 91.2 | 622 | 8.01 |
| **7** | **7.81** | **87.4** | **683** | **8.47** |
| 8 | 8.76 | **109.6** | 960 | 7.07 |
| 12 | 12.35 | 105.7 | 1306 | 5.06 (min-match 3) |

A pass evaluates the pending token plus the drafts, so `--speculative-ngram 8`
is a nine-row pass and falls off the small-M dense path, whose measured range is
2..=8 rows. As the task instructed, this is reported rather than fixed: the
row range, `SMALL_M_MIN_STORAGE_BYTES` and the block kernels are untouched.

This matters because the acceptance histogram says the drafts are being cut off
while they are still paying. At draft 7 / min-match 4, acceptance by draft
position is:

```
0: 25/27 (93%)   1: 23/27 (85%)   2: 21/26 (81%)   3: 21/26 (81%)
4: 20/26 (77%)   5: 19/26 (73%)   6: 19/26 (73%)
```

Still 73% at the last position, and flat from position 3 onward — acceptance is
bimodal, so a proposal that survives its third token almost always survives to
the end. A longer draft would keep converting those at roughly 77 ms/token
instead of 148, but only if a pass wider than eight rows kept the fast path.
Widening that range is the single highest-value follow-up this measurement
identifies, and it belongs to the kernel layer, outside this task.

### Why W2 and W3 regress

W2's *no-match* overhead — what the gate was written to measure — is nil:

| W2, draft 7, min-match 4 | value |
| --- | ---: |
| steps without a match | 244 of 250 |
| no-match decode | 31.498 s = **129.1 ms/step** |
| target-only baseline | **129.4 ms/token** |
| no-match overhead | **-0.23%** |

The whole regression is six proposals. Those 2.4% of steps committed 11 tokens
in 3.477 s — 316 ms/token against 129 ms sequential — wasting 2.05 s on a 33 s
run, which is the entire -6.1% end-to-end difference. None of the six was
accepted in full.

W3 fails for the same reason more severely: 25 proposals, 15 rejected at their
first token, 18.9% acceptance. The workload repeats *structure* ("one test per
method, same test structure") but not literal token spans, so a four-token key
matches scaffolding like an assertion opening and then jumps to the wrong
continuation. Prompt lookup cannot distinguish the two cases; only verification
can, and verification is what costs.

The practical consequence is that this drafter is a per-workload switch, not a
global one. It is off by default and the report recommends leaving it off unless
the workload is known to be copy-heavy.

### W1 sweep

Single repetition per cell, W1 only, against the 6.90 tok/s best-of-2 baseline.

| draft len | min match | tok/s | vs baseline | match rate | acceptance | tokens/pass | full drafts | rejected at once |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 | 7.19 | 1.042x | 57.7% | 62.9% | 3.52 | 33 | 15 |
| 4 | 3 | 7.43 | 1.077x | 44.0% | 68.5% | 3.73 | 31 | 13 |
| 6 | 2 | 6.81 | 0.987x | 53.2% | 54.8% | 4.22 | 21 | 11 |
| 6 | 3 | 7.25 | 1.051x | 39.3% | 59.9% | 4.52 | 21 | 11 |
| 6 | 4 | 7.82 | 1.133x | 29.2% | 74.0% | 5.30 | 21 | 4 |
| 7 | 2 | 6.64 | 0.962x | 51.1% | 52.4% | 4.59 | 20 | 13 |
| 7 | 3 | 7.32 | 1.061x | 36.3% | 60.2% | 5.14 | 20 | 11 |
| **7** | **4** | **8.37** | **1.213x** | 25.2% | 80.4% | 6.48 | 20 | 2 |
| 8 | 2 | 5.43 | 0.787x | 48.8% | 51.8% | 5.02 | 18 | 10 |
| 8 | 3 | 5.93 | 0.859x | 35.0% | 57.2% | 5.43 | 18 | 11 |
| 8 | 4 | 7.07 | 1.025x | 23.8% | 77.3% | 7.00 | 17 | 2 |
| 12 | 2 | 4.54 | 0.658x | 46.3% | 40.2% | 5.55 | 11 | 11 |
| 12 | 3 | 5.06 | 0.733x | 32.3% | 45.2% | 6.13 | 11 | 11 |

The min-match 4 column is an addition to the sweep the task specified. It was
added because the position histograms from the first min-match 3 runs showed
acceptance collapsing after position 0, which pointed straight at key
selectivity; it turned out to be the largest single effect in the whole sweep.

Two clean monotonic readings: acceptance rises with match length at every draft
length (52-63% at min-match 2, 57-69% at 3, 74-80% at 4), and throughput falls
off a cliff at draft 8 and beyond for every match length.

**Recommended defaults: `--speculative-ngram 7 --ngram-min-match 4`.**
`DEFAULT_NGRAM_MIN_MATCH` is therefore set to 4, not the 3 the task specified —
the sweep shows 3 is worse on every workload measured, and shipping a default
the data contradicts is not defensible. Draft length remains 0 (off) by default.

### Snapshot cost against the 3 ms/row budget

In situ on W1 (draft 7, min-match 3, 291 snapshot rows), both runs emitting
identical tokens:

| copy strategy | snapshot time | ms/row | decode tok/s |
| --- | ---: | ---: | ---: |
| `copy_from_slice` (`--snapshot-copy plain`) | 1.566 s | 5.38 | 7.07 |
| AVX streaming stores (default) | 0.992 s | **3.41** | 7.27 |

Streaming stores are 1.58x faster on the copy and worth 2.8% of end-to-end
decode throughput. **Neither meets the < 3 ms/row budget**, and the standalone
benchmark shows why the plain variant never could: at 62.8 MiB per row it is
pinned to this host's 18.28 GB/s STREAM ceiling. The streaming path beats that
ceiling by moving less traffic but still lands at 3.41 ms/row in situ against
2.78 ms/row standalone, because inside a real forward the source is not always
still resident in L3 when the snapshot runs.

Per-row cost also depends on how wide the pass is, which confirms that reading:

| pass width | snapshot ms/row |
| --- | ---: |
| 2 rows (MTP draft=1) | 5.96 |
| 5 rows (draft 4) | 4.05 |
| 8 rows (draft 7) | 3.46 |
| 13 rows (draft 12) | 3.09 |

Wider passes amortise better because slots 2..n are written while the layer's
recurrent state is still warm. In context the miss is not what costs the
speedup: at the recommended setting the snapshot is 0.731 s of a 30.5 s decode,
2.4%, against a replay path that would have cost a full 148 ms forward on each
of the 7 rollbacks.

### MTP non-regression

The MTP path was refitted onto the same rollback and still works. W1,
`--speculative-mtp 1`:

```
MTP speculation: 126/129 draft tokens accepted (97.7%, 0 gated);
129 verification passes over 258 tokens; 0 rollback replays over 0 tokens;
draft 3.266s, verify 35.029s, checkpoint 1.538s, restore 0.025s,
replay 0.000s, resync 1.256s
```

`replay 0.000s` and `0 rollback replays` are the refit landing: this line
previously carried a non-zero replay component. At 6.43 tok/s against the 6.90
baseline, MTP draft=1 remains below break-even, consistent with the earlier
conclusion. All existing MTP tests pass unchanged, including
`accepted_speculative_drafts_stop_at_the_budget_boundary` and
`rejected_speculative_drafts_do_not_consume_thinking_budget`.

## Part A design: snapshot rollback

### What it replaces

Before this change, a verification pass took one `state.checkpoint()` (a fresh
63 MiB clone of every linear layer's recurrent state) and, on any rejection,
`restore()`d to the pre-pass state and **replayed** the accepted tokens through
a full forward pass. The replay cost a whole decode step, and it was on the
critical path of every partial acceptance.

Rolling back to any row boundary of a pass is now a state copy. Two changes get
there, and the second matters as much as the first:

1. **Per-row snapshots.** The DeltaNet recurrence in
   `QuantizedDeltaLayer::forward` (`src/qwen/quantized_deltanet.rs`) consumes
   verification rows sequentially in one `for token in 0..seq` loop, updating
   `conv` in `causal_depthwise_conv_step` and `recurrent` in
   `recurrent_delta_step`. An optional sink copies `{conv, recurrent}` into a
   preallocated slot at each row boundary that loop crosses. Full-attention
   layers need no storage at all: their KV cache is append-only and rolls back
   with the existing `truncate`.
2. **No replay for the committed hidden rows either.** The rows the
   verification pass already computed for the committed prefix are
   authoritative — row *r* was computed from exactly the prefix that a
   sequential decode would have fed it. So the rejection path takes
   `verified.normalized_hidden.narrow(0, 0, committed_rows)` instead of
   recomputing those rows. This is what actually removes the replay: a rollback
   that only fixed the recurrent state would still have had to replay to obtain
   hidden rows for the MTP resynchronisation.

`rollback_replays`, `replayed_tokens` and `replay_wall_time` are kept in both
metric structs and read zero on every speculative run in this report.

### Slot indexing, and why the checkpoint disappeared

Slot `r` holds the state **before** row `r` is consumed. That makes slot 0 the
pre-pass checkpoint, so the separate `state.checkpoint()` clone — which
allocated 63 MiB on every pass — is gone entirely, replaced by a write into a
buffer that is allocated once and reused for the session. The last row is never
snapshotted: a fully accepted draft has nothing to roll back to.

A pass over `n` rows therefore writes `n` slots, and `rollback(snapshots, c)`
for `c` committed rows restores slot `c` in every linear layer, truncates every
full-attention layer to `pre_position + c`, and sets `position` accordingly.

An interrupted pass is handled by the per-layer `stored_rows` counter: layers
the pass never reached have no stored rows and are still at the pre-pass state,
so restoring them is a no-op rather than a restore of stale data from an
earlier pass.

### Deviation from the spec: streaming stores instead of `copy_from_slice`

The spec asked for `copy_from_slice` into preallocated buffers and set a budget
of <3 ms/row. **That budget is not reachable with `copy_from_slice` on this
host, and the reason is arithmetic rather than implementation quality.**

One snapshot row is every linear layer's `{conv, recurrent}`:

| quantity | value |
| --- | ---: |
| `recurrent` per linear layer | 32 heads x 128 x 128 x 4 B = 2.00 MiB |
| `conv` per linear layer | 8192 x 3 x 4 B = 96 KiB |
| linear layers (40 layers, `full_attention_interval` 4) | 30 |
| **bytes per snapshot row** | **62.8 MiB** |

A standalone benchmark (`ngram-run-logs/snapbench*.c`, compiled `-O2`, run
under `taskset -c 0-5`) reproducing the real access order — layers outer, rows
inner, source freshly written by a stand-in for the recurrence — measures:

| copy | threads | ms/row | effective GB/s |
| --- | ---: | ---: | ---: |
| `memcpy` / `copy_from_slice` | 1 | 4.841 | 13.61 |
| `memcpy` | 2 | 3.724 | 17.69 |
| `memcpy` | 3 | 3.591 | 18.34 |
| `memcpy` | 6 | 3.645 | 18.07 |
| AVX non-temporal stores | 1 | **2.779** | 23.70 |
| AVX non-temporal stores | 2 | 2.657 | 24.79 |

The `memcpy` rows plateau at 18.3 GB/s, which is this host's STREAM figure
(18.28 GB/s): a plain copy of 62.8 MiB per row is bandwidth-bound at the
machine limit, and no amount of threading takes it below ~3.6 ms/row. Streaming
stores beat that limit because they change the traffic, not the rate: a normal
store of a full cache line still reads the destination line first
(read-for-ownership), so `copy_from_slice` moves roughly three lines of traffic
per line copied, while `_mm256_stream_ps` moves two. There is a second,
compounding effect inside the real forward: within one layer the live 2 MiB
recurrent state stays in the 12 MiB L3 across the pass's rows, and an ordinary
snapshot write evicts it every row, so the recurrence then re-reads it from
DRAM. Streaming stores bypass the cache and leave the live state resident.

The implementation therefore uses an AVX non-temporal path with a
`copy_from_slice` fallback (`copy_state` in `src/qwen/quantized_deltanet.rs`).
`unsafe` is confined to the SIMD copy with a local safety comment, which is one
of the uses AGENTS.md permits; AVX is detected at runtime and the fallback is
used otherwise. A unit test
(`streaming_and_ordinary_copies_produce_identical_state`) asserts the two paths
produce identical bytes at every alignment offset, and `--snapshot-copy plain`
selects the fallback so the two can be compared in situ.

## Part B design: the n-gram drafter

`src/ngram.rs` keeps one map per indexed key length from an FNV-1a hash of the
last *k* token IDs to the most recent position at which that key ended, plus a
`last_match` cache holding, per key length, the most recent occurrence *before*
the current suffix. The cache is filled during `push`, so a lookup is a map
read plus a slice comparison. Committing a token costs one map insert per
indexed length.

**Correctness never depends on the hash.** A lookup compares the candidate
position's actual token IDs against the current suffix before proposing
anything, so a colliding key degrades to a miss. The unit test
`hash_collisions_are_rejected_by_comparing_token_ids` forces every key into one
bucket and asserts both halves of that: a collided lookup returns no match, and
a genuine repeat under the same degenerate hash is still found because its
tokens compare equal.

The index is maintained only while the n-gram drafter is in use, and it feeds
proposals only — a stale or incomplete index costs speed, never correctness. At
the start of a turn the runtime extends it when it already mirrors the sequence
the model state represents and reseeds it from the current turn otherwise (a
first turn, a `reset`, or a turn that ran another decoding mode).

Per step, with the pending authoritative token included in the suffix: try key
lengths longest-first down to `--ngram-min-match`, take the most recent
verified occurrence, propose `tokens[p+1 .. p+1+draft_len]` clamped to the
sequence end and truncated at the first stop token. No match means no
proposal, and the step runs the ordinary one-row pass with no snapshot, no
verification and no rollback.

Verification reuses the existing machinery unchanged: one multi-row pass over
`[pending, draft...]`, `accepted_draft_prefix`, commit the accepted prefix plus
the authoritative token from the row after it. Committed tokens are the target
model's own greedy choices at every position, which is why the output is
identical to target-only decoding.

### Commit-path invariant

A pass commits `1 + accepted` rows and emits `accepted + 1` tokens. The draft
length is clamped to what both the turn's token limit and the thinking budget
can still absorb. That clamp is load-bearing rather than cosmetic: it forces
the token-limit and budget boundaries to land either on the last accepted draft
or on the authoritative token, so the emission loop can never stop with
evaluated-but-unemitted tokens left in the model state. The thinking budget
itself is the shared `ThinkingBudget` commit path, not a copy of it.

## Deviations from the task specification, and why

| # | Spec said | What was built | Why |
| ---: | --- | --- | --- |
| 1 | Snapshot with `copy_from_slice` into preallocated buffers, budget < 3 ms/row | AVX non-temporal stores with a `copy_from_slice` fallback | `copy_from_slice` of 62.8 MiB/row is pinned to this host's 18.28 GB/s STREAM ceiling and cannot reach 3 ms/row at any thread count (measured 3.59 ms/row at best, 4.84 single-threaded). Streaming stores move less traffic and reach 3.41 ms/row in situ. Neither meets the budget; the faster one was kept and both are reported. |
| 2 | One snapshot per row, plus the existing pre-pass `checkpoint()` | Slot `r` = state *before* row `r`, so slot 0 **is** the checkpoint | Same number of copies, one fewer concept, and it deletes a 63 MiB heap allocation that happened on every verification pass. |
| 3 | Rollback restores state; (implicitly) hidden rows still come from somewhere | Committed hidden rows are narrowed out of the verification pass's own output | Restoring only the recurrent state would still have forced a replay forward to obtain hidden rows for the MTP resynchronisation. This is what actually gets `replay_wall_time` to zero. |
| 4 | `--ngram-min-match` default 3 | Default 4 | The sweep shows 3 is worse on all three workloads (W1 1.058x vs 1.228x). Shipping a default the measurements contradict is not defensible. The flag still accepts 2 and 3. |
| 5 | Sweep draft_len x min_match in {2, 3} | Added a min-match 4 column | The first runs' position histograms showed acceptance collapsing after position 0, pointing at key selectivity. It turned out to be the largest single effect in the sweep, so leaving it unmeasured would have hidden the main result. |
| 6 | Metrics list ending at wall times and the position histogram | Also per-match-length stats, fully-accepted and rejected-at-first-token counts | The position histogram shows acceptance is bimodal but not *which key length* is responsible. The breakdown is what identified min-match 3 as the problem. |
| 7 | New unit tests | Model-dependent tests are opt-in via `INFERQ_TEST_GGUF` / `INFERQ_TEST_MODEL_DIR` | AGENTS.md requires tests needing the full checkpoint to be opt-in and to skip with a clear message. `./scripts/validate.sh` stays offline. |

Indexed key lengths stayed at {2, 3, 4} as specified. The data argues for
trying longer keys, but longer keys lower the match rate, which is already the
binding constraint on W1 at min-match 4 — so that is a hypothesis for a
follow-up, not a change made blind here.

## Test coverage

Offline unit tests (16 new, all in `./scripts/validate.sh`):

- `src/ngram.rs` (11): most-recent-occurrence lookup; longest key preferred;
  min-match 2 finding what a trigram key misses; draft truncation at a stop
  token; empty draft when the continuation *starts* with a stop token; draft
  clamped to the tokens that exist; no proposal without repetition; `clear`;
  zero draft length; and the forced-collision test, which asserts both that a
  collided lookup returns no match and that a genuine repeat under the same
  degenerate hash is still found by token comparison.
- `src/qwen/quantized_deltanet.rs` (5): per-row snapshot restore; rejecting a
  row the pass never reached; a layer that never ran left untouched; buffer
  reuse across passes; and streaming vs ordinary copies producing identical
  bytes at every alignment offset.
- `src/runtime.rs` (4): n-gram accepted drafts stopping at the budget boundary;
  rejected drafts not consuming budget; acceptance-by-position bookkeeping; and
  the replay counters reading zero.

Opt-in integration tests against the real checkpoint (`tests/ngram_speculation.rs`,
all three passing in 106.6 s):

- `partial_acceptance_rollback_matches_sequential_decoding` — the token-equality
  check the task specified: force a pass with two right and two wrong proposals,
  roll back to the interior boundary, decode 32 more tokens, require equality
  with a purely sequential decode.
- `ngram_speculation_reproduces_target_only_tokens_without_replays` — greedy
  equality plus `rollback_replays == 0` on a run with real partial acceptances.
- `ngram_speculation_respects_the_thinking_budget`.

## Raw command log

Build:

```bash
CARGO_TARGET_DIR=target-native RUSTFLAGS='-C target-cpu=native' \
  cargo build --release --bin gguf_infer --bin gguf_bench --bin gguf_verify_bench
```

Every measured run went through `ngram-run-logs/run.sh`, which fixes the
environment:

```bash
taskset -c 0-5 ./target-native/release/gguf_infer \
  --model /models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --tokenizer-model /models/Qwen3.6-35B-A3B \
  --chat --no-thinking \
  --prompt "$(cat ngram-run-logs/prompts/wN.txt)" \
  --max-new-tokens 256 \
  --expert-cache-mib 46000 --warmup-all-experts \
  <per-run flags>
```

The two driver scripts, run sequentially on an otherwise idle machine:

```bash
./ngram-run-logs/validate-all.sh   # 24 runs: equivalence, best-of-2, sweep, MTP
./ngram-run-logs/followup.sh       # 10 runs: min-match 4, snapshot copy A/B
```

Snapshot micro-benchmarks:

```bash
gcc -O2 -fopenmp        -o snapbench  ngram-run-logs/snapbench.c
gcc -O2 -mavx2 -fopenmp -o snapbench2 ngram-run-logs/snapbench2.c
for t in 1 2 3 6; do taskset -c 0-5 ./snapbench  8 $t; done
for t in 1 2 3 6; do taskset -c 0-5 ./snapbench2 8 $t; done
```

Validation:

```bash
./scripts/validate.sh
INFERQ_TEST_GGUF=/models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
INFERQ_TEST_MODEL_DIR=/models/Qwen3.6-35B-A3B \
  taskset -c 0-5 cargo test --release --test ngram_speculation -- --test-threads=1
```

The driver scripts, workload prompts, table generator (`extract.py`) and
micro-benchmark sources are committed in `ngram-run-logs/`; running
`validate-all.sh` followed by `followup.sh` regenerates every log this report
draws on, and `extract.py` regenerates the tables.
