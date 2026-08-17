# Qwen3.6 speculative decoding

Inferq has two greedy draft sources. Both propose tokens, both verify every
proposal through the same multi-row target pass, and both commit only what the
target model would have decoded on its own, so neither changes output.
Speculation stays disabled by default.

- **prompt lookup (the n-gram arm)**: when the recent token tail repeats an
  earlier tail in the same context, the continuation that followed it last time
  is proposed. No model runs to produce a draft.
- **MTP**: the auxiliary multi-token-prediction transformer block
  Qwen3.6-35B-A3B ships inside the same GGUF.

`--speculative {off,auto,ngram,mtp}` selects between them. `auto` is the
unified policy described in the next section, which uses both under their own
controllers; `ngram` and `mtp` restrict it to one arm. The two arms never draft
in the same step.

`--speculative-ngram N` and `--speculative-mtp N` remain as deprecated aliases
that select the corresponding single-arm mode with `N` as that arm's ceiling,
and print a deprecation warning.

## The unified speculative policy

Neither draft source is a general win, which is why neither could be a default.
The n-gram arm is free when it fires but blind to structural repetition; the
MTP arm sees structure but pays a draft forward per drafted token whether or
not the draft is accepted. `--speculative auto` puts both behind one loop that
decides per decode step:

1. **n-gram evidence, if the index has it** — an active span continuation, or a
   key match tried longest-first down to `--ngram-min-match`. Free either way.
2. **an MTP draft**, if that arm is not currently suspended.
3. **plain decode otherwise** — exactly the one-row pass an unspeculated run
   makes, with no draft, snapshot, rollback or resynchronisation.

An n-gram match takes the step outright; the arms never both run in one step,
and an n-gram draft is never extended with MTP tokens.

### The controllers

Each arm owns independent state. Suspending one never affects the other, and
both suspended is plain decode at the cost of two integer comparisons per step.

| | n-gram arm | MTP arm |
| --- | --- | --- |
| draft length | `[4, 7]`, starts at the cap | depth `[2, 7]`, starts at 4 |
| after a fully accepted draft | `+2` | `+1` |
| after acceptance below half | halve | halve |
| suspends when the acceptance EWMA falls below | 0.4 | 0.5 |
| suspension | 64 committed tokens, doubling to 512 after a failed probe | same |
| probes at | its current length | depth 2 |

The EWMA has alpha 0.2 and starts from an optimistic prior of 1.0, so an arm
withdraws after about five consecutive total rejections rather than after a
single unlucky draft — the copy-heavy workload accepts 80% of proposed tokens
while still rejecting whole drafts occasionally.

Tuning flags: `--ngram-draft-cap`, `--ngram-draft-floor`, `--mtp-depth-cap`,
`--mtp-depth-floor`, `--mtp-depth-start`, `--ngram-suspend-below`,
`--mtp-suspend-below`, `--ewma-alpha`, `--backoff-tokens`, `--backoff-cap`.
Individual mechanisms switch off with `--no-span-continuation`,
`--no-adaptive-length` and `--no-ewma-backoff`.

### Span continuation

After a fully accepted n-gram draft the loop keeps copying from the same source
span without asking the index for a fresh key match. A copied region normally
re-matches on its own suffix, but the index holds one position per key, so a
later occurrence of the same short suffix displaces it; the chain keeps the
copy running across those gaps. It continues only while every proposed token
verified *and* the token the span predicts next is the one the target chose.
Anything else — a rejection, a step the arm did not win, a span that has run off
the end of the index — ends it.

### MTP state, and lazy catch-up

The MTP block keeps its own sequence state, and under the policy it is no
longer the only thing committing tokens: the n-gram arm and plain decode commit
too, and while the MTP arm is suspended it may commit nothing for hundreds of
tokens.

The runtime therefore tracks the block's synced position separately from the
committed position, and retains for the tokens in between the authoritative
target hidden rows the committing passes already produced. Every committing
pass appends to that gap — prefill, n-gram passes, MTP passes, plain decode
steps, and the tokens injected by a forced thinking closure. Nothing
resynchronises eagerly.

Catch-up runs **only when the MTP arm is about to draft**, and closes a gap of
any length in one batched pass, which is possible precisely because the rows
were retained rather than recomputed. So while the arm is suspended its state
lags at no cost; a run that never uses the arm never touches the block at all,
including for the prompt. A `debug_assert` and a release-mode check both
confirm the block's position equals the target's before every draft.

`--eager-mtp-resync` restores the previous behaviour of resynchronising after
every committing pass. It decodes identically — the integration test asserts
that the MTP arm proposes and accepts the same token counts under both — and
only costs more time.

### Metrics

The end-of-run report prints, per arm, proposals and acceptance, drafts fully
accepted and rejected at once, suspensions, suspended steps and probes; the
split of MTP acceptance by whether literal evidence also existed for that step;
and resynchronisation time, passes, rows and longest gap.
`--speculative-trace PATH` writes one JSON object per decode step carrying the
arm that fired, what it proposed and what verified, and both controllers' draft
length, EWMA and suspension state.

### Measured results

`policy-report-702d043633e0.md` has the full tables. The short version: the
policy is the only configuration measured that wins on both the copy-heavy and
the structurally repetitive workload, and it is token-identical to target-only
everywhere — but it is **not** the default, because the MTP arm's break-even
acceptance on this host is 0.68-0.74 against a shipped suspend bar of 0.5, and
prose workloads still regress.

## n-gram (prompt-lookup) speculation

### Try it

Build host-native binaries first:

```bash
CARGO_TARGET_DIR=target-native \
RUSTFLAGS='-C target-cpu=native' \
cargo build --release \
  --bin gguf_infer --bin gguf_bench --bin gguf_verify_bench
```

```bash
MODEL_ROOT=/models/Qwen3.6-35B-A3B

./target-native/release/gguf_infer \
  --model "${MODEL_ROOT}/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  --tokenizer-model "${MODEL_ROOT}" \
  --chat --no-thinking \
  --prompt 'Rewrite this function, adding error context to each failure path: ...' \
  --max-new-tokens 256 \
  --expert-cache-mib 46000 --warmup-all-experts \
  --speculative ngram --ngram-draft-cap 7 --ngram-min-match 4
```

`--speculative off`, or omitting the flag, keeps target-only generation.
`--ngram-min-match N` sets the shortest token suffix the drafter will match on.
Speculation requires greedy sampling and does not support routing traces or
censuses.

The numbers in the rest of this section were measured with the draft length
pinned at 7. Under `--speculative ngram` the controller starts there and moves
it; add `--no-adaptive-length --no-ewma-backoff --no-span-continuation` to
reproduce the pinned behaviour exactly.

### Why prompt lookup rather than a draft model

Verification amortises well on this host but drafting does not. A batched
target pass costs about 148 ms for its first row and about 77 ms for each
additional row, so extra rows are worth roughly half a decode step. MTP
drafting never converted that into a win because every step paid a draft
forward through the MTP block plus a resynchronisation, whether or not
speculation was going to help.

A prompt-lookup draft costs a hash-map read and a slice comparison — measured
at 0.000 s of lookup time across whole runs — and it is issued only when the
index actually has a match. A step without a match runs exactly the one-row
pass ordinary decoding would run: no draft, no snapshot, no verification, no
rollback.

That makes the economics entirely a question of **match precision**. A
correct proposal of *d* tokens turns *d+1* decode steps into one pass; an
incorrect one pays for *d* extra rows and commits a single token. The measured
results below are dominated by this trade, not by any overhead in the drafter.

### The index

One map per indexed key length (2, 3 and 4 tokens) from an FNV-1a hash of the
last *k* token IDs to the most recent position at which that key ended.
Committing a token costs one map insert per length. Prompt tokens are indexed
at prefill and every committed token thereafter.

Correctness does not depend on the hash: a lookup compares the candidate
position's actual token IDs against the current suffix before proposing
anything, so a colliding key produces a miss rather than a wrong draft. The
index feeds proposals only — a stale or incomplete index costs speed, never
correctness — so a turn that follows a `reset`, or that follows a turn decoded
in another mode, simply reseeds it.

### The draft policy

Per decode step, with the pending authoritative token included in the suffix:

1. Try key lengths longest-first, down to `--ngram-min-match`.
2. On a verified match at position `p`, propose
   `tokens[p+1 .. p+1+draft_len]`, clamped to the end of the sequence and
   truncated at the first stop token it contains.
3. On no match, decode normally.
4. Verify `[pending, draft...]` in one target pass, accept the longest
   greedy-matching prefix, commit it plus the authoritative token from the row
   after it, and roll back the rest.

Accepted drafts respect the thinking budget through the same `ThinkingBudget`
commit path the MTP route uses. The draft length is additionally clamped to
what the turn's token limit and the remaining thinking budget can absorb, which
keeps the committed rows and the emitted tokens in step at every boundary.

### Metrics

The end-of-run report prints steps with and without a match, drafts issued,
draft tokens proposed and accepted, tokens committed per verification pass,
rollbacks, replays (always zero, see below), the lookup, verification,
snapshot, rollback and no-match decode wall times, an **acceptance histogram by
draft position**, and a per-match-length breakdown carrying how many drafts
were accepted in full and how many were rejected at their first token.

The histogram and the match-length breakdown are the tuning signal. Acceptance
is bimodal in practice — a proposal is usually either right to its end or wrong
immediately — so a flat histogram means longer drafts are paying off, while a
histogram that collapses after position 0 or 1 means the key length is too
short to be selective and is buying rejected passes.

## Snapshot rollback

Both speculation paths share one rollback mechanism. Rolling a verification
pass back to any of its row boundaries costs a state copy and never a replayed
forward pass.

`QuantizedModelState` holds, per layer, either a full-attention KV cache or a
DeltaNet `{conv, recurrent}` pair. The KV cache is append-only and rolls back
with `truncate`. The recurrent state cannot, so the multi-row forward takes an
optional per-row snapshot sink: the DeltaNet recurrence consumes verification
rows sequentially, and the sink copies `{conv, recurrent}` into a preallocated
slot at each row boundary that loop crosses.

Slot `r` holds the state **before** row `r` is consumed. Slot 0 is therefore
the pre-pass checkpoint, which is why the separate `state.checkpoint()` clone —
a fresh 63 MiB allocation on every pass — is gone. The final row is never
snapshotted, since a fully accepted draft has nothing to roll back to.

The replay disappears for a second reason as well. The rows a verification pass
computed for the committed prefix are authoritative: row *r* was computed from
exactly the prefix a sequential decode would have fed it. The rejection path
therefore narrows the pass's own hidden rows to the committed prefix rather
than recomputing them, which is what lets the MTP resynchronisation drop its
replay too.

`rollback_replays`, `replayed_tokens` and `replay_wall_time` are retained in
the metrics for comparability with earlier measurements and now read zero.

### Snapshot cost

One snapshot row is 62.8 MiB on this model: 30 linear layers x (2.00 MiB
recurrent + 96 KiB conv). Copying that with `copy_from_slice` is bandwidth-
bound at this host's STREAM limit and cannot be brought under 3 ms/row by
threading it. The copy therefore uses AVX non-temporal stores, which move two
lines of traffic per line copied instead of three and, inside the forward,
avoid evicting the live recurrent state from L3 between rows. A
`copy_from_slice` fallback is used where AVX is unavailable and can be selected
with `--snapshot-copy plain`; a unit test asserts the two produce identical
bytes.

## Measured results

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

A pass evaluates the pending token plus the drafts, so a draft length of 8
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

### Draft length and match length sweep

Full sweep tables, the raw command log and the design notes are in
`ngram-report-702d043633e0.md`. The recommendation from it is draft length 7 at
`--ngram-min-match 4`: acceptance rises with match length at every draft length
(52-63% at min-match 2, 74-80% at min-match 4), and throughput falls off a
cliff at draft 8 and beyond for every match length. Those are now the n-gram
controller's cap and starting length.

`DEFAULT_NGRAM_MIN_MATCH` is 4. Speculation is off unless asked for. Used on
its own, the n-gram arm should stay off unless the workload is known to be
copy-heavy: the two non-repetitive workloads measured both regress, not from
drafter overhead but from the handful of wrong proposals they do produce.
`--speculative auto` exists to remove that per-workload judgement call, and
`policy-report-702d043633e0.md` reports how far it gets.

## Bounded thinking

Chat generation preserves the model's normal thinking behavior when neither
of these options is supplied:

- `--no-thinking` renders the assistant prefix as
  `<think>\n\n</think>\n\n`, matching the Qwen template's non-thinking form.
- `--thinking-budget N` starts in normal thinking mode. If the tokenizer's
  complete `</think>` token sequence has not been committed after `N` generated
  thinking tokens, Inferq injects the tokenizer's complete
  `</think>\n\n` sequence into the output and evaluates every injected token
  through the target model before answer generation continues.

The budget is per assistant turn, including in `--interactive` mode. Only
authoritative output tokens count: rejected drafts are neither emitted nor
charged to the budget, under either drafter. A real two-turn Q4 smoke test with
`--thinking-budget 2 --speculative mtp` force-closed both turns
independently, kept 100% of the exercised drafts accepted, and continued with
`Hi!` and `Bye!` after the evaluated closures.

Example:

```bash
INFERQ_NUM_THREADS=4 \
./target-native/release/gguf_infer \
  --model "${MODEL_ROOT}/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  --tokenizer-model "${MODEL_ROOT}" \
  --interactive --chat \
  --thinking-budget 64 \
  --max-new-tokens 256 \
  --expert-cache-mib 46000 --warmup-all-experts \
  --speculative mtp --mtp-depth-cap 1 --mtp-depth-floor 1
```

A cap below the default floor is clamped rather than rejected, so the
deprecated `--speculative-mtp 1` keeps working; it just warns.

`--no-thinking` and `--thinking-budget` are mutually exclusive and require a
Qwen chat template that declares thinking support.

## Qwen3.6 MTP speculative decoding

Qwen3.6-35B-A3B includes one auxiliary multi-token-prediction (MTP) transformer
block in the same GGUF as the target model. Inferq can use it as a greedy draft
predictor with `--speculative mtp`, either on its own or as one arm of
`--speculative auto`.

> **Superseded.** The measurements in this section were taken on the Intel
> i7-6700 qualified host before the pool unification and before snapshot
> rollback. Two of its conclusions no longer describe the code: rejection is no
> longer handled by restoring and replaying, so the "1.01 seconds replaying
> rejected prefixes" and "0.52 seconds checkpointing" components of the draft=1
> breakdown below no longer exist, and the checkpoint/restore/replay figures
> from `gguf_verify_bench` describe a path only that benchmark still uses. The
> section's central finding stands and is the reason the n-gram drafter exists:
> MTP drafting pays its draft and resynchronisation cost on **every** step
> regardless of whether speculation was going to help, which is what kept
> draft=1 below break-even.

The optional `--speculative-mtp-min-margin 0.3` gate (any mode that uses the
MTP arm) falls back to a one-row
target pass when the MTP top-1/top-2 raw-logit margin is below 0.3. It improved
that particular workload, but remains experimental and is not enabled
automatically.

Speculation requires greedy sampling and does not support routing traces or
censuses. The final report includes draft acceptance, gated proposals,
verification rows, rollback/replay counts, and draft, verification,
checkpoint, restore, replay, and MTP-resynchronization time. `gguf_bench`
records the same data in JSONL schema version 6.

### Execution and state semantics

The MTP predictor matches the architecture encoded by the Qwen config and
GGUF:

```text
current token embedding -- RMSNorm --+
                                      +-- concatenate -- eh_proj -- block 40
previous target hidden -- RMSNorm ---+                         |
                                                               +-- shared final norm/head -- draft logits
```

At a speculation boundary the runtime drafts up to `N` tokens, evaluates the
pending target token plus the drafts in one target pass, accepts the longest
greedy-matching prefix, and rolls target recurrent state back to the accepted
row boundary. Output is transactional: rejected draft bytes are never emitted.
Stop tokens come only from the authoritative target result.

After every verification, MTP state is truncated to the pre-draft boundary and
rebuilt from the committed tokens and their authoritative target hidden rows.
Retaining predictor state made from approximated draft hidden rows can preserve
the target token sequence while silently changing later MTP predictions, so it
is deliberately not treated as synchronized state. Forced thinking-closure
tokens use the same authoritative target-plus-MTP path, keeping target state,
MTP position, and the interactive pending token aligned.

### Reproducible verifier benchmark

`gguf_verify_bench` holds one deterministic prefetched context, keeps every
expert resident, and evaluates the same fixed tokens at K=1,2,4,8. It reports
total and per-row stage time, checkpoint/restore/rejection replay, and all five
expert-reuse metrics for every layer.

```bash
MODEL_ROOT=/data/projects/localllm/models/Qwen3.6-35B-A3B

INFERQ_NUM_THREADS=4 \
./target-native/release/gguf_verify_bench \
  --model "${MODEL_ROOT}/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  --tokenizer-model "${MODEL_ROOT}" \
  --batch-sizes 1,2,4,8 \
  --repetitions 3 \
  --expert-cache-mib 46000 \
  --output verifier-q4.json
```

The default deterministic verification IDs are
`[8160,579,264,7047,1817,25,271,16]`. The JSON contains, per layer and K,
token-to-expert assignments, unique selected experts, duplicate assignment
rate, average rows per selected expert, and the maximum rows assigned to one
expert.

### What changed in the verifier

The routed MoE small-batch path is now expert-major. It computes all routes,
groups `(row, route weight)` records by expert, gathers each expert's input
rows, executes gate/up and down once for that group, stores each route result,
then performs the final weighted accumulation in original token/route order.
K=1 retains the original token-major path, and experts stay in their resident
compressed representation.

Measurement also isolated a separate dense-matrix problem. Candle's existing
quantized CPU matmul traversed a large quantized matrix once per input row on
this workload: the LM head nearly doubled from 17.4 ms at K=1 to 35.8 ms at
K=2. Inferq therefore adds a measured small-M path for Q4_K, Q5_K, Q6_K, and
Q8_0 matrices of at least 4 MiB. It quantizes the M input rows once, traverses
each compressed weight row once, applies that row to every input while it is
cache-hot, and transposes the output. The size threshold is important: using
the same path for the much smaller expert matrices regressed their stages by
18-42%, so routed experts continue to use Candle's grouped multi-row path.

**This path covers 2 to 8 rows.** A verification pass of more than 8 rows —
a draft length of 8 or higher, since a pass evaluates the pending token
plus the drafts — falls back to the slower per-row traversal. The n-gram
results below show what that costs.

### K=1/2/4/8 verification scaling

Qualified host: Intel i7-6700, four physical cores, four Candle/Rayon threads,
native release build, fully resident Q4_K_M, three repetitions.

| K | Before total | Before / row | After total | After / row | Per-row change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 122.09 ms | 122.09 ms | 121.20 ms | 121.20 ms | -0.7% |
| 2 | 227.23 ms | 113.61 ms | 189.88 ms | 94.94 ms | -16.4% |
| 4 | 428.74 ms | 107.18 ms | 335.20 ms | 83.80 ms | -21.8% |
| 8 | 807.39 ms | 100.92 ms | 595.30 ms | 74.41 ms | -26.3% |

After optimization, total stage times were:

| Stage | K=1 | K=2 | K=4 | K=8 |
| --- | ---: | ---: | ---: | ---: |
| DeltaNet projections | 25.04 ms | 36.41 ms | 60.91 ms | 106.45 ms |
| DeltaNet recurrence | 10.31 ms | 17.29 ms | 29.51 ms | 52.88 ms |
| Full attention | 10.00 ms | 15.87 ms | 27.56 ms | 51.40 ms |
| MoE router | 3.15 ms | 5.44 ms | 7.14 ms | 9.88 ms |
| MoE top-k | 0.48 ms | 0.86 ms | 1.62 ms | 3.30 ms |
| Routed expert gate/up | 18.99 ms | 33.81 ms | 66.07 ms | 119.40 ms |
| Expert activation | 1.36 ms | 2.51 ms | 5.08 ms | 8.74 ms |
| Routed expert down | 11.22 ms | 20.06 ms | 39.23 ms | 69.27 ms |
| Routed accumulation | 0.70 ms | 0.97 ms | 2.09 ms | 4.47 ms |
| Shared expert | 5.98 ms | 9.03 ms | 15.55 ms | 26.74 ms |
| Dense projections outside MoE | 43.29 ms | 62.84 ms | 105.25 ms | 186.80 ms |
| Final norm | 0.007 ms | 0.010 ms | 0.015 ms | 0.021 ms |
| LM head | 17.75 ms | 21.23 ms | 34.52 ms | 60.47 ms |

Checkpoint averaged 41.5-45.9 ms, restore 6.7-7.3 ms, and one-row rejection
replay 122.0-122.9 ms in the standalone harness. End-to-end draft=1 measured
0.52 seconds of checkpoint time over the whole 128-token run, so allocator and
process context materially affect the standalone checkpoint number.

Aggregating the benchmark's per-layer reuse records gives:

| K | Assignments | Unique experts | Duplicate rate | Rows / selected expert | Maximum rows / expert |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 320 | 320 | 0.0% | 1.000 | 1 |
| 2 | 640 | 506 | 20.9% | 1.265 | 2 |
| 4 | 1,280 | 867 | 32.3% | 1.476 | 4 |
| 8 | 2,560 | 1,261 | 50.7% | 2.030 | 8 |

The JSON report retains all 40 individual layer records rather than hiding
layer-to-layer variation behind these aggregates.

### MTP end-to-end result

The workload is the same 25-token rendered chat prompt followed by 128 greedy
output tokens. All runs emitted the exact same complete target token sequence.

| Mode | Before tok/s | After tok/s | Decode time | Accepted / drafted | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target only | 8.10 | **8.09** | 15.699 s | 0 / 0 | n/a |
| MTP draft=1 | 6.31 | 7.17 | 17.722 s | 59 / 67 | 88.1% |
| MTP draft=2 | 5.14 | 6.09 | 20.844 s | 76 / 100 | 76.0% |
| MTP draft=3 | 4.63 | 5.64 | 22.511 s | 85 / 126 | 67.5% |
| MTP draft=1, margin 0.3 | n/a | 7.51 | 16.905 s | 58 / 68; 6 gated | 85.3% |

Draft=1 is substantially faster than the earlier implementation, but has not
crossed break-even: the ungated result is 11.4% slower than target-only, and
the measured margin gate is still 7.1% slower. Consequently speculation stays
opt-in.

For ungated draft=1, target verification is the largest measured component at
14.15 seconds. The avoidable work separating it from target-only includes
1.49 seconds in the MTP block, 1.01 seconds replaying rejected prefixes, 0.52
seconds checkpointing, 0.44 seconds resynchronizing, and 0.06 seconds restoring.
At K=2 the largest verifier stages are dense non-MoE projections (62.84 ms),
routed gate/up plus down (53.88 ms), LM head (21.23 ms), DeltaNet recurrence
(17.29 ms), and full attention (15.87 ms). These measurements isolate several
contributors; they do not support declaring any single remaining kernel the
sole bottleneck.

Architecture references:

- [Qwen3.6-35B-A3B config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)
- [llama.cpp Qwen3.5/3.6 MoE graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35moe.cpp)
- [llama.cpp speculative-decoding documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
