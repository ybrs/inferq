# Cutting the MTP draft cost

Two changes, measured together: a confidence gate on what the MTP predictor
submits, and a vocabulary-prefix LM head for what it scores drafts against.

Measured on host `702d043633e0` (Intel i7-8700, six physical cores,
`taskset -c 0-5`, INFERQ default threading, fully resident Q4_K_M, greedy
decoding). Regenerable with `draft-run-logs/campaign.sh`; raw logs and the
per-drafted-token calibration dumps are in `draft-run-logs/`.

**Baselines were re-measured on this branch in the same session.** Absolute
throughput on this host moves several percent between sessions (W1 target-only
read 6.83 tok/s in the afternoon and 7.06 tok/s overnight), so ratios are the
comparable quantity and no number here should be compared to
`policy-report-702d043633e0.md` in absolute tok/s.

## Result

With both changes, best of two, on the single-process harness:

| workload | gate | target-only | policy, before either change | **policy, both** | ratio | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| W1 copy-heavy | 1.25x | 7.13 | 8.83 | **9.04** | **1.268x** | **pass** |
| W2 prose | 0.97x | 8.01 | 7.75 | **8.70** | **1.086x** | **pass** |
| W3 self-repetitive | 1.02x | 7.90 | 8.77 | **10.34** | **1.309x** | **pass** |
| W4 mixed, 768 tokens | 1.10x | 6.32 | 6.66 | **7.01** | **1.108x** | **pass** |

**All four gates pass, and W2 is now a win rather than a loss** — the workload
that regressed 9% under the policy alone gains 8.6%. 64 speculative runs across
the campaign emitted token ids identical to target-only, with zero mismatches.

Per the original task's gate 8 this is the point at which the default flips to
`--speculative auto`. It has **not** been flipped, for a reason that is not
about these numbers: greedy equivalence is currently probabilistic rather than
guaranteed (see below), and `AGENTS.md` is explicit that approximate behaviour
must never become the default. Fixing that is now the only thing standing
between this work and default-on speculation.

## The vocabulary-prefix draft head

`output.weight` is [248320, 2048] Q6_K = **397.9 MiB**, and streaming it is the
entire draft cost: 397.9 MiB at this host's measured bandwidth is ~26 ms, which
is the 24-26 ms/drafted-token measured three separate ways.

Drafting does not need the whole vocabulary. BPE gives frequent tokens low ids,
and this model's output confirms it — median emitted token id is 485-1332
against a vocabulary of 248,320. So `{id < K}` is a **contiguous byte prefix** of
the head: a shorter matmul over sequential memory, sliced with
`QuantizedMatrix::leading_rows` at no requantisation cost.

The context-token union the design originally called for was dropped after
measuring what it adds over the prefix alone at K=32768: **+1.6 pts on W1,
+1.2 on W2, +0.8 on W3, +2.9 on W4**. It is not worth a gather kernel and the
bookkeeping to maintain a dynamic set.

| workload | draft ms/token, full head | K=32768 | single-arm MTP uplift | acceptance full -> K |
| --- | ---: | ---: | ---: | ---: |
| W1 | 25.3 | **8.1** | 1.090x | 95.0% -> 92.1% |
| W2 | 23.9 | **6.9** | 1.111x | 93.2% -> 89.5% |
| W3 | 23.9 | **6.7** | 1.209x | 93.0% -> 98.0% |

The residual ~7 ms is the MTP transformer block itself; the LM head component
fell from ~24 ms to ~3.5 ms as the byte count predicts.

Acceptance drops slightly where coverage is imperfect, which is the trade being
made and it is small. **This is safe only because drafting is a proposal**: a
token the prefix gets wrong is rejected by the target exactly like any other
wrong draft. The target model keeps its own full head and never touches this
path; `a_shortlisted_draft_head_changes_speed_but_not_output` pins both halves.

### K sweep

| K | W1 | W2 | W3 |
| --- | ---: | ---: | ---: |
| 8192 | 1.187x | 0.989x | 1.267x |
| 16384 | **1.259x** | **1.077x** | **1.327x** |
| **32768** | **1.268x** | **1.086x** | 1.309x |
| 65536 | 1.261x | 1.060x | 1.282x |
| full | 1.238x | 0.967x | 1.110x |

16384 and 32768 are within noise of each other and both clearly beat the
extremes: 8192 loses coverage faster than it saves bytes, 65536 saves too few.
32768 is kept as the default on the strength of W1 and W2.

### The confidence gate, recalibrated against the shortlisted drafter

A prefix softmax has the wrong normaliser — the missing mass is exactly what the
prefix dropped — so the threshold measured against the full-head drafter could
not be assumed to carry. Re-measured:

| confidence | W1 | W2 | W3 |
| --- | ---: | ---: | ---: |
| 0.5 | 1.246x | 1.008x | 1.257x |
| 0.6 | 1.270x | 1.021x | 1.282x |
| **0.7** | 1.268x | **1.086x** | **1.309x** |
| 0.8 | **1.337x** | 1.045x | 1.289x |

0.7 remains the right default: best on W2, which is the binding workload, and
within noise of best on W3. W1 prefers 0.8, which is worth revisiting if W1-like
work becomes the priority. The two mechanisms compose rather than fight — the
gate declines precisely the drafts the prefix gets wrong, which is why
acceptance holds up at 89-98%.

## Earlier result: the confidence gate alone

| workload | gate | target-only | policy, no confidence gate | **policy, gate 0.70** | ratio | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| W1 copy-heavy | 1.25x | 7.06 | 8.77 | **9.15** | **1.296x** | **pass** |
| W2 prose | 0.97x | 7.98 | 7.16 | **7.73** | **0.969x** | miss by 0.001 |
| W3 self-repetitive | 1.02x | 7.95 | 8.57 | **8.79** | **1.106x** | **pass** |
| W4 mixed, 768 tokens | 1.10x | 6.23 | 6.38 | **6.75** | **1.083x** | miss |

Token ids identical to target-only on all four (but see "Greedy equivalence is
probabilistic" below — that claim is weaker than it looks, and not because of
this change).

Against the previous branch (1.215x / 0.906x / 1.098x / 1.012x) every workload
improved, and **W1 clears its gate for the first time**. W2 lands 0.001 short of
its gate on best-of-two and 0.002 over it on the single-rep sweep cell at the
same setting; the honest statement is that W2 is *at* its gate, indistinguishable
from it at this host's noise level, and it is recorded as a miss because the
best-of-two protocol says so.

## What the gate does

The MTP head's own top-1 softmax probability now ends a chained draft at the
first token below a threshold, **inside the drafting loop**, so declined tokens
are never drafted and their ~25 ms each is never paid. The pre-existing
`--speculative-mtp-min-margin` truncated after `draft_mtp` had already drafted
all *d* tokens, so it saved verification rows only, and it gated on a raw logit
margin, which is not on a scale anything can be compared against.

| workload | drafted | submitted | declined | chains cut | MTP acceptance | uplift from the gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 92 | 78 | 15% | 14 | **97.4%** | 1.043x |
| W2 | 255 | 147 | 42% | 108 | **93.2%** | 1.080x |
| W3 | 224 | 192 | 14% | 32 | **92.7%** | 1.026x |
| W4 | 541 | 471 | 13% | 70 | **94.3%** | 1.058x |

MTP acceptance was 59.9–82.3% before the gate and is **92.7–97.4%** after. The
uplift is largest on W2, the workload the arm was losing money on, which is what
the gate was built for.

**The MTP arm now suspends on no workload at all** — zero suspensions, zero
probes, on every one of W1–W4. The EWMA backoff has become dead weight in these
measurements because the arm it was there to withdraw is now profitable
everywhere. It should stay: it is the safety net for a workload none of these
four represents, and the calibration below shows what happens without it.

## Part 0 — the threshold was derived, then confirmed

Two independent routes to the same number.

**Derived.** A drafted token is worth submitting only if the probability the
target agrees exceeds the cost of the marginal row plus the draft that produced
it, over the plain decode step it replaces:

```text
p* = (draft_ms + row_ms) / plain_step_ms
```

which on measured stage costs is **0.739 (W1), 0.696 (W2), 0.702 (W3)**.

**Measured.** `--draft-calibration` records every drafted token's confidence,
depth and whether the target accepted it. Over 1,092 drafted tokens at depth 7,
acceptance is strongly monotone in confidence:

| workload | bottom decile | top decile | overall |
| --- | ---: | ---: | ---: |
| W1 | 0.385 | 1.000 | 0.835 |
| W2 | 0.000 | 0.679 | 0.314 |
| W3 | 0.296 | 1.000 | 0.776 |

Replaying each recorded chain offline — stopping at the first token below *t*,
charging the unavoidable extra draft whose confidence is *why* it stopped, and
costing passes as `fixed + marginal x rows` fitted to the measured 1-row and
~8-row points — puts the optimum at **0.70 on all three workloads**:

| gate | W1 | W2 | W3 |
| --- | ---: | ---: | ---: |
| off | 1.112 | 0.565 | 1.114 |
| **0.70** | **1.244** | **1.043** | **1.222** |
| decline everything | 0.837 | 0.830 | 0.832 |

Derivation and measurement agreeing to within 0.04 is the reason the default is
0.70 rather than a tuned constant.

That third row is the load-bearing one. **Declining every draft is worse than
plain decode**, because the gate must always draft one token to learn its
confidence. The gate optimises *within* a step; only the backoff can withdraw
the arm *across* steps. They are complementary, and a workload harsher than W2
still needs both.

## Threshold sweep, measured

Single rep per cell, policy `auto`.

| threshold | W1 | W2 | W3 |
| --- | ---: | ---: | ---: |
| 0.5 | 8.65 (1.225x) | 7.55 (0.946x) | 9.11 (1.146x) |
| 0.6 | 8.69 (1.231x) | 7.48 (0.937x) | 8.99 (1.131x) |
| **0.70** | 8.91 (1.262x) | **7.76 (0.972x)** | 8.59 (1.081x) |
| 0.8 | **9.02 (1.278x)** | 7.40 (0.927x) | 8.68 (1.092x) |
| 0.9 | 8.77 (1.242x) | 7.31 (0.916x) | 8.58 (1.079x) |

0.70 is best on W2 and within noise of best on W1. W3 prefers 0.5 in this sweep
(1.146x against 1.081x), which is the one disagreement with the model; W3 clears
its gate at every setting, so 0.70 is kept as the compromise that maximises the
binding constraint. A per-workload optimum is not available to a default.

## Greedy equivalence is probabilistic, and always was

Validation surfaced a pre-existing problem that three reports have stated as a
guarantee. It is filed as its own task and is **not** caused by this change.

`QuantizedMatrix::forward` dispatches on `MULTI_ROW_RANGE` plus a size
threshold: at M=1 it falls through to Candle's kernel, and at M>=2 it takes the
small-M/fused path, **which quantises its input rows**. Those are different
algorithms, so the same position evaluated at different pass widths gets
different arithmetic. Measured against the real checkpoint
(`batching_perturbs_logits_but_not_the_greedy_choice`):

- every one of the 248,320 logits differs between a 1-row pass and row 0 of a
  wider pass
- worst |delta| **1.28**
- tightest top1/top2 margin over 24 sampled positions **0.25**

**The perturbation is about five times the tightest decision margin.** So a
speculative run accepts on the multi-row argmax while a target-only run decides
on the one-row argmax, and at a tight margin they can disagree. Every measured
comparison in this report and the previous one came out identical, but that is
an empirical observation, not a property of the design.

It does bite. The integration suite failed once during this work with two tests
reporting a divergence, then passed on two subsequent full runs of the same
binary in the same order. That fits exactly: a fixed configuration is
bit-reproducible (8/8 paired campaign runs identical), while comparing two
configurations with different pass widths is a dice roll on tight margins. This
change altered pass widths and rolled a bad one.

The fix is to make M=1 take the same path as M>=2 so equivalence becomes exact.
It changes target-only numerics too, so it needs its own re-baseline, and it is
tracked separately rather than smuggled in here.

## The verification row is at the memory floor

Measured with `gguf_verify_bench` while closing the expert-batching task, and
it settles the question of where the remaining headroom is.

Expert weights are 498.0 MiB per layer across 256 experts. Per verification
pass, counting only the experts actually selected:

| K | routed gate/up + down | ms/row | unique experts | bytes | GB/s | duplicate rate | rows/expert |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36.2 ms | 36.2 | 320 | 622 MiB | **18.03** | 0.0% | 1.00 |
| 2 | 59.6 ms | 29.8 | 504 | 980 MiB | 17.24 | 21.2% | 1.28 |
| 4 | 107.5 ms | 26.9 | 862 | 1,677 MiB | 16.36 | 32.7% | 1.52 |
| 8 | 183.1 ms | 22.9 | 1,277 | 2,484 MiB | 14.23 | 50.1% | 2.08 |
| 16 | 335.3 ms | 21.0 | 2,225 | 4,328 MiB | 13.54 | 56.5% | 2.38 |

Against a host that measures 15.599 GB/s on STREAM Triad. Triad is two reads
and a write, so it understates a pure read stream — which is why the ratio
exceeds 100% and why the practical read ceiling is at least the 18.03 GB/s
observed at K=1. **The stage is sitting on the memory floor.**

The duplicate expert loads that an expert-side batching pass would have removed
are already gone: 16x the rows touch only 7x the bytes, because the expert-major
path serves all 56.5% of duplicate assignments from a single load.

So the ~0.55 verification-row-to-decode-step ratio is a property of this MoE on
this machine, not an implementation gap, and the "biggest remaining lever"
estimate in `policy-report-702d043633e0.md` — a 30% row-cost cut worth more than
the other two levers combined — is retired. **Further speculative gains have to
come from fewer rows, not cheaper ones**: better drafting, higher acceptance,
cheaper drafts. Which is what the two changes in this report did.

## Deviations and notes

1. **The gate is on by default** at 0.70, including in single-arm `mtp` mode.
   That changes `--speculative mtp` against the numbers in
   `policy-report-702d043633e0.md`; `--mtp-min-confidence 0` restores the old
   behaviour and is what the `nogate` rows above use.
2. **`--speculative-mtp-min-margin` is retained** but superseded. It is kept for
   comparability with the pre-existing measurement in
   `docs/speculative-decoding.md`, not because it is recommended.
3. **The default is still `--speculative off`.** Two of four gates pass. W2 is
   at its gate and W4 is 0.017 short, so the never-lose property this work is
   aiming at is close but not established.
4. The calibration curve is specific to the current Q6_K drafter. Anything that
   changes the drafting head — a shortlisted or lower-precision LM head — moves
   the softmax and requires re-running Part 0.

## Reproducing

```bash
CARGO_TARGET_DIR=target-native RUSTFLAGS='-C target-cpu=native' \
  cargo build --release --bin gguf_infer

./draft-run-logs/campaign.sh        # the confidence-gate campaign, one process per cell

# The configuration matrix, loading and warming the model once. 72 cells in
# 54 min against 52 cells in 58 min for the per-process harness.
taskset -c 0-5 ./target-native/release/gguf_policy_bench \
  --model "${MODEL_ROOT}/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  --tokenizer-model "${MODEL_ROOT}" \
  --prompts draft-run-logs/prompts.json \
  --matrix draft-run-logs/matrix-shortlist.json \
  --repetitions 2 --output draft-run-logs/shortlist.jsonl
```

Calibration data for a fresh threshold:

```bash
./draft-run-logs/run.sh cal_w1 draft-run-logs/prompts/w1.txt 256 \
  --speculative mtp --mtp-depth-cap 7 --mtp-depth-start 7 \
  --no-adaptive-length --no-ewma-backoff --no-span-continuation \
  --draft-calibration draft-run-logs/cal_w1.jsonl
```
