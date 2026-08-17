# The MTP draft confidence gate

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

./draft-run-logs/campaign.sh        # baselines, arm, gate, sweep
```

Calibration data for a fresh threshold:

```bash
./draft-run-logs/run.sh cal_w1 draft-run-logs/prompts/w1.txt 256 \
  --speculative mtp --mtp-depth-cap 7 --mtp-depth-start 7 \
  --no-adaptive-length --no-ewma-backoff --no-span-continuation \
  --draft-calibration draft-run-logs/cal_w1.jsonl
```
