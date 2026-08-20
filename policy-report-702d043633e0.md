# The unified speculative policy

One default-on speculative loop over three draft sources, measured on host
`702d043633e0` (Intel i7-8700, six physical cores, `taskset -c 0-5`, INFERQ
default threading, fully resident Q4_K_M, greedy decoding). Every table below
is regenerable with `policy-run-logs/campaign.sh` and
`policy-run-logs/extract.py`; the raw logs and per-step traces are in
`policy-run-logs/`.

## The one number that explains this report

The MTP block costs **24-26 ms per drafted token** on this host and it pays that
cost whether or not the draft is accepted. Setting that against a measured
verification row (70-85 ms, rising with context length) and a plain decode step
(135-150 ms, likewise) gives the acceptance rate at which the MTP arm starts
paying for itself:

| single-arm run | draft ms/token | verify ms/row | other ms/pass | depth | plain ms/token | **break-even acceptance** | measured acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1, MTP d7 | 25.6 | 84.5 | 72 | 6.84 | 150.3 | **0.739** | 0.835 |
| W2, MTP d7 | 24.1 | 69.7 | 46 | 6.85 | 134.7 | **0.675** | 0.314 |
| W3, MTP d7 | 24.1 | 71.6 | 50 | 6.92 | 136.3 | **0.686** | 0.776 |

The model behind that column is
`cost = draft·d + verify·(1+d) + other`, `tokens = 1 + a·d`, break-even where
`cost / tokens = plain`. It predicts each single-arm run's measured throughput
to within about 1%, so it is a description of this host rather than a guess.

**The MTP arm's break-even acceptance is 0.68-0.74. The suspend threshold the
task specifies is 0.5.** Every miss below follows from that gap: at 0.5 the arm
is allowed to keep drafting through a whole band of acceptance rates in which
each proposal loses time. The specified sweep — 0.4, 0.5, 0.6 — brackets
break-even from below and never crosses it, so an extended sweep at 0.7 and 0.8
was added.

## Verdict against the task's gates

| gate | required | measured | verdict |
| --- | --- | --- | --- |
| 1. `validate.sh` and the new unit tests | pass | fmt, check, 93 unit + 10 integration tests, clippy `-D warnings`, release build all clean | **pass** |
| 2. W1 copy-heavy | >= 1.25x | **1.215x** | miss |
| 3. W2 prose | >= 0.97x | **0.906x** | miss |
| 4. W3 self-repetitive | >= 1.02x (set by Part 0.1) | **1.098x** | **pass** |
| 5. W4 mixed | >= 1.10x | **1.012x** | miss |
| 6. Single-arm modes reproduce the previous reports | within noise | all six reproduce; W1 n-gram is exact | **pass** |
| 7. MTP smoke and existing tests | pass | pass | **pass** |
| 8. Flip the default if 2-5 all pass | — | three miss, so **the default stays `off`** | n/a |
| Greedy equivalence, every workload and mode | token ids identical | identical in every comparison | **pass** |

Three gates miss. Section "The one number" above is the whole explanation, and
the sections below test it rather than assert it: the threshold sweep the task
specified, an extended sweep past the measured break-even, and a targeted
experiment on the two constants that turned out to matter.

## Part 0

**0.1 — MTP depth 7 on W3, best of two.** 8.43 and 8.41 tok/s against 7.35
target-only: **1.147x**. That is comfortably above 1.05, so **the W3 gate is
1.02x**, the strict one, and it is the gate the policy passes.

**0.2 — MTP acceptance on W1 conditioned on n-gram evidence.** MTP depth 4,
acceptance split by whether the index also held a match for that step:

| steps | tokens proposed | accepted | acceptance |
| --- | ---: | ---: | ---: |
| where the index also had a match | 140 | 137 | **97.9%** |
| where it did not | 80 | 63 | **78.8%** |
| unconditional | 220 | 200 | 90.9% |

The answer is **adverse selection, not inheritance**. The steps where literal
evidence exists are exactly the steps where MTP is nearly perfect, and the tie
rule gives all of them to the n-gram arm. The policy's MTP arm therefore works
on the 78.8% residue, not the 90.9% headline — which, against the 0.739
break-even measured on W1, leaves almost nothing to convert into a speedup.
This single number predicted the W1 miss before it was measured.

The split is computed in-run from the same index the policy consults, and is
reported in every mode, which is why a single-arm MTP run can answer it at all.

## Decode throughput, best of two

| workload | target-only | policy `auto` | ratio | tokens identical |
| --- | ---: | ---: | ---: | --- |
| W1 copy-heavy | 6.83 tok/s | 8.30 tok/s | **1.215x** | yes |
| W2 prose | 7.46 tok/s | 6.76 tok/s | **0.906x** | yes |
| W3 self-repetitive | 7.35 tok/s | 8.07 tok/s | **1.098x** | yes |
| W4 mixed, 768 tokens | 5.92 tok/s | 5.99 tok/s | **1.012x** | yes |

Against the single arms, best of two throughout:

| workload | target-only | n-gram only | MTP only | **policy** |
| --- | ---: | ---: | ---: | ---: |
| W1 copy-heavy | 1.000x | **1.234x** | 1.110x | 1.215x |
| W2 prose | 1.000x | **0.940x** | 0.576x | 0.906x |
| W3 self-repetitive | 1.000x | 0.837x | **1.147x** | 1.098x |
| worst case | — | 0.837x | 0.576x | **0.906x** |

The policy is **never the best configuration on any single workload and never
the worst**. It gives up about 2 points to n-gram-only on W1, 5 to MTP-only on
W3, and 3 to n-gram-only on W2 — and in exchange its worst case is -9% instead
of n-gram's -16% or MTP's -42%.

That is exactly the trade a default-on policy should make, and it is worth
being precise about why it cannot do better. On a workload where one arm is
right, the policy spends its opening proposals discovering which one, and pays
for the other arm's losing proposals until backoff withdraws it. The gap to the
better single arm *is* that discovery cost. It is bounded and small; what it is
not is zero, and the gates were written as though it would be.

## Per-arm behaviour

| workload | steps | n-gram (span) | MTP | plain | evidence | n-gram acc | MTP acc | suspensions n/m | probes n/m | tokens/pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 46 | 25 (19) | 21 | 0 | 54.3% | 84.8% | 78.1% | 0/0 | 0/0 | 5.54 |
| W2 | 165 | 3 (0) | 46 | 116 | 1.8% | 13.3% | 59.9% | 0/2 | 0/1 | 2.84 |
| W3 | 52 | 9 (3) | 42 | 1 | 19.2% | 41.3% | 80.7% | 1/0 | 0/0 | 4.98 |
| W4 | 214 | 93 (29) | 70 | 51 | 43.5% | 61.0% | 82.3% | 0/1 | 0/1 | 4.39 |

Span continuation carries real weight: **19 of W1's 25 n-gram steps and 29 of
W4's 93** are continuations rather than fresh key matches. The index's
one-position-per-key limit was costing more than it looked like it was.

The arms specialise as intended. W1 and W4 are mostly n-gram work; W3 is almost
entirely MTP (42 of 52 steps) because it repeats structure without repeating
spans; W2 is mostly plain decode (116 of 165 steps) because neither arm has
anything to work with.

## Where the time goes

| workload | decode | lookup | MTP draft | verify | snapshot | rollback | plain | MTP resync | resync passes / rows (longest) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 30.72s | 0.000s | 2.50s | 26.49s | 1.14s | 0.110s | 0.00s | 1.51s | 21 / 525 (277) |
| W2 | 37.71s | 0.000s | 3.54s | 17.60s | 0.93s | 0.229s | 15.63s | 0.60s | 46 / 221 (65) |
| W3 | 31.61s | 0.000s | 5.55s | 24.88s | 1.24s | 0.163s | 0.14s | 0.76s | 42 / 283 (33) |
| W4 | 128.00s | 0.000s | 9.39s | 104.20s | 3.74s | 0.578s | 8.30s | 5.17s | 70 / 1132 (367) |

The n-gram lookup remains free — 0.000 s across every whole run, as in the
previous report.

**Lazy resynchronisation costs 1.6-4.9% of decode** (W2 0.60 s of 37.71 s,
W3 2.4%, W4 4.0%, W1 4.9%), and its single largest
component is the one-time prompt catch-up: 277 rows on W1, 367 on W4, visible
as the "longest" column and as the `resync rows` figure at step 1 of every
trajectory below. Two things follow, and the second is a genuine property of
this design rather than a measurement artefact:

- The comparison here is **unfavourable to the lazy scheme by construction.**
  The old MTP path resynchronised during prefill and charged that catch-up to
  prefill time; this one charges it to decode, where the tok/s figure lives.
  About 0.8 s of W1's 30.72 s decode is work the old numbers hid. It is
  strictly less total work — one batched pass instead of one per token, and
  none at all when the arm never fires — and it still reads worse. No number in
  this report is adjusted for that.
- Because the cost is front-loaded, **partial suspension is the worst case for
  the MTP arm.** Once the prompt catch-up is paid it cannot be recovered by
  suspending, and resuming pays a fresh gap. The extended sweep shows this
  directly.

## Gate 6 — single-arm modes against the previous reports

| workload | mode | tok/s | ratio | previous report |
| --- | --- | ---: | ---: | ---: |
| W1 | ngram | 8.43 | 1.234x | 8.47 (1.228x) |
| W2 | ngram | 7.01 | 0.940x | 7.26 (0.937x) |
| W3 | ngram | 6.15 | 0.837x | 6.23 (0.809x) |
| W1 | mtp | 7.58 | 1.110x | 1.14x |
| W2 | mtp | 4.30 | 0.576x | 0.55x |
| W3 | mtp | 8.40 | 1.143x | 1.07x (single run; 1.147x best-of-two here) |

W1 n-gram reproduces the previous report **exactly** — 27 drafts, 25.2% match
rate, 6.48 tokens per verification pass — which is the strongest available
evidence that the single-arm path really is the policy loop with one arm
disabled and not a re-implementation. The MTP rows carry the prefill-resync
relocation described above, which is the direction of their small shortfall.

## Sweep — MTP suspend threshold x start depth

Single rep per cell, as specified.

### W2 (target-only 7.46 tok/s)

| suspend below | start 3 | start 4 | start 5 |
| --- | ---: | ---: | ---: |
| 0.4 | 6.65 (0.891x) | 6.75 (0.905x) | 6.63 (0.889x) |
| 0.5 | 6.74 (0.903x) | 6.58 (0.882x) | 6.56 (0.879x) |
| 0.6 | 6.87 (0.921x) | 6.66 (0.893x) | 6.71 (0.899x) |

### W4 (target-only 5.92 tok/s)

| suspend below | start 3 | start 4 | start 5 |
| --- | ---: | ---: | ---: |
| 0.4 | 5.97 (1.008x) | 5.98 (1.010x) | 6.07 (1.025x) |
| 0.5 | 5.97 (1.008x) | 6.05 (1.022x) | 6.09 (1.029x) |
| 0.6 | 5.95 (1.005x) | 5.87 (0.992x) | 5.99 (1.012x) |

**Start depth is not a lever.** Nine W2 cells span 0.879x to 0.921x and nine W4
cells span 0.992x to 1.029x, with no ordering by start depth in either. The
adaptive controller moves the depth away from its start within a few
proposals, which is what it is for.

## Extended sweep — past the measured break-even

The specified sweep brackets the 0.68-0.74 break-even from below without
crossing it, so these cells were added.

| suspend below | W1 | W2 | W3 | W4 |
| --- | ---: | ---: | ---: | ---: |
| 0.5 (default, best of two) | **8.30 (1.215x)** | 6.76 (0.906x) | **8.07 (1.098x)** | 5.99 (1.012x) |
| 0.7 | 8.03 (1.176x) | 6.67 (0.894x) | 8.07 (1.098x) | 6.09 (1.030x) |
| 0.8 | 7.89 (1.155x) | **6.95 (0.932x)** | 7.76 (1.056x) | 5.89 (0.995x) |

**Raising the bar past break-even does not work.** It fails to fix W2 (best
0.932x, still short of 0.97x) and it costs W1 and W3. The break-even model is
correct about which individual proposals lose money; it is wrong to conclude
that a threshold can act on that, for two measured reasons:

1. **W2's cost is detection and re-entry, not steady-state drafting.** At 0.7
   the arm is out for 192 of 219 steps and still loses 12%. It spends about
   five proposals discovering failure, suspends four times, and probes three
   times — and every probe must resynchronise the whole suspended gap before it
   can draft.
2. **Partial suspension is worse than either extreme on W1.** At 0.5 the arm
   never suspends and returns 1.215x. At 0.7 it suspends once, 20 steps fall
   back to plain decode, and throughput drops to 1.176x — *below* both
   always-on and n-gram-only. The prompt catch-up has already been paid and
   suspending cannot recover it.

## Withdrawal and re-entry constants

Since the threshold is not the lever, the two constants that control detection
cost and re-entry frequency were varied instead. Single rep per cell.

| configuration | W1 | W2 | W3 | W4 |
| --- | ---: | ---: | ---: | ---: |
| defaults (alpha 0.2, first suspension 64) | **8.30 (1.215x)** | 6.76 (0.906x) | 8.07 (1.098x) | 5.99 (1.012x) |
| alpha 0.4 — withdraw in ~3 proposals | 7.67 (1.123x) | 6.16 (0.826x) | 8.24 (1.121x) | 6.14 (1.037x) |
| first suspension 256 tokens — fewer re-entries | 8.23 (1.205x) | 6.99 (0.937x) | 8.16 (1.110x) | 5.91 (0.998x) |
| alpha 0.4 and 256 tokens | 8.00 (1.171x) | **7.05 (0.945x)** | **8.16 (1.110x)** | **6.25 (1.056x)** |

| configuration | W2 MTP steps | W2 MTP acc | suspensions | probes (resumed) | plain steps | W2 resync |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| defaults | 46 | 59.9% | 2 | 1 (1) | 116 | 0.60s |
| alpha 0.4 | 18 | 51.0% | 3 | 2 (1) | 203 | 0.57s |
| suspension 256 | 31 | 61.9% | 1 | 0 (0) | 151 | 0.30s |
| both | **14** | 54.8% | **1** | **0 (0)** | **212** | **0.16s** |

This isolates the mechanism cleanly. **Rarer re-entry helps** — lengthening the
first suspension alone improves W2 and W3 at no measurable cost on W1, and cuts
W2's suspension count from 2 to 1 and its resynchronisation from 0.60 s to
0.30 s. **Faster withdrawal on its own hurts** — alpha 0.4 costs W1 nearly 8%,
because it suspends an arm that was profitable there. Together they give W2's
ideal shape: withdraw once, at 14 proposals instead of 46, and stay out for the
rest of the run.

## Closest passing configuration

`--speculative auto --ewma-alpha 0.4 --backoff-tokens 256`, best of two, under
the same protocol as the defaults:

| workload | gate | target-only | defaults | verdict | recommended | verdict |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| W1 | 1.25x | 6.83 | 8.30 (1.215x) | miss | 7.96 (1.165x) | miss |
| W2 | 0.97x | 7.46 | 6.76 (0.906x) | miss | 7.07 (0.948x) | miss |
| W3 | 1.02x | 7.35 | 8.07 (1.098x) | **pass** | 8.29 (1.128x) | **pass** |
| W4 | 1.10x | 5.92 | 5.99 (1.012x) | miss | 6.21 (1.049x) | miss |

Token ids identical to target-only on all four.

It improves three workloads and costs one, and it reduces the worst single
shortfall from 0.088 (W4 under defaults) to 0.085 (W1). It is closer, and it
still does not pass. **The recommendation is therefore to raise
`DEFAULT_BACKOFF_TOKENS` from 64 to 256 and leave alpha at 0.2** — the
re-entry finding is clean and mechanism-backed on every workload, whereas
alpha 0.4 helps only in combination and hurts badly alone, which is the shape
of noise rather than of an effect. That change is *not* applied in this branch:
with the default `off`, changing a controller constant on the strength of
single-rep cells would be tuning without a gate to check it against.

## W4 — the arm hand-off

**Deviation, stated up front.** The W4 prompt as first written sequenced the
three phases but did not bound their length, and the model spent all 768 tokens
inside part 1. The workload could not show a hand-off because it never left the
first phase. The prompt now bounds each part — part 2 "in exactly three
sentences", part 3 with the literal test skeleton to repeat — and every W4 cell
in this report was measured against it. The corrected workload is *harder*: it
contains a genuine prose stretch where neither arm can win, and W4 fell from
1.066x on the unbounded prompt to 1.012x on the bounded one.

| window (committed tokens) | n-gram | span | MTP | plain | n-gram acc | MTP acc | depth at end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-255 | 10 | 14 | 25 | 41 | 74% | 66% | 2 |
| 256-511 | 22 | 8 | 24 | 10 | 57% | 88% | 6 |
| 512-767 | 32 | 7 | 21 | 0 | 53% | 90% | 7 |

The hand-off is visible: n-gram steps grow 24 -> 30 -> 39 as the answer becomes
more repetitive, MTP acceptance climbs 66% -> 88% -> 90% and its depth with it
(2 -> 6 -> 7), and the 41 plain steps — the state both arms withdraw to — are
concentrated in the first window, which contains the hardest, least repetitive
prose. By the unit-test phase in the last window there are no plain steps left
at all.

What the table also shows is why W4 misses its gate. The arms hand off
correctly and the policy still only reaches 1.012x, because the phase where
both arms work well is also the phase where the n-gram arm alone would have
done nearly as well, and the phase where the MTP arm is needed is the one where
its 25 ms per drafted token is hardest to earn back.

## Controller trajectories

W2, sampled every 16th step — the backoff doing its job:

| step | committed | arm | proposed/accepted | n-gram len | n-gram ewma | susp | MTP depth | MTP ewma | susp | resync rows |
| ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| 1 | 1 | mtp | 4/0 | 7 | 1.00 | n | 2 | 0.80 | n | 24 |
| 17 | 45 | mtp | 4/4 | 7 | 1.00 | n | 5 | 0.78 | n | 4 |
| 33 | 98 | plain | 0/0 | 7 | 1.00 | n | 2 | 0.48 | y | 0 |
| 49 | 115 | ngram | 4/0 | 4 | 0.66 | n | 2 | 0.48 | y | 0 |
| 65 | 131 | plain | 0/0 | 4 | 0.66 | n | 2 | 0.48 | y | 0 |
| 81 | 148 | plain | 0/0 | 4 | 0.58 | n | 2 | 0.48 | y | 0 |
| 97 | 170 | mtp | 3/1 | 4 | 0.58 | n | 2 | 0.77 | n | 3 |
| 113 | 203 | plain | 0/0 | 4 | 0.58 | n | 2 | 0.45 | y | 0 |
| 145 | 235 | plain | 0/0 | 4 | 0.58 | n | 2 | 0.45 | y | 0 |
| 161 | 251 | plain | 0/0 | 4 | 0.58 | n | 2 | 0.45 | y | 0 |

The MTP arm withdraws by step 33 and the run settles into plain decode; the one
probe at step 97 succeeds on its own terms (three proposed, one accepted lifts
the EWMA to 0.77) and the arm re-suspends immediately afterwards. That probe is
the flapping the withdrawal experiment above removes.

W3, where the opposite happens — the n-gram arm is the one that withdraws:

| step | committed | arm | proposed/accepted | n-gram len | n-gram ewma | susp | MTP depth | MTP ewma | susp | resync rows |
| ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| 1 | 1 | mtp | 4/4 | 7 | 1.00 | n | 5 | 1.00 | n | 33 |
| 17 | 83 | mtp | 3/3 | 7 | 1.00 | n | 4 | 0.83 | n | 3 |
| 33 | 168 | mtp | 7/7 | 4 | 0.59 | n | 7 | 0.92 | n | 8 |
| 49 | 243 | mtp | 5/5 | 4 | 0.38 | y | 6 | 0.82 | n | 5 |

The MTP depth climbs 5 -> 7 while acceptance holds above 0.82, and the n-gram
arm shrinks 7 -> 4 and suspends. This is the case the whole task exists for: a
workload where the previously shipped drafter loses 16% and the policy gains
10%.

W4, the mixed workload, showing both arms alternating and one MTP suspension
around the prose phase:

| step | committed | arm | proposed/accepted | n-gram len | n-gram ewma | susp | MTP depth | MTP ewma | susp | resync rows |
| ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| 1 | 1 | mtp | 4/3 | 7 | 1.00 | n | 4 | 0.95 | n | 367 |
| 17 | 83 | ngram | 4/4 | 6 | 0.75 | n | 4 | 0.77 | n | 0 |
| 33 | 158 | ngram-span | 7/0 | 4 | 0.65 | n | 6 | 0.85 | n | 0 |
| 49 | 214 | mtp | 2/0 | 7 | 0.71 | n | 2 | 0.47 | y | 2 |
| 81 | 246 | plain | 0/0 | 7 | 0.71 | n | 2 | 0.47 | y | 0 |
| 113 | 307 | mtp | 4/4 | 6 | 0.64 | n | 5 | 0.76 | n | 4 |
| 145 | 470 | ngram | 4/4 | 6 | 0.57 | n | 7 | 0.93 | n | 0 |
| 177 | 624 | ngram | 4/1 | 4 | 0.50 | n | 7 | 0.94 | n | 0 |
| 209 | 746 | mtp | 5/5 | 7 | 0.92 | n | 6 | 0.89 | n | 21 |

## What this says about the next step

The three Part B mechanisms all work as specified and all ship on by default;
none needed to be withdrawn behind a flag. Span continuation earns its place
outright. The finding is about the **signal the backoff controller uses**, not
about the loop or the mechanisms.

Acceptance fraction is a proxy for profitability, and on this host it is not a
good enough one. Profitability depends on depth, on context length (a
verification row costs 70 ms at 289 tokens of context and 85 ms at 533), and —
under lazy catch-up — on how much gap a resumption has to close. The measured
break-evens differ by 10 percentage points across three workloads for that
reason. A single acceptance threshold cannot separate "78% and profitable" on
W1 from "60% and unprofitable" on W2, and the extended sweep shows what happens
when you try: the bar that suspends W2's arm also suspends W1's and W3's.

The controller already has everything it needs to measure the real quantity.
The loop times every stage per arm; **an arm could compare its own milliseconds
per committed token against the plain-decode step time it can also measure, and
suspend when that ratio exceeds one.** That is a direct measurement of the
thing the EWMA is approximating, it needs no new constant per host, and on this
data it would have suspended W2's arm after two proposals (206 ms/token against
134) while never suspending W3's (122 against 136). It is a small change to
`ArmController` and it is the change this report recommends before any further
tuning of thresholds.

The other lever is the one the task ruled out of scope and the data keeps
pointing at: the MTP arm's 25 ms per drafted token is LM-head-bound, and every
break-even above is a direct function of it. Halving it would move the
break-even from ~0.70 to ~0.55 and put all four workloads on the other side of
their gates without touching the policy at all.

<!-- RESULTS -->

## Part A design — one loop, three sources

The previous dispatch chose a drafter once, at the top of generation, and both
choices were wrong somewhere. The n-gram arm is free when it fires but blind to
structural repetition; the MTP arm sees structure but pays a draft forward per
drafted token whether or not the draft will be accepted. Neither is a global
win, so neither could be a default.

The loop now decides per step, in a fixed order:

1. **n-gram arm.** An active span continuation, or a key match tried
   longest-first down to `--ngram-min-match`. Free either way: a slice
   comparison against the index. If it produces a draft and the arm is not
   suspended, it takes the step.
2. **MTP arm.** Otherwise, if that arm is not suspended, a chained draft at the
   controller's current depth.
3. **Plain decode.** Otherwise, the one-row pass unspeculated decoding runs —
   no draft, no snapshot, no rollback, no resynchronisation.

The tie rule is absolute: an n-gram match takes the step outright, the two arms
never run in one step, and an n-gram draft is never extended with MTP tokens.
`auto_uses_both_arms_and_never_runs_them_in_the_same_step` asserts it directly,
through the metric that would have to be non-zero if it were ever violated.

Both arms share the machinery unchanged — multi-row verification,
`accepted_draft_prefix`, snapshot rollback, the thinking-budget and stop-token
clamps — and single-arm `ngram` and `mtp` modes are this same loop with the
other arm's configuration disabled. There is no second code path to keep in
step, which is what makes gate 6 a test of the refactor rather than a
comparison of two implementations.

### MTP state consistency, and why catch-up is lazy

The MTP block keeps its own sequence state. Previously that state was
resynchronised from the verification pass's hidden rows after every MTP pass,
which was sound because every committed token came from an MTP pass. Under the
policy that is no longer true: the n-gram arm and plain decode commit tokens
too, and while the MTP arm is suspended it may commit none for hundreds of
tokens.

The runtime therefore tracks the block's synced position separately from the
committed position and retains, for the tokens in between, the authoritative
target hidden rows the committing passes already produced:

```text
mtp_synced_position ──── gap tokens + their hidden rows ──── state.position
        ▲                                                          ▲
   MTP block state                                     target model state
```

Every committing pass appends to that gap — the prefill pass, n-gram passes,
MTP passes, plain decode steps, and the tokens injected by a forced thinking
closure. Nothing resynchronises eagerly. `catch_up_mtp` runs only when the MTP
arm is about to draft, and closes the whole gap in **one batched pass**, which
is possible precisely because the rows were retained rather than recomputed.

Three consequences worth stating:

- While the MTP arm is suspended its state lags at **zero cost**. A run that
  spends its life in plain decode never touches the block at all.
- The prefill gap is lazy too. An `auto` run whose MTP arm never fires never
  pays the prefill resynchronisation the old MTP path paid unconditionally.
- One batched pass over *n* gap rows is strictly cheaper than *n* single-row
  passes, so laziness is not a trade of latency for throughput — it is cheaper
  on both counts. What it costs is memory: one hidden row per un-synced token.

Correctness rests on two checks. A `debug_assert` compares the synced position
with the target position before every draft, and a release-mode `ensure!`
re-checks it against the block's own reported position — the MTP arm cannot
draft from a stale position without one of them firing.

`--eager-mtp-resync` restores the old behaviour, resynchronising after every
committing pass. It exists so the two can be compared directly, and
`lazy_mtp_catch_up_decodes_exactly_like_eager_resynchronisation` does exactly
that against the real checkpoint. That test asserts more than token equality:
verification would mask a stale predictor, so equal output alone would prove
nothing. It asserts that the MTP arm **proposed and accepted identical token
counts** under both schemes, which is a property only a correctly caught-up
predictor has.

One boundary is handled by withdrawal rather than repair. If an output callback
fails midway through emitting a pass's committed tokens, the session rolls back
to that token's boundary — but the retained gap describes more tokens than
survive, and its hidden rows cannot be trimmed to match. The gap is marked
invalid and the MTP arm sits out until the session is reset. This costs speed
on a path that is already an error path, and never correctness.

## Part B design — three mechanisms, per-arm state

Each arm owns an independent `ArmController`. Suspension of one never affects
the other; both suspended is plain decode, and the cost of that state is two
integer comparisons per step.

### 1. Span continuation (n-gram arm)

After a fully accepted n-gram pass from source position *p* of length *L*, the
next step drafts the continuation of the same source span without a fresh key
match. A copied region normally re-matches on its own suffix, but only while a
key covering it survives in the index — the maps hold one position per key, so
a later occurrence of the same short suffix displaces it. The chain keeps the
copy running across those gaps.

The chain advances only when both conditions hold: every proposed token
verified, **and** the token the span predicts next is the one the target
actually chose. The second condition matters more than the first. A pass
commits its accepted drafts *and* the authoritative token after them; a fully
accepted draft whose successor diverges has already left the span, and
continuing it would propose tokens from a region the text no longer follows.
Any other outcome — a rejection, a step the arm did not win, a span that has run
off the end of the index — ends the chain. `chain_span` is a pure function and
carries its own unit tests for each of those cases.

Continuation drafts count as n-gram-arm proposals for the EWMA, and are
reported separately as `ngram_span_steps`.

### 2. Adaptive draft length and depth

| arm | range | start | on full acceptance | on acceptance below half |
| --- | --- | --- | --- | --- |
| n-gram | `[4, 7]` | cap (7) | `+2` | halve, floored |
| MTP | `[2, 7]` | 4 | `+1` | halve, floored |

The n-gram arm starts at its cap because 7 is the draft length the previous
report recommended and the length at which its acceptance histogram was still
flat; the controller only ever shortens below that when acceptance has actually
collapsed. The cap is 7 rather than higher because a pass evaluates the pending
token plus the drafts, and passes wider than eight rows leave the measured fast
path — that is a kernel-layer boundary, out of scope here, and the cap is a
flag rather than a constant so it can move when that changes.

The MTP arm starts at 4 and grows one at a time because its draft cost is
linear in depth at about 25 ms per drafted token. Measured acceptance by depth
on this model is 97.7% at depth 1, 94.5% at depth 3 and 55.9% at depth 15, so a
deep fixed depth is self-defeating; the controller exists to keep depth where
acceptance still earns it.

### 3. EWMA backoff

Per-proposal acceptance fraction, alpha 0.2, per-arm state and constants:

| arm | suspend below | first window | probe | escalation |
| --- | --- | --- | --- | --- |
| n-gram | 0.4 | 64 committed tokens | at current length | double on a failed probe, cap 512 |
| MTP | 0.5 | 64 committed tokens | at depth 2 | double on a failed probe, cap 512 |

The MTP bar is higher because that arm pays its draft cost unconditionally,
and its probe is at the floor because depth 2 is the cheapest informative
question it can ask.

Two decisions here are not in the specification and are worth defending.

**The EWMA starts from an optimistic prior of 1.0** rather than being seeded
from the first proposal. Seeding from the first proposal suspends an arm on its
first fully rejected draft — and on the copy-heavy workload the n-gram arm
accepts 80% of the tokens it proposes while still rejecting whole drafts
occasionally. W1 would lose its arm to noise. The prior decays as
`(1 - alpha)^n`, so it buys a failing arm about five consecutive total
rejections at the 0.4 bar and four at the 0.5 bar, and cannot hold one open
beyond that. Both trial lengths are asserted in unit tests.

**A probe replaces the EWMA rather than blending into it.** Blending would
carry the acceptance collapse that caused the suspension, and since the arm
gets exactly one proposal per probe, a workload that has moved on could not
resume for several more suspensions. A successful MTP probe also leaves the
depth at the probe depth, so re-entry is conservative and climbs back through
the growth rule rather than jumping to where it failed.

## Deviations from the specification, and why

1. **Part 0 ran after the implementation, not before it.** The two numbers it
   settles are the W3 gate (reporting only) and a calibration figure; no code
   depends on either, and both were measured through the refactored binary in
   single-arm mode, which gate 6 independently validates. Doing it in this
   order cost nothing and let the ~1 h of machine time run against a finished
   binary.

2. **Part 0.2 was answered in-run rather than offline.** The task suggested
   computing n-gram match existence offline from the token stream. Instead the
   loop records, in every mode, the key length the index would have matched —
   a slice comparison, measured at 0.000 s per run. It is the same answer from
   the same index the policy actually consults, with no second implementation
   of the lookup to disagree with the first.

3. **The W4 prompt was rewritten mid-campaign** and every W4 cell re-measured.
   See the W4 section: the original did not bound its three phases and never
   left the first one, so it could not show a hand-off. The corrected prompt is
   harder and W4's number is worse for it.

4. **An extended sweep and a withdrawal experiment were added** beyond the
   specified 3x3. The specified sweep brackets the measured break-even from
   below without crossing it, so on its own it could only have shown that
   nothing helps, without showing why. The task asks for constants recommended
   from the data; these are the cells the data pointed at.

5. **The acceptance EWMA starts from an optimistic prior of 1.0.** The
   specification gives alpha and the thresholds but not the initial value, and
   seeding from the first proposal would suspend an arm on its first fully
   rejected draft. On W1 the n-gram arm accepts 80% of proposed tokens while
   still rejecting whole drafts occasionally, so that seeding would have cost
   W1 its best arm to noise. See the Part B section for the trial lengths this
   implies, both of which are asserted in unit tests.

6. **`--eager-mtp-resync` was added.** It is not in the specification, but the
   task requires a test that lazy catch-up decodes identically to eager
   resynchronisation, and that test needs an eager mode to compare against.

7. **No Part B mechanism needed the default-off escape hatch.** All three pass
   their unit gates and ship on. The per-mechanism flags exist anyway, because
   the gate-6 reproduction needs them to pin the controllers.

8. **`--speculative auto` is not the default**, per gate 8: gates 2, 3 and 5
   miss. The recommended controller constant change (`DEFAULT_BACKOFF_TOKENS`
   64 -> 256) is likewise *not* applied — see "Closest passing configuration"
   for why changing a constant on single-rep evidence, with no gate left to
   check it against, would be tuning rather than measurement.

## Reproducing

```bash
CARGO_TARGET_DIR=target-native RUSTFLAGS='-C target-cpu=native' \
  cargo build --release --bin gguf_infer

./policy-run-logs/campaign.sh          # baselines, auto, Part 0, gate 6, sweep
./policy-run-logs/extended-sweep.sh    # suspend threshold past break-even
./policy-run-logs/withdrawal.sh        # detection cost and re-entry frequency
./policy-run-logs/recommended.sh --ewma-alpha 0.4 --backoff-tokens 256
./policy-run-logs/extract.py           # every table in this report
```

Integration tests against the real checkpoint:

```bash
INFERQ_TEST_GGUF=/models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
INFERQ_TEST_MODEL_DIR=/models/Qwen3.6-35B-A3B \
  cargo test --release --test speculative_policy --test ngram_speculation \
  -- --test-threads=1
```

## Scoping

No kernel, dispatch, quantization or LM-head change is in this branch. The
~25 ms per drafted token the MTP block costs is a constraint the controllers
work around, not a target; the two-stage LM head is a separate task.
