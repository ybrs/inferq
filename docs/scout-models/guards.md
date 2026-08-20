# Prompt guards: how much of the fabrication is promptable?

Follow-up to [`report.md`](report.md) and [`alternates.md`](alternates.md),
answering "can stricter prompting raise the faithfulness score and lower the
hallucinations?"

**Short answer: yes, and by a lot — but where the guard goes matters more than
what it says.** The same guard text that takes Qwen3-1.7B from 16/24 to 22/24
makes a 0.6B summarize the guard's own worked example instead of the ticket it
was given. Two rules came out of this: keep the anti-refusal clause in, and keep
task-specific examples out of the shared system prompt.

## How this run differs from the earlier attempt

`alternates.md` reported a first pass at this (guards recover Q2_K_XL from 4/10
to 8/10; a 0.6B just learns to refuse). Those guards were hand-rewritten per
probe and the outputs were not kept, so they could not be re-scored. This run is
set up so the numbers are comparable:

- One shared **system** prompt ([`harness/tests-guarded/system-guard.txt`](harness/tests-guarded/system-guard.txt)).
  The user prompts stay byte-identical to the unguarded suite, so the system
  message is the only variable.
- Every answer is written to `outputs-guarded*/` and scored by the same graders
  as the baseline.
- Two **control** probes run alongside, `c1` and `c2`, whose answers *are*
  present in the text. A guard that buys its score with blanket refusal loses
  points there instead of hiding.

## Results

Faithfulness (`/24`) under three guard placements. Speed is decode t/s from
`alternates.md`, for what each row costs.

| Model | t/s | No guard | Shared rules | Shared rules + example | Rules shared, example per-task |
| --- | ---: | ---: | ---: | ---: | ---: |
| granite-4.1-3b Q4_K_M | 9.8 | **24** | **24** | **24** | **24** |
| granite-4.1-3b Q2_K_XL | 13.0 | 15 | **21** | 19 | 19 |
| Qwen3.5-2B Q4_K_XL | 14.9 | 16 | 13 | 17 | 19 |
| Qwen3-1.7B Q4_K_M | 19.1 | 16 | 16 | 19 | **22** |
| Qwen3-0.6B Q4_K_M | 49.8 | 7 | 16 | 17 | **24** |

Controls (`/10`), same runs — this is the column that says whether a score is
real:

| Model | No guard | Shared rules | Shared rules + example | Rules shared, example per-task |
| --- | ---: | ---: | ---: | ---: |
| granite-4.1-3b Q4_K_M | 10 | 10 | 10 | 10 |
| granite-4.1-3b Q2_K_XL | 10 | 10 | 10 | 10 |
| Qwen3.5-2B Q4_K_XL | 10 | 10 | 10 | 10 |
| Qwen3-1.7B Q4_K_M | 10 | 10 | **7** | 10 |
| Qwen3-0.6B Q4_K_M | 5 | 8 | **5** | 8 |

## What the guard fixed

**A general rule set is enough for grounded Q&A and summarization.** Qwen3-0.6B
went from inventing `TLS 1.2 and TLS 1.3` on the absent-fact probe to refusing
it correctly, and its summary score on the *task* suite went 2/7 → 7/7: unguarded
it blamed a customer-side change the ticket explicitly denies, guarded it stopped.
No model lost control points to the shared rules — the earlier "guards only
install a refusal" result did not reproduce.

The clause that makes the difference is the one pointing the other way:

> 7. If the message DOES state the answer, answer it. Refusing an answerable
>    question is as wrong as inventing one.

Without it a guard is all downside pressure, which is what the first pass ran
into. With it, `c1` stayed 3/3 for every model under the shared rules, including
the 0.6B that previously refused an answerable question.

**Structured extraction needs a worked example, and only a worked example.** The
rules alone never fixed `h3` for the fast models: Qwen3-1.7B kept merging the
invoice export and the migration script into one item and handing it to Emily —
it was losing a task, not just misattributing an owner. Adding one worked example
that demonstrates exactly that shape (a blocker that is not an owner, a second
person who owns only the second task, an availability line that is not an item)
took it to a clean 10/10:

```json
[{"title": "Fix the invoice export", "assignee": null, "due": null, "blocked_by": "legal signs off on the data retention question"},
 {"title": "Take the migration script", "assignee": "Emily", "due": null, "blocked_by": null},
 {"title": "Bump the Postgres version", "assignee": null, "due": null, "blocked_by": null}]
```

## What the guard broke

Putting that example in the **shared** system prompt is what the "+ example"
column costs. At ≤2B an example is not an illustration of a rule, it is a
template for the next answer:

- **Qwen3-1.7B** answered the plain question "what is the sample delivery ID?"
  with `{"title": "Sample delivery ID", "assignee": null, "due": null,
  "blocked_by": null}` — extraction schema, for a question that asked for a
  string. Control 10/10 → 7/10.
- **Qwen3-0.6B** summarized *the example* instead of the ticket: its answer
  discusses the nightly report, Ben, and the Node upgrade — none of which appear
  in the ticket it was handed. The guard became the hallucination source.
- **granite-4.1-3b Q2_K_XL** leaked the refusal phrase into JSON, emitting
  `"due": "NOT IN TICKET"` where the rules asked for `null`, and dropped from
  21/24 to 19/24. The low quant is the one build where the plain rule set beats
  every richer guard.

Moving the example into the extraction call only — general rules in the shared
system prompt, the example appended for the job that needs it — recovers both:
controls return to 10/10 and `h3` keeps the 10/10.

## What this changes

**Not the recommendation.** granite-4.1-3b Q4_K_M scores 24/24 with no guard at
all and cannot be improved by one. Guards are for the models below it.

**The fallback, though, is now real.** Qwen3-1.7B Q4_K_M at 19 t/s — roughly
twice granite's decode — reaches 22/24 with clean controls and no task-suite
regression (21/23 either way). If a scout call is latency-bound, that is a
defensible choice in a way it was not before; the two points it still drops are
both on summary attribution, not on extraction.

**Do not read the 0.6B's 24/24 as competence.** Three probes is a small suite
and it is now tuned for them. The same model scores 17/23 on the task suite
(guarded; 14/23 unguarded), writes a Python script that fails on the fixture, and
drops an entire task in control `c2`. It stopped falling into these three traps.
It did not become a scout.

**Low quants are still not worth it.** Guards close part of the gap — Q2_K_XL
goes 15 → 21 — but it still turns "Kevin is out until the 14th" into a task, and
it is the build most confused by a richer prompt. `alternates.md`'s conclusion
stands.

**Guards cost prefill, not decode.** The shared rules take the prompt from ~180
to ~540 tokens, and the version with the example to ~860. Measured on `h1`,
whose answer is five tokens, so the wall-clock is almost pure prefill:

| Model | No guard | Shared rules | + example |
| --- | ---: | ---: | ---: |
| granite-4.1-3b Q4_K_M | 3.1s | 8.3s | 13.3s |
| Qwen3-1.7B Q4_K_M | 1.6s | 4.0s | 6.8s |
| Qwen3-0.6B Q4_K_M | 0.9s | 1.6s | 2.6s |

That is +5.2s on granite for the shared rules and +10.2s with the example —
a real tax on short answers, and the reason the example belongs only in the call
that needs it. It scales with prefill speed, so it hurts least where the guard
helps most.

## A grader fix landed with this

Scoring the guarded runs surfaced two lexical gaps in `grade_h.py`'s
contradiction check. It only accepted `customer says/claims/reports`, so
"the customer, Bolt Industries, **reported**" scored as unattributed, and it only
accepted "root cause not identified", so "root cause **still unidentified**"
scored as dropped. Both patterns are widened. Every table in `report.md` and
`alternates.md` has been rescored: most rows gain a point, granite-4.1-3b and
granite-4.0-h-micro reach 24/24, and no ranking moved. The model outputs on disk
are unchanged — only the scoring of them.

## Reproducing

```bash
cd harness

./run-guarded.sh                       # 5 models x {control unguarded, faith guarded, control guarded}

python3 grade_h.py outputs-guarded     # guarded faithfulness
python3 grade_c.py outputs-control     # controls without the guard
python3 grade_c.py outputs-guarded     # controls with it — compare these two
```

`GUARD=tests-guarded/system-guard-v2.txt ./run-guarded.sh` runs the variant with
the worked example. `grade.py`, `grade_h.py` and `grade_c.py` all take an
outputs directory as their first argument, so any of the sets can be scored in
place:

| Directory | What is in it |
| --- | --- |
| `outputs/` | the original unguarded suites |
| `outputs-lowbit/` | low-quant builds, unguarded |
| `outputs-control/` | control probes, unguarded |
| `outputs-guarded/` | shared rules |
| `outputs-guarded2/` | shared rules + worked example |
| `outputs-guarded3/` | rules shared, example per-task (composed from the two above) |
| `outputs-guarded-task/` | task suite under the shared rules |

To iterate on guard wording, `serve.sh` plus `ask.sh` is still the fast loop —
but check any improvement against `tests/c1-answerable-fact.txt` and
`tests/c2-owned-tasks.txt` before believing it.
