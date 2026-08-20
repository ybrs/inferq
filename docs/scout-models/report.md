# Scout models: eleven small models on the i7-8700

Evaluation of small CPU models for a scout role beside the main engine —
organizing tasks, summarizing tickets, writing throwaway Python.

**Run `granite-4.1-3b-Q4_K_M`.** The two fastest models fabricate task
assignees. The whole 4B class decodes no faster than the 35B-A3B already
running, so it buys nothing.

Host: Intel i7-8700, 6 cores, AVX2, no VNNI, ~19–21 GB/s effective bandwidth.
llama.cpp `a3b1eff`, Q4 GGUF, 6 threads pinned to cores 0–5, greedy decoding.

## Recommendation

| | Model | Why |
| --- | --- | --- |
| **Run this** | `granite-4.1-3b` Q4_K_M | 23/23 tasks, 23/24 faithfulness, 11.1 t/s, Apache 2.0, 1.95 GiB |
| Speed alternative | `LFM2-2.6B` Q4_K_M | 13.7 t/s, 23/23 tasks, perfect on the extraction trap; blunter on summary attribution |
| Long ticket threads | `granite-4.0-h-micro` Q4_K_M | Ties on quality, hybrid Mamba2 with constant KV memory; 10.6 t/s at short context |
| **Do not use** | `Qwen3-1.7B`, `Qwen3.5-2B` | Fastest here, and both invent a task owner on realistic notes |
| **Do not use** | `LFM2.5-2.6B` | Ignores `enable_thinking:false`; 242s for three tasks, no summary produced |

## Speed

`llama-bench`, pp512 prefill and tg128 decode. File size in GiB.

| Model | File | Prefill t/s | Decode t/s |
| --- | ---: | ---: | ---: |
| Qwen3-1.7B | 1.03 | 151.8 | 20.64 |
| Qwen3.5-2B UD-Q4_K_XL | 1.24 | 91.1 | 14.89 |
| **LFM2-2.6B** | 1.45 | 78.4 | 13.69 |
| LFM2.5-2.6B QAD-Q4_0 | 1.48 | 79.3 | 13.41 |
| SmolLM3-3B | 1.78 | 76.1 | 12.05 |
| **granite-4.1-3b** | 1.95 | 67.6 | 11.08 |
| **granite-4.0-h-micro** | 1.81 | 68.8 | 10.64 |
| *Qwen3.6-35B-A3B — already running* | *—* | *~36* | *8–10* |
| Qwen3-4B-Instruct-2507 | 2.32 | 56.1 | 9.31 |
| gemma-3-4b-it-qat Q4_0 | 2.35 | 64.3 | 9.07 |
| Phi-4-mini | 2.31 | 52.5 | 8.83 |
| Qwen3.5-4B Q3_K_M | 2.13 | 39.2 | 8.82 |

Everything below the baseline row decodes at 8.8–9.3 t/s — the same rate as the
35B-A3B. **The 4B class is a dead end on this host.** A real speedup exists only
at 3B and under, and the honest margin is 1.2–1.7×, not the 3–5× that file size
suggests: the box streams at ~19–21 GB/s, not the 31.8 GB/s a raw memory
benchmark reports.

Qwen3.5-4B at Q3_K_M is *slower* than Qwen3-4B at Q4_K_M despite the smaller
file. Q3 unpacking costs compute an AVX2-only core cannot spare. Don't use Q3
quants here.

## Task suite

JSON action-item extraction (`/8`), grounded ticket summary (`/7`), and a small
Python script that is compiled and executed against a fixture CSV including its
error path (`/8`). `sec` and `tok` cover all three answers.

| Model | t1 | t2 | t3 | Total | sec | tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LFM2-2.6B | 8 | 7 | 8 | **23/23** | 38.7 | 473 |
| SmolLM3-3B | 8 | 7 | 8 | **23/23** | 40.8 | 390 |
| granite-4.1-3b | 8 | 7 | 8 | **23/23** | 49.0 | 451 |
| granite-4.0-h-micro | 8 | 7 | 8 | **23/23** | 53.7 | 470 |
| Qwen3-4B-Instruct-2507 | 8 | 7 | 8 | **23/23** | 59.6 | 453 |
| gemma-3-4b-it-qat | 8 | 7 | 8 | **23/23** | 70.9 | 541 |
| Qwen3.5-4B Q3_K_M | 7 | 7 | 8 | 22/23 | 56.0 | 374 |
| Qwen3-1.7B | 8 | 7 | 6 | 21/23 | 22.5 | 374 |
| Qwen3.5-2B | 8 | 7 | 6 | 21/23 | 35.0 | 439 |
| Phi-4-mini | 8 | 7 | 4 | 19/23 | 51.6 | 400 |
| LFM2.5-2.6B | 7 | 0 | 8 | 15/23 | 242.2 | 3259 |

Six models tie at the top, so **this suite does not discriminate** — it only
rules out the clearly unfit. No model hallucinated on the summary: none invented
a certificate expiry, firewall, or DNS cause, and none contradicted the ticket.

## Faithfulness suite

Three traps: a question whose answer is absent from the ticket (`/6`), a
customer claim contradicted by our own deploy log (`/8`), and standup notes
containing an unowned blocked task, an owner attached to a *different* task, an
item with no date, and a line that looks like a task but is not (`/10`).

| Model | Absent fact | Contradiction | Ambiguous tasks | Total | sec | tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **granite-4.0-h-micro** | 6/6 | 7/8 | **10/10** | **23/24** | 29.1 | 211 |
| **granite-4.1-3b** | 6/6 | 7/8 | **10/10** | **23/24** | 27.6 | 217 |
| **LFM2-2.6B** | 6/6 | 5/8 | **10/10** | 21/24 | 20.9 | 218 |
| Phi-4-mini | 6/6 | 7/8 | 8/10 | 21/24 | 31.2 | 216 |
| Qwen3-4B-Instruct-2507 | 6/6 | 6/8 | 8/10 | 20/24 | 37.6 | 254 |
| SmolLM3-3B | 6/6 | 5/8 | 8/10 | 19/24 | 24.7 | 204 |
| Qwen3-1.7B | 6/6 | 6/8 | **4/10** | 16/24 | 13.2 | 189 |
| Qwen3.5-2B | 6/6 | 6/8 | **4/10** | 16/24 | 17.7 | 181 |

Every model refused the absent-fact question rather than inventing TLS details.
**Grounded Q&A is not where these models fail — extraction under ambiguity is.**

### The case that decides it

Given these standup notes:

> the invoice export is still broken — we can't start on it until legal signs
> off on the data retention question. **Emily said she'd take the migration
> script** once the export is fixed. We should also bump the Postgres version at
> some point. **Kevin is out until the 14th.**

`granite-4.1-3b` — export unowned and correctly blocked, Emily on the migration
script where the notes put her, no task invented for Kevin:

```json
[{"title": "invoice export is still broken", "assignee": null, "due": null,
  "blocked_by": "legal signs off on the data retention question"},
 {"title": "take the migration script", "assignee": "Emily", "due": null, "blocked_by": null},
 {"title": "bump the Postgres version", "assignee": null, "due": null, "blocked_by": null}]
```

`Qwen3-1.7B` — Emily moved onto the export, which was never hers; her actual
task dropped entirely; the blocker promoted to a task of its own, losing the
dependency:

```json
[{"title": "Fix invoice export", "assignee": "Emily", "due": null, "blocked_by": null},
 {"title": "Bump Postgres version", "assignee": null, "due": null, "blocked_by": null},
 {"title": "Wait for legal sign-off", "assignee": null, "due": null, "blocked_by": null}]
```

`Qwen3.5-2B` — same wrong owner, plus a due date pulled from an unrelated
sentence, plus a task conjured from Kevin's absence:

```json
[{"title": "Fix invoice export", "assignee": "Emily", "due": "14th",
  "blocked_by": "Legal data retention question"},
 {"title": "Bump Postgres version", "assignee": null, "due": null, "blocked_by": null},
 {"title": "Kevin availability", "assignee": "Kevin", "due": "14th", "blocked_by": null}]
```

Both failures are the kind a casual test misses: fluent, well-formed JSON with
the wrong person in it. For anything that writes into a tracker, a fabricated
owner costs more than a slow answer.

`Qwen3-4B-Instruct-2507` and `Phi-4-mini` also turned Kevin's absence into a
task (8/10). Only the two Granites and LFM2-2.6B got this clean.

## Thinking mode, not throughput

The first run nearly wrote off Qwen3.5-2B. Asked to extract four action items,
it spent its entire 1400-token budget inside a `<think>` block — visibly
second-guessing itself, *"Wait, check the JSON keys…"* — and emitted no answer.

Same model, same prompt, one parameter:

| | Elapsed | Tokens | Result |
| --- | ---: | ---: | --- |
| Thinking on (default) | 96.7s | 1400 | no answer |
| `enable_thinking: false` | 9.3s | 109 | correct |

So judge these models by **tokens emitted per answer**, not benchmark t/s. A
20 t/s model that deliberates for 800 tokens loses to a 13 t/s model that
answers in 80. This is also what disqualifies `LFM2.5-2.6B`: its chat template
always thinks and ignores the flag. The older non-thinking `LFM2-2.6B` is
unaffected.

## Binary and ternary

Settled, and negative. `bitnet.cpp`'s AVX2 kernels are genuine — its flagship
benchmark CPU is itself AVX2-only — so the earlier Ternary-Bonsai failure was
that PrismML fork's scalar fallback, **not an AVX2 ecosystem limit**. But
BitNet-b1.58-2B-4T scores IFEval 53.5, far below every dense model here. Not
worth revisiting until someone ships ternary with modern post-training.

## Reproducibility and caveats

- Decoding is greedy (`temperature: 0`), so a re-run on the same build
  reproduces these numbers. An earlier pass at `temperature 0.2` moved
  individual scores by several points between runs — enough to reorder the
  middle of the table, though never enough to change the conclusion. Treat
  single-point differences as noise; the fabrication findings reproduced across
  every run.
- Each cell is one sample of one prompt. This suite is a screening tool, not a
  benchmark: it is designed to *disqualify*, and the wide ties at the top mean
  it cannot rank the survivors finely.
- The "does not attribute claims" flag on the Granites in the grader output is a
  regex artifact, not a real weakness — granite-4.1's summary attributes
  properly ("*despite stating that no deployments were made*") and preserves
  "*the root cause has not yet been identified*".
- Not measured: long-context behavior. `granite-4.0-h-micro` has constant KV
  memory and should hold its rate as ticket threads grow while the dense models
  sag. Benchmark both Granites at depth 4k/16k before committing if threads run
  long.
