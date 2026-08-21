# Scout models: everything we found

One place for all of it. The detail lives in [`report.md`](report.md) (the
eleven-model evaluation), [`alternates.md`](alternates.md) (low quants),
[`guards.md`](guards.md) (prompt guards), and
[`capabilities.md`](capabilities.md) (translation, code, DAGs, diagrams); this
file is the whole picture with the numbers that decided each call.

**Host:** Intel i7-8700, 6 cores, AVX2, no VNNI, ~19–21 GB/s effective
bandwidth. llama.cpp `a3b1eff`, Q4 GGUF unless noted, 6 threads pinned to cores
0–5, greedy decoding, thinking disabled. The engine this sits beside is
Qwen3.6-35B-A3B at ~8–10 t/s.

---

## The decision

| | Model | Why |
| --- | --- | --- |
| **Run this** | `granite-4.1-3b` Q4_K_M | 23/23 tasks, 24/24 faithfulness, 11.1 t/s, Apache 2.0, 1.95 GiB |
| Speed alternative | `LFM2-2.6B` Q4_K_M | 13.7 t/s, 23/23 tasks, 23/24 faithfulness |
| Long ticket threads | `granite-4.0-h-micro` Q4_K_M | Ties granite on quality, hybrid Mamba2 with constant KV memory |
| Latency-bound fallback | `Qwen3-1.7B` **+ prompt guard** | 22/24 at 19 t/s, but only with the guard and only for extraction |
| Verifiable narrow jobs | `Qwen3-0.6B` | 50 t/s. File routing and one-line file summaries only |
| **Don't** | `Qwen3-1.7B` / `Qwen3.5-2B` unguarded | Fabricate task owners |
| **Don't** | `LFM2.5-2.6B` | Ignores `enable_thinking:false`; 242s for three tasks |
| **Don't** | anything 4B | Decodes no faster than the 35B-A3B already running |

---

## 1. Speed

`llama-bench`, pp512 prefill / tg128 decode.

| Model | File GiB | Prefill t/s | Decode t/s |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 0.36 | 356.3 | 49.8 |
| Qwen3-1.7B | 1.03 | 151.8 | 20.6 |
| Qwen3.5-2B UD-Q4_K_XL | 1.24 | 91.1 | 14.9 |
| **LFM2-2.6B** | 1.45 | 78.4 | 13.7 |
| SmolLM3-3B | 1.78 | 76.1 | 12.1 |
| **granite-4.1-3b** | 1.95 | 67.6 | 11.1 |
| **granite-4.0-h-micro** | 1.81 | 68.8 | 10.6 |
| *Qwen3.6-35B-A3B — already running* | *—* | *~36* | *8–10* |
| Qwen3-4B-Instruct-2507 | 2.32 | 56.1 | 9.3 |
| gemma-3-4b-it-qat Q4_0 | 2.35 | 64.3 | 9.1 |
| Phi-4-mini | 2.31 | 52.5 | 8.8 |
| Qwen3.5-4B Q3_K_M | 2.13 | 39.2 | 8.8 |

- **The 4B class is a dead end here.** Everything below the baseline row decodes
  at the same rate as the 35B MoE. Real speedup exists only at ≤3B, and it is
  1.2–1.7×, not the 3–5× file size suggests.
- **Effective bandwidth is ~19–21 GB/s**, not the 31.8 GB/s a streaming
  benchmark reports, so `32 ÷ file-size-GB` over-predicts decode by ~35%.
- **No Q3 quants.** Qwen3.5-4B Q3_K_M is *slower* than the larger Qwen3-4B
  Q4_K_M. Q3 unpacking costs compute an AVX2 core without VNNI cannot spare.
- **Judge by tokens per answer, not t/s.** Thinking mode dominates wall clock:
  Qwen3.5-2B spent a 1400-token budget deliberating over one extraction and
  emitted no answer; the same prompt with `enable_thinking:false` finished in
  9.3s.

## 2. Task suite — does not discriminate

JSON extraction (`/8`), grounded summary (`/7`), a Python script that is compiled
and executed against a fixture including its error path (`/8`).

Six models tie at **23/23**: LFM2-2.6B, SmolLM3-3B, granite-4.1-3b,
granite-4.0-h-micro, Qwen3-4B-Instruct-2507, gemma-3-4b-it-qat. Qwen3-1.7B and
Qwen3.5-2B score 21/23, Phi-4-mini 19/23 (script), Qwen3-0.6B 14/23. Nothing
here separates the leaders — it only rules out the unfit.

## 3. Faithfulness suite — this is the one that decides

An absent fact (`/6`), a customer claim contradicted by our deploy log (`/8`),
and standup notes with an unowned blocked task, an owner attached to a
*different* task, and a line that looks like a task but is not (`/10`).

| Model | Absent fact | Contradiction | Ambiguous tasks | Total |
| --- | ---: | ---: | ---: | ---: |
| **granite-4.0-h-micro** | 6/6 | 8/8 | **10/10** | **24/24** |
| **granite-4.1-3b** | 6/6 | 8/8 | **10/10** | **24/24** |
| LFM2-2.6B | 6/6 | 7/8 | **10/10** | 23/24 |
| Phi-4-mini | 6/6 | 7/8 | 8/10 | 21/24 |
| Qwen3-4B-Instruct-2507 | 6/6 | 6/8 | 8/10 | 20/24 |
| SmolLM3-3B | 6/6 | 6/8 | 8/10 | 20/24 |
| Qwen3-1.7B | 6/6 | 6/8 | **4/10** | 16/24 |
| Qwen3.5-2B | 6/6 | 6/8 | **4/10** | 16/24 |

Every model refused the absent-fact question rather than inventing TLS details.
**Grounded Q&A is not where these models fail — extraction under ambiguity is.**
Given standup notes where Emily owns the migration script and the invoice export
is blocked on legal, the two fastest models put **Emily on the export**.
Qwen3.5-2B also invented a whole task and emitted the *string* `"null"`.

## 4. Low quants — don't go below Q4

| Build | Size | Prefill | Decode | Faithfulness |
| --- | ---: | ---: | ---: | ---: |
| granite-4.1-3b Q4_K_M | 1.95 | 60.9 | 9.78 | **24/24** |
| granite-4.1-3b Q2_K_XL | 1.31 | 30.6 | **12.98** | 15/24 |
| granite-4.1-3b IQ2_M | 1.19 | 15.1 | 10.91 | 20/24 |
| Qwen3-1.7B Q2_K | 0.72 | 97.0 | **25.43** | 17/24 |
| Qwen3-1.7B IQ1_S | 0.50 | 40.4 | 23.32 | 3/24 |

- **K-quants scale, I-quants don't.** `IQ1_S` is 31% smaller than `Q2_K` and
  decodes *slower*. Codebook lookup costs more compute than the bandwidth it
  saves — 12–14 GB/s effective against Q4's ~20.
- **Prefill collapses**: granite at IQ2_M prefills 4× slower than at Q4.
  Summarization is prefill-heavy, so low-bit optimizes the wrong half.
- **The quant that would actually buy speed breaks the reason we chose granite.**
  Q2_K_XL assigned the invoice export to "legal" — the blocker became the owner.
- Sub-2-bit at 1.7B is not degraded, it is broken: IQ1_S repeated
  `"The 10, the 14th."` to the token cap, 219s, invalid JSON.

## 5. Prompt guards — how much of this is promptable?

One shared system prompt, user prompts byte-identical to the unguarded suite,
plus two **control** probes whose answers *are* present, so a guard that buys its
score by refusing loses points instead of hiding.

| Model | t/s | No guard | Shared rules | + example in shared prompt | Example per-task |
| --- | ---: | ---: | ---: | ---: | ---: |
| granite-4.1-3b Q4_K_M | 9.8 | **24** | 24 | 24 | 24 |
| granite-4.1-3b Q2_K_XL | 13.0 | 15 | **21** | 19 | 19 |
| Qwen3.5-2B | 14.9 | 16 | 13 | 17 | 19 |
| Qwen3-1.7B | 19.1 | 16 | 16 | 19 | **22** |
| Qwen3-0.6B | 49.8 | 6 | 15 | 17 | **23** |

Controls (`/10`) stay at 10 for everything except the "+ example in shared
prompt" column, where Qwen3-1.7B drops to 7 and the 0.6B to 5.

- **General grounding rules work, and the anti-refusal clause is what makes them
  work.** "If the message does state the answer, answer it — refusing an
  answerable question is as wrong as inventing one." Without it, a guard just
  installs refusal.
- **Never put a worked example in a shared system prompt at ≤2B.** It becomes a
  format template: Qwen3-1.7B answered a plain question in extraction JSON, and
  Qwen3-0.6B **summarized the example instead of the ticket** — reporting on a
  nightly report, Ben and Node, none of which were in the input. The guard became
  the hallucination source. Put the example in the call that needs it.
- **Guards cost prefill, not decode.** ~180 → ~540 tokens for the rules, ~860
  with an example: +5.2s and +10.2s per granite call before the first token.

## 6. What else can they do?

| Job | Qwen3-0.6B | Qwen3-1.7B | granite-4.1-3b |
| --- | --- | --- | --- |
| Dutch → English | fluent and **wrong** | mostly right | reliable |
| "What is this file about?" | works | works | works |
| "Which files should I open?" | right, sloppy list | **invents filenames** | right, under-fills |
| Workflow → JSON DAG | **serialises parallel branches** | invalid JSON | correct |
| Working Python script | no | no | yes |
| Mermaid diagram | degenerates | no edges at all | valid, wrong topology |
| matplotlib chart | renders a **misleading** chart | crashes | crashes |
| Picking the right MCP tool | **invents tool names** | correct | correct |

- **Translation:** the 0.6B turned "could you get back to us **today**" into
  "**tomorrow**" and "retrieving shipments" into "processing incoming messages",
  while preserving the delivery ID — the parts a human spot-checks look right.
  On false friends it wrote *magazijnmeester* → "magazine editor" and inverted
  "that saves a lot of hassle" into "that's a hope gone". granite got all of them.
- **Code reading works**, including at 0.6B, on a Rust file with no docstring.
  This is the job the 0.6B is actually for.
- **Small models serialise parallelism.** Given two independent pulls feeding one
  validation feeding two independent loads, the 0.6B emitted valid JSON that
  chained the loads and left a branch unconnected. granite got all seven edges
  right in JSON — and lost the same parallelism when asked for the same graph as
  Mermaid, inventing error branches nobody described. **Generate DAGs as JSON and
  render diagrams from them.**
- **Nobody writes a working chart one-shot.** granite and the 1.7B crash on
  typos, which a repair loop would fix. The 0.6B's *runs* — and draws overlapping
  bars that read as a stacked chart. A chart that renders and misleads is worse
  than one that crashes.

## 7. Tool selection

46 taskq MCP tools with one-line descriptions, three requests, answer as a JSON
array of tool names. The listing is full of near-misses on purpose:
`set_task_status` / `set_task_summary` / `update_task`, `search_taskq` /
`list_tasks` / `task_queue`.

| Model | update a ticket | search for one | create one | Total |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B Q4_K_M | 6/6 | 6/6 | 5/6 | **17/18** |
| granite-4.1-3b Q4_K_M | 5/6 | 6/6 | 6/6 | **17/18** |
| Qwen3-0.6B Q4_K_M | 1/6 | 2/6 | 3/6 | 6/18 |

**The 0.6B fabricated a tool name in all three probes** — `ticket_status`,
`get_taskq`, `request_task` — every one a plausible blend of real names in the
listing. Worse, asked to *mark ticket 405 done*, it chose `create_task` and
`claim_task`: the request to update a ticket would have created a new one.

The two larger models both found the right tool every time. Their shared failure
is different and more interesting: **both bolt on an unrequested side effect.**
granite followed `set_task_status` with `task_to_issue` on the update request,
and the 1.7B followed `create_task` with `task_to_issue` on the create request —
neither request mentioned GitHub, and `task_to_issue` opens a real issue.

That is the distinction worth building around:

- a **fabricated name fails closed** — the server rejects it, the agent sees an
  error and can retry;
- an **extra real call fails open** — it runs, and something exists in GitHub
  that nobody asked for.

So a small model driving tools needs an allowlist per request type more than it
needs a bigger listing. And the 0.6B does not belong anywhere near a tool loop
that writes.

## 8. Rules that generalise

1. **Route by verifiability, not difficulty.** The 0.6B is safe where the caller
   can check the answer mechanically — file routing against a real listing, JSON
   against a schema. It is unsafe wherever the output is fluent prose whose
   errors have no validator: translation, summaries feeding decisions, tool
   arguments.
2. **Small models under-segment.** They do not count the units of work in the
   input. Three tasks in, two out; two parallel branches in, one chain out; a
   compound request in, an answer about its first clause out. Split compound
   inputs before handing them over.
3. **Prompt guards are worth about one model size — in one direction.** They fix
   grounding, refusal calibration and extraction shape. They do not fix
   segmentation, and they cannot be shared across task types below 3B.
4. **Structured output is where small models are accurate.** The same model that
   nails a JSON DAG will lose its topology in Mermaid and lose the meaning
   entirely in prose.
5. **Always run controls.** Any faithfulness score can be bought with refusal.
   Any capability probe can be passed by ignoring the interface. Both showed up
   here, and only the control probes caught them.

## 9. Eval hygiene learned the hard way

- **Run at `temperature 0`.** At 0.2, individual scores moved several points
  between runs — enough to reorder the middle of a table.
- **Lexical graders under-count.** The `h2` checker accepted "customer reports"
  but not "the customer, Bolt Industries, reported", and "root cause not
  identified" but not "still unidentified" — worth about a point per row across
  every table. Widening it then over-counted: it matched the ticket's own header
  `(customer: Bolt Industries) states` and credited attribution the model had not
  done. Both fixed; check a grader's misses *and* its new hits.
- **Sandbox generated code.** The 0.6B's chart script hardcodes `output.png`, and
  the first grading run wrote a PNG into the repo.
- **Keep the raw outputs.** The first pass at prompt guards was hand-run and
  discarded its answers, so none of it could be re-scored when the grader
  changed.

## 10. Open

- `granite-4.0-h-micro` vs `granite-4.1-3b` on **long** ticket threads — the
  constant-KV claim of hybrid Mamba2 is untested past short context.
- **Routing accuracy under concurrency.** Everything above was measured idle. The
  0.6B's 50 t/s and granite's 11 t/s both come out of the same ~20 GB/s the
  engine is already using; a scout that runs while the engine decodes taxes the
  thing it is meant to protect.
- Wiring the chosen model into the scout role, with the per-task guards from
  §5 and the tool allowlist from §7.
