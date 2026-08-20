# Alternates: lower quants and smaller models

Follow-up to [`report.md`](report.md), answering "what if we push to Q1 quants or
smaller models?"

**Short answer: don't go below Q4 on this host.** The best case buys ~3 t/s and
costs the property that made granite the pick. I-quants (`IQ*`) are actively
counterproductive on AVX2. Prompt guards recover most of the damage at 3B, and
none of it at 0.6B.

## Measured speed

One session, so these are comparable to each other but not to `report.md`
(background load drifts a few percent between sessions).

| Model | Quant | Size | Prefill t/s | Decode t/s | Effective GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| granite-4.1-3b | Q4_K_M | 1.95 GiB | 60.9 | 9.78 | 20.5 |
| granite-4.1-3b | Q2_K_XL | 1.31 | 30.6 | **12.98** | 18.3 |
| granite-4.1-3b | IQ2_M | 1.19 | 15.1 | 10.91 | 13.9 |
| Qwen3-1.7B | Q4_K_M | 1.03 | 121.5 | 19.05 | 21.1 |
| Qwen3-1.7B | Q2_K | 0.72 | 97.0 | **25.43** | 19.6 |
| Qwen3-1.7B | IQ1_S | 0.50 | 40.4 | 23.32 | 12.4 |
| Qwen3-0.6B | Q4_K_M | 0.36 | 356.3 | **49.77** | 19.4 |

**K-quants scale, I-quants don't.** Q2_K converts its size reduction into decode
at roughly the expected rate, holding ~19–20 GB/s effective. The I-quants break
the pattern: `IQ1_S` is 31% *smaller* than `Q2_K` and decodes *slower* (23.3 vs
25.4 t/s). Their codebook lookup costs more compute than the bandwidth it saves,
dropping effective throughput to 12–14 GB/s. An AVX2 core without VNNI has no
spare compute to trade — the same effect that made Qwen3.5-4B `Q3_K_M` slower
than a larger `Q4_K_M` file in the main report.

**Prefill collapses.** granite at `IQ2_M` prefills at 15 t/s against 61 at Q4 —
4× slower. Ticket summarization is prefill-heavy (long thread in, short summary
out), so low-bit quants optimize the half of the workload that matters least.

## Measured quality

Faithfulness suite (`/24`, see `report.md` for what h1–h3 test):

| Build | h1 | h2 | h3 | Total |
| --- | ---: | ---: | ---: | ---: |
| granite-4.1-3b Q4_K_M *(reference)* | 6/6 | 8/8 | 10/10 | **24/24** |
| granite-4.1-3b IQ2_M | 6/6 | 6/8 | 8/10 | 20/24 |
| Qwen3-1.7B Q2_K | 6/6 | 5/8 | 6/10 | 17/24 |
| granite-4.1-3b Q2_K_XL | 6/6 | 5/8 | 4/10 | 15/24 |
| Qwen3-0.6B Q4_K_M | 0/6 | 5/8 | 2/10 | 7/24 |
| Qwen3-1.7B IQ1_S | 2/6 | 1/8 | 0/10 | 3/24 |

<sub>Scored with the widened h2 patterns described in
[`guards.md`](guards.md) — one point higher per row than first published, same
outputs, same order.</sub>

The failure modes matter more than the totals:

- **granite Q2_K_XL** — the option that would actually buy speed — assigned the
  invoice export to **"legal"**, promoting the blocker to the owner, and turned
  Kevin's absence into a task. The model chosen precisely because it does not
  fabricate owners starts fabricating them.
- **Qwen3-1.7B IQ1_S** degenerated: `"The 10, the 14th. The 10, the 14th."`
  repeated to the 1400-token cap — 219s and invalid JSON. Sub-2-bit at 1.7B is
  not degraded, it is broken.
- **Qwen3-0.6B** invented `TLS 1.2 and TLS 1.3` on the absent-fact question —
  the one probe every model in the main report passed.

## Can prompt guards fix it?

Partly, and only above a capacity threshold. A first pass here found that guards
recover most of the quantization damage at 3B and none of it at 0.6B, where they
only converted over-claiming into blanket refusal.

That result was re-run properly — one shared system prompt, unchanged user
prompts, saved outputs, and control probes whose answers *are* present — and it
half survives. The refusal trade is avoidable: a guard that also says "if the
answer is present, answer it" lifts Q2_K_XL from 15/24 to **21/24** with no
control loss, and lifts the 0.6B from 7/24 to 16/24. What does not change is the
conclusion for this file: guarded, the low quant still turns "Kevin is out until
the 14th" into a task, and it is the build most easily confused by a richer
prompt — adding a worked example *lowers* its score to 19/24 and leaks the
refusal phrase into JSON as `"due": "NOT IN TICKET"`.

Full experiment, including where guards backfire at ≤2B: [`guards.md`](guards.md).

## Reproducing the failures

`serve.sh` starts a model and leaves it up; `ask.sh` posts one prompt file to it.
Editing a prompt and re-running takes seconds on a 0.6B.

```bash
cd harness
./serve.sh Qwen3-0.6B-Q4_K_M.gguf     # ~1s to load

./ask.sh tests/h1-absent-fact.txt      # invents "TLS 1.2 and TLS 1.3"
./ask.sh tests/h3-ambiguous-tasks.txt  # puts Emily on the export, Kevin on Postgres

cp tests/h1-absent-fact.txt /tmp/guarded.txt
$EDITOR /tmp/guarded.txt               # add your guard
./ask.sh /tmp/guarded.txt

./serve.sh --stop
```

To check a guard did not simply install a blanket refusal, run it against a
question the ticket *does* answer:

```bash
sed 's|Question: Which TLS version.*|Question: What is the sample delivery ID the customer provided?|' \
  tests/h1-absent-fact.txt > /tmp/control.txt
./ask.sh /tmp/control.txt              # must still answer dlv_9f3k2
```

That control is the one that exposed the 0.6B few-shot result as fake. Any guard
that improves a refusal score should be checked against it.

To score a guarded output the same way the suite does, drop it in `outputs/`
under a model name and run `python3 grade_h.py`.
