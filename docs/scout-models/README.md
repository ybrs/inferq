# Scout models

Evaluation of small CPU models for a **scout** role next to the main engine:
organizing tasks, summarizing tickets, and writing throwaway Python — work that
does not justify loading Qwen3.6-35B-A3B.

Full findings: [`report.md`](report.md). Lower quants and smaller models:
[`alternates.md`](alternates.md). How far strict prompting moves the numbers:
[`guards.md`](guards.md).

**Result:** run `granite-4.1-3b-Q4_K_M` — 24/24 on faithfulness with no prompt
guard at all. The two fastest models (`Qwen3-1.7B`, `Qwen3.5-2B`) fabricate task
assignees out of the box; a strict system prompt plus a worked example in the
extraction call takes `Qwen3-1.7B` to 22/24 at twice the decode speed, which
makes it a usable fallback where latency dominates ([`guards.md`](guards.md)).

## What was evaluated

11 GGUF models, 0.6B–4B, Q4 unless noted, on the i7-8700 (6 cores, AVX2, no
VNNI, ~19–21 GB/s effective bandwidth).

| Round | What it measures |
| --- | --- |
| `llama-bench` | pp512 prefill and tg128 decode, 6 threads pinned to cores 0–5 |
| Task suite (`t1`–`t3`, 23 pts) | JSON action-item extraction; grounded ticket summary; a small Python script that is compiled and **executed** against a fixture CSV, including its error path |
| Faithfulness suite (`h1`–`h3`, 24 pts) | A question whose answer is absent from the ticket; a customer claim contradicted by our own deploy log; standup notes with an unowned task, an owner attached to a *different* task, and a line that looks like a task but is not |

The task suite does not discriminate — 10 of 11 models score near-perfect. The
faithfulness suite is the one that separates them.

## Running the benchmarks

Prerequisites: mainline `llama.cpp` built at `/models/llamacpp-main`, GGUFs in
`/models/small-models/`. Both are host-local and not vendored here.

```bash
cd harness

./bench.sh              # llama-bench sweep -> bench-results.md
./run-quality-all.sh    # task suite,        11 models -> outputs/
./run-halluc.sh         # faithfulness suite, 8 models -> outputs/
./run-guarded.sh        # prompt-guard experiment -> outputs-guarded*/

python3 report.py       # scoreboard: speed + task suite
python3 grade_h.py      # scoreboard: faithfulness suite
python3 grade_c.py      # scoreboard: control probes (over-refusal check)
```

The three graders take an outputs directory as their first argument
(`python3 grade_h.py outputs-lowbit`), defaulting to `outputs/`.

Each `quality*.sh` starts a `llama-server`, drives it over
`/v1/chat/completions` so the model's own chat template applies, and writes one
file per prompt to `outputs/`. The graders are deterministic: they parse JSON,
check assignees and nulls against the fixture, and actually run the generated
Python.

Paths are absolute at the top of each script — edit those if your layout differs.

## Two things worth knowing before you trust a number

**Disable thinking.** Reasoning-capable models default to a `<think>` block that
dominates wall-clock. Qwen3.5-2B spent its entire 1400-token budget deliberating
over one extraction and emitted no answer; the same prompt with
`chat_template_kwargs: {"enable_thinking": false}` finished in 9.3s. Judge these
models by **tokens emitted per answer**, not benchmark t/s.

**Effective bandwidth is ~19–21 GB/s**, not the 31.8 GB/s a streaming benchmark
reports, so `32 ÷ file-size-GB` over-predicts decode by roughly 35%. Also avoid
Q3_K on this host: Q3 unpacking costs compute AVX2 cannot spare, and
Qwen3.5-4B Q3_K_M measured *slower* than the larger Qwen3-4B Q4_K_M.

## Serving

```bash
taskset -c 0-5 /models/llamacpp-main/build/bin/llama-server \
  -m /models/small-models/granite-4.1-3b-Q4_K_M.gguf \
  -t 6 -c 8192 --jinja --host 127.0.0.1 --port 8099
```

Clients must send `"chat_template_kwargs": {"enable_thinking": false}` and a hard
`max_tokens`, so a runaway think loop times out instead of hanging.

## Layout

```
report.md              findings, tables, and the raw outputs that decided it
alternates.md          low quants and smaller models
guards.md              prompt guards: what they fix, and what they break
harness/
  bench.sh             llama-bench sweep
  quality.sh           one model, one suite (task or faith)
  run-quality-all.sh   task suite across all models
  run-halluc.sh        faithfulness suite across the leaders
  serve.sh / ask.sh    start a model, send it one prompt — the loop for
                       iterating on prompt guards
  grade.py             task-suite grader
  grade_h.py           faithfulness grader
  report.py            speed + task-suite scoreboard
  tests/               the six suite prompts plus the two control probes
  tests-guarded/       system guards; -v2 adds the extraction example
  outputs/             raw model responses, one file per model per prompt
  outputs-lowbit/      the same for the low-quant builds
  outputs-guarded*/    the prompt-guard runs (see guards.md)
  bench-results.md     raw llama-bench output
```
