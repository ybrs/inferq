# Small CPU models as a scout: state of this evaluation

**Status: incomplete. The evaluation did not produce a ranking. Do not use the
numbers in `harness/attic/` — they are invalid, for the reasons below.**

Task: taskq #405 (original, invalid) and #406 (this redo, not finished).

---

## Conclusion

This did not get done. Two attempts, three days, no answer to the question that
was asked — which small model on this host picks the right MCP tool and how fast
does it run.

The first attempt produced a confident report built on a test that did not test
tool calling and graders that were regular expressions. The second attempt
replaced both, and then spent its time on harness scaffolding and pre-flight
checks instead of running models — including a check on CPU hyperthread siblings
that blocked its own sweep twice, and a "does this model work" probe that did a
7,427-token prefill per model and took an hour to get through five of seventeen.
When the sweep finally started it was killed after two of twenty runs.

The work below is real and reusable. The ranking is not there.

---

## What was established

These hold independently of the unfinished sweep. Each was measured, not
inferred; the commands and raw output are in `harness/`.

### 1. The original tool test did not test tool calling

`harness/attic/tests-capability/w1-w3` pasted 46 tool names into a user message
as plain text and asked for "a JSON array of the tool names you would call".
That measures list-copying and JSON formatting.

Sent as a real OpenAI `tools` array instead, through `llama-server --jinja`, the
model that test scored worst answers correctly on the first try:

```
recorded in #405:  ["create_task","claim_task","pr_status","ticket_status"]   "fabricated a tool"
actually happens:  set_task_status(task_id=405, status="done",
                                   comment="benchmark is committed on branch scout-models")
```

`ticket_status` — the fabrication that test built a whole scoring category
around — **cannot occur**. llama.cpp constrains tool-name generation with a
grammar. The headline failure mode did not exist in production.

### 2. Every grader was a regular expression

`attic/grade.py` credits a summary for containing the substring `tls` and
subtracts points if `dns` appears anywhere in the text. It also reads
`/tmp/.claude/jobs/f2433e14/tmp/quality`, a scratch path that no longer exists,
so the committed grader cannot reproduce the committed numbers.

### 3. Flash attention is a 2.7x loss on this CPU at depth

Measured on Qwen3-0.6B with the 46-tool prompt resident (`verify-config-results.txt`):

| config | deep decode |
|---|---:|
| `-fa on` | 5.56 t/s |
| `-fa off` | **14.91 t/s** |
| `-fa on`, q8_0 KV | 5.97 t/s |

llama.cpp's default is `-fa auto`, which turns it **on**. Any benchmark on this
host that did not pass `-fa off` was measuring a 2.7x penalty it never mentioned.

Two other settings were being silently ignored: `--no-mmap` is deprecated in
this build (`-lm none` is the replacement), and llama-server defaults to
`-np 4`, splitting the KV cache across four slots.

### 4. The 46-tool block costs 7,427 tokens, and decode collapses under it

Measured through `/apply-template` and `/tokenize`:

```
system + user message only ..............     89 tokens
+ all 46 tool schemas ................... 7,427 tokens   (+7,338)
```

98.8% of every prompt is the tool catalogue. Consequences for Qwen3-0.6B:

| context | decode | implied bandwidth |
|---|---:|---:|
| ~55 tokens (no tools) | 52.7 t/s | 20.7 GiB/s |
| 7,427 tokens (46 tools) | 14.9 t/s | 17.6 GiB/s |

Both land on the same ~18-21 GB/s memory ceiling. The model is not slower; it
re-reads 809 MiB of KV cache per generated token instead of 6 MiB. Qwen3-0.6B
carries 112 KiB of KV per token — `(128+128) x 8 kv-heads x 28 layers x 2 bytes`.

**Every speed number in the #405 report is a `tg128` figure** — decode at 128
tokens of context. For a scout with tools loaded, they are all roughly 3x
optimistic, uniformly.

### 5. Three models cannot hold the role at all

- `gemma-3-4b-it-qat` — the string `tools` never appears in its chat template.
- `Phi-4-mini-instruct` — its 398-character template reads `tools` only from
  inside a system-message object; the top-level array is never rendered. Any
  tool score for it measures llama.cpp's generic fallback, not the model.
- `Qwen3-1.7B-UD-IQ1_S` and `Qwen3-1.7B-Q2_K` emit no tool call at all.
- `granite-4.1-3b-UD-IQ2_M` prefills at 15 t/s — ~495 s for the tool block,
  past any usable timeout.

`roster.json` originally recorded Phi-4-mini as tool-capable because the string
`tools` occurs in its template. That was a substring false positive in this
harness, found by review before it reached a score.

### 6. Roster corrections

- `SmolLM3-3B-Q4_K_M.gguf` and `SmolLM3-Q4_K_M.gguf` are the same file
  (md5 `e7713ec819a55089d3b0cb4ebdcbbf7e`). #405 counted them as two models.
- `Qwen3.5-0.8B-Q4_K_M` added. In the partial data it is the fastest thing
  measured — 199.5 t/s prefill, 25.4 t/s decode at tool depth, against
  Qwen3-0.6B's 134.2 / 14.5 at half the size.

---

## The one run that completed

`Qwen3-0.6B-Q4_K_M`, thinking off, all 20 scenarios, no errors:
`harness/results/20260821T120245Z-4ce68ac/`.

Cold prefill 135.5 t/s over 7,463 tokens; decode 14.86 t/s at that depth;
shallow decode canary 46.2 t/s.

Tool calls emitted (ungraded — the LLM graders were never run):

| scenario | called |
|---|---|
| s01 status + note | `set_task_status` |
| s02 semantic search | `search_taskq` |
| s03 restraint (explain) | *(none)* |
| s04 "on hold" -> blocked | `release_task` |
| s05 create subtask | `create_task` |
| s06 assign not claim | *(none)* |
| s07 cancel not delete | *(none)* |
| s08 multi-turn follow-up | *(none)* |
| s09 summary not comment | `set_task_summary` |
| s10 project overview | *(none)* |
| s11 link relation | `task_to_issue` |
| s12 restraint (no capability) | *(none)* |
| s13 pick next | *(none)* |
| s14 read not create | `task_to_issue` |

One data point, one model, one mode. It is not a result.

---

## What is here and how to run it

```
harness/
  serverctl.py          process control by PID and process group, pre-flight, step cap
  hostcheck.py          governor, cgroup, cpuset, per-CPU busy, swap, thermals
  gguf_meta.py          GGUF header reader with no numpy dependency
  build_roster.py       roster.json - distinct models, md5, template facts
  fetch_tools.py        pulls the 46 live taskq MCP schemas
  verify_templates.py   per-model checks; writes roster-verified.json
  validate_rubrics.py   checks every rubric claim against the live schemas
  run_eval.py           the capability sweep
  bench_grid.py         llama-bench at 128 / 901 / 7427 depth
  speed_grid.py         llama-server at the same depths, cold and warm
  run_fixtures.py       executes the q3 scripts models wrote
  grade_llm.py          prepare / collect / merge for LLM grading
  report.py             renders the report, refuses inputs it cannot stand behind
  scenarios-tools.json    14 tool scenarios + rubrics
  scenarios-quality.json   6 quality scenarios + rubrics
  calibration.json      19 known-answer probes for the graders
  FAILURE-MODES.md      50 ways this kind of benchmark silently produces wrong numbers
  attic/                #405's graders, prompts and outputs. Invalid, see attic/README.md
```

Order:

```
python3 fetch_tools.py taskq-tools.json
python3 build_roster.py
python3 verify_templates.py            # writes roster-verified.json
python3 validate_rubrics.py            # must exit 0
python3 run_eval.py                    # the sweep; ~2 min/run without thinking
python3 run_fixtures.py results/<run>
python3 grade_llm.py prepare results/<run> --pass 1
#   ... LLM graders fill results/<run>/grading/pass1/grades/
python3 grade_llm.py collect results/<run> --pass 1
python3 grade_llm.py merge results/<run>
python3 report.py results/<run>
```

Settled config, by measurement: `-np 1 -fa off -lm none`, `-t 6`,
`taskset -c 0-5`, temperature 0, seed 42.

### Known problems in this harness

- `verify_templates.py` took ~5 minutes per model until the 7,427-token prefill
  was taken out of it. It is faster now but was never re-run to completion, so
  `roster-verified.json` is currently generated from `roster.json` plus
  established facts rather than from a full verification pass.
- `run_eval.py` creates its timestamped output directory before pre-flight, so a
  blocked run leaves an empty directory behind. There are three such directories.
- The sweep must be launched with `setsid`. Backgrounding it from a shell that
  later exits kills it with the process group.
- Thinking-on runs are roughly 20x slower than thinking-off (a 1,200-token
  budget per scenario at ~15 t/s). Budget for that before enabling it.
- `grade_llm.py` has never been run against real grades. `collect` and `merge`
  are untested against a grader's actual output.
