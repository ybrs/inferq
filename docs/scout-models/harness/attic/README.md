# Invalidated — do not use

Everything in this directory is from the first scout evaluation (taskq #405) and
is kept only so the claims in that report can be traced back to what produced
them. None of it may be run against new results.

Two independent reasons it is invalid:

**The tool test did not test tool calling.** `tests-capability/w1-w3` pasted 46
tool names into a user message as plain text and asked for a JSON array of
names. That measures list-copying and JSON formatting. Sent as a real `tools`
array instead, Qwen3-0.6B — scored here as fabricating a tool name — emits a
correct `set_task_status(task_id=405, status="done", comment=...)` on the first
try. The fabricated-name failure mode the grader was built around cannot occur
at all: llama.cpp constrains tool names with a grammar.

**Every grader is a regex.** `grade.py` credits a summary for containing the
substring `tls` and penalises `dns` appearing anywhere. `grade.py` also reads
`/tmp/.claude/jobs/f2433e14/tmp/quality`, a scratch path that no longer exists,
so the committed grader cannot reproduce the committed numbers.

The replacement is one entry point, `../grade_llm.py`, which grades with an LLM
against the written rubrics in `../scenarios-*.json` and stamps
`grading_method: llm-rubric-v2` into its output. `../report.py` refuses grade
files without that stamp.
