# FAILURE-MODES — 50 ways this harness can silently produce a wrong table

Scope: `/workspace/docs/scout-models/harness/` — `run_eval.py`, `build_roster.py`, `gguf_meta.py`,
`fetch_tools.py`, the two scenario files, `roster.json`, `taskq-tools.json`, and the planned LLM
grading step. The six already-known defects (-np, -fa, free_port, idle gate, mmap warmup, drift
check) are assumed fixed; items below are what remains, including failures the fixes themselves
can introduce. Findings marked **VERIFIED** were confirmed on this box on 2026-08-21.

---

## A. Host & environment

### 1. CPU governor left on powersave
- **What happens:** **VERIFIED:** `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` is `powersave` right now. Under intel_pstate this usually still turbos, but if it (or `no_turbo`, or a clamped `scaling_max_freq`) changes between runs or mid-run, every t/s number shifts 10-40% while remaining plausible.
- **How it shows up:** SILENT — decode rates are internally consistent, just scaled by an unmeasured factor.
- **Guard:** Pre-flight in `run_eval.py` `main()` (new `preflight()` function): read and record `scaling_governor`, `scaling_min/max_freq`, and `/sys/devices/system/cpu/intel_pstate/no_turbo` for cpu0-5 into the run manifest; assert all six governors are identical and `no_turbo == 0`, abort otherwise. Per-model check: re-read at each `run_one()` start, assert unchanged from the manifest.

### 2. Docker cgroup CPU quota or cpuset shrinking the six cores
- **What happens:** The container's cgroup can carry a CPU quota (`cpu.max` like `60000 100000`) or a cpuset excluding cores 0-5. `taskset -c 0-5` succeeds anyway; the kernel throttles the server invisibly. **VERIFIED today:** `cpu.max` is `max 100000` and cpuset is `0-11` — fine now, but nothing pins it.
- **How it shows up:** SILENT — throttled t/s look like slow models.
- **Guard:** Pre-flight in `preflight()`: read `/sys/fs/cgroup/cpu.max` and assert the quota field is literally `max`; read `/sys/fs/cgroup/cpuset.cpus.effective` and assert it contains 0-5. Record both in the manifest. Abort on failure.

### 3. Host load on HT siblings 6-11
- **What happens:** CPUs 6-11 are the hyperthread siblings of 0-5. A host-side or other-container process pinned nowhere lands there and steals execution ports and L1/L2 from the very cores the eval uses, without ever showing load on 0-5.
- **How it shows up:** SILENT — 10-25% decode degradation for whichever models ran during the contention.
- **Guard:** Pre-flight and per-model: sample `/proc/stat` twice 2s apart in `preflight()` and at each `run_one()` start; compute busy% per CPU; assert busy% on CPUs 6-11 < 5% (and on 0-5 < 5% before server start). Abort pre-flight, or mark the model run `"host_contended": true` in `_run.json` if it appears mid-sweep.

### 4. Thermal / turbo drift across the 3 hours
- **What happens:** Sustained 6-core AVX2 load heats the package; effective turbo ratio decays over hours (and with ambient temperature). Model 1 is measured on a cold chip, model 17 on a hot one.
- **How it shows up:** SILENT — a systematic speed bias correlated with run order, i.e. with model size, since the roster is size-sorted.
- **Guard:** Canary: in `main()`, after the last run, re-run the *first* model's cold prefill plus one fixed decode scenario and compare t/s to its original numbers; assert within 10%, else stamp `"drift_exceeded"` in the manifest and flag every speed column. Per-model: record `/proc/cpuinfo` MHz mean over cpus 0-5 and the coretemp reading into each `_run.json`.

### 5. Memory pressure and the already-full swap
- **What happens:** **VERIFIED:** swap is 3/3 GB used. If anything pushes the box into reclaim mid-run (another container ballooning), kswapd competes on the eval cores and the server's pages can be evicted between requests.
- **How it shows up:** SILENT — sporadic slow scenarios attributed to the model.
- **Guard:** Pre-flight: assert `MemAvailable` > 3x the largest model size + 4 GB. Per-model: record `/proc/vmstat` `pswpin`/`pswpout` at `run_one()` start and end; assert delta == 0, else mark the run's timings tainted in `_run.json`.

### 6. Model load time and disk contention bleeding into measurements
- **What happens:** With `--no-mmap` the 0.4-2.5 GB read happens at startup. If load duration is not recorded separately, or another process is hammering the disk, the health-wait and first request absorb it and nobody can later prove timings were clean.
- **How it shows up:** SILENT if load creep contaminates anything timed; the number that suffers is the cold prefill.
- **Guard:** In `run_one()`: record `load_seconds` (time from `Popen` to `/health` ok) into `_run.json` as its own field, never summed into any t/s. Pre-flight: one `iostat -x 1 2`-equivalent sample from `/proc/diskstats`; assert disk utilization < 10%.

### 7. Wall clock instead of monotonic clock
- **What happens:** `post()` and `main()` use `time.time()`. An NTP step or slew during a 3-hour run shifts elapsed_s for whichever scenarios straddle it.
- **How it shows up:** SILENT — a few elapsed values off by the step size, unexplainable later.
- **Guard:** In `run_eval.py`, replace every `time.time()` used for durations (`post()`, `wait_health()`, `main()`'s `t0`) with `time.monotonic()`. Keep one `time.time()` only as a timestamp in the manifest. One-line per-request property: elapsed must be >= server-reported total ms.

## B. Serving configuration

### 8. The new warmup request destroys the cold-prefill measurement
- **What happens:** Defect fix #5 adds a discarded warmup request. If that warmup uses the same system+tools messages, the "cold" prefill request then hits the prefix cache: `prompt_n` drops from ~7000 to ~5 and `prompt_per_second` becomes garbage (or absurdly high). The headline prefill column — a key scout metric — is then wrong for every model.
- **How it shows up:** SILENT — a t/s number is still printed and recorded; it just measures nothing.
- **Guard:** Per-model check in `run_one()`: the warmup request must use a short, disjoint prompt (no `tools`, different system string). Then assert on the cold prefill response: `timings["prompt_n"] >= 5000` and (if present) `cache_n == 0` / `usage["prompt_tokens"] >= 5000`. If the assert fails, restart the server and redo the cold prefill; if it fails twice, abort the run.

### 9. Server flags silently not applied
- **What happens:** llama-server ignores or renames flags across versions (`-fa` semantics have changed; `--no-mmap` vs `--mlock`; `-np`). A flag that stops being recognized, or is overridden by an env var, changes slots/FA/ctx without any request failing.
- **How it shows up:** SILENT — this is exactly how the 5.5-vs-43.6 t/s defect happened; the fix must be verified, not assumed.
- **Guard:** Per-model check: after `wait_health()`, GET `/props`; assert `total_slots == 1`, `default_generation_settings.n_ctx == 16384`, and that flash-attention and no-mmap show as enabled (field names per this llama.cpp commit — pin them once by hand). Record the full `/props` JSON into `_run.json`. Abort the run on mismatch.

### 10. Sampler settings leaking from server defaults
- **What happens:** The request sets only `temperature: 0`. `top_k`, `top_p`, `min_p`, `repeat_penalty`, and `seed` come from server defaults, which differ across llama.cpp versions; a nonzero repeat penalty measurably changes greedy output on long JSON/tool sequences.
- **How it shows up:** SILENT — outputs differ from what the same model would produce elsewhere, and from a rerun on a newer binary.
- **Guard:** Per-request: in `post()`'s payload construction (both tool and quality bodies in `run_one()`), explicitly set `"top_k": 1, "seed": 42, "repeat_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0`. Per-model: read `default_generation_settings` from `/props` into `_run.json` so the effective sampler chain is auditable.

### 11. Server dies mid-run and the rest of the scenarios become fake zeros
- **What happens:** If llama-server crashes (OOM, assert, NaN) at scenario 5, every subsequent `post()` returns a connection error. `run_one()` happily records 15 error entries and writes a normal-looking `_run.json`; the grader later scores 15 "no call" zeros.
- **How it shows up:** SILENT at the table level — the model looks like it failed 15 scenarios rather than the harness losing the server.
- **Guard:** Per-request check in `run_one()`: after any `err` from `post()`, call `proc.poll()`; if the process is dead, set `meta["invalid"] = "server_died_at_" + sc["id"]`, stop the loop, and record the remaining scenario ids under `meta["not_run"]`. `report.py`/grading must render these as NO-DATA, never 0.

### 12. Context set beyond the model's trained context
- **What happens:** `-c 16384` is applied to all models. If any GGUF has `n_ctx_train < 16384`, llama-server extends RoPE with only a log warning; with a ~7k-token prompt the model may already be in degraded territory, producing worse tool calls than the model is capable of.
- **How it shows up:** SILENT — quality quietly worse for the affected models only.
- **Guard:** Pre-flight: extend `gguf_meta.probe()` to also read `{arch}.context_length`; add it to `roster.json` via `build_roster.py`; in `run_eval.py` `preflight()`, assert `context_length >= int(CTX)` for every model, else use `min(context_length, CTX)` for that model and record the effective ctx in `_run.json`.

### 13. Grammar-constrained tool decoding masking real failures
- **What happens:** With `--jinja` + `tools`, llama.cpp constrains tool-call output with a grammar built from the schema. Required keys are then *forced*: s05's "PARTIAL: project omitted, which the server rejects" is unreachable, because the grammar makes the model emit `project` even when it had no idea it was required. Conversely `status` and `relation` have no `enum` in taskq's schemas (verified: plain strings), so invented values still get through — s04/s11 still discriminate, but only by accident of the schema.
- **How it shows up:** SILENT — capability differences are compressed; models get credit the grammar earned.
- **Guard:** Pre-flight: a `validate_rubrics.py` step that loads `taskq-tools.json` and asserts, per scenario, which rubric branches are reachable under grammar (required keys always present; no-enum strings unconstrained); annotate the rubric or grading prompt accordingly. Per-model: grep the server log for the chat-format/grammar line and record it in `_run.json` so it is known whether constrained decoding was active.

### 14. Truncation at max_tokens graded as a wrong answer
- **What happens:** think-off tool requests get `max_tokens: 700`. A model whose thinking switch is a no-op (or that reasons in `content`) burns 700 tokens of deliberation and is cut off before emitting a call. `run_one()` prints `(no call)`; the grader scores ZERO for "no call at all" when the truthful label is "did not fit the budget".
- **How it shows up:** SILENT — a plausible zero with a specific wrong cause; it also poisons the think-off vs think-on comparison.
- **Guard:** Per-request: `finish_reason` is already recorded — act on it: in `run_one()`, when `finish_reason == "length"` and `tool_calls` is empty, set `out["truncated"] = true` and print `TRUNCATED` instead of `(no call)`. Grading input must carry `finish_reason` and the instruction: score truncated answers as a distinct `TRUNCATED` category, reported separately from wrong answers.

## C. Model & chat template

### 15. `template_has_tools` is a substring lie — Phi-4-mini is a verified false positive
- **What happens:** **VERIFIED:** Phi-4-mini's 398-char template contains `'tools' in message` — it only renders a `tools` key *on a message object*, which the OpenAI-style request never populates. The top-level `tools` array is never rendered by this template; roster says `template_has_tools: true` anyway. Same class of risk for any template where the substring appears in prose.
- **How it shows up:** SILENT — a model recorded as tool-capable is actually being served by llama.cpp's generic fallback (see 17), and its score is attributed to the wrong mechanism.
- **Guard:** Pre-flight: replace the substring probe. In `build_roster.py` (or a new `verify_templates.py`), start llama-server per model and POST `/apply-template` with one message plus a one-tool `tools` array; set `template_has_tools = (tool name appears in the rendered prompt)`. Rebuild `roster.json` from that. Assert Phi-4-mini and LFM2-2.6B flip or are explicitly annotated.

### 16. A no-op thinking kwarg creating two identical runs labeled as different modes
- **What happens:** `thinking_kwarg` is also a substring guess. If the key is wrong for a template (the exact failure that fooled the previous run about LFM2.5), `chat_template_kwargs` is silently ignored and "think-on" and "think-off" are the *same* configuration — but the table shows two rows and the think-off row gets the 700-token budget (see 14), so the two rows differ for the wrong reason.
- **How it shows up:** SILENT — a fabricated on/off comparison.
- **Guard:** Pre-flight per model with a kwarg: POST `/apply-template` twice, once with `{kwarg: true}` and once with `{kwarg: false}`, and assert the rendered prompts differ. If they do not, set `thinking_kwarg = null` in the roster and run the model once in `default` mode. Implement in the new `verify_templates.py`; `run_eval.py` refuses to start if `roster.json` predates the templates' verification stamp.

### 17. Tools sent to models whose template cannot render them — fallback measured as capability
- **What happens:** `run_one()` never consults `template_has_tools`. gemma-3 (verified: no tools in template), Phi-4-mini and LFM2 (no `tool_call` parsing) get the 46 schemas anyway; llama.cpp falls back to its generic handler, which injects the schemas as text and parses a house JSON format the model was never trained on. The resulting score measures llama.cpp's fallback prompt, not the model, and lands in the same column as native-tool models.
- **How it shows up:** SILENT — plausible low-to-middling scores with the wrong meaning; the docstring in `gguf_meta.py` even states the correct policy ("cannot hold the role") and the runner ignores it.
- **Guard:** Per-model in `run_one()`: if `not model["template_has_tools"]` (post-fix per item 15), skip tool scenarios and write `meta["tool_suite"] = "N/A: template has no tool support"`. For the fallback-handled middle cases, parse the server log's `Chat format:` line into `_run.json["chat_format"]` and make the report show it next to every tool score.

### 18. s08's multi-turn shape breaks or degrades in several templates
- **What happens:** s08 sends an assistant message with `tool_calls` and `content: ""`, then a `role: "tool"` message. **VERIFIED:** gemma-3's template calls `raise_exception` unless roles strictly alternate user/assistant — the request 500s. LFM2's and Phi-4-mini's templates have no `tool_call` rendering, so the assistant's search call is dropped or mangled and the model never sees what it supposedly did. The scenario then measures template round-tripping, not "acting on a tool result".
- **How it shows up:** Mixed — gemma is LOUD per-request but SILENT in the table (an error graded as zero); LFM2/Phi are fully SILENT.
- **Guard:** Pre-flight per model: render *every* scenario's full message list via `/apply-template`; assert no HTTP error and that the literal strings `388` and `search_taskq` from the tool result survive into the rendered prompt. On failure, record `s08: "N/A: template cannot represent tool turns"` in `_run.json` and exclude it from that model's denominator (report n/13 with a footnote, never a zero).

### 19. Reasoning not separated from content
- **What happens:** If llama.cpp's reasoning parser doesn't recognize a template's think markers (or `--reasoning-format` defaults change), the chain-of-thought lands in `content`. Quality rubrics then dock "prose wrapping the JSON" (q1: -1, -2 for unparseable), and tool answers look like rambling with no call.
- **How it shows up:** SILENT — penalties applied for a plumbing artifact.
- **Guard:** Pass `--reasoning-format auto` explicitly in `start_server()` and record it. Per-request check in `run_one()`: if `reasoning_content == ""` and `content` matches `^\s*<think|<\|thinking\|>|<seed:think>`, set `out["reasoning_leak"] = true`; pre-flight one think-on smoke request per thinking model and abort if the leak fires there.

### 20. Double BOS from template plus tokenizer
- **What happens:** LFM2 and gemma-3 templates emit `{{ bos_token }}` themselves; if the server also prepends BOS on tokenization, small models get a degraded, out-of-distribution prompt start. llama.cpp only logs a warning.
- **How it shows up:** SILENT — a per-model quality haircut nobody can see in the outputs.
- **Guard:** Per-model pre-flight: after warmup, grep the server log for the known `duplicate leading token` / double-BOS warning string and assert absent; alternatively POST `/apply-template` then `/tokenize` and assert the BOS id appears at most once in the first 3 tokens. Record the verdict in `_run.json`.

### 21. Model files drifting from roster.json between hashing and running
- **What happens:** `build_roster.py` computed md5s at some earlier time. If a GGUF was re-downloaded, replaced, or a new quant added since, `run_eval.py` serves whatever bytes are on disk under the roster's name; results are attributed to the wrong artifact. (Corollary: the previous SmolLM3 duplicate fiasco.)
- **How it shows up:** SILENT — right filename, wrong weights.
- **Guard:** Per-model in `run_one()`: before `start_server()`, re-compute the file's md5 (reuse `build_roster.md5`) and assert it equals `model["md5"]`; write the verified hash into `_run.json`. Pre-flight: also assert no `*.gguf` exists in MODEL_DIR that is absent from the roster, so a new model can't be silently skipped.

## D. Measurement

### 22. Cached-prefix prefill rates treated as prefill numbers
- **What happens:** After the cold request, every tool scenario reuses the ~7k-token prefix; its `prompt_per_second` covers only the ~30 new suffix tokens and is dominated by fixed overhead. Any aggregation that averages `timings.prompt_per_second` across scenarios (the obvious spreadsheet move) yields a nonsense prefill column.
- **How it shows up:** SILENT — a numeric column that means nothing.
- **Guard:** In `report.py` (grading/aggregation): assert prefill t/s is taken *only* from `cold_prefill` and only when its `prompt_n >= 5000` (item 8's check). Store `prompt_n` next to every rate; any code that reads `prompt_per_second` where `prompt_n < 1000` must raise.

### 23. Decode rates compared across unequal context depths and modes
- **What happens:** Decode t/s at 7.4k tokens of context is structurally lower than llama-bench `tg128`; think-on runs decode 1900+ tokens ending much deeper than think-off's 700. Comparing a model's think-on decode rate against another's think-off rate, or against previously published bench numbers, ranks configurations, not models.
- **How it shows up:** SILENT — a defensible-looking speed ranking with mixed denominators.
- **Guard:** In `report.py`: derive the decode column from one designated scenario (e.g. s01) per mode, report think-on and think-off decode as separate columns, and print the depth (`prompt_n + predicted_n`) beside each rate. Add an assertion that no cell mixes modes. Document "not comparable to tg128" in the table header.

### 24. Single-sample timing noise
- **What happens:** Each speed number is measured once. A scheduler hiccup, TLB/page-fault burst, or brief host activity makes one model 15% slower with no way to detect it.
- **How it shows up:** SILENT — reordered speed ranks among near-tied models.
- **Guard:** Per-model in `run_one()`: after warmup, run a fixed decode canary (identical short prompt, `max_tokens: 128`) three times; record all three and use the median in `meta["decode_canary"]`; assert (max-min)/median < 10%, else set `"timing_noisy": true` and have the report flag that row.

### 25. Wall-clock elapsed conflated with inference speed
- **What happens:** `elapsed_s` from `post()` includes JSON serialization of the 27KB tools array, HTTP, jinja rendering of a 5-8KB template over 46 schemas, and queueing. Using elapsed to derive any rate, or comparing elapsed across models with different template sizes, misattributes overhead to the model.
- **How it shows up:** SILENT — a few hundred ms of per-request skew, worst for the largest templates.
- **Guard:** Per-request: keep `elapsed_s` for orchestration only; all rates must come from the server's `timings` object. Add a check in `run_one()`: `overhead = elapsed_s - (prompt_ms + predicted_ms)/1000`; record it and warn if > 2s (indicates queueing or a sick server).

### 26. Failed cold-prefill request silently shifting the cold cost onto s01
- **What happens:** If the `cold_prefill` request errors (it is printed but the run continues), s01 becomes the request that pays the full 7k-token prefill; its `elapsed_s` and timings become an outlier and the prefill measurement for that model is simply missing.
- **How it shows up:** SILENT — one weird scenario timing and an empty cell someone later backfills from the wrong place.
- **Guard:** Per-model in `run_one()`: if `cold_prefill` errored, restart the server and retry once; if it errors again, set `meta["invalid"] = "cold_prefill_failed"` and skip the model's speed columns entirely (grading may still use answer content). Never let s01's timings stand in for prefill.

## E. Orchestration & process control

### 27. No process group — the watchdog can't actually kill a step
- **What happens:** `start_server()` uses a plain `Popen`; the mandated 5-minute watchdog has nothing to `os.killpg` because the child is in the harness's own process group, and any helper the server spawns would be missed by `proc.kill()`.
- **How it shows up:** LOUD if it hangs — but a half-killed step that leaves the server alive becomes item 29's SILENT stale-server problem.
- **Guard:** In `start_server()`: `subprocess.Popen(..., start_new_session=True)`; keep `p.pid` and kill with `os.killpg(os.getpgid(p.pid), SIGKILL)` in both `stop_server()` escalation and the watchdog thread. The watchdog: a `threading.Timer`-based deadline armed around every step in `run_one()`, 300s, that killpgs and raises. Pre-flight assertion: `os.getpgid(p.pid) != os.getpgid(0)`.

### 28. wait_health can legally take ~15 minutes
- **What happens:** `wait_health()` loops 300 times, each iteration up to ~3s (2s HTTP timeout + 1s sleep) — up to 900s, triple the 5-minute cap, all before a single scenario runs. A model that loads in 6 minutes also quietly violates the cap.
- **How it shows up:** LOUD if it times out; SILENT as a cap violation and as unrecorded load-time variance.
- **Guard:** Rewrite `wait_health()` with a `time.monotonic()` deadline: `deadline = start + 240`; loop while `monotonic() < deadline`. Record actual load time in `_run.json` (item 6). This is a per-model check; the 300s watchdog (item 27) is the backstop.

### 29. Orphaned server after a harness crash poisons the next launch
- **What happens:** A Python exception outside the `try/finally`, or a SIGKILL of the harness, leaves llama-server running with the old model. The next invocation's port-8099 collision is exactly the mechanism that previously recorded model A's answers under model B's name.
- **How it shows up:** SILENT — the fix for defect #3 covers in-run staleness; this is the across-invocation path.
- **Guard:** Write `results/<run>/server.pid` on spawn; `preflight()` refuses to start if port 8099 accepts a connection OR a recorded pidfile points at a live process (check `/proc/<pid>/cmdline` contains `llama-server` before killing by that exact PID). Install `atexit` + SIGTERM/SIGINT handlers in `main()` that killpg the current server. Per-request (defect-3 fix, reinforced): assert the response's `model` field matches the intended gguf path on *every* request, not just the first.

### 30. A timed-out request poisons the next scenario's timing
- **What happens:** When `post()` times out client-side, llama-server is still generating. The next scenario queues behind the leftover work; its `elapsed_s` and possibly its slot cache state are contaminated, and with `-np 1` the whole pipeline serializes behind a zombie job.
- **How it shows up:** SILENT — one timeout produces two or three bad rows.
- **Guard:** Per-request in `run_one()`: on any `err` containing `timed out`, record the scenario as `TIMEOUT`, then restart the server (`stop_server` + `start_server` + warmup) before the next scenario, and note the restart in `_run.json["restarts"]`.

### 31. REQ_TIMEOUT (900s) vs the 5-minute hard cap — one slow scenario aborts the whole run
- **What happens:** The request timeout is 15 min but the mandated step cap is 5 min: a legitimately slow scenario (2.6 GB model, think-on, 1900 tokens at 5 t/s ≈ 380s) trips the global watchdog, which by the stated policy kills everything — three hours of results at stake for one scenario; or, mis-set the other way, slow-but-valid answers get recorded as failures.
- **How it shows up:** LOUD at abort time, but the *silent* version is the budget mismatch nobody computed: budgets that cannot fit make truncation/timeouts systematic for the slowest models only.
- **Guard:** Pre-flight budget assertion in `preflight()`: for the worst case, `(7500_prompt_tokens / min_expected_prefill_tps) + (max_tokens_max / min_expected_decode_tps) < REQ_TIMEOUT < step_cap`, with floors taken from the slowest plausible model (e.g. 40 t/s prefill, 4 t/s decode). Set `EVAL_REQ_TIMEOUT=280` and step cap 300 so a slow scenario becomes a recorded per-scenario TIMEOUT (item 30), never a global abort.

## F. Data integrity & provenance

### 32. Non-atomic JSON writes
- **What happens:** `record()` writes directly to the final path. A crash or the watchdog's SIGKILL mid-write leaves a truncated `s07.json` that `json.load` in the grading step either crashes on (best case) or that gets regenerated inconsistently.
- **How it shows up:** LOUD on load if you are lucky; SILENT if a partial older file survives alongside newer data.
- **Guard:** In `record()`: write to `path + ".tmp"`, `f.flush(); os.fsync(f.fileno())`, then `os.replace(tmp, path)`. Grading pre-flight: `json.load` every file it will read and fail fast listing unparseable ones.

### 33. Results directory reuse mixes two runs
- **What happens:** **VERIFIED:** `harness/results/` already exists and is non-empty, and `run_one()` uses `os.makedirs(..., exist_ok=True)`. A new run writes into the same tree; any scenario that errors this time leaves *last* run's file in place, and the grader reads a chimera of two configurations. The many stale `outputs-*` directories show this has already been the working style.
- **How it shows up:** SILENT — every file individually looks fine.
- **Guard:** In `main()`: default outdir becomes `results/<UTC-timestamp>-<git-short-sha>`; if a given `--outdir` exists and is non-empty, refuse to start unless `--resume` is passed. Per-model: `run_one()` refuses an existing `tag` directory that lacks a valid complete `_run.json` (crash residue) unless resuming.

### 34. `--only tools` then `--only quality` overwrites `_run.json`
- **What happens:** `run_one()` writes `_run` at the end from a fresh `meta`; running the two halves in separate invocations (which `--only` invites) leaves `_run.json` containing only the second half's results and settings, while the first half's per-scenario files sit beside it unanchored.
- **How it shows up:** SILENT — the manifest no longer describes the data next to it.
- **Guard:** In `run_one()`: before writing `_run`, if the file exists, load it and merge `results` dicts (assert the `model`, `mode`, `md5`, and config fields are identical, abort if not). Grading pre-flight: assert `_run.json["results"]` keys cover every expected scenario id for the suites being graded.

### 35. A failed scenario indistinguishable from a real zero downstream
- **What happens:** Errors are recorded as `{"error": ...}` with no `content`/`tool_calls` keys. If the grading pipeline feeds answers by field access with defaults (the natural `out.get("content","")`), an HTTP 500 becomes an empty answer and is scored ZERO — the exact "confident wrong table" outcome.
- **How it shows up:** SILENT — the most dangerous single pattern in the pipeline.
- **Guard:** In the grading driver: partition results into `answered` / `errored` / `truncated` / `not_run` *before* building any grader prompt; only `answered` (and `truncated`, labeled) reach the grader. The final table must print coverage per cell (`13/14 graded, 1 error`) and `report.py` must assert `graded + errored + truncated + not_run == expected_count` for every run.

### 36. No provenance manifest
- **What happens:** Three weeks later nobody can say which llama.cpp commit, build flags, roster hash, scenario hash, or host state produced the table — so a suspicious number can be neither trusted nor reproduced, which is how the previous two days were lost.
- **How it shows up:** SILENT — until the first dispute, then fatal.
- **Guard:** `preflight()` writes `<outdir>/manifest.json`: output of `llama-server --version`, the pinned commit `a3b1eff`, sha256 of `taskq-tools.json`, `scenarios-tools.json`, `scenarios-quality.json`, `roster.json`, `run_eval.py` itself, `lscpu` summary, governor, `free`, cgroup values, env overrides (CTX/THREADS/THINK_BUDGET). Grading refuses an outdir without a manifest.

### 37. A typo'd `--models` filter yields a successful empty run
- **What happens:** `--models qwen3.5` matching nothing (case-sensitive substring) produces `0 models -> 0 runs` and then prints `EVAL-DONE` with exit code 0. In a scripted overnight pipeline this reads as success.
- **How it shows up:** SILENT for anyone reading only the exit code / final line.
- **Guard:** In `main()`: `assert runs, "model filter matched nothing"`; and when no filter is given, assert `len(runs) == 26` (recomputed from the roster: 9 dual-mode + 8 single) so a roster regression also fails loudly. Exit nonzero if any run recorded `invalid`.

### 38. No resumability — a crash at hour 3 loses everything or forces a contaminating rerun
- **What happens:** With no resume logic, the operator's realistic move after a crash is rerunning with `--models` for the remainder into the same directory — re-entering item 33's mixing hazard under time pressure, or rerunning everything on a now-hotter machine (item 4).
- **How it shows up:** SILENT — the recovered dataset is a patchwork of two host states with no marker.
- **Guard:** In `main()`: with `--resume <outdir>`, skip any run whose `_run.json` exists, is complete (all scenario ids present, no `invalid`), and whose recorded config hash matches the manifest; rerun the rest; append `"resumed_at"` timestamps into the manifest so the seam is visible in the report.

## G. Scenario & rubric validity

### 39. Rubric claims drifting from the real tool schemas
- **What happens:** Every rubric asserts schema facts (status vocabulary, `link_nodes` relation list, `create_task` requiring project+title, assign-vs-claim semantics, omit-clears behavior for `assign_task`/`set_task_summary`). I verified all of these against `taskq-tools.json` today and they currently match — but the next `fetch_tools.py` refresh can silently falsify a rubric, and a grader will then mark correct answers wrong.
- **How it shows up:** SILENT — the grader trusts the rubric.
- **Guard:** Pre-flight script `validate_rubrics.py`: machine-check, per scenario, that every tool name mentioned in the rubric exists; that every enum value listed (todo/in_progress/blocked/done/cancelled; blocked_by/requires/documented_by/references) appears in the tool's description or schema; that every "required" claim matches `parameters.required`. Run it in `preflight()`; abort on any mismatch.

### 40. taskq schemas changing between fetch, run, and grading
- **What happens:** `taskq-tools.json` is a snapshot of a live server. If taskq is redeployed before grading (or between two compared runs), the models were prompted with one contract and graded/interpreted against another.
- **How it shows up:** SILENT — no artifact records which contract was live.
- **Guard:** Pre-flight: re-run `fetch_tools.py` into the scratchpad and `diff` against the committed `taskq-tools.json`; abort on difference (re-pin deliberately if intended). Record the file's sha256 in the manifest (item 36); the grading driver asserts it grades against the same hash the run recorded.

### 41. Restraint scenarios graded from prose instead of the tool_calls array
- **What happens:** s03/s12 say *any* tool call is zero. Thinking models often narrate calls ("I'll use claim_task...") in reasoning or content without emitting one; other models emit a call *and* a correct explanation. Graders left to read the whole blob will drift — some scoring the narration as a call, some excusing an actual call because the prose was good.
- **How it shows up:** SILENT — inconsistent application of the harshest rule in the suite.
- **Guard:** In the grading driver: for tool scenarios, present the grader with a structured block — `tool_calls` (parsed JSON, authoritative), `content`, `finish_reason` — plus the fixed instruction "a tool call exists iff the tool_calls array is non-empty; text mentioning a tool is not a call." Also pre-parse each `tool_calls[].function.arguments` string in `run_one()` and record `arguments_parsed` / `arguments_parse_error` so graders never re-parse raw strings differently.

### 42. Tool ordering in the prompt changing between compared runs
- **What happens:** All models see the 46 tools in the same order (sorted by name in `fetch_tools.py`) — fair within a run. But if the file is ever regenerated without sorting, or tools are added, position effects shift small-model choices, and cross-run comparisons ("after the fix, Qwen3 improved") measure the shuffle.
- **How it shows up:** SILENT — plausible deltas between runs.
- **Guard:** Pre-flight in `preflight()`: assert `[t["function"]["name"] for t in tools] == sorted(...)`, assert `len(tools) == 46`, and compare the file hash against the manifest of any baseline run being compared. Comparisons across differing hashes must be refused by `report.py`.

### 43. q3's execution fixture doesn't exist — the rubric assumes evidence nobody produces
- **What happens:** The q3 rubric states "the execution result is supplied to you and is authoritative", but nothing in `run_eval.py` or the harness runs the generated script against fixtures. If grading proceeds anyway, the grader eyeballs the code and scores plausibility, exactly the failure class LLM grading was adopted to avoid.
- **How it shows up:** SILENT — scores appear on schedule with no execution behind them.
- **Guard:** Add `run_fixtures.py` (grading pre-step): extract q3 code from each run's `q3-python-script.json`, execute in the scratchpad (`/tmp/claude-.../scratchpad/q3/<tag>/`) with `timeout 10`, no network, cwd outside the repo, against a good CSV and a missing-column CSV; record stdout/stderr/exit codes to `q3-execution.json`. The grading driver asserts this file exists for every run before grading q3.

## H. Grading

### 44. Position and anchoring bias with 26 answers in one context
- **What happens:** One grader sees all 26 answers to a scenario. LLM judges systematically favor early positions, drift stricter/laxer through a long list, anchor later scores on earlier ones, and can be unblinded by model self-identification or file-name mentions in answers.
- **How it shows up:** SILENT — a consistent few-point tilt correlated with presentation order.
- **Guard:** In the grading driver: shuffle answer order with a recorded seed; relabel answers `answer-01..answer-26` with the tag→label mapping kept out of the grader's context; instruct the grader to ignore any self-identification inside an answer; grade a second pass in a different shuffle (see 46). Store the mapping and seed alongside the grades for de-anonymization.

### 45. Grader inventing, dropping, or misattributing scores
- **What happens:** Given 26 blobs, a grader can emit 25 scores, score a label twice, or hallucinate a score for an answer that was NO-DATA (item 35). One misattributed row silently swaps two models' results.
- **How it shows up:** SILENT — the output table has the right shape.
- **Guard:** Grading driver contract: the grader must return, per answer, `{label, verbatim_quote, score, rubric_branch}` where `verbatim_quote` is <=15 words copied from the answer. Post-parse assertions: the label set equals the submitted set exactly; each quote is a substring of the labeled answer (`quote in answer_text`); any violation discards the batch and regrades.

### 46. Grader nondeterminism treated as signal
- **What happens:** The same grader at nonzero temperature (or a different snapshot) moves borderline scores by 1-2 points; with 26 runs x 20 scenarios, rank flips among adjacent models are near-certain.
- **How it shows up:** SILENT — a ranking that would not reproduce.
- **Guard:** Pin the grader model id and temperature 0 in the grading driver; grade every scenario twice with different shuffles (item 44); flag any answer whose two scores differ by more than 1 point (or by any grade step for the FULL/PARTIAL/ZERO scenarios) for a third adjudication pass; report the disagreement rate in the final table's footnotes. If disagreement exceeds 10% of cells, the grading configuration is invalid — stop.

### 47. No calibration probes — a miscalibrated grader passes unnoticed
- **What happens:** If the grader misreads a rubric (e.g. treats s12's add_comment as full credit), every score it emits is systematically shifted and nothing in the pipeline can notice.
- **How it shows up:** SILENT — internally consistent wrongness.
- **Guard:** Grading driver: inject two synthetic answers into every scenario batch under anonymous labels — one hand-written known-FULL-CREDIT and one known-ZERO (for s04, e.g., a correct `set_task_status(412,"blocked")` call and an invented `"on_hold"` call). Assert the grader scores them FULL and ZERO respectively; otherwise discard the batch, fix the grading prompt, regrade. Strip probes before aggregation.

### 48. Summing scores across per-scenario graders with different scales
- **What happens:** Each scenario's grader has its own severity; a grand total sums 20 incomparable scales, so a model strong on harshly-graded scenarios loses to one strong on leniently-graded ones — an artifact, not a capability difference.
- **How it shows up:** SILENT — the headline "overall score" column.
- **Guard:** In `report.py`: publish the per-scenario score matrix as the primary artifact; aggregate by within-scenario rank (mean rank across scenarios) or within-scenario z-score, never by raw sum across scenarios; assert the aggregation function refuses to sum raw scores from different scenario ids. Tool suite and quality suite are reported as separate rankings.

### 49. Stale regex graders from the invalidated attempt getting reused
- **What happens:** **VERIFIED:** `grade.py`, `grade_c.py`, `grade_cap.py`, `grade_h.py`, `grade_tools.py` from the previous (regex-based, invalidated) attempt still sit in the harness directory, alongside a dozen `outputs-*` result trees. Under time pressure, a tired operator or an agent "finding the grading script" will run one of them on the new results.
- **How it shows up:** SILENT — it produces a full, familiar-looking table by the exact method already proven invalid.
- **Guard:** Before the run: `git mv` the old graders and `outputs-*` into `harness/attic/` with a README saying "invalidated, do not use". The new grading entry point is a single `grade_llm.py` that asserts the results manifest exists (item 36) and stamps `"grading_method": "llm-rubric-v2"` plus grader model id into its own output; `report.py` refuses grade files without that stamp.

### 50. No audit trail from score back to evidence
- **What happens:** A week later, "granite got PARTIAL on s06" cannot be checked without regrading: no record of which rubric branch fired, what the grader saw, or which shuffle/mapping was used. Unauditable scores are unfalsifiable — the previous benchmark died of exactly this.
- **How it shows up:** SILENT — until the first challenged number.
- **Guard:** The grading driver writes, per scenario, `grades/<scenario_id>.json` containing: grader model id, grading prompt hash, rubric text hash, shuffle seed, label→run mapping, and per-answer `{score, rubric_branch, verbatim_quote, grader_justification}` plus the full grader transcript path. `report.py` asserts every cell in the final table links to one of these records; a `spotcheck.py` samples 10 random cells and prints score + evidence side by side for a human pass before publication.

---

*50 items. Verified-on-box findings: items 1, 2, 5, 15, 18, 33, 49.*
