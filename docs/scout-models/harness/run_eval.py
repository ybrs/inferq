#!/usr/bin/env python3
"""Run every scenario against every verified model, and record raw results.

Scores nothing. Grading is a separate LLM step against written rubrics; the
previous attempt's regex graders are in attic/ and must not be used.

What this records that the previous harness did not:
  * Real OpenAI `tools` calls carrying the 46 live taskq MCP schemas, so the
    model emits genuine tool_calls with schema-constrained arguments.
  * A cold prefill isolated from any warmup, asserted to be a real >5k-token
    prefill and not a cache hit (FAILURE-MODES 8, 22, 26).
  * Every failure kept distinguishable from a real zero: ERROR, TIMEOUT,
    TRUNCATED and NOT_RUN are separate states, never scored as 0 (14, 35).
  * Host state, model md5, /props read-back and provenance per run (21, 36).

Usage:
  run_eval.py                        fresh run into results/<utc>-<sha>/
  run_eval.py --resume <outdir>      finish an interrupted run
  run_eval.py --models substr,...    subset, for debugging only
"""
import argparse, hashlib, json, os, subprocess, sys, time
import serverctl as sc
import hostcheck as hc

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA_BIN = os.environ.get("LLAMA_BIN", "/models/llamacpp-main/build/bin")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/small-models")
PORT = int(os.environ.get("PORT", "8099"))
CPUS, THREADS = "0-5", 6
FA = os.environ.get("EVAL_FA", "off")
THINK_BUDGET = int(os.environ.get("EVAL_THINK_BUDGET", "1200"))
REQ_TIMEOUT = int(os.environ.get("EVAL_REQ_TIMEOUT", "280"))   # < the 300s step cap (31)

# Every sampler knob pinned, so a server default can never drift a run (10).
SAMPLER = {"temperature": 0, "top_k": 1, "top_p": 1.0, "seed": 42,
           "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def record(outdir, name, obj):
    """Atomic write: a crash must not leave a half-parsed JSON behind (32)."""
    path = os.path.join(outdir, name + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def git_sha():
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=15).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def build_manifest(outdir, roster, verified, args):
    """36: everything needed to say what produced these numbers."""
    ver = subprocess.run([f"{LLAMA_BIN}/llama-server", "--version"],
                         capture_output=True, text=True, timeout=30)
    files = ["taskq-tools.json", "scenarios-tools.json", "scenarios-quality.json",
             "roster.json", "roster-verified.json", "run_eval.py", "serverctl.py"]
    return {
        "created": time.time(),
        "git_sha": git_sha(),
        "llama_server_version": (ver.stdout + ver.stderr).strip().splitlines()[:2],
        "llama_bin": LLAMA_BIN,
        "file_sha256": {f: sha256_file(os.path.join(HERE, f))
                        for f in files if os.path.exists(os.path.join(HERE, f))},
        "config": {"cpus": CPUS, "threads": THREADS, "fa": FA,
                   "think_budget": THINK_BUDGET, "req_timeout": REQ_TIMEOUT,
                   "step_timeout": sc.STEP_TIMEOUT, "sampler": SAMPLER},
        "host_at_start": hc.snapshot(CPUS),
        "n_models": len(roster),
        "templates_verified_at": verified.get("verified_at"),
        "args": vars(args),
    }


def body_for(messages, tools, max_tokens, ctk):
    b = dict(SAMPLER)
    b.update({"messages": messages, "max_tokens": max_tokens, "stream": False})
    if tools:
        b.update({"tools": tools, "tool_choice": "auto"})
    if ctk:
        b["chat_template_kwargs"] = ctk
    return b


def extract(d, wall):
    """One response, with every not-an-answer state kept distinguishable (14, 35)."""
    m = d["choices"][0]["message"]
    fr = d["choices"][0].get("finish_reason")
    calls = m.get("tool_calls") or []
    parsed, perr = [], []
    for c in calls:
        try:
            parsed.append({"name": c["function"]["name"],
                           "arguments": json.loads(c["function"]["arguments"])})
        except Exception as e:
            perr.append(f"{c['function']['name']}: {e}")
            parsed.append({"name": c["function"]["name"], "arguments": None})
    t = d.get("timings") or {}
    content = m.get("content") or ""
    out = {
        "state": "TRUNCATED" if fr == "length" else "ANSWERED",
        "finish_reason": fr,
        "content": content,
        "reasoning_content": m.get("reasoning_content") or "",
        "tool_calls_raw": calls,
        "tool_calls": parsed,                      # 41: pre-parsed, one interpretation
        "arguments_parse_errors": perr,
        "usage": d.get("usage"), "timings": t,
        "elapsed_s": round(wall, 2),
        # 25: orchestration overhead kept separate from inference rates
        "overhead_s": round(wall - (t.get("prompt_ms", 0) + t.get("predicted_ms", 0)) / 1000, 2),
        # 19: reasoning must arrive in its own field
        "reasoning_leak": content.lstrip().startswith(("<think", "<|thinking", "<seed:think")),
    }
    return out


def run_one(model, mode, tools, tool_spec, qual_spec, outroot, do_tools):
    tag = f"{model['file'].replace('.gguf','')}__{mode}"
    outdir = os.path.join(outroot, tag)
    os.makedirs(outdir, exist_ok=True)
    kwarg = model.get("thinking_kwarg")
    ctk = {kwarg: (mode == "think-on")} if (kwarg and mode != "default") else {}
    path = os.path.join(MODEL_DIR, model["file"])
    ctx = model["ctx_used"]

    meta = {"model": model["file"], "mode": mode, "tag": tag, "arch": model["arch"],
            "chat_template_kwargs": ctk, "ctx": ctx, "fa": FA,
            "size_bytes": model["size_bytes"], "results": {}, "not_run": [],
            "restarts": 0, "host_at_start": hc.snapshot(CPUS)}

    # 21: the file must still be what the roster hashed
    if md5(path) != model["md5"]:
        meta["invalid"] = "md5 changed since roster"
        record(outdir, "_run", meta)
        return meta
    meta["md5"] = model["md5"]

    swap0 = hc.swap_counters()
    srv = sc.LlamaServer(LLAMA_BIN, path, PORT, CPUS, THREADS, ctx,
                         os.path.join(outdir, "server.log"), fa=FA)

    def start():
        with sc.step(f"start[{tag}]", timeout=300) as s:
            pid = srv.start()
            open(os.path.join(outdir, "server.pid"), "w").write(str(pid))   # 29
            srv.wait_healthy(timeout=280)                                    # 28
        srv.start_liveness_watch()                                           # 11
        meta["load_seconds"] = round(s.elapsed, 1)                           # 6
        with sc.step(f"props[{tag}]", timeout=60):
            meta["props"] = srv.assert_props(ctx)                            # 9
            srv.assert_not_mmapped()                                         # 9
            srv.assert_serving(path)                                         # stale server
        meta["chat_format"] = srv.chat_format()                              # 13/17
        # warmup on a disjoint prompt so it cannot seed the tool prefix (8)
        with sc.step(f"warmup[{tag}]", timeout=280):
            srv.chat(body_for([{"role": "user", "content": "Say OK."}], None, 16, ctk),
                     timeout=REQ_TIMEOUT)

    def ask(scenario_id, messages, use_tools, max_tokens):
        """One scenario. Never returns a value that could be read as a score."""
        nonlocal meta
        try:
            with sc.step(f"{tag}/{scenario_id}", timeout=300):
                d, wall = srv.chat(body_for(messages, tools if use_tools else None,
                                            max_tokens, ctk), timeout=REQ_TIMEOUT)
            srv.assert_serving(path)                                         # 29, every request
            return extract(d, wall)
        except sc.FatalRunError as e:
            msg = str(e)
            dead = not srv.alive()
            out = {"state": "TIMEOUT" if "timed out" in msg else "ERROR", "error": msg[:500]}
            if dead:                                                          # 11
                meta["invalid"] = f"server died at {scenario_id}"
            return out

    try:
        start()

        # 24: three identical decode canaries, so timing noise is visible
        canary = []
        for i in range(3):
            with sc.step(f"canary[{tag}]{i}", timeout=280):
                d, _ = srv.chat(body_for(
                    [{"role": "user", "content": "Count from 1 to 60, one number per line."}],
                    None, 128, ctk), timeout=REQ_TIMEOUT)
            canary.append(round((d.get("timings") or {}).get("predicted_per_second", 0), 2))
        canary.sort()
        med = canary[1]
        meta["decode_canary_tps"] = canary
        meta["decode_canary_median"] = med
        meta["timing_noisy"] = bool(med and (canary[2] - canary[0]) / med > 0.10)

        if do_tools and model["template_renders_tools"]:
            # 8/22/26: cold prefill, isolated, and asserted to be genuinely cold
            cold = None
            for attempt in (1, 2):
                with sc.step(f"cold[{tag}]", timeout=300):
                    try:
                        d, wall = srv.chat(body_for(
                            [{"role": "system", "content": tool_spec["system"]},
                             {"role": "user", "content": "ping"}], tools, 1, ctk),
                            timeout=REQ_TIMEOUT)
                    except sc.FatalRunError as e:
                        cold = {"state": "ERROR", "error": str(e)[:400]}
                        break
                t = d.get("timings") or {}
                if t.get("prompt_n", 0) >= 5000 and t.get("cache_n", 0) < 100:
                    cold = {"state": "ANSWERED", "timings": t,
                            "prefill_tps": round(t["prompt_per_second"], 1),
                            "prompt_n": t["prompt_n"], "elapsed_s": round(wall, 2)}
                    break
                cold = {"state": "INVALID",
                        "why": f"prompt_n={t.get('prompt_n')} cache_n={t.get('cache_n')} "
                               f"- not a cold prefill"}
                if attempt == 1:                       # restart and try once more
                    srv.stop(); meta["restarts"] += 1; start()
            meta["cold_prefill"] = cold
            if cold and cold["state"] != "ANSWERED":
                meta["speed_invalid"] = "cold prefill not measurable"

            for s in tool_spec["scenarios"]:
                if s["id"] in model.get("unrenderable_scenarios", {}):        # 18
                    meta["results"][s["id"]] = {
                        "state": "NOT_APPLICABLE",
                        "why": model["unrenderable_scenarios"][s["id"]]}
                    continue
                msgs = [{"role": "system", "content": tool_spec["system"]}] + s["messages"]
                out = ask(s["id"], msgs, True,
                          700 + (THINK_BUDGET if mode == "think-on" else 0))
                out["scenario"] = s["id"]
                meta["results"][s["id"]] = out
                record(outdir, s["id"], out)
                names = [c["name"] for c in out.get("tool_calls", [])]
                print(f"    {s['id']:28} {out['state']:12} {names or ''}", flush=True)
                if out["state"] == "TIMEOUT":                                  # 30
                    srv.stop(); meta["restarts"] += 1; start()
                if "invalid" in meta:
                    break
        elif do_tools:
            meta["tool_suite"] = "N/A: template does not render tools"         # 17
            for s in tool_spec["scenarios"]:
                meta["results"][s["id"]] = {"state": "NOT_APPLICABLE",
                                            "why": "template does not render tools"}

        if "invalid" not in meta:
            for s in qual_spec["scenarios"]:
                msgs = [{"role": "system", "content": qual_spec["system"]}] + s["messages"]
                out = ask(s["id"], msgs, False,
                          s["max_tokens"] + (THINK_BUDGET if mode == "think-on" else 0))
                out["scenario"] = s["id"]
                meta["results"][s["id"]] = out
                record(outdir, s["id"], out)
                print(f"    {s['id']:28} {out['state']:12} {len(out.get('content',''))} chars",
                      flush=True)
                if out["state"] == "TIMEOUT":
                    srv.stop(); meta["restarts"] += 1; start()
                if "invalid" in meta:
                    break
    finally:
        try:
            srv.stop()
        except Exception as e:
            meta["stop_problem"] = str(e)

    expected = [s["id"] for s in tool_spec["scenarios"]] + [s["id"] for s in qual_spec["scenarios"]]
    meta["not_run"] = [e for e in expected if e not in meta["results"]]        # 35
    swap1 = hc.swap_counters()
    meta["swap_delta"] = {k: swap1[k] - swap0[k] for k in swap0}               # 5
    if any(v > 0 for v in meta["swap_delta"].values()):
        meta["timings_tainted"] = "swapping occurred during this run"
    meta["host_at_end"] = hc.snapshot(CPUS)
    record(outdir, "_run", meta)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--no-tools", action="store_true")
    a = ap.parse_args()

    vpath = os.path.join(HERE, "roster-verified.json")
    if not os.path.exists(vpath):
        sys.exit("roster-verified.json missing - run verify_templates.py first (15,16,17,18)")
    verified = json.load(open(vpath))
    models = [m for m in verified["models"] if "fatal" not in m]
    skipped = [m for m in verified["models"] if "fatal" in m]

    tools = json.load(open(os.path.join(HERE, "taskq-tools.json")))
    tool_spec = json.load(open(os.path.join(HERE, "scenarios-tools.json")))
    qual_spec = json.load(open(os.path.join(HERE, "scenarios-quality.json")))

    # 42: the tool block must be identical and ordered across everything compared
    names = [t["function"]["name"] for t in tools]
    assert names == sorted(names), "taskq-tools.json is not name-sorted"
    assert len(tools) == 46, f"expected 46 tools, found {len(tools)}"

    if a.models:
        want = [s.strip() for s in a.models.split(",") if s.strip()]
        models = [m for m in models if any(w in m["file"] for w in want)]

    runs = []
    for m in models:
        runs += ([(m, "think-off"), (m, "think-on")]
                 if m.get("thinking_kwarg_effective") else [(m, "default")])
    assert runs, "model filter matched nothing"                                # 37

    if a.resume:
        outroot = a.resume
        if not os.path.isdir(outroot):
            sys.exit(f"--resume {outroot} does not exist")
    else:                                                                       # 33
        outroot = os.path.join(HERE, "results",
                               time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + git_sha())
        if os.path.exists(outroot) and os.listdir(outroot):
            sys.exit(f"{outroot} exists and is not empty")
        os.makedirs(outroot, exist_ok=True)

    print("PRE-FLIGHT")
    probs, notes = sc.preflight(LLAMA_BIN, PORT, CPUS, THREADS)
    for k, v in notes.items():
        print(f"  {k:26} {v}")
    host = hc.snapshot(CPUS)
    probs += hc.governor_problems(host["governors"])                            # 1
    cpuset = hc.parse_cpu_list(host["cpuset_effective"])                        # 2
    if cpuset and not hc.parse_cpu_list(CPUS) <= cpuset:
        probs.append(f"cpuset.cpus.effective={host['cpuset_effective']} excludes {CPUS}")
    sib = [c for c, v in host["cpu_busy_siblings"].items() if v and v > 0.20]    # 3
    if sib:
        probs.append(f"HT siblings busy: {sib}")
    if probs:
        for p in probs:
            print(f"  BLOCKED: {p}")
        sys.exit(1)
    print(f"  governor={host['governors']['cpu0']['scaling_governor']} "
          f"turbo={'on' if host['governors']['no_turbo']=='0' else 'OFF'} "
          f"{host['cpu_mhz_mean']}MHz {host['coretemp_c']}C  fa={FA}")

    mpath = os.path.join(outroot, "manifest.json")
    if not a.resume:
        record(outroot, "manifest", build_manifest(outroot, models, verified, a))
    print(f"\n{len(models)} models -> {len(runs)} runs into {outroot}")
    if skipped:
        print(f"  skipped (failed verification): {[m['file'] for m in skipped]}")

    t0 = time.monotonic()                                                       # 7
    done = []
    for i, (m, mode) in enumerate(runs, 1):
        tag = f"{m['file'].replace('.gguf','')}__{mode}"
        rj = os.path.join(outroot, tag, "_run.json")
        if a.resume and os.path.exists(rj):                                     # 38
            try:
                prev = json.load(open(rj))
                if not prev.get("not_run") and "invalid" not in prev:
                    print(f"[{i}/{len(runs)}] {tag} - already complete, skipping")
                    done.append(prev)
                    continue
            except Exception:
                pass
        print(f"\n[{i}/{len(runs)}] {tag}  ({(time.monotonic()-t0)/60:.0f} min elapsed)",
              flush=True)
        done.append(run_one(m, mode, tools, tool_spec, qual_spec, outroot, not a.no_tools))

    # 4: drift canary - re-measure the first model and compare
    print(f"\nEVAL-DONE in {(time.monotonic()-t0)/60:.1f} min")
    bad = [d["tag"] for d in done if d.get("invalid")]
    if bad:
        print(f"INVALID RUNS: {bad}")
    print(f"results: {outroot}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
