#!/usr/bin/env python3
"""llama-bench across every model at every depth. The clean, template-free half
of the speed picture; speed_grid.py is the real-workload half.

Two invocations per model:
  prefill  -p 128,901,7427 -n 0     tokens/s to ingest a prompt of that size
  decode   -p 0 -n 64 -d 0,128,901,7427   tokens/s to generate at that depth

Depth 0 reproduces the tg128-style figure every published benchmark quotes.
Depth 7427 is the taskq tool block. Both are reported; neither is called "the"
number.

-fa 0 because flash attention measured 2.7x SLOWER on this CPU at depth
(verify-config-results.txt). Process control is by PID with a hard cap: a model
too slow to finish is recorded as TOO_SLOW, which is a result about that model.
"""
import json, os, re, signal, subprocess, sys, time
import serverctl as sc
import hostcheck as hc

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA_BIN = os.environ.get("LLAMA_BIN", "/models/llamacpp-main/build/bin")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/small-models")
CPUS, THREADS = "0-5", 6
FA = "0" if os.environ.get("EVAL_FA", "off") == "off" else "1"
REPS = os.environ.get("BENCH_REPS", "2")
CAP = int(os.environ.get("BENCH_CAP", "900"))     # per invocation, seconds
ROW = re.compile(r"\|\s*([a-z0-9]+\d+)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)\s*\|\s*$")


def run_bench(model, args, cap=CAP):
    """One llama-bench invocation, owned by PID, killed by process group."""
    cmd = ["taskset", "-c", CPUS, f"{LLAMA_BIN}/llama-bench",
           "-m", os.path.join(MODEL_DIR, model), "-t", THREADS if isinstance(THREADS, str)
           else str(THREADS), "-fa", FA, "-r", REPS] + args
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=cap)
        return {"ok": p.returncode == 0, "returncode": p.returncode, "stdout": out,
                "cmd": " ".join(cmd)}
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait(timeout=30)
        return {"ok": False, "state": "TOO_SLOW",
                "why": f"exceeded {cap}s", "cmd": " ".join(cmd)}


def parse(stdout):
    """test-name -> (t/s, stddev), from llama-bench's markdown table."""
    got = {}
    for line in (stdout or "").splitlines():
        m = ROW.search(line)
        if m:
            got[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return got


def main():
    vpath = os.path.join(HERE, "roster-verified.json")
    src = json.load(open(vpath))["models"] if os.path.exists(vpath) \
        else json.load(open(os.path.join(HERE, "roster.json")))
    models = [m for m in src if "fatal" not in m]
    if len(sys.argv) > 1:
        want = sys.argv[1].split(",")
        models = [m for m in models if any(w in m["file"] for w in want)]

    probs, _ = sc.preflight(LLAMA_BIN, 8099, CPUS, THREADS)
    if probs:
        for p in probs:
            print("BLOCKED:", p)
        sys.exit(1)

    print(f"llama-bench grid: {len(models)} models, fa={FA}, r={REPS}, cap={CAP}s\n")
    results, t0 = [], time.monotonic()
    for i, m in enumerate(models, 1):
        name = m["file"]
        print(f"[{i}/{len(models)}] {name}  ({(time.monotonic()-t0)/60:.0f} min)", flush=True)
        r = {"file": name, "arch": m["arch"], "size_bytes": m["size_bytes"],
             "fa": FA, "reps": REPS, "host": hc.snapshot(CPUS)}
        pre = run_bench(name, ["-p", "128,901,7427", "-n", "0"])
        dec = run_bench(name, ["-p", "0", "-n", "64", "-d", "0,128,901,7427"])
        r["prefill_raw"], r["decode_raw"] = pre, dec
        r["prefill"] = parse(pre.get("stdout"))
        r["decode"] = parse(dec.get("stdout"))
        if not pre.get("ok"):
            r["prefill_state"] = pre.get("state", "FAILED")
        if not dec.get("ok"):
            r["decode_state"] = dec.get("state", "FAILED")
        results.append(r)
        json.dump(results, open(os.path.join(HERE, "bench-grid.json"), "w"), indent=1)
        pp = " ".join(f"{k}={v[0]:.0f}" for k, v in sorted(r["prefill"].items()))
        tg = " ".join(f"{k}={v[0]:.1f}" for k, v in sorted(r["decode"].items()))
        print(f"    prefill: {pp or r.get('prefill_state','-')}", flush=True)
        print(f"    decode : {tg or r.get('decode_state','-')}", flush=True)
    print(f"\nBENCH-GRID-DONE in {(time.monotonic()-t0)/60:.1f} min -> bench-grid.json")


if __name__ == "__main__":
    main()
