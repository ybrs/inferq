#!/usr/bin/env python3
"""Compact scoreboard: llama-bench speed joined with task-suite scores."""

import json, subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

BENCH = {}
cur = None
for line in open(os.path.join(HERE, "bench-results.md")):
    if line.startswith("=== "):
        cur = line[4:].strip().replace(".gguf", "")
    elif line.startswith("| ") and cur and ("pp512" in line or "tg128" in line):
        parts = [p.strip() for p in line.split("|")]
        test = "pp" if "pp512" in line else "tg"
        BENCH.setdefault(cur, {})[test] = parts[-2].split("±")[0].strip()

out = subprocess.run([sys.executable, os.path.join(HERE, "grade.py")],
                     capture_output=True, text=True).stdout
objs, dec, i = [], json.JSONDecoder(), 0
while i < len(out):
    while i < len(out) and out[i] in " \n\t": i += 1
    if i >= len(out): break
    o, j = dec.raw_decode(out, i); objs.append(o); i = j

hdr = f"{'model':32} {'pp t/s':>7} {'tg t/s':>7} {'t1':>4} {'t2':>4} {'t3':>4} {'total':>6} {'sec':>6} {'tok':>6}"
print(hdr); print("-" * len(hdr))
for o in objs:
    m = o["model"]; b = BENCH.get(m, {})
    p = o["perf"]
    sec = sum(v.get("elapsed", 0) for v in p.values())
    tok = sum(v.get("tokens", 0) for v in p.values())
    s = o["scores"]
    g = lambda k: (s.get(k) or "-").split("/")[0]
    print(f"{m[:32]:32} {b.get('pp','-'):>7} {b.get('tg','-'):>7} {g('t1'):>4} {g('t2'):>4} {g('t3'):>4} "
          f"{o['total']:>6} {sec:6.1f} {tok:6d}")
print("\n(t1 task-JSON /8, t2 grounded summary /7, t3 python script /8; sec+tok = all 3 answers combined)\n")
for o in objs:
    bad = [n for k in ("t1", "t2", "t3") for n in o["notes"].get(k, [])
           if any(w in n for w in ("HALLUC", "BAD", "WRONG", "MISSING", "invalid", "TOO LONG",
                                   "does not", "NO ANSWER", "expected"))]
    if bad: print(f"  {o['model'][:32]:32} {'; '.join(bad)}")
