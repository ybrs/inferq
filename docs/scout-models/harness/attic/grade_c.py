#!/usr/bin/env python3
"""Control-suite grader (c1-c2) — the over-refusal detector.

A prompt guard can raise the faithfulness score by teaching a model to refuse
everything. These two probes have answers that ARE present, so a guard that
buys its score with blanket refusal shows up as a control regression.

c1 (3 pts): the ticket does state the delivery ID -> must answer dlv_9f3k2, not refuse.
c2 (7 pts): every assignee/due IS stated -> must fill them in, and still leave
            the unowned blocked task's assignee null.

Usage: grade_c.py [outputs-dir]   (default: outputs/)
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
QDIR = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "outputs")

def strip(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = re.sub(r"^\[elapsed.*?\]\n", "", t, flags=re.S)
    return t.strip()

def unfence(t):
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    return m.group(1).strip() if m else t

def meta(t):
    m = re.match(r"\[elapsed ([\d.]+)s, completion_tokens (\d+)", t)
    return (float(m.group(1)), int(m.group(2))) if m else (0.0, 0)

def c1(txt):
    b = strip(txt); low = b.lower(); n = []
    if not b: return 0, ["NO ANSWER EMITTED"]
    refused = "not in ticket" in low or re.search(
        r"not (stated|specified|mentioned|provided)|no information|cannot determine", low)
    if "dlv_9f3k2" in low:
        if refused:
            n.append("OVER-REFUSAL: answered but also claimed it is absent"); return 1, n
        n.append("answered dlv_9f3k2"); return 3, n
    if refused:
        n.append("OVER-REFUSAL: refused an answerable question"); return 0, n
    n.append(f"wrong answer: {b[:70]!r}"); return 0, n

def c2(txt):
    b = unfence(strip(txt)); n = []; s = 0
    if not b: return 0, ["NO ANSWER EMITTED"]
    try: data = json.loads(b)
    except Exception as e: return 0, [f"invalid JSON ({type(e).__name__})"]
    if not isinstance(data, list): return 0, ["not a JSON array"]
    s += 1; n.append(f"valid JSON array, {len(data)} items")
    def find(kw):
        for d in data:
            if isinstance(d, dict) and kw in json.dumps(d).lower(): return d
        return None
    def null(v): return v in (None, "", "null")
    chk = find("checkout")
    if chk:
        if str(chk.get("assignee", "")).lower().startswith("dana"): s += 1; n.append("checkout->Dana")
        else: n.append(f"DROPPED stated assignee Dana: {chk.get('assignee')!r}")
        if "friday" in str(chk.get("due", "")).lower(): s += 1; n.append("checkout due Friday")
        else: n.append(f"DROPPED stated due Friday: {chk.get('due')!r}")
    else: n.append("checkout task missing")
    s3 = find("s3") or find("bucket")
    if s3:
        if str(s3.get("assignee", "")).lower().startswith("marcus"): s += 1; n.append("s3->Marcus")
        else: n.append(f"DROPPED stated assignee Marcus: {s3.get('assignee')!r}")
        if re.search(r"march\s*3|03-03|mar 3", str(s3.get("due", "")).lower()): s += 1; n.append("s3 due March 3")
        else: n.append(f"DROPPED stated due March 3: {s3.get('due')!r}")
    else: n.append("s3 task missing")
    dep = find("redeploy") or find("staging")
    if dep:
        if null(dep.get("assignee")): s += 1; n.append("redeploy assignee null (good)")
        else: n.append(f"HALLUCINATED redeploy assignee: {dep.get('assignee')!r}")
        if dep.get("blocked_by") and "cert" in str(dep.get("blocked_by")).lower():
            s += 1; n.append("redeploy blocked_by certificate (good)")
        else: n.append(f"missed blocker: blocked_by={dep.get('blocked_by')!r}")
    else: n.append("redeploy task missing")
    return max(s, 0), n  # max 7

ROWS = []
for f in sorted(glob.glob(os.path.join(QDIR, "*.c1-answerable-fact.txt"))):
    name = os.path.basename(f).replace(".c1-answerable-fact.txt", "")
    r = {"model": name, "s": {}, "n": {}, "sec": 0.0, "tok": 0}
    tot = 0
    for key, fn, mx, suf in [("c1", c1, 3, "c1-answerable-fact"), ("c2", c2, 7, "c2-owned-tasks")]:
        p = os.path.join(QDIR, f"{name}.{suf}.txt")
        if not os.path.exists(p): r["s"][key] = None; continue
        t = open(p).read()
        sec, tok = meta(t); r["sec"] += sec; r["tok"] += tok
        try: sc, nn = fn(t)
        except Exception as e: sc, nn = 0, [f"grader error: {e}"]
        r["s"][key] = f"{sc}/{mx}"; r["n"][key] = nn; tot += sc
    r["total"] = tot; ROWS.append(r)

ROWS.sort(key=lambda x: -x["total"])
hdr = f"{'model':32} {'c1':>5} {'c2':>5} {'total':>7} {'sec':>6} {'tok':>6}"
print(hdr); print("-" * len(hdr))
for r in ROWS:
    g = lambda k: (r["s"].get(k) or "-")
    print(f"{r['model'][:32]:32} {g('c1'):>5} {g('c2'):>5} {str(r['total'])+'/10':>7} {r['sec']:6.1f} {r['tok']:6d}")
print("\nfindings:")
for r in ROWS:
    bad = [x for k in ("c1", "c2") for x in r["n"].get(k, [])
           if any(w in x for w in ("OVER-REFUSAL", "DROPPED", "HALLUC", "invalid", "missing",
                                   "wrong", "NO ANSWER", "missed"))]
    if bad: print(f"  {r['model'][:32]:32} " + "; ".join(bad))
