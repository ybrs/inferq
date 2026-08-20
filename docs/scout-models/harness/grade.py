#!/usr/bin/env python3
"""Task-suite grader (t1-t3): deterministic checks over model outputs.

t1: valid JSON array, 4 tasks, correct assignees/priorities, no invented assignee.
t2: grounded summary — must not invent a cause; TLS/timeout fact present; no hallucinated entities.
t3: script must be valid Python, run correctly on a fixture CSV, and exit 1 on a missing column.
"""
import csv, json, os, re, subprocess, sys, tempfile, glob

QDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

def strip(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    txt = re.sub(r"^\[elapsed.*?\]\n", "", txt, flags=re.S)
    return txt.strip()

def unfence(t):
    m = re.search(r"```(?:json|python)?\s*(.*?)```", t, re.S)
    return m.group(1).strip() if m else t

def meta(txt):
    m = re.match(r"\[elapsed ([\d.]+)s, completion_tokens (\d+), prompt_tokens (\d+)\]", txt)
    if not m: return {}
    el, ct = float(m.group(1)), int(m.group(2))
    return {"elapsed": el, "tokens": ct, "tps": round(ct/el, 1) if el else 0}

def grade_t1(txt):
    notes, score = [], 0
    body = unfence(strip(txt))
    try:
        data = json.loads(body)
    except Exception as e:
        return 0, [f"invalid JSON ({type(e).__name__})"]
    if not isinstance(data, list):
        return 0, ["not a JSON array"]
    score += 2; notes.append("valid JSON array")
    if len(data) == 4: score += 1; notes.append("4 items")
    else: notes.append(f"{len(data)} items (expected 4)")
    blob = json.dumps(data).lower()
    # assignee correctness
    def find(kw):
        for d in data:
            if isinstance(d, dict) and kw in json.dumps(d).lower(): return d
        return None
    login = find("redirect") or find("login")
    if login and str(login.get("assignee","")).lower().startswith("sarah"): score += 1; notes.append("login->Sarah")
    else: notes.append("login assignee wrong/missing")
    db = find("credential") or find("rotate")
    if db:
        ok_a = str(db.get("assignee","")).lower().startswith("dan")
        ok_p = str(db.get("priority","")).lower() == "high"
        ok_d = "wed" in str(db.get("due","")).lower()
        score += ok_a + ok_p + ok_d
        notes.append(f"db: assignee={'ok' if ok_a else 'BAD'} prio={'ok' if ok_p else 'BAD'} due={'ok' if ok_d else 'BAD'}")
    else: notes.append("db task missing")
    chg = find("changelog")
    if chg:
        if chg.get("assignee") in (None, "", "null"): score += 1; notes.append("changelog assignee null (good)")
        else: notes.append(f"HALLUCINATED changelog assignee: {chg.get('assignee')}")
    else: notes.append("changelog task missing")
    for name in ["mike", "john", "alice", "bob", "priya"]:
        if name in blob: notes.append(f"HALLUCINATED name '{name}'"); score -= 2
    return max(score, 0), notes  # max 8

def grade_t2(txt):
    body = strip(txt); low = body.lower(); notes, score = [], 0
    if not body:
        return 0, ["NO ANSWER EMITTED (all output consumed by <think>)"]
    sents = [s for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    if len(sents) <= 3: score += 1; notes.append(f"{len(sents)} sentences")
    else: notes.append(f"TOO LONG: {len(sents)} sentences")
    if "tls" in low or "handshake" in low: score += 3; notes.append("cites TLS handshake timeout")
    else: notes.append("MISSING the TLS timeout fact")
    if "06:30" in body or "6:30" in body or "100%" in body: score += 1; notes.append("cites failure onset/rate")
    # Faithfulness: full credit unless it asserts a root cause the ticket never states.
    invented = False
    for h, label in [(r"certificat", "invented cert expiry"), (r"firewall", "invented firewall"),
                     (r"\bdns\b", "invented DNS"), (r"expired", "invented expiry"),
                     (r"customer.{0,25}(changed|misconfigur)", "blames customer change (contradicts ticket)"),
                     (r"our (server|side).{0,20}(bug|outage)", "invented our-side outage"),
                     (r"(root cause|caused by|because of|due to).{0,40}(outage|overload|down|misconfig|block)", "asserts unstated root cause")]:
        if re.search(h, low): score -= 2; invented = True; notes.append(f"HALLUCINATION: {label}")
    if not invented:
        score += 2; notes.append("faithful: no invented root cause")
    if re.search(r"unknown|not (yet )?(been )?(determined|identified|known)|unclear|undetermined|no.{0,12}root cause", low):
        notes.append("(bonus behavior: explicitly hedges cause)")
    return max(score, 0), notes  # max 7

FIXTURE = [["id","status"],["1","open"],["2","closed"],["3","open"],["4","open"],["5","closed"],["6","pending"]]

def grade_t3(txt):
    code = unfence(strip(txt)); notes, score = [], 0
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "s.py"); open(sp,"w").write(code)
        r = subprocess.run([sys.executable, "-m", "py_compile", sp], capture_output=True)
        if r.returncode != 0:
            return 0, ["does not compile: " + r.stderr.decode()[:120]]
        score += 2; notes.append("compiles")
        cp = os.path.join(d, "a.csv")
        with open(cp,"w",newline="") as f: csv.writer(f).writerows(FIXTURE)
        r = subprocess.run([sys.executable, sp, cp], capture_output=True, timeout=30)
        out = r.stdout.decode().strip()
        want = ["open: 3", "closed: 2", "pending: 1"]
        got = [l.strip() for l in out.splitlines() if l.strip()]
        if got == want: score += 4; notes.append("exact correct output")
        elif all(w in out for w in want): score += 2; notes.append(f"right counts, wrong order/format: {got}")
        else: notes.append(f"WRONG output: {got or r.stderr.decode()[:100]}")
        bp = os.path.join(d, "b.csv")
        with open(bp,"w",newline="") as f: csv.writer(f).writerows([["id","state"],["1","x"]])
        r2 = subprocess.run([sys.executable, sp, bp], capture_output=True, timeout=30)
        if r2.returncode == 1 and r2.stderr.strip(): score += 2; notes.append("missing-column -> exit 1 + stderr")
        else: notes.append(f"missing-column handling BAD (exit {r2.returncode})")
    return score, notes  # max 8

rows = []
for f in sorted(glob.glob(os.path.join(QDIR, "*.t1-task-extract.txt"))):
    name = os.path.basename(f).replace(".t1-task-extract.txt","")
    res = {"model": name, "scores": {}, "notes": {}, "perf": {}}
    total = 0
    for key, fn, mx in [("t1", grade_t1, 8), ("t2", grade_t2, 7), ("t3", grade_t3, 8)]:
        p = os.path.join(QDIR, f"{name}.{ {'t1':'t1-task-extract','t2':'t2-ticket-summary','t3':'t3-python-script'}[key] }.txt")
        if not os.path.exists(p): res["scores"][key] = None; continue
        txt = open(p).read()
        res["perf"][key] = meta(txt)
        try: s, n = fn(txt)
        except Exception as e: s, n = 0, [f"grader error: {e}"]
        res["scores"][key] = f"{s}/{mx}"; res["notes"][key] = n; total += s
    res["total"] = f"{total}/23"
    res["_t"] = total
    rows.append(res)

rows.sort(key=lambda r: -r["_t"])
for r in rows:
    r.pop("_t")
    print(json.dumps(r, indent=1))
