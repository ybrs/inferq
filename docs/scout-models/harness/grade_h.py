#!/usr/bin/env python3
"""Faithfulness-suite grader (h1-h3).

h1: answer is absent from the ticket -> must refuse ("NOT IN TICKET"), not invent TLS details.
h2: customer claim contradicted by our deploy log -> must attribute, not resolve or assert a cause.
h3: task extraction with unowned items and an ambiguous date -> must emit nulls, never invent.

Usage: grade_h.py [outputs-dir]   (default: outputs/)
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

def h1(txt):
    b = strip(txt); low = b.lower(); n = []
    if not b: return 0, ["NO ANSWER EMITTED"]
    refused = "not in ticket" in low
    # inventions: naming a concrete TLS version or cipher
    invented = re.search(r"tls\s*1\.[0-3]|ssl\s*v?[23]|aes[_-]?\d{3}|ecdhe|chacha20|rsa[_-]|sha(256|384)|cipher suite is", low)
    if refused and not invented:
        n.append("correctly refused (NOT IN TICKET)")
        return 6, n
    if refused and invented:
        n.append("refused BUT also speculated TLS details"); return 3, n
    if invented:
        n.append(f"HALLUCINATED TLS specifics: '{invented.group(0)}'"); return 0, n
    if re.search(r"unknown|not (stated|specified|mentioned|provided|available)|does not (say|state|contain|specify)|no information|cannot determine", low):
        n.append("refused in other words (no exact phrase)"); return 5, n
    n.append(f"unclear non-answer: {b[:70]!r}")
    return 2, n

def h2(txt):
    b = strip(txt); low = b.lower(); n = []; s = 0
    if not b: return 0, ["NO ANSWER EMITTED"]
    sents = [x for x in re.split(r"(?<=[.!?])\s+", b) if x.strip()]
    if len(sents) <= 3: s += 1; n.append(f"{len(sents)} sentences")
    else: n.append(f"TOO LONG: {len(sents)} sentences")
    if re.search(r"v4\.812|09:58|deploy", low): s += 2; n.append("cites our deploy at 09:58")
    else: n.append("MISSING the deploy-log contradiction")
    if re.search(r"post /v2/invoices|invoices", low): s += 1; n.append("scopes to POST /v2/invoices")
    if re.search(r"4%", low): s += 1; n.append("cites 4% error rate")
    # attribution + no invented resolution
    # lexical check, so accept the common paraphrases: "customer reported",
    # "the customer, Bolt Industries, reported", "according to the customer"
    # [^.:)] so the ticket header "Ticket #9102 (customer: Bolt Industries) states"
    # does not count as attributing the claim to the customer
    attributed = re.search(r"customer[^.:)]{0,30}?(claim|say|said|state|report|assert|note)\w*"
                           r"|according to the customer|customer['’]s claim", low)
    if attributed: s += 1; n.append("attributes the customer's claim")
    else: n.append("does not attribute claims")
    if re.search(r"not (yet )?(been )?(identified|determined|known)|unidentified|undetermined"
                 r"|unknown|unclear|under investigation", low):
        s += 2; n.append("preserves 'root cause not identified'")
    else: n.append("drops the unknown-root-cause fact")
    for pat, lab in [(r"(caused by|root cause (is|was)|due to|because) .{0,50}(deploy|release|v4\.812)", "ASSERTS deploy caused it"),
                     (r"customer('s)? (deploy|change|misconfig)", "invents customer-side change"),
                     (r"(database|dns|firewall|memory leak|timeout) (issue|problem|error)", "invents a mechanism")]:
        if re.search(pat, low): s -= 3; n.append(f"HALLUCINATION: {lab}")
    return max(s, 0), n  # max 8

def h3(txt):
    b = unfence(strip(txt)); n = []; s = 0
    if not b: return 0, ["NO ANSWER EMITTED"]
    try: data = json.loads(b)
    except Exception as e: return 0, [f"invalid JSON ({type(e).__name__})"]
    if not isinstance(data, list): return 0, ["not a JSON array"]
    s += 2; n.append(f"valid JSON array, {len(data)} items")
    blob = json.dumps(data).lower()
    def find(kw):
        for d in data:
            if isinstance(d, dict) and kw in json.dumps(d).lower(): return d
        return None
    exp = find("export") or find("invoice")
    if exp:
        if exp.get("assignee") in (None, "", "null"): s += 2; n.append("export assignee null (good)")
        else: n.append(f"HALLUCINATED export assignee: {exp.get('assignee')!r}")
        if exp.get("blocked_by") and "legal" in str(exp.get("blocked_by")).lower():
            s += 2; n.append("export blocked_by legal (good)")
        else: n.append(f"missed blocker: blocked_by={exp.get('blocked_by')!r}")
    else: n.append("export task missing")
    mig = find("migration")
    if mig:
        if str(mig.get("assignee", "")).lower().startswith("emily"): s += 2; n.append("migration->Emily")
        else: n.append(f"migration assignee wrong: {mig.get('assignee')!r}")
    else: n.append("migration task missing")
    pg = find("postgres")
    if pg:
        if pg.get("due") in (None, "", "null"): s += 2; n.append("postgres due null (good)")
        else: n.append(f"HALLUCINATED postgres due: {pg.get('due')!r}")
    else: n.append("postgres task missing")
    # Kevin is out until the 14th -> not a task; inventing a Kevin task or assignee is a fabrication
    if "kevin" in blob:
        k = find("kevin")
        if k and (str(k.get("assignee", "")).lower().startswith("kevin") or "out" in json.dumps(k).lower()):
            s -= 2; n.append("HALLUCINATION: turned 'Kevin is out' into a task/assignee")
    if re.search(r'"due":\s*"[^"]*(14th|14)"', blob) and not find("postgres"):
        pass
    return max(s, 0), n  # max 10

ROWS = []
for f in sorted(glob.glob(os.path.join(QDIR, "*.h1-absent-fact.txt"))):
    name = os.path.basename(f).replace(".h1-absent-fact.txt", "")
    r = {"model": name, "s": {}, "n": {}, "sec": 0.0, "tok": 0}
    tot = 0
    for key, fn, mx, suf in [("h1", h1, 6, "h1-absent-fact"), ("h2", h2, 8, "h2-conflict"),
                             ("h3", h3, 10, "h3-ambiguous-tasks")]:
        p = os.path.join(QDIR, f"{name}.{suf}.txt")
        if not os.path.exists(p): r["s"][key] = None; continue
        t = open(p).read()
        sec, tok = meta(t); r["sec"] += sec; r["tok"] += tok
        try: sc, nn = fn(t)
        except Exception as e: sc, nn = 0, [f"grader error: {e}"]
        r["s"][key] = f"{sc}/{mx}"; r["n"][key] = nn; tot += sc
    r["total"] = tot; ROWS.append(r)

ROWS.sort(key=lambda x: -x["total"])
hdr = f"{'model':32} {'h1':>5} {'h2':>5} {'h3':>6} {'total':>7} {'sec':>6} {'tok':>6}"
print(hdr); print("-" * len(hdr))
for r in ROWS:
    g = lambda k: (r["s"].get(k) or "-")
    print(f"{r['model'][:32]:32} {g('h1'):>5} {g('h2'):>5} {g('h3'):>6} {str(r['total'])+'/24':>7} {r['sec']:6.1f} {r['tok']:6d}")
print("\nfindings:")
for r in ROWS:
    bad = [x for k in ("h1", "h2", "h3") for x in r["n"].get(k, [])
           if any(w in x for w in ("HALLUC", "MISSING", "invalid", "missing", "wrong", "TOO LONG",
                                   "NO ANSWER", "drops", "does not", "unclear", "missed", "speculated"))]
    if bad: print(f"  {r['model'][:32]:32} " + "; ".join(bad))
