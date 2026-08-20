#!/usr/bin/env python3
"""Capability-suite grader (z1-z4): the mechanically checkable half of the
capability probes — DAG topology, script execution, Mermaid edges, chart code.

The translation probes (x*) and code-summary probes (y1, y2) need a reader and
are written up in capabilities.md instead. The file-routing probes (y3, y4) are
checked here for shape only: a path that is not in the listing is a fabrication,
and a duplicate wastes one of the three picks.

Usage: grade_cap.py [outputs-dir]   (default: outputs-cap/)
"""
import ast, csv, glob, json, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
QDIR = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "outputs-cap")

def strip(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = re.sub(r"^\[elapsed.*?\]\n", "", t, flags=re.S)
    return t.strip()

def unfence(t):
    m = re.search(r"```(?:json|python|mermaid)?\s*(.*?)```", t, re.S)
    return m.group(1).strip() if m else t

# --- the workflow every z-probe describes -------------------------------------
# two independent pulls -> one validate -> two independent loads -> rebuild -> email
def classify(node_id, label=""):
    # id first: a node's own run-text often says "after validation", which would
    # otherwise classify every downstream node as the validation step
    for l in (str(node_id).lower(), f"{node_id} {label}".lower()):
        k = _classify(l)
        if k: return k
    return None

def _classify(l):
    if "email" in l or "send" in l: return "email"
    if "valid" in l: return "validate"
    if "load" in l and "order" in l: return "load_orders"
    if "load" in l and "return" in l: return "load_returns"
    if "summary" in l or "rebuild" in l: return "rebuild"
    if "order" in l: return "pull_orders"
    if "return" in l: return "pull_returns"
    return None

EXPECTED_EDGES = {("pull_orders","validate"), ("pull_returns","validate"),
                  ("validate","load_orders"), ("validate","load_returns"),
                  ("load_orders","rebuild"), ("load_returns","rebuild"),
                  ("rebuild","email")}
EXPECTED_NODES = {"pull_orders","pull_returns","validate","load_orders","load_returns","rebuild","email"}

def score_graph(nodes, edges, pts_nodes, pts_edges):
    """nodes: {id: label}; edges: [(from_id, to_id)]. Returns (score, notes)."""
    n = []; s = 0
    kind = {i: classify(i, lab) for i, lab in nodes.items()}
    got = {k for k in kind.values() if k}
    s += pts_nodes * len(got & EXPECTED_NODES) // len(EXPECTED_NODES)
    missing = EXPECTED_NODES - got
    n.append(f"{len(got & EXPECTED_NODES)}/7 nodes" + (f", missing {sorted(missing)}" if missing else ""))
    sem = {(kind.get(a), kind.get(b)) for a, b in edges
           if kind.get(a) and kind.get(b) and kind.get(a) != kind.get(b)}
    hit = sem & EXPECTED_EDGES
    s += pts_edges * len(hit) // len(EXPECTED_EDGES)
    n.append(f"{len(hit)}/7 edges")
    wrong = sem - EXPECTED_EDGES
    if wrong: n.append("INVENTED EDGES: " + ", ".join(f"{a}->{b}" for a, b in sorted(wrong)))
    # the failure this suite is looking for: parallel branches serialised into a chain
    if ("load_orders","load_returns") in sem or ("load_returns","load_orders") in sem \
       or ("pull_orders","pull_returns") in sem or ("pull_returns","pull_orders") in sem:
        n.append("LINEARISED: chained two independent branches")
    return s, n

def z1(txt):  # DAG as JSON, /10
    b = unfence(strip(txt))
    if not b: return 0, ["NO ANSWER EMITTED"]
    try: d = json.loads(b)
    except Exception as e: return 0, [f"invalid JSON ({type(e).__name__})"]
    if not isinstance(d, dict) or "nodes" not in d or "edges" not in d:
        return 0, ["not a {nodes, edges} object"]
    nodes = {}
    for x in d["nodes"]:
        if isinstance(x, dict) and "id" in x: nodes[str(x["id"])] = str(x.get("run", ""))
    edges = [(str(a), str(b_)) for e in d["edges"] if isinstance(e, (list, tuple)) and len(e) == 2
             for a, b_ in [e]]
    s, n = score_graph(nodes, edges, 4, 4)
    return s + 2, ["valid JSON DAG"] + n

def z3(txt):  # mermaid, /8
    b = unfence(strip(txt))
    if not b: return 0, ["NO ANSWER EMITTED"]
    if "flowchart" not in b.split("\n")[0].lower():
        return 0, ["does not start with a flowchart header"]
    body = "\n".join(b.split("\n")[1:])
    if re.search(r"^\s*flowchart\s*$", body, re.M):
        return 0, ["DEGENERATE: repeats the header instead of drawing"]
    labels = dict(re.findall(r"([A-Za-z_][\w]*)\s*[\[\({]([^\]\)}]*)[\]\)}]", body))
    edges = [(a, b_) for a, _, b_ in re.findall(r"([A-Za-z_][\w]*)\s*(\[[^\]]*\]|\([^)]*\)|\{[^}]*\})?\s*-[-.=]*>\s*(?:\|[^|]*\|\s*)?([A-Za-z_][\w]*)", body)]
    if not edges: return 2, ["parses, but NO EDGES — a list, not a graph"]
    nodes = {i: labels.get(i, i) for i in {x for e in edges for x in e}}
    s, n = score_graph(nodes, edges, 3, 3)
    return s + 2, [f"{len(edges)} edges drawn"] + n

FIXTURE = [["id","status","amount"],["1","open","10.50"],["2","closed","3.25"],
           ["3","open","1.25"],["4","pending","7.00"],["5","closed","0.75"]]
EXPECT = ["closed: 2 orders, 4.00 total", "open: 2 orders, 11.75 total", "pending: 1 orders, 7.00 total"]

def z2(txt):  # CSV aggregation script, /8
    code = unfence(strip(txt)); n = []; s = 0
    if not code: return 0, ["NO ANSWER EMITTED"]
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "s.py"); open(sp, "w").write(code)
        if subprocess.run([sys.executable, "-m", "py_compile", sp], capture_output=True).returncode:
            return 0, ["does not compile"]
        s += 2; n.append("compiles")
        cp = os.path.join(d, "f.csv")
        with open(cp, "w", newline="") as f: csv.writer(f).writerows(FIXTURE)
        try: r = subprocess.run([sys.executable, sp, cp], capture_output=True, timeout=30, text=True, cwd=d)
        except subprocess.TimeoutExpired: return s, n + ["TIMEOUT"]
        out = [x.strip() for x in r.stdout.strip().split("\n") if x.strip()]
        if out == EXPECT: s += 4; n.append("correct output")
        else: n.append(f"WRONG output: {(r.stderr.strip()[:60] or out)!r}")
        bp = os.path.join(d, "bad.csv")
        with open(bp, "w", newline="") as f: csv.writer(f).writerows([["id","amount"],["1","2.00"]])
        try: r2 = subprocess.run([sys.executable, sp, bp], capture_output=True, timeout=30, text=True, cwd=d)
        except subprocess.TimeoutExpired: return s, n + ["TIMEOUT on missing-column path"]
        if r2.returncode == 1 and r2.stderr.strip(): s += 2; n.append("missing column -> exit 1 + stderr")
        else: n.append(f"missing-column path BAD (exit {r2.returncode}, stderr={bool(r2.stderr.strip())})")
    return s, n

def z4(txt):  # matplotlib chart, /6
    code = unfence(strip(txt)); n = []; s = 0
    if not code: return 0, ["NO ANSWER EMITTED"]
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "p.py"); open(sp, "w").write(code)
        if subprocess.run([sys.executable, "-m", "py_compile", sp], capture_output=True).returncode:
            return 0, ["does not compile"]
        s += 2; n.append("compiles")
        png = os.path.join(d, "out.png")
        env = dict(os.environ, MPLBACKEND="Agg")
        py = os.environ.get("MPL_PYTHON", sys.executable)  # a venv with matplotlib installed
        try: r = subprocess.run([py, sp, png], capture_output=True, timeout=120, text=True, env=env, cwd=d)
        except subprocess.TimeoutExpired: return s, n + ["TIMEOUT"]
        stray = [x for x in os.listdir(d) if x.endswith(".png") and x != os.path.basename(png)]
        if os.path.exists(png) and open(png, "rb").read(8) == b"\x89PNG\r\n\x1a\n":
            s += 3; n.append(f"renders a PNG ({os.path.getsize(png)} bytes)")
        elif stray:
            s += 1; n.append(f"IGNORES THE PATH ARGUMENT: wrote {stray[0]} instead")
        else:
            last = r.stderr.strip().splitlines()[-1][:70] if r.stderr.strip() else "silent"
            if "No module named 'matplotlib'" in last:
                n.append("SKIPPED: no matplotlib — set MPL_PYTHON to an interpreter that has it")
            else: n.append(f"NO PNG WRITTEN: {last}")
        if re.search(r"(set_yscale|yscale)\s*\(\s*['\"]log", code): s += 1; n.append("log y axis")
        else: n.append("missing log y axis")
    return s, n

LISTING = {"src/config.rs","src/gguf.rs","src/loader.rs","src/ngram.rs","src/profile.rs",
           "src/qgemm.rs","src/runtime.rs","src/sampling.rs","src/speculative.rs",
           "src/tokenizer.rs","src/tool_calls.rs","src/trace.rs"}

def routing(txt, expect):
    """Shape check only: fabricated paths and duplicates. /4"""
    b = unfence(strip(txt)); n = []; s = 0
    if not b: return 0, ["NO ANSWER EMITTED"]
    try: picks = json.loads(b)
    except Exception as e: return 0, [f"invalid JSON ({type(e).__name__})"]
    if not isinstance(picks, list): return 0, ["not a JSON array"]
    picks = [str(p) for p in picks]
    s += 1; n.append(f"{len(picks)} picks")
    if len(picks) != 3: n.append(f"WRONG COUNT: asked for 3, got {len(picks)}")
    else: s += 1
    ghosts = [p for p in picks if p not in LISTING]
    if ghosts: n.append(f"FABRICATED PATHS not in the listing: {ghosts}")
    else: s += 1
    if len(set(picks)) != len(picks): n.append("DUPLICATE pick wastes a slot")
    elif expect in picks: s += 1; n.append(f"includes {expect}")
    elif len(set(picks)) == len(picks) and expect not in picks: n.append(f"MISSES {expect}")
    return s, n

PROBES = [("z1", z1, 10, "z1-dag-json"), ("z2", z2, 8, "z2-python-script"),
          ("z3", z3, 8, "z3-mermaid"), ("z4", z4, 6, "z4-matplotlib"),
          ("y3", lambda t: routing(t, "src/runtime.rs"), 4, "y3-file-routing"),
          ("y4", lambda t: routing(t, "src/tool_calls.rs"), 4, "y4-file-routing-2")]

ROWS = []
for f in sorted(glob.glob(os.path.join(QDIR, "*.z1-dag-json.txt"))):
    name = os.path.basename(f).replace(".z1-dag-json.txt", "")
    r = {"model": name, "s": {}, "n": {}}; tot = 0
    for key, fn, mx, suf in PROBES:
        p = os.path.join(QDIR, f"{name}.{suf}.txt")
        if not os.path.exists(p): r["s"][key] = None; continue
        try: sc, nn = fn(open(p).read())
        except Exception as e: sc, nn = 0, [f"grader error: {e}"]
        r["s"][key] = f"{sc}/{mx}"; r["n"][key] = nn; tot += sc
    r["total"] = tot; ROWS.append(r)

ROWS.sort(key=lambda x: -x["total"])
hdr = f"{'model':30} {'dag':>6} {'script':>7} {'mermaid':>8} {'chart':>6} {'route1':>7} {'route2':>7} {'total':>7}"
print(hdr); print("-" * len(hdr))
for r in ROWS:
    g = lambda k: (r["s"].get(k) or "-")
    print(f"{r['model'][:30]:30} {g('z1'):>6} {g('z2'):>7} {g('z3'):>8} {g('z4'):>6} "
          f"{g('y3'):>7} {g('y4'):>7} {str(r['total'])+'/40':>7}")
print("\nfindings:")
for r in ROWS:
    print(f"  {r['model']}")
    for k, _, _, _ in PROBES:
        for x in r["n"].get(k, []):
            if any(w in x for w in ("INVENTED","LINEAR","WRONG","invalid","DEGENERATE","NO ",
                                    "FABRICATED","DUPLICATE","MISSES","missing","does not","BAD","TIMEOUT")):
                print(f"    {k}: {x}")
