#!/usr/bin/env python3
"""Tool-selection grader (w1-w3): can a model pick the right MCP call?

The listing is the 46 taskq tools in tests-capability/taskq-tools.txt. Three
requests, each with one obviously correct tool and several near-misses sharing a
naming convention (set_task_status / set_task_summary / update_task, search_taskq
/ list_tasks / task_queue).

Two failure modes are scored separately because they behave differently in
production:
  - a FABRICATED name fails closed: the MCP server rejects it, the agent sees an
    error and can retry.
  - an EXTRA call to a real side-effecting tool fails open: it runs. task_to_issue
    opens a GitHub issue nobody asked for; create_task on an update request makes
    a duplicate ticket.

Usage: grade_tools.py [outputs-dir]   (default: outputs-tools/)
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
QDIR = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "outputs-tools")
LISTING = {l.split(" — ")[0].strip() for l in
           open(os.path.join(HERE, "tests-capability", "taskq-tools.txt")) if " — " in l}

# tools that change something outside taskq, or create records, if called wrongly
SIDE_EFFECTS = {"create_task", "delete_task", "delete_comment", "delete_document",
                "task_to_issue", "create_pull_request", "git_push", "git_push_all",
                "restart_server", "restart_agent", "claim_task", "assign_task",
                "create_project", "create_document", "set_handoff", "link_nodes"}

PROBES = {
    "w1": dict(suf="w1-tool-update", need={"set_task_status", "update_task"},
               ok={"set_task_status", "update_task", "add_comment", "get_task",
                   "list_comments", "set_task_summary"}),
    "w2": dict(suf="w2-tool-search", need={"search_taskq"},
               ok={"search_taskq", "list_tasks", "get_task", "task_queue", "project_status"}),
    "w3": dict(suf="w3-tool-create", need={"create_task"},
               ok={"create_task", "get_project", "list_projects"}),
}

def strip(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = re.sub(r"^\[elapsed.*?\]\n", "", t, flags=re.S)
    return t.strip()

def unfence(t):
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    return m.group(1).strip() if m else t

def score(txt, need, ok):
    b = unfence(strip(txt)); n = []; s = 0
    if not b: return 0, ["NO ANSWER EMITTED"]
    try: picks = json.loads(b)
    except Exception as e: return 0, [f"invalid JSON ({type(e).__name__})"]
    if not isinstance(picks, list): return 0, ["not a JSON array"]
    picks = [str(p).strip() for p in picks]
    s += 1; n.append(f"picked {picks}")
    ghosts = [p for p in picks if p not in LISTING]
    if ghosts: n.append(f"FABRICATED TOOL: {ghosts} (fails closed — server rejects it)")
    else: s += 2
    if set(picks) & need: s += 2; n.append(f"found {sorted(set(picks) & need)[0]}")
    else: n.append(f"MISSES the tool for the job ({'/'.join(sorted(need))})")
    stray = [p for p in picks if p in LISTING and p not in ok]
    danger = [p for p in stray if p in SIDE_EFFECTS]
    if danger: n.append(f"UNREQUESTED SIDE EFFECT: {danger} (fails open — it runs)")
    elif stray: n.append(f"extra calls: {stray}")
    else: s += 1
    return s, n

ROWS = []
for f in sorted(glob.glob(os.path.join(QDIR, "*.w1-tool-update.txt"))):
    name = os.path.basename(f).replace(".w1-tool-update.txt", "")
    r = {"model": name, "s": {}, "n": {}}; tot = 0
    for key, cfg in PROBES.items():
        p = os.path.join(QDIR, f"{name}.{cfg['suf']}.txt")
        if not os.path.exists(p): r["s"][key] = None; continue
        sc, nn = score(open(p).read(), cfg["need"], cfg["ok"])
        r["s"][key] = f"{sc}/6"; r["n"][key] = nn; tot += sc
    r["total"] = tot; ROWS.append(r)

ROWS.sort(key=lambda x: -x["total"])
hdr = f"{'model':30} {'update':>7} {'search':>7} {'create':>7} {'total':>7}"
print(hdr); print("-" * len(hdr))
for r in ROWS:
    g = lambda k: (r["s"].get(k) or "-")
    print(f"{r['model'][:30]:30} {g('w1'):>7} {g('w2'):>7} {g('w3'):>7} {str(r['total'])+'/18':>7}")
print("\nfindings:")
for r in ROWS:
    print(f"  {r['model']}")
    for k in PROBES:
        for x in r["n"].get(k, []):
            if any(w in x for w in ("FABRICATED", "MISSES", "UNREQUESTED", "invalid", "NO ", "extra")):
                print(f"    {k}: {x}")
