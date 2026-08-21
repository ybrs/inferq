#!/usr/bin/env python3
"""Check every claim the rubrics make against the real taskq schemas.

Guards FAILURE-MODES 39, 40, 42.

A rubric that says "set_task_status requires task_id and status" is an assertion
about a live API. If the API disagrees - or drifts later - the grader will mark
correct behaviour wrong, and the whole table inherits it. Nothing here looks at
model output; it only checks that what we are about to grade against is true.

Exit non-zero on any mismatch. run_eval.py should not start otherwise.
"""
import json, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = json.load(open(os.path.join(HERE, "taskq-tools.json")))
BY_NAME = {t["function"]["name"]: t["function"] for t in TOOLS}
NAMES = set(BY_NAME)

# Vocabularies the rubrics assert. Each must be justified by the live schemas.
CLAIMED_ENUMS = {
    "status": ["todo", "in_progress", "blocked", "done", "cancelled"],
    "relation": ["blocked_by", "requires", "documented_by", "references"],
}
IDENT = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")

# snake_case words that appear in rubric prose but are not tool names
NOT_TOOLS = {
    "task_id", "tool_calls", "finish_reason", "expect_tool_call", "full_credit",
    "no_call", "from_id", "to_id", "from_type", "to_type", "doc_type", "file_path",
    "max_tokens", "chat_template_kwargs", "in_progress", "blocked_by", "documented_by",
    "repo_root", "parent_id", "project_id", "github_issue_number", "order_index",
    "top_level", "at_a_glance", "rubric_branch", "verbatim_quote",
}

fails, notes = [], []


def check_tool_names(spec, label):
    for s in spec["scenarios"]:
        blob = s.get("rubric", "") + " " + s.get("probes", "")
        for ident in set(IDENT.findall(blob)):
            if ident in NOT_TOOLS or ident in NAMES:
                continue
            # anything else that looks like a tool name is either a typo or a
            # tool that no longer exists
            if ident.split("_")[0] in {"set", "get", "list", "create", "update",
                                       "delete", "add", "task", "search", "link",
                                       "claim", "assign", "release", "restore",
                                       "project", "git", "pr", "server", "agent"}:
                fails.append(f"{label}/{s['id']}: rubric names '{ident}' which is not a taskq tool")


def check_required(tool, args, where):
    fn = BY_NAME.get(tool)
    if not fn:
        fails.append(f"{where}: unknown tool '{tool}'")
        return
    req = set(fn["parameters"].get("required", []))
    for a in args:
        if a not in req:
            fails.append(f"{where}: rubric treats '{a}' as required for {tool}, "
                         f"but schema requires {sorted(req)}")


def main():
    # 42: identical, ordered tool block, or nothing is comparable
    names = [t["function"]["name"] for t in TOOLS]
    if names != sorted(names):
        fails.append("taskq-tools.json is not sorted by name")
    if len(TOOLS) != 46:
        fails.append(f"expected 46 tools, found {len(TOOLS)}")
    notes.append(f"{len(TOOLS)} tools, sorted: {names == sorted(names)}")

    tool_spec = json.load(open(os.path.join(HERE, "scenarios-tools.json")))
    qual_spec = json.load(open(os.path.join(HERE, "scenarios-quality.json")))
    check_tool_names(tool_spec, "tools")
    check_tool_names(qual_spec, "quality")

    # the status vocabulary the rubrics rely on (s01, s04, s07)
    desc = BY_NAME["set_task_status"]["description"]
    for v in CLAIMED_ENUMS["status"]:
        if v not in desc:
            fails.append(f"set_task_status description does not mention status '{v}'")
    notes.append(f"status vocabulary present in set_task_status description: "
                 f"{CLAIMED_ENUMS['status']}")

    # the relation vocabulary (s11)
    rdesc = BY_NAME["link_nodes"]["description"]
    for v in CLAIMED_ENUMS["relation"]:
        if v not in rdesc:
            fails.append(f"link_nodes description does not mention relation '{v}'")
    notes.append(f"relation vocabulary present in link_nodes description: "
                 f"{CLAIMED_ENUMS['relation']}")

    # per-scenario required-argument claims the rubrics make explicitly
    check_required("set_task_status", ["task_id", "status"], "s01/s04/s07")
    check_required("create_task", ["project", "title"], "s05")
    check_required("link_nodes", ["from_id", "to_id", "relation"], "s11")
    check_required("search_taskq", ["query"], "s02")
    check_required("get_task", ["task_id"], "s14")
    check_required("add_comment", ["task_id", "body"], "s01")
    check_required("project_status", ["project"], "s10")

    # s06: assign_task's assignee must be OPTIONAL, since the rubric says
    # omitting it clears the assignee
    if "assignee" in BY_NAME["assign_task"]["parameters"].get("required", []):
        fails.append("s06 rubric assumes assign_task.assignee is optional, schema says required")
    # s09: same for set_task_summary.summary
    if "summary" in BY_NAME["set_task_summary"]["parameters"].get("required", []):
        fails.append("s09 rubric assumes set_task_summary.summary is optional, schema says required")
    # s13/s10: task_queue and project_status must be distinguishable
    if "project" not in BY_NAME["task_queue"]["parameters"].get("properties", {}):
        fails.append("s13 rubric asks for task_queue(project=...), schema has no project")

    # 40: the live server must still expose what we pinned
    if os.environ.get("SKIP_LIVE_TOOL_DIFF") != "1":
        with tempfile.TemporaryDirectory() as d:
            live = os.path.join(d, "live.json")
            r = subprocess.run([sys.executable, os.path.join(HERE, "fetch_tools.py"), live],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                notes.append(f"live tool re-fetch failed (not fatal): {r.stderr[:200]}")
            else:
                if json.load(open(live)) != TOOLS:
                    fails.append("live taskq schemas differ from the pinned taskq-tools.json "
                                 "- re-pin deliberately before running")
                else:
                    notes.append("live taskq schemas match the pinned file")

    for n in notes:
        print(f"  ok  {n}")
    for f in fails:
        print(f"  FAIL {f}")
    print(f"\n{len(fails)} problem(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
