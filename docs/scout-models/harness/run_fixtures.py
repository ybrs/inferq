#!/usr/bin/env python3
"""Execute the q3 Python scripts each model wrote, against fixtures.

Guards FAILURE-MODES 43. The q3 rubric tells the grader "the execution result is
supplied to you and is authoritative" - so it has to actually exist, or the
grader invents a judgment about code it never ran.

Runs outside the repo, with no network and a hard timeout, because this is
model-written code.

Usage: run_fixtures.py <results-dir>
"""
import csv, json, os, subprocess, sys, tempfile

SCRATCH = os.environ.get("SCRATCHPAD",
    "/tmp/claude-1000/-workspace/62d6604f-a22b-42cb-b619-7e4ad5686d32/scratchpad")
GOOD = [["id", "status"], ["1", "open"], ["2", "closed"], ["3", "open"],
        ["4", "open"], ["5", "closed"], ["6", "pending"]]
BAD = [["id", "state"], ["1", "x"]]
WANT = ["open: 3", "closed: 2", "pending: 1"]


def unfence(t):
    """Take the code out of a markdown fence if there is one."""
    if "```" in t:
        parts = t.split("```")
        for p in parts[1:]:
            body = p.split("\n", 1)
            if len(body) == 2:
                head, rest = body
                if head.strip() in ("", "python", "py", "python3"):
                    return rest.rsplit("```", 1)[0].strip()
    return t.strip()


def execute(code, tag):
    d = os.path.join(SCRATCH, "q3", tag)
    os.makedirs(d, exist_ok=True)
    sp = os.path.join(d, "script.py")
    open(sp, "w").write(code)
    res = {"code_chars": len(code)}

    c = subprocess.run([sys.executable, "-m", "py_compile", sp],
                       capture_output=True, text=True, timeout=60)
    res["compiles"] = (c.returncode == 0)
    if not res["compiles"]:
        res["compile_error"] = c.stderr[-400:]
        return res

    good = os.path.join(d, "good.csv")
    with open(good, "w", newline="") as f:
        csv.writer(f).writerows(GOOD)
    bad = os.path.join(d, "bad.csv")
    with open(bad, "w", newline="") as f:
        csv.writer(f).writerows(BAD)

    def run(args, label):
        try:
            r = subprocess.run([sys.executable, sp] + args, capture_output=True,
                               text=True, timeout=10, cwd=d,
                               env={"PATH": "/usr/bin:/bin", "HOME": d})
            return {"exit": r.returncode, "stdout": r.stdout[-2000:],
                    "stderr": r.stderr[-1000:]}
        except subprocess.TimeoutExpired:
            return {"exit": None, "state": "TIMEOUT", "stdout": "", "stderr": ""}
        except Exception as e:
            return {"exit": None, "state": f"{type(e).__name__}: {e}",
                    "stdout": "", "stderr": ""}

    res["good_csv"] = run([good], "good")
    res["missing_column_csv"] = run([bad], "bad")
    res["no_args"] = run([], "noargs")

    out = res["good_csv"]["stdout"].strip()
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    res["good_exact_match"] = (lines == WANT)
    res["good_counts_present"] = all(w in out for w in WANT)
    b = res["missing_column_csv"]
    res["missing_column_exit_1"] = (b.get("exit") == 1)
    res["missing_column_stderr"] = bool((b.get("stderr") or "").strip())
    return res


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: run_fixtures.py <results-dir>")
    out = {}
    for tag in sorted(os.listdir(root)):
        p = os.path.join(root, tag, "q3-python-script.json")
        if not os.path.isfile(p):
            continue
        d = json.load(open(p))
        if d.get("state") not in ("ANSWERED", "TRUNCATED"):
            out[tag] = {"state": d.get("state"), "executed": False}
            continue
        code = unfence(d.get("content") or "")
        if not code:
            out[tag] = {"state": "EMPTY", "executed": False}
            continue
        try:
            out[tag] = {"state": d.get("state"), "executed": True, **execute(code, tag)}
        except Exception as e:
            out[tag] = {"state": d.get("state"), "executed": False,
                        "harness_error": f"{type(e).__name__}: {e}"}
        r = out[tag]
        print(f"  {tag:46} compiles={r.get('compiles')} "
              f"exact={r.get('good_exact_match')} exit1={r.get('missing_column_exit_1')}")
    dest = os.path.join(root, "q3-execution.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nwrote {dest} ({len(out)} runs)")


if __name__ == "__main__":
    main()
