#!/usr/bin/env python3
"""Grading driver. Builds anonymised packets for LLM graders, then validates
what they return. Guards FAILURE-MODES 35, 41, 44, 45, 46, 47, 49, 50.

The previous evaluation graded with regexes; those are in attic/ and this is the
only entry point that may be used against new results. Its output is stamped
`grading_method: llm-rubric-v2` and report.py refuses grade files without it.

Split into two commands because the grading itself is done by LLM agents, not by
this script:

  prepare <results-dir> --pass N
      Partitions every run's answer by state - only ANSWERED and TRUNCATED are
      gradeable; ERROR, TIMEOUT, NOT_RUN and NOT_APPLICABLE are carried through
      as coverage facts and must never become a score of zero. Shuffles the
      gradeable answers under a recorded seed, relabels them answer-01.., mixes
      in the two calibration probes, and writes one self-contained packet per
      scenario. The label->run mapping is written to a separate keys file that
      the grader never sees.

  collect <results-dir> --pass N
      Reads the graders' output and refuses it unless: the label set matches
      exactly, every verbatim_quote really is a substring of the answer it is
      attributed to, and both calibration probes landed where they had to.
      Then de-anonymises.

  merge <results-dir>
      Compares pass 1 and pass 2, flags disagreements, and writes grades-final.json.
"""
import argparse, hashlib, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRADEABLE = ("ANSWERED", "TRUNCATED")

INSTRUCTIONS = """You are grading answers from small local models against one rubric.

Rules that override any instinct to be generous or harsh:

1. Apply ONLY the rubric below. Do not invent criteria. Do not reward length,
   politeness, or formatting the rubric does not ask for.
2. For tool scenarios, the `tool_calls` array is authoritative. A tool call
   exists if and only if that array is non-empty. Text that merely NAMES a tool
   is NOT a tool call. An empty array means no call was made - which for some
   scenarios is the correct answer.
3. `finish_reason: "length"` means the answer was cut off by a token budget, not
   that the model refused or got it wrong. Score what is there and set
   "truncated": true.
4. The answers are anonymous and shuffled. If an answer names a model, ignore it.
5. Score every label you are given, exactly once. Do not add labels. Do not skip
   any, including ones you find hard to judge.

Return ONLY a JSON object, no prose around it:

{"grades": [
  {"label": "answer-01",
   "score": <number, per the rubric's scale>,
   "max_score": <the rubric's maximum>,
   "rubric_branch": "<which rubric clause you applied, quoted or paraphrased in <=12 words>",
   "verbatim_quote": "<<=15 words copied EXACTLY from that answer, as evidence>",
   "truncated": <true|false>,
   "justification": "<one sentence>"}
]}

The verbatim_quote must be a literal substring of the answer you are scoring; it
is checked mechanically and a mismatch discards your whole batch. For an answer
whose content is empty and whose evidence is a tool call, quote from the tool
call JSON shown to you.
"""


def load_runs(root):
    runs = {}
    for tag in sorted(os.listdir(root)):
        p = os.path.join(root, tag, "_run.json")
        if os.path.isfile(p):
            try:
                runs[tag] = json.load(open(p))
            except Exception as e:
                print(f"  WARN unreadable {p}: {e}", file=sys.stderr)
    return runs


def answer_view(scenario_id, res, q3exec=None):
    """What the grader sees. Tool calls pre-parsed so every grader reads them
    identically (41); nothing else about the response is summarised away."""
    v = {"finish_reason": res.get("finish_reason"),
         "content": res.get("content", ""),
         "truncated": res.get("state") == "TRUNCATED"}
    if scenario_id.startswith("s"):
        v["tool_calls"] = res.get("tool_calls", [])
        if res.get("arguments_parse_errors"):
            v["argument_parse_errors"] = res["arguments_parse_errors"]
    if scenario_id == "q3-python-script" and q3exec is not None:
        v["execution_result"] = q3exec
    return v


def cmd_prepare(a):
    root = a.results
    runs = load_runs(root)
    if not runs:
        sys.exit(f"no _run.json found under {root}")
    tool_spec = json.load(open(os.path.join(HERE, "scenarios-tools.json")))
    qual_spec = json.load(open(os.path.join(HERE, "scenarios-quality.json")))
    calib = json.load(open(os.path.join(HERE, "calibration.json")))
    q3path = os.path.join(root, "q3-execution.json")
    q3 = json.load(open(q3path)) if os.path.exists(q3path) else None
    if q3 is None:
        print("  WARN q3-execution.json missing - run run_fixtures.py first (43)",
              file=sys.stderr)

    pdir = os.path.join(root, "grading", f"pass{a.pass_no}", "packets")
    kdir = os.path.join(root, "grading", f"pass{a.pass_no}", "keys")
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(kdir, exist_ok=True)

    scenarios = [(s, tool_spec["system"]) for s in tool_spec["scenarios"]] + \
                [(s, qual_spec["system"]) for s in qual_spec["scenarios"]]
    summary = []
    for s, system in scenarios:
        sid = s["id"]
        rng = random.Random(f"{a.seed}:{a.pass_no}:{sid}")
        gradeable, coverage = [], {}
        for tag, run in runs.items():
            res = (run.get("results") or {}).get(sid)
            if res is None:
                coverage[tag] = "NOT_RUN"
                continue
            st = res.get("state")
            if st in GRADEABLE:
                gradeable.append((tag, answer_view(sid, res, (q3 or {}).get(tag))))
            else:
                coverage[tag] = st or "UNKNOWN"       # 35: never a zero

        items = [{"kind": "real", "tag": t, "view": v} for t, v in gradeable]
        probe = calib.get(sid)
        if probe:                                      # 47
            for k in ("full", "zero"):
                items.append({"kind": f"probe-{k}", "tag": f"__probe_{k}__",
                              "view": dict(probe[k], finish_reason="stop", truncated=False)})
        rng.shuffle(items)                             # 44
        labels = {}
        packet_answers = []
        for i, it in enumerate(items, 1):
            lab = f"answer-{i:02d}"
            labels[lab] = {"kind": it["kind"], "tag": it["tag"]}
            packet_answers.append({"label": lab, **it["view"]})

        packet = {
            "scenario_id": sid,
            "probes": s.get("probes"),
            "system_prompt_the_models_saw": system,
            "user_messages": s["messages"],
            "expect_tool_call": s.get("expect_tool_call"),
            "rubric": s["rubric"],
            "instructions": INSTRUCTIONS,
            "answers": packet_answers,
        }
        json.dump(packet, open(os.path.join(pdir, f"{sid}.json"), "w"), indent=1)
        json.dump({"scenario_id": sid, "seed": f"{a.seed}:{a.pass_no}:{sid}",
                   "labels": labels, "coverage": coverage},
                  open(os.path.join(kdir, f"{sid}.json"), "w"), indent=1)
        summary.append((sid, len(gradeable), len(coverage), bool(probe)))

    print(f"pass {a.pass_no}: {len(scenarios)} packets -> {pdir}")
    for sid, n, nc, p in summary:
        print(f"  {sid:30} gradeable={n:3}  not-gradeable={nc:3}  probes={'yes' if p else 'no'}")


def cmd_collect(a):
    root = a.results
    base = os.path.join(root, "grading", f"pass{a.pass_no}")
    gdir, kdir, pdir = (os.path.join(base, x) for x in ("grades", "keys", "packets"))
    out, problems = {}, []
    for f in sorted(os.listdir(kdir)):
        sid = f[:-5]
        key = json.load(open(os.path.join(kdir, f)))
        packet = json.load(open(os.path.join(pdir, f)))
        gpath = os.path.join(gdir, f)
        if not os.path.exists(gpath):
            problems.append(f"{sid}: no grades returned")
            continue
        try:
            grades = json.load(open(gpath))["grades"]
        except Exception as e:
            problems.append(f"{sid}: unreadable grades ({e})")
            continue

        got = {g["label"] for g in grades}
        want = set(key["labels"])
        if got != want:                                     # 45
            problems.append(f"{sid}: label set mismatch "
                            f"missing={sorted(want-got)} extra={sorted(got-want)}")
            continue
        answers = {x["label"]: x for x in packet["answers"]}
        bad_quotes = []
        for g in grades:
            hay = json.dumps(answers[g["label"]], ensure_ascii=False)
            q = (g.get("verbatim_quote") or "").strip()
            if q and q not in hay:                          # 45
                bad_quotes.append(g["label"])
        if bad_quotes:
            problems.append(f"{sid}: verbatim_quote not found in answer for {bad_quotes}")
            continue

        by_label = {g["label"]: g for g in grades}
        cal = {}
        for lab, meta in key["labels"].items():
            if meta["kind"] == "probe-full":
                g = by_label[lab]
                cal["full"] = (g["score"], g.get("max_score"))
                if g.get("max_score") and g["score"] < 0.9 * g["max_score"]:
                    problems.append(f"{sid}: calibration FULL probe scored "
                                    f"{g['score']}/{g.get('max_score')} - grader is miscalibrated")
            elif meta["kind"] == "probe-zero":
                g = by_label[lab]
                cal["zero"] = (g["score"], g.get("max_score"))
                if g["score"] > 0.15 * (g.get("max_score") or 10):
                    problems.append(f"{sid}: calibration ZERO probe scored "
                                    f"{g['score']}/{g.get('max_score')} - grader is miscalibrated")

        scores = {}
        for lab, meta in key["labels"].items():
            if meta["kind"] != "real":
                continue
            g = by_label[lab]
            scores[meta["tag"]] = {k: g.get(k) for k in
                                   ("score", "max_score", "rubric_branch",
                                    "verbatim_quote", "truncated", "justification")}
        out[sid] = {"scores": scores, "coverage": key["coverage"],
                    "calibration": cal, "seed": key["seed"]}

    dest = os.path.join(base, "collected.json")
    json.dump({"grading_method": "llm-rubric-v2", "pass": a.pass_no,
               "grader_model": a.grader, "scenarios": out}, open(dest, "w"), indent=1)
    print(f"collected {len(out)} scenarios -> {dest}")
    for p in problems:
        print(f"  PROBLEM {p}")
    return 1 if problems else 0


def cmd_merge(a):
    root = a.results
    passes = []
    for n in (1, 2):
        p = os.path.join(root, "grading", f"pass{n}", "collected.json")
        if os.path.exists(p):
            passes.append(json.load(open(p)))
    if not passes:
        sys.exit("no collected.json in any pass")
    if len(passes) == 1:
        print("WARNING: only one grading pass; disagreement cannot be measured (46)")

    final, disagreements, cells = {}, [], 0
    for sid in passes[0]["scenarios"]:
        a1 = passes[0]["scenarios"][sid]
        a2 = passes[1]["scenarios"].get(sid) if len(passes) > 1 else None
        merged = {}
        for tag, g1 in a1["scores"].items():
            cells += 1
            s1 = g1["score"]
            if a2 and tag in a2["scores"]:
                s2 = a2["scores"][tag]["score"]
                delta = abs(s1 - s2)
                merged[tag] = {"score": (s1 + s2) / 2, "pass1": s1, "pass2": s2,
                               "delta": delta, "max_score": g1.get("max_score"),
                               "evidence": g1.get("verbatim_quote"),
                               "rubric_branch": g1.get("rubric_branch"),
                               "justification": g1.get("justification"),
                               "truncated": g1.get("truncated")}
                if delta > 1:
                    disagreements.append((sid, tag, s1, s2))
            else:
                merged[tag] = {"score": s1, "pass1": s1, "pass2": None, "delta": None,
                               "max_score": g1.get("max_score"),
                               "evidence": g1.get("verbatim_quote"),
                               "rubric_branch": g1.get("rubric_branch"),
                               "justification": g1.get("justification"),
                               "truncated": g1.get("truncated")}
        final[sid] = {"scores": merged, "coverage": a1["coverage"],
                      "calibration": a1.get("calibration")}

    rate = len(disagreements) / cells if cells else 0
    dest = os.path.join(root, "grades-final.json")
    json.dump({"grading_method": "llm-rubric-v2",
               "grader_model": passes[0].get("grader_model"),
               "n_passes": len(passes), "cells": cells,
               "disagreement_rate": round(rate, 4),
               "disagreements": disagreements,
               "scenarios": final}, open(dest, "w"), indent=1)
    print(f"merged {cells} cells -> {dest}")
    print(f"disagreement rate: {rate:.1%} ({len(disagreements)} cells differ by >1)")
    if rate > 0.10:                                          # 46
        print("  DISAGREEMENT RATE ABOVE 10% - grading configuration is not trustworthy")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "collect"):
        s = sub.add_parser(name)
        s.add_argument("results")
        s.add_argument("--pass", dest="pass_no", type=int, default=1)
        s.add_argument("--seed", default="scout-v2")
        s.add_argument("--grader", default="claude-haiku-4-5-20251001")
    s = sub.add_parser("merge")
    s.add_argument("results")
    a = ap.parse_args()
    return {"prepare": cmd_prepare, "collect": cmd_collect, "merge": cmd_merge}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
