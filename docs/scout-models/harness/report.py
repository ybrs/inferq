#!/usr/bin/env python3
"""Turn a results directory into the report. Guards FAILURE-MODES 22, 23, 35, 48, 49, 50.

Refuses inputs it cannot stand behind:
  * grade files not stamped grading_method: llm-rubric-v2 (the regex graders in
    attic/ produced no such stamp)
  * a prefill rate taken from anything but a genuinely cold >=5000-token prefill
  * a decode rate whose context depth is not carried alongside it
  * any attempt to sum raw scores across scenarios with different scales

Every cell in the capability table carries its coverage, so a model that errored
is never printed as a zero.

Usage: report.py <results-dir> [--out report.md]
"""
import argparse, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DECODE_SCENARIO = "s01-status-and-note"   # fixed, so every decode cell is one depth


def load(root):
    runs = {}
    for tag in sorted(os.listdir(root)):
        p = os.path.join(root, tag, "_run.json")
        if os.path.isfile(p):
            runs[tag] = json.load(open(p))
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    gpath = os.path.join(root, "grades-final.json")
    grades = json.load(open(gpath)) if os.path.exists(gpath) else None
    if grades and grades.get("grading_method") != "llm-rubric-v2":   # 49
        sys.exit("grades-final.json is not stamped llm-rubric-v2 - refusing to report")
    return runs, manifest, grades


def prefill_of(run):
    """22: only a real cold prefill counts as a prefill rate."""
    c = run.get("cold_prefill")
    if not c or c.get("state") != "ANSWERED":
        return None, run.get("speed_invalid") or (c or {}).get("why") or "not measured"
    if (c.get("prompt_n") or 0) < 5000:
        return None, f"prompt_n={c.get('prompt_n')} - not a cold prefill"
    return c["prefill_tps"], None


def decode_of(run):
    """23: one designated scenario, and the depth reported with it."""
    r = (run.get("results") or {}).get(DECODE_SCENARIO)
    if not r or r.get("state") not in ("ANSWERED", "TRUNCATED"):
        return None, None
    t = r.get("timings") or {}
    depth = (t.get("prompt_n") or 0) + (t.get("cache_n") or 0)
    return round(t.get("predicted_per_second") or 0, 2), depth


def rank_aggregate(grades, scenario_ids, tags):
    """48: never sum raw scores across scenarios. Rank within each scenario, then
    average the ranks. A 0-6 scenario and a 0-10 scenario cannot be added."""
    ranks = {t: [] for t in tags}
    for sid in scenario_ids:
        sc = (grades["scenarios"].get(sid) or {}).get("scores", {})
        pairs = [(t, sc[t]["score"] / (sc[t].get("max_score") or 1))
                 for t in tags if t in sc and sc[t].get("score") is not None]
        if len(pairs) < 2:
            continue
        pairs.sort(key=lambda x: -x[1])
        # average rank for ties
        i = 0
        while i < len(pairs):
            j = i
            while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
                j += 1
            r = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[pairs[k][0]].append(r)
            i = j + 1
    return {t: (round(statistics.mean(v), 2) if v else None, len(v)) for t, v in ranks.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs, manifest, grades = load(a.results)
    tool_ids = [s["id"] for s in json.load(open(os.path.join(HERE, "scenarios-tools.json")))["scenarios"]]
    qual_ids = [s["id"] for s in json.load(open(os.path.join(HERE, "scenarios-quality.json")))["scenarios"]]

    L = []
    w = L.append
    w("# Small-model scout evaluation\n")
    cfg = manifest["config"]
    w(f"llama.cpp `{manifest['llama_server_version'][0] if manifest['llama_server_version'] else '?'}`, "
      f"harness `{manifest['git_sha']}`, "
      f"`-np 1 -fa {cfg['fa']} -lm none`, {cfg['threads']} threads pinned to CPUs {cfg['cpus']}, "
      f"temperature 0, seed {cfg['sampler']['seed']}.\n")
    h = manifest["host_at_start"]
    w(f"Host at start: governor `{h['governors']['cpu0']['scaling_governor']}`, "
      f"turbo {'on' if h['governors']['no_turbo'] == '0' else 'OFF'}, "
      f"{h['cpu_mhz_mean']} MHz, {h['coretemp_c']}&deg;C.\n")

    w("\n## Speed\n")
    w("Prefill is the cold ingest of the 46-tool block (>=5000 tokens, cache empty). "
      "Decode is measured on one fixed scenario so every cell is at the same depth; "
      "that depth is printed. **These are not comparable to `tg128`**, which is decode "
      "at 128 tokens of context - a condition a scout with tools loaded is never in.\n")
    w("| run | GiB | load s | cold prefill t/s | decode t/s | depth | canary t/s | noisy |")
    w("|---|---:|---:|---:|---:|---:|---:|:--:|")
    for tag, r in sorted(runs.items()):
        pf, why = prefill_of(r)
        dc, depth = decode_of(r)
        w(f"| {tag} | {r['size_bytes']/2**30:.2f} | {r.get('load_seconds','-')} "
          f"| {pf if pf else '_' + str(why)[:28] + '_'} | {dc if dc else '-'} "
          f"| {depth if depth else '-'} | {r.get('decode_canary_median','-')} "
          f"| {'yes' if r.get('timing_noisy') else ''} |")

    if grades:
        w(f"\n## Capability\n")
        w(f"Graded by `{grades.get('grader_model')}` against the written rubrics, "
          f"{grades['n_passes']} independent passes with different shuffles. "
          f"Disagreement rate {grades['disagreement_rate']:.1%} "
          f"({len(grades['disagreements'])} of {grades['cells']} cells differ by more than 1 point).\n")
        tags = sorted(runs)
        for label, ids in (("Tool calling", tool_ids), ("Quality", qual_ids)):
            w(f"\n### {label}\n")
            w("Scores are normalised within each scenario and aggregated by mean rank; "
              "raw scores are never summed across scenarios with different maxima.\n")
            w("| run | " + " | ".join(i.split("-")[0] for i in ids) + " | mean rank | graded |")
            w("|---" * (len(ids) + 3) + "|")
            agg = rank_aggregate(grades, ids, tags)
            rows = []
            for t in tags:
                cells = []
                for sid in ids:
                    s = (grades["scenarios"].get(sid) or {}).get("scores", {}).get(t)
                    if s and s.get("score") is not None:
                        cells.append(f"{s['score']:g}/{s.get('max_score','?')}"
                                     + ("&#42;" if s.get("truncated") else ""))
                    else:
                        cov = (grades["scenarios"].get(sid) or {}).get("coverage", {}).get(t)
                        cells.append(f"_{cov or 'n/a'}_")       # 35: never a 0
                mr, n = agg.get(t, (None, 0))
                rows.append((mr if mr is not None else 99, t, cells, mr, n))
            for _, t, cells, mr, n in sorted(rows):
                w(f"| {t} | " + " | ".join(cells) + f" | {mr if mr else '-'} | {n}/{len(ids)} |")
            w("\n&#42; = truncated at the token budget, scored on what was produced.\n")
    else:
        w("\n## Capability\n\n_Not graded yet - run grade_llm.py._\n")

    w("\n## Provenance\n")
    w("| file | sha256 |")
    w("|---|---|")
    for f, s in manifest["file_sha256"].items():
        w(f"| `{f}` | `{s[:16]}...` |")

    text = "\n".join(L) + "\n"
    dest = a.out or os.path.join(a.results, "report.md")
    open(dest, "w").write(text)
    print(f"wrote {dest} ({len(text)} chars)")


if __name__ == "__main__":
    main()
