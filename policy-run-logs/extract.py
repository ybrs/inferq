#!/usr/bin/env python3
"""Turn the raw policy-run-logs/*.log files into the tables the report needs.

Nothing here recomputes a measurement; it reformats what the binary reported.
Run it after policy-run-logs/campaign.sh:

    ./policy-run-logs/extract.py            # every section
    ./policy-run-logs/extract.py windows    # just the W4 arm-handoff table
"""
import json
import pathlib
import re
import sys

LOGS = pathlib.Path(__file__).resolve().parent


def parse(path):
    text = path.read_text(errors="replace")
    row = {"tag": path.stem}
    m = re.search(
        r"^input: (\d+) tokens evaluated in ([\d.]+)s \(([\d.]+) tok/s\); "
        r"decode: (\d+) passes in ([\d.]+)s \(([\d.]+) tok/s\); context: (\d+)",
        text,
        re.M,
    )
    if not m:
        return None
    row["prefill_s"] = float(m.group(2))
    row["decode_s"] = float(m.group(5))
    row["decode_tps"] = float(m.group(6))

    m = re.search(r"^generated token ids: (\[.*?\])$", text, re.M | re.S)
    row["tokens"] = json.loads(m.group(1)) if m else None

    m = re.search(
        r"^speculative policy: mode (\w+); (\d+) steps = (\d+) n-gram \((\d+) span\) \+ "
        r"(\d+) MTP \+ (\d+) plain; (\d+) steps had literal evidence \(([\d.]+)%\); "
        r"([\d.]+) tokens per verification pass; (\d+) verification passes over (\d+) tokens; "
        r"(\d+) rollbacks; lookup ([\d.]+)s, draft ([\d.]+)s, verify ([\d.]+)s, "
        r"snapshot ([\d.]+)s, rollback ([\d.]+)s, plain decode ([\d.]+)s, "
        r"MTP resync ([\d.]+)s over (\d+) passes / (\d+) rows \(longest (\d+)\)",
        text,
        re.M,
    )
    if m:
        g = m.groups()
        row.update(
            mode=g[0], steps=int(g[1]), ngram_steps=int(g[2]), span_steps=int(g[3]),
            mtp_steps=int(g[4]), plain_steps=int(g[5]), evidence_steps=int(g[6]),
            evidence_pct=float(g[7]), tokens_per_pass=float(g[8]),
            passes=int(g[9]), pass_tokens=int(g[10]), rollbacks=int(g[11]),
            lookup_s=float(g[12]), draft_s=float(g[13]), verify_s=float(g[14]),
            snapshot_s=float(g[15]), rollback_s=float(g[16]), plain_s=float(g[17]),
            resync_s=float(g[18]), resync_passes=int(g[19]), resync_rows=int(g[20]),
            max_resync=int(g[21]),
        )

    for arm in ("ngram", "mtp"):
        m = re.search(
            rf"^policy arm {arm}: (\d+) proposals, (\d+)/(\d+) tokens accepted \(([\d.]+)%\), "
            r"(\d+) fully accepted, (\d+) rejected at once; (\d+) suspensions over (\d+) steps, "
            r"(\d+) probes \((\d+) resumed\)",
            text,
            re.M,
        )
        if m:
            g = m.groups()
            row.update({
                f"{arm}_proposals": int(g[0]), f"{arm}_accepted": int(g[1]),
                f"{arm}_proposed": int(g[2]), f"{arm}_acceptance": float(g[3]),
                f"{arm}_full": int(g[4]), f"{arm}_rejected": int(g[5]),
                f"{arm}_suspensions": int(g[6]), f"{arm}_suspended_steps": int(g[7]),
                f"{arm}_probes": int(g[8]), f"{arm}_resumed": int(g[9]),
            })

    m = re.search(
        r"^policy arm mtp by literal evidence: (\d+)/(\d+) accepted on an n-gram match "
        r"\(([\d.]+)%\), (\d+)/(\d+) accepted on a miss \(([\d.]+)%\)",
        text,
        re.M,
    )
    if m:
        g = m.groups()
        row.update(
            mtp_hit_accepted=int(g[0]), mtp_hit_proposed=int(g[1]), mtp_hit_pct=float(g[2]),
            mtp_miss_accepted=int(g[3]), mtp_miss_proposed=int(g[4]), mtp_miss_pct=float(g[5]),
        )
    return row


def load():
    rows = {}
    for path in sorted(LOGS.glob("*.log")):
        row = parse(path)
        if row:
            rows[row["tag"]] = row
    return rows


def best(rows, prefix, reps=(1, 2)):
    """Fastest of the repetitions of one measurement."""
    candidates = [rows[f"{prefix}_r{rep}"] for rep in reps if f"{prefix}_r{rep}" in rows]
    return max(candidates, key=lambda row: row["decode_tps"]) if candidates else None


def identical(rows, a, b):
    left, right = rows.get(a), rows.get(b)
    if not left or not right or left["tokens"] is None or right["tokens"] is None:
        return "n/a"
    return "yes" if left["tokens"] == right["tokens"] else "**NO**"


def throughput(rows):
    print("## Decode throughput, best of two\n")
    print("| workload | target-only | policy `auto` | ratio | tokens identical |")
    print("| --- | ---: | ---: | ---: | --- |")
    for w, name in ((1, "W1 copy-heavy"), (2, "W2 prose"), (3, "W3 self-repetitive"),
                    (4, "W4 mixed (768 tokens)")):
        base, auto = best(rows, f"base_w{w}"), best(rows, f"auto_w{w}")
        if not base or not auto:
            continue
        same = identical(rows, base["tag"], auto["tag"])
        print(f"| {name} | {base['decode_tps']:.2f} tok/s | {auto['decode_tps']:.2f} tok/s "
              f"| **{auto['decode_tps'] / base['decode_tps']:.3f}x** | {same} |")
    print()


def arms(rows):
    print("## Per-arm behaviour under `auto`, best-of-two run\n")
    print("| workload | steps | n-gram (span) | MTP | plain | evidence | n-gram acc | MTP acc "
          "| suspensions n/m | probes n/m | tokens/pass |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for w in (1, 2, 3, 4):
        row = best(rows, f"auto_w{w}")
        if not row or "steps" not in row:
            continue
        print(
            f"| W{w} | {row['steps']} | {row['ngram_steps']} ({row['span_steps']}) "
            f"| {row['mtp_steps']} | {row['plain_steps']} | {row['evidence_pct']:.1f}% "
            f"| {row.get('ngram_acceptance', 0):.1f}% | {row.get('mtp_acceptance', 0):.1f}% "
            f"| {row.get('ngram_suspensions', 0)}/{row.get('mtp_suspensions', 0)} "
            f"| {row.get('ngram_probes', 0)}/{row.get('mtp_probes', 0)} "
            f"| {row['tokens_per_pass']:.2f} |"
        )
    print()


def costs(rows):
    print("## Where the policy's own time goes, best-of-two `auto` run\n")
    print("| workload | decode | lookup | MTP draft | verify | snapshot | rollback | plain "
          "| MTP resync | resync passes / rows (longest) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for w in (1, 2, 3, 4):
        row = best(rows, f"auto_w{w}")
        if not row or "steps" not in row:
            continue
        print(
            f"| W{w} | {row['decode_s']:.2f}s | {row['lookup_s']:.3f}s | {row['draft_s']:.2f}s "
            f"| {row['verify_s']:.2f}s | {row['snapshot_s']:.2f}s | {row['rollback_s']:.3f}s "
            f"| {row['plain_s']:.2f}s | {row['resync_s']:.2f}s "
            f"| {row['resync_passes']} / {row['resync_rows']} ({row['max_resync']}) |"
        )
    print()


def part0(rows):
    print("## Part 0\n")
    base = best(rows, "base_w3")
    mtp = best(rows, "p0_mtp_d7_w3")
    if base and mtp:
        ratio = mtp["decode_tps"] / base["decode_tps"]
        gate = "1.02x" if ratio >= 1.05 else "0.98x (never-lose)"
        print(f"**0.1** MTP depth 7 on W3, best of two: {mtp['decode_tps']:.2f} tok/s against "
              f"{base['decode_tps']:.2f} tok/s target-only = **{ratio:.3f}x**. "
              f"The W3 gate is therefore **{gate}**.\n")
        reps = [rows[t]['decode_tps'] for t in (f"p0_mtp_d7_w3_r{r}" for r in (1, 2)) if t in rows]
        print(f"Repetitions: {', '.join(f'{tps:.2f}' for tps in reps)} tok/s.\n")
    run = rows.get("p0_mtp_d4_w1")
    if run and "mtp_miss_proposed" in run:
        print(f"**0.2** MTP depth 4 on W1, acceptance split by whether the n-gram index also "
              f"held a match for that step: **{run['mtp_hit_pct']:.1f}%** on "
              f"{run['mtp_hit_proposed']} tokens proposed at steps with literal evidence, "
              f"**{run['mtp_miss_pct']:.1f}%** on {run['mtp_miss_proposed']} tokens proposed "
              f"at steps without it. Unconditional acceptance was "
              f"{run.get('mtp_acceptance', 0):.1f}%.\n")


def gate6(rows):
    print("## Gate 6 — single-arm modes against the previous reports\n")
    print("| workload | mode | tok/s | ratio to target-only | previous report |")
    print("| --- | --- | ---: | ---: | ---: |")
    previous = {
        ("ngram", 1): "8.47 (1.228x)", ("ngram", 2): "7.26 (0.937x)",
        ("ngram", 3): "6.23 (0.809x)",
        ("mtp", 1): "1.14x", ("mtp", 2): "0.55x", ("mtp", 3): "1.07x",
    }
    for mode in ("ngram", "mtp"):
        for w in (1, 2, 3):
            row = rows.get(f"g6_{mode}_w{w}")
            base = best(rows, f"base_w{w}")
            if not row or not base:
                continue
            print(f"| W{w} | {mode} | {row['decode_tps']:.2f} "
                  f"| {row['decode_tps'] / base['decode_tps']:.3f}x "
                  f"| {previous.get((mode, w), '')} |")
    print()


def sweep(rows):
    print("## Sweep — MTP suspend threshold x start depth\n")
    for w in (2, 4):
        base = best(rows, f"base_w{w}")
        if not base:
            continue
        print(f"### W{w} (target-only {base['decode_tps']:.2f} tok/s)\n")
        print("| suspend below | start 3 | start 4 | start 5 |")
        print("| --- | ---: | ---: | ---: |")
        for threshold in ("0.4", "0.5", "0.6"):
            cells = []
            for start in ("3", "4", "5"):
                row = rows.get(f"sweep_t{threshold}_d{start}_w{w}")
                cells.append(
                    f"{row['decode_tps']:.2f} ({row['decode_tps'] / base['decode_tps']:.3f}x)"
                    if row else "-"
                )
            print(f"| {threshold} | {' | '.join(cells)} |")
        print()


def xsweep(rows):
    """The suspend threshold extended past the measured break-even."""
    print("## Extended sweep — MTP suspend threshold past break-even\n")
    print("| suspend below | W1 | W2 | W3 | W4 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    defaults = {}
    for w in (1, 2, 3, 4):
        base, auto = best(rows, f"base_w{w}"), best(rows, f"auto_w{w}")
        if base and auto:
            defaults[w] = (auto["decode_tps"], auto["decode_tps"] / base["decode_tps"])
    if defaults:
        cells = [f"{tps:.2f} ({ratio:.3f}x)" for _, (tps, ratio) in sorted(defaults.items())]
        print(f"| 0.5 (default, best of two) | {' | '.join(cells)} |")
    for threshold in ("0.7", "0.8"):
        cells = []
        for w in (1, 2, 3, 4):
            row = rows.get(f"xsweep_t{threshold}_w{w}")
            base = best(rows, f"base_w{w}")
            cells.append(
                f"{row['decode_tps']:.2f} ({row['decode_tps'] / base['decode_tps']:.3f}x)"
                if row and base else "-"
            )
        print(f"| {threshold} | {' | '.join(cells)} |")
    print()
    print("| suspend below | workload | MTP steps | MTP acc | suspensions | probes (resumed) "
          "| plain steps | MTP resync |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for threshold in ("0.7", "0.8"):
        for w in (1, 2, 3, 4):
            row = rows.get(f"xsweep_t{threshold}_w{w}")
            if not row or "steps" not in row:
                continue
            print(
                f"| {threshold} | W{w} | {row['mtp_steps']} "
                f"| {row.get('mtp_acceptance', 0):.1f}% | {row.get('mtp_suspensions', 0)} "
                f"| {row.get('mtp_probes', 0)} ({row.get('mtp_resumed', 0)}) "
                f"| {row['plain_steps']} | {row['resync_s']:.2f}s |"
            )
    print()


def withdrawal(rows):
    """Do faster withdrawal and rarer re-entry help where the threshold did not?"""
    print("## Withdrawal and re-entry constants\n")
    variants = [
        ("auto", "defaults (alpha 0.2, first suspension 64)"),
        ("wd_fast", "alpha 0.4 — withdraw in ~3 proposals"),
        ("wd_long", "first suspension 256 tokens — fewer re-entries"),
        ("wd_both", "alpha 0.4 and 256 tokens"),
    ]
    print("| configuration | W1 | W2 | W3 | W4 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for tag, label in variants:
        cells = []
        for w in (1, 2, 3, 4):
            row = best(rows, f"{tag}_w{w}") if tag == "auto" else rows.get(f"{tag}_w{w}")
            base = best(rows, f"base_w{w}")
            cells.append(
                f"{row['decode_tps']:.2f} ({row['decode_tps'] / base['decode_tps']:.3f}x)"
                if row and base else "-"
            )
        print(f"| {label} | {' | '.join(cells)} |")
    print()
    print("| configuration | workload | MTP steps | MTP acc | suspensions | probes (resumed) "
          "| plain steps | MTP resync |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for tag, label in variants:
        for w in (1, 2, 3, 4):
            row = best(rows, f"{tag}_w{w}") if tag == "auto" else rows.get(f"{tag}_w{w}")
            if not row or "steps" not in row:
                continue
            print(
                f"| {tag} | W{w} | {row['mtp_steps']} | {row.get('mtp_acceptance', 0):.1f}% "
                f"| {row.get('mtp_suspensions', 0)} | {row.get('mtp_probes', 0)} "
                f"({row.get('mtp_resumed', 0)}) | {row['plain_steps']} | {row['resync_s']:.2f}s |"
            )
    print()


def recommended(rows):
    """The closest-to-passing configuration, best of two, against the gates."""
    print("## Closest passing configuration, best of two\n")
    gates = {1: 1.25, 2: 0.97, 3: 1.02, 4: 1.10}
    print("| workload | gate | target-only | defaults | verdict | recommended | verdict |")
    print("| --- | ---: | ---: | ---: | --- | ---: | --- |")
    for w in (1, 2, 3, 4):
        base = best(rows, f"base_w{w}")
        auto = best(rows, f"auto_w{w}")
        rec = best(rows, f"rec_w{w}")
        if not base:
            continue
        cells = []
        for row in (auto, rec):
            if not row:
                cells += ["-", "-"]
                continue
            ratio = row["decode_tps"] / base["decode_tps"]
            verdict = "**pass**" if ratio >= gates[w] else "miss"
            cells += [f"{row['decode_tps']:.2f} ({ratio:.3f}x)", verdict]
        print(f"| W{w} | {gates[w]:.2f}x | {base['decode_tps']:.2f} | {' | '.join(cells)} |")
    print()
    for w in (1, 2, 3, 4):
        base, rec = best(rows, f"base_w{w}"), best(rows, f"rec_w{w}")
        if base and rec:
            print(f"- W{w} token ids identical to target-only: "
                  f"{identical(rows, base['tag'], rec['tag'])}")
    print()


def windows(rows, width=256):
    """Per-window arm-fire counts, the evidence that the arms hand off."""
    print(f"## W4 arm hand-off, per {width}-token window\n")
    for rep in (1, 2):
        path = LOGS / f"auto_w4_r{rep}.trace.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not records:
            continue
        print(f"Trace `{path.name}`, {len(records)} steps.\n")
        print("| window (committed tokens) | n-gram | span | MTP | plain | n-gram acc | MTP acc "
              "| depth at end |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        buckets = {}
        for record in records:
            buckets.setdefault(record["committed"] // width, []).append(record)
        for index in sorted(buckets):
            group = buckets[index]
            counts = {"ngram": 0, "ngram-span": 0, "mtp": 0, "plain": 0}
            for record in group:
                counts[record["arm"]] += 1
            def acc(kinds):
                proposed = sum(r["proposed"] for r in group if r["arm"] in kinds)
                accepted = sum(r["accepted"] for r in group if r["arm"] in kinds)
                return f"{100 * accepted / proposed:.0f}%" if proposed else "-"
            print(
                f"| {index * width}-{(index + 1) * width - 1} | {counts['ngram']} "
                f"| {counts['ngram-span']} | {counts['mtp']} | {counts['plain']} "
                f"| {acc(('ngram', 'ngram-span'))} | {acc(('mtp',))} "
                f"| {group[-1]['mtp_depth']} |"
            )
        print()
        break


def trajectory(rows, tag, every=16):
    """Controller state sampled along one run."""
    path = LOGS / f"{tag}.trace.jsonl"
    if not path.exists():
        return
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"## Controller trajectory, `{tag}` (every {every}th step)\n")
    print("| step | committed | arm | proposed/accepted | n-gram len | n-gram ewma | susp "
          "| MTP depth | MTP ewma | susp | resync rows |")
    print("| ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |")
    for record in records[::every]:
        print(
            f"| {record['step']} | {record['committed']} | {record['arm']} "
            f"| {record['proposed']}/{record['accepted']} | {record['ngram_len']} "
            f"| {record['ngram_ewma']:.2f} | {'y' if record['ngram_suspended'] else 'n'} "
            f"| {record['mtp_depth']} | {record['mtp_ewma']:.2f} "
            f"| {'y' if record['mtp_suspended'] else 'n'} | {record['resync_tokens']} |"
        )
    print()


def main():
    rows = load()
    sections = sys.argv[1:] or [
        "part0", "throughput", "arms", "costs", "windows", "gate6", "sweep", "xsweep",
        "withdrawal", "recommended", "trajectories",
    ]
    for section in sections:
        if section == "throughput":
            throughput(rows)
        elif section == "arms":
            arms(rows)
        elif section == "costs":
            costs(rows)
        elif section == "part0":
            part0(rows)
        elif section == "gate6":
            gate6(rows)
        elif section == "sweep":
            sweep(rows)
        elif section == "xsweep":
            xsweep(rows)
        elif section == "withdrawal":
            withdrawal(rows)
        elif section == "recommended":
            recommended(rows)
        elif section == "windows":
            windows(rows)
        elif section == "trajectories":
            for tag in ("auto_w1_r1", "auto_w2_r1", "auto_w3_r1", "auto_w4_r1"):
                trajectory(rows, tag)
        else:
            print(f"unknown section {section}", file=sys.stderr)


if __name__ == "__main__":
    main()
