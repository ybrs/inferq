#!/usr/bin/env python3
"""Turn the raw gguf_infer logs into the tables the n-gram report needs.

Reads every ngram-run-logs/*.log, pulls the decode rate, the generated token
ids, and the n-gram metrics line, and prints markdown. Nothing here recomputes
a measurement; it only reformats what the binary reported.
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
    row["prefill_tokens"] = int(m.group(1))
    row["prefill_s"] = float(m.group(2))
    row["decode_passes"] = int(m.group(4))
    row["decode_s"] = float(m.group(5))
    row["decode_tps"] = float(m.group(6))
    row["context"] = int(m.group(7))

    m = re.search(r"^generated token ids: (\[.*?\])$", text, re.M | re.S)
    row["tokens"] = json.loads(m.group(1)) if m else None

    m = re.search(
        r"^n-gram speculation: draft (\d+) / min match (\d+); (\d+)/(\d+) steps matched "
        r"\(([\d.]+)%\); (\d+) drafts, (\d+)/(\d+) draft tokens accepted \(([\d.]+)%\); "
        r"([\d.]+) tokens per verification pass; (\d+) verification passes over (\d+) tokens; "
        r"(\d+) rollbacks \((\d+) replays over (\d+) tokens\); lookup ([\d.]+)s, verify ([\d.]+)s, "
        r"snapshot ([\d.]+)s, rollback ([\d.]+)s, replay ([\d.]+)s, no-match decode ([\d.]+)s",
        text,
        re.M,
    )
    if m:
        g = m.groups()
        row.update(
            draft_len=int(g[0]), min_match=int(g[1]),
            steps_matched=int(g[2]), steps=int(g[3]), match_rate=float(g[4]),
            drafts=int(g[5]), accepted=int(g[6]), proposed=int(g[7]),
            acceptance=float(g[8]), tokens_per_pass=float(g[9]),
            passes=int(g[10]), pass_rows=int(g[11]), rollbacks=int(g[12]),
            replays=int(g[13]), replayed_tokens=int(g[14]),
            lookup_s=float(g[15]), verify_s=float(g[16]), snapshot_s=float(g[17]),
            rollback_s=float(g[18]), replay_s=float(g[19]), nomatch_s=float(g[20]),
        )
    m = re.search(r"^n-gram acceptance by draft position: (.+)$", text, re.M)
    if m:
        row["histogram"] = [
            (int(p), int(a), int(t))
            for p, a, t in re.findall(r"(\d+):(\d+)/(\d+)", m.group(1))
        ]
    m = re.search(r"^n-gram drafts by match length: (.+)$", text, re.M)
    if m:
        row["by_len"] = m.group(1)
    m = re.search(r"^n-gram draft outcomes: (\d+) fully accepted, (\d+) rejected at once", text, re.M)
    if m:
        row["full_drafts"] = int(m.group(1))
        row["zero_drafts"] = int(m.group(2))
    m = re.search(r"^MTP speculation: (.+)$", text, re.M)
    if m:
        row["mtp"] = m.group(1)
    m = re.search(r"snapshots (\d+) rows x ([\d.]+) MiB", text)
    if m:
        row["snapshot_rows"] = int(m.group(1))
        row["snapshot_mib"] = float(m.group(2))
    return row


def main():
    rows = {}
    for path in sorted(LOGS.glob("*.log")):
        parsed = parse(path)
        if parsed:
            rows[parsed["tag"]] = parsed

    print("## Raw run index\n")
    print("| run | decode tok/s | decode s | passes | context |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for tag, r in sorted(rows.items()):
        print(f"| {tag} | {r['decode_tps']:.2f} | {r['decode_s']:.3f} | "
              f"{r['decode_passes']} | {r['context']} |")

    print("\n## Greedy equivalence (token ids)\n")
    print("| workload | target-only run | n-gram run | tokens | identical |")
    print("| --- | --- | --- | ---: | --- |")
    for w in ("w1", "w2", "w3"):
        for rep in ("r1", "r2"):
            base, spec = f"{w}_base_{rep}", f"{w}_ngram7_{rep}"
            if base in rows and spec in rows:
                same = rows[base]["tokens"] == rows[spec]["tokens"]
                n = len(rows[base]["tokens"] or [])
                print(f"| {w.upper()} | {base} | {spec} | {n} | "
                      f"{'**yes**' if same else '**NO**'} |")

    print("\n## Decode throughput (best of 2)\n")
    print("| workload | target-only tok/s | n-gram=7 tok/s | speedup | match rate | "
          "acceptance | tokens/pass |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for w in ("w1", "w2", "w3"):
        bases = [rows[f"{w}_base_{r}"]["decode_tps"] for r in ("r1", "r2")
                 if f"{w}_base_{r}" in rows]
        specs = [rows[f"{w}_ngram7_{r}"] for r in ("r1", "r2")
                 if f"{w}_ngram7_{r}" in rows]
        if not bases or not specs:
            continue
        best_base = max(bases)
        best_spec = max(specs, key=lambda r: r["decode_tps"])
        print(f"| {w.upper()} | {best_base:.2f} | {best_spec['decode_tps']:.2f} | "
              f"{best_spec['decode_tps'] / best_base:.3f}x | "
              f"{best_spec.get('match_rate', 0):.1f}% | "
              f"{best_spec.get('acceptance', 0):.1f}% | "
              f"{best_spec.get('tokens_per_pass', 0):.2f} |")

    print("\n## W1 sweep\n")
    print("| draft len | min match | tok/s | speedup vs base | match rate | acceptance | "
          "tokens/pass | full drafts | zero drafts |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    w1_base = max((rows[f"w1_base_{r}"]["decode_tps"] for r in ("r1", "r2")
                   if f"w1_base_{r}" in rows), default=None)
    sweep = [(r["draft_len"], r["min_match"], r) for tag, r in rows.items()
             if tag.startswith("sweep_w1_")]
    for dl, mm, r in sorted(sweep):
        speed = f"{r['decode_tps'] / w1_base:.3f}x" if w1_base else "n/a"
        print(f"| {dl} | {mm} | {r['decode_tps']:.2f} | {speed} | {r['match_rate']:.1f}% | "
              f"{r['acceptance']:.1f}% | {r['tokens_per_pass']:.2f} | "
              f"{r.get('full_drafts', 0)} | {r.get('zero_drafts', 0)} |")

    print("\n## Acceptance histograms by draft position\n")
    for tag in sorted(rows):
        r = rows[tag]
        if "histogram" in r:
            cells = " ".join(f"{p}:{a}/{t}" for p, a, t in r["histogram"])
            print(f"- `{tag}`: {cells}")

    print("\n## Snapshot cost\n")
    print("| run | snapshot rows | snapshot s | ms/row | MiB/row | verify s | snapshot share |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for tag, r in sorted(rows.items()):
        if r.get("snapshot_rows"):
            ms = r["snapshot_s"] * 1000 / r["snapshot_rows"]
            share = r["snapshot_s"] / r["verify_s"] * 100 if r["verify_s"] else 0
            print(f"| {tag} | {r['snapshot_rows']} | {r['snapshot_s']:.3f} | {ms:.2f} | "
                  f"{r['snapshot_mib']:.1f} | {r['verify_s']:.3f} | {share:.1f}% |")

    print("\n## Match-length breakdown\n")
    for tag in sorted(rows):
        if "by_len" in rows[tag]:
            print(f"- `{tag}`: {rows[tag]['by_len']}")

    print("\n## MTP runs\n")
    for tag in sorted(rows):
        if "mtp" in rows[tag]:
            print(f"- `{tag}` ({rows[tag]['decode_tps']:.2f} tok/s): {rows[tag]['mtp']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
