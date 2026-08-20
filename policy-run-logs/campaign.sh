#!/bin/bash
# Regenerate every measurement in policy-report-$(hostname).md.
#
#   ./policy-run-logs/campaign.sh [section ...]
#
# with no arguments it runs every section in order. Sections:
#   baseline  target-only reference for W1-W4, best of two
#   auto      the policy at its defaults, best of two, with per-step traces
#   part0     the two numbers Part 0 settles
#   gate6     single-arm modes pinned to the previous reports' settings
#   sweep     MTP suspend threshold x start depth on W2 and W4
#
# Runs are serialised on purpose: this host is memory-bandwidth bound, so a
# concurrent build or a second run would change the number being measured.
set -u

cd "$(dirname "$0")/.."
RUN=./policy-run-logs/run.sh
P=policy-run-logs/prompts

# The controllers pinned off, which is what makes a single-arm mode reproduce
# the fixed-draft behaviour the earlier reports measured.
PINNED="--no-adaptive-length --no-ewma-backoff --no-span-continuation"

baseline() {
  for rep in 1 2; do
    "${RUN}" "base_w1_r${rep}" "${P}/w1.txt" 256 --speculative off
    "${RUN}" "base_w2_r${rep}" "${P}/w2.txt" 256 --speculative off
    "${RUN}" "base_w3_r${rep}" "${P}/w3.txt" 256 --speculative off
    "${RUN}" "base_w4_r${rep}" "${P}/w4.txt" 768 --speculative off
  done
}

auto() {
  for rep in 1 2; do
    for w in 1 2 3; do
      "${RUN}" "auto_w${w}_r${rep}" "${P}/w${w}.txt" 256 \
        --speculative auto \
        --speculative-trace "policy-run-logs/auto_w${w}_r${rep}.trace.jsonl"
    done
    "${RUN}" "auto_w4_r${rep}" "${P}/w4.txt" 768 \
      --speculative auto \
      --speculative-trace "policy-run-logs/auto_w4_r${rep}.trace.jsonl"
  done
}

part0() {
  # 0.1 — MTP depth 7 on W3, best of two, sets which W3 gate applies.
  for rep in 1 2; do
    "${RUN}" "p0_mtp_d7_w3_r${rep}" "${P}/w3.txt" 256 \
      --speculative mtp --mtp-depth-cap 7 --mtp-depth-start 7 ${PINNED}
  done
  # 0.2 — MTP depth 4 on W1, acceptance split by whether the index also held
  # literal evidence for that step. The split is computed in-run from the same
  # index the policy would have consulted.
  "${RUN}" p0_mtp_d4_w1 "${P}/w1.txt" 256 \
    --speculative mtp --mtp-depth-cap 4 --mtp-depth-start 4 ${PINNED} \
    --speculative-trace policy-run-logs/p0_mtp_d4_w1.trace.jsonl
}

gate6() {
  for w in 1 2 3; do
    "${RUN}" "g6_ngram_w${w}" "${P}/w${w}.txt" 256 \
      --speculative ngram --ngram-draft-cap 7 --ngram-min-match 4 ${PINNED}
    "${RUN}" "g6_mtp_w${w}" "${P}/w${w}.txt" 256 \
      --speculative mtp --mtp-depth-cap 7 --mtp-depth-start 7 ${PINNED}
  done
}

sweep() {
  for threshold in 0.4 0.5 0.6; do
    for start in 3 4 5; do
      tag="sweep_t${threshold}_d${start}"
      "${RUN}" "${tag}_w2" "${P}/w2.txt" 256 \
        --speculative auto --mtp-suspend-below "${threshold}" --mtp-depth-start "${start}"
      "${RUN}" "${tag}_w4" "${P}/w4.txt" 768 \
        --speculative auto --mtp-suspend-below "${threshold}" --mtp-depth-start "${start}"
    done
  done
}

sections=("$@")
if [[ ${#sections[@]} -eq 0 ]]; then
  sections=(baseline auto part0 gate6 sweep)
fi
for section in "${sections[@]}"; do
  echo "=== ${section} ==="
  "${section}"
done
