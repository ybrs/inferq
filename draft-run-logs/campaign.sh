#!/bin/bash
# Validation for the MTP draft confidence gate (taskq 390).
#
# Same protocol as policy-run-logs: 6 physical cores, default threading, fully
# resident experts, greedy decoding, runs strictly serialised.
#
#   ./draft-run-logs/campaign.sh [section ...]
#     baseline  target-only reference, best of two
#     gate      policy `auto` with and without the gate, best of two
#     arm       single-arm MTP with and without the gate, the isolated effect
#     sweep     confidence threshold, single rep
set -u

cd /workspace
RUN=./draft-run-logs/run.sh
P=draft-run-logs/prompts

baseline() {
  for rep in 1 2; do
    for w in 1 2 3; do
      "${RUN}" "base_w${w}_r${rep}" "${P}/w${w}.txt" 256 --speculative off
    done
    "${RUN}" "base_w4_r${rep}" "${P}/w4.txt" 768 --speculative off
  done
}

gate() {
  for rep in 1 2; do
    for w in 1 2 3; do
      "${RUN}" "gate_w${w}_r${rep}" "${P}/w${w}.txt" 256 --speculative auto
      "${RUN}" "nogate_w${w}_r${rep}" "${P}/w${w}.txt" 256 \
        --speculative auto --mtp-min-confidence 0
    done
    "${RUN}" "gate_w4_r${rep}" "${P}/w4.txt" 768 --speculative auto
    "${RUN}" "nogate_w4_r${rep}" "${P}/w4.txt" 768 \
      --speculative auto --mtp-min-confidence 0
  done
}

arm() {
  # The MTP arm on its own is where the gate's effect is largest and cleanest,
  # because nothing else is taking steps away from it.
  for w in 1 2 3; do
    "${RUN}" "armgate_w${w}" "${P}/w${w}.txt" 256 --speculative mtp
    "${RUN}" "armnogate_w${w}" "${P}/w${w}.txt" 256 \
      --speculative mtp --mtp-min-confidence 0
  done
}

sweep() {
  for t in 0.5 0.6 0.7 0.8 0.9; do
    for w in 1 2 3; do
      "${RUN}" "sweep_c${t}_w${w}" "${P}/w${w}.txt" 256 \
        --speculative auto --mtp-min-confidence "${t}"
    done
  done
}

sections=("$@")
if [[ ${#sections[@]} -eq 0 ]]; then
  sections=(baseline arm gate sweep)
fi
for section in "${sections[@]}"; do
  echo "=== ${section} ==="
  "${section}"
done
echo "=== draft campaign complete ==="
