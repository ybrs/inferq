#!/bin/bash
# The specified sweep brackets the MTP arm's break-even acceptance from below
# without crossing it. Measured stage costs on this host put break-even at
# 0.65-0.72 depending on workload context length, so this extends the suspend
# threshold past it and re-measures the recommended configuration best-of-two
# on every workload.
#
#   ./policy-run-logs/extended-sweep.sh [threshold ...]
set -u

cd "$(dirname "$0")/.."
RUN=./policy-run-logs/run.sh
P=policy-run-logs/prompts

thresholds=("$@")
if [[ ${#thresholds[@]} -eq 0 ]]; then
  thresholds=(0.7 0.8)
fi

for threshold in "${thresholds[@]}"; do
  for w in 1 2 3; do
    "${RUN}" "xsweep_t${threshold}_w${w}" "${P}/w${w}.txt" 256 \
      --speculative auto --mtp-suspend-below "${threshold}"
  done
  "${RUN}" "xsweep_t${threshold}_w4" "${P}/w4.txt" 768 \
    --speculative auto --mtp-suspend-below "${threshold}"
done
