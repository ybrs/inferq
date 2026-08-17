#!/bin/bash
# Best-of-two for the configuration the sweeps recommend, on every workload,
# under the same protocol as the defaults. This is the "closest passing
# configuration" gate 8 asks for when the defaults miss.
#
#   ./policy-run-logs/recommended.sh [extra gguf_infer flags...]
set -u

cd /workspace
RUN=./policy-run-logs/run.sh
P=policy-run-logs/prompts

for rep in 1 2; do
  for w in 1 2 3; do
    "${RUN}" "rec_w${w}_r${rep}" "${P}/w${w}.txt" 256 \
      --speculative auto \
      --speculative-trace "policy-run-logs/rec_w${w}_r${rep}.trace.jsonl" "$@"
  done
  "${RUN}" "rec_w4_r${rep}" "${P}/w4.txt" 768 \
    --speculative auto \
    --speculative-trace "policy-run-logs/rec_w4_r${rep}.trace.jsonl" "$@"
done
