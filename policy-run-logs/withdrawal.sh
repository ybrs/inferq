#!/bin/bash
# Test what actually costs W2, now that the threshold sweep has ruled itself out.
#
# Raising the suspend threshold from 0.5 to 0.7 made W2 *worse*, and every cell
# of the specified 3x3 sweep lands between 0.879x and 0.921x. So the cost is not
# a long tail of bad drafts that a higher bar would cut — it is concentrated in
# (a) the proposals it takes to detect failure and (b) each probe's re-entry,
# which under lazy catch-up has to resynchronise the whole suspended gap.
#
# This varies the two constants that control exactly those: a faster EWMA
# (withdraw in ~3 proposals instead of ~5) and a longer first suspension
# (fewer re-entries).
set -u

cd /workspace
RUN=./policy-run-logs/run.sh
P=policy-run-logs/prompts

run_all() {
  local tag="$1"; shift
  for w in 1 2 3; do
    "${RUN}" "${tag}_w${w}" "${P}/w${w}.txt" 256 --speculative auto "$@"
  done
  "${RUN}" "${tag}_w4" "${P}/w4.txt" 768 --speculative auto "$@"
}

run_all wd_fast --ewma-alpha 0.4
run_all wd_long --backoff-tokens 256
run_all wd_both --ewma-alpha 0.4 --backoff-tokens 256
