#!/bin/bash
# Prompt-guard experiment: does a strict system prompt raise the faithfulness
# score, and does it pay for that with over-refusal?
#
# Three runs per model, all with the user prompts byte-identical to the
# unguarded suite, so only the system message varies:
#
#   outputs-control/   c1-c2, no guard   -> baseline: does it answer what IS there?
#   outputs-guarded/   h1-h3, guarded    -> did the guard fix the fabrication?
#   outputs-guarded/   c1-c2, guarded    -> did the guard break the control?
#
# Score with:  python3 grade_h.py outputs-guarded
#              python3 grade_c.py outputs-control
#              python3 grade_c.py outputs-guarded
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD="${GUARD:-$HERE/tests-guarded/system-guard.txt}"

MODELS="${*:-
granite-4.1-3b-Q4_K_M.gguf
granite-4.1-3b-UD-Q2_K_XL.gguf
Qwen3-1.7B-Q4_K_M.gguf
Qwen3.5-2B-UD-Q4_K_XL.gguf
Qwen3-0.6B-Q4_K_M.gguf
}"

for m in $MODELS; do
  echo "=== $m"
  OUTDIR="$HERE/outputs-control" "$HERE/quality.sh" "$m" control
  OUTDIR="$HERE/outputs-guarded" SYSTEM="$GUARD" "$HERE/quality.sh" "$m" faith
  OUTDIR="$HERE/outputs-guarded" SYSTEM="$GUARD" "$HERE/quality.sh" "$m" control
done
echo "GUARDED-RUN-DONE"
