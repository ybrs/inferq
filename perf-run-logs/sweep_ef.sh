#!/bin/bash
set -u
source /workspace/perf-run-logs/paths.env
BIN=/workspace/target-native/release/gguf_infer
PROMPT='Write a Rust function that parses a semver string into (major, minor, patch) with unit tests.'
LOG=/workspace/perf-run-logs/sweep_ef.log
: > "$LOG"

run_cfg() {
  local name="$1" candle="$2" rayon="$3" cores="$4" rep="$5"
  echo "=== run $name rep$rep CANDLE=$candle RAYON=$rayon cores=[$cores] ===" | tee -a "$LOG"
  if [ -n "$cores" ]; then
    CMD=(taskset -c "$cores" env CANDLE_NUM_THREADS="$candle" RAYON_NUM_THREADS="$rayon" "$BIN")
  else
    CMD=(env CANDLE_NUM_THREADS="$candle" RAYON_NUM_THREADS="$rayon" "$BIN")
  fi
  "${CMD[@]}" \
    --model "$MODEL" --tokenizer-model "$TOK" \
    --chat --no-thinking \
    --prompt "$PROMPT" \
    --max-new-tokens 128 \
    --expert-cache-mib 24000 --warmup-all-experts \
    --speculative-mtp 0 >> "$LOG" 2>&1
  echo "--- end $name rep$rep ---" | tee -a "$LOG"
}

# e: best-so-far (CANDLE=6 RAYON=6) with NO taskset
run_cfg e 6 6 "" 1
run_cfg e 6 6 "" 2
# f: best N=6 threads for CANDLE, RAYON=1 (pool double-subscription test), same cores as best run (0-5)
run_cfg f 6 1 "0,1,2,3,4,5" 1
run_cfg f 6 1 "0,1,2,3,4,5" 2

echo "ALL DONE" | tee -a "$LOG"
