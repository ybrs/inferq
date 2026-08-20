#!/bin/bash
set -u
source /workspace/perf-run-logs/paths.env
BIN=/workspace/target-native/release/gguf_infer
PROMPT='Write a Rust function that parses a semver string into (major, minor, patch) with unit tests.'
LOG=/workspace/perf-run-logs/sweep_ab_cd.log
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

# a: 4 physical cores
run_cfg a 4 4 "0,1,2,3" 1
run_cfg a 4 4 "0,1,2,3" 2
# b: 6 physical cores
run_cfg b 6 6 "0,1,2,3,4,5" 1
run_cfg b 6 6 "0,1,2,3,4,5" 2
# c: 8 threads (6 physical + 2 HT siblings, only 6 physical exist on this host)
run_cfg c 8 8 "0,1,2,3,4,5,6,7" 1
run_cfg c 8 8 "0,1,2,3,4,5,6,7" 2
# d: 12 logical (all threads)
run_cfg d 12 12 "0-11" 1
run_cfg d 12 12 "0-11" 2

echo "ALL DONE" | tee -a "$LOG"
