#!/bin/bash
# One measured gguf_infer run for the unified speculative policy validation.
#
# usage: run.sh <tag> <prompt-file> <max-new-tokens> [extra gguf_infer flags...]
#
# Environment is fixed here so every row of every table is comparable, and it
# matches ngram-run-logs/run.sh exactly so the numbers in this report can be
# compared with ngram-report-702d043633e0.md: 6 physical cores, INFERQ default
# threading (no thread env vars), fully resident experts, greedy decoding.
set -u

MODEL_ROOT=/models/Qwen3.6-35B-A3B
BIN=/workspace/target-native/release/gguf_infer
LOGDIR=/workspace/draft-run-logs

tag="$1"
prompt_file="$2"
max_new="$3"
shift 3

log="${LOGDIR}/${tag}.log"
started=$(date +%s)
taskset -c 0-5 "${BIN}" \
  --model "${MODEL_ROOT}/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  --tokenizer-model "${MODEL_ROOT}" \
  --chat --no-thinking \
  --prompt "$(cat "${prompt_file}")" \
  --max-new-tokens "${max_new}" \
  --expert-cache-mib 46000 --warmup-all-experts \
  "$@" > "${log}" 2>&1

status=$?
elapsed=$(( $(date +%s) - started ))
{
  echo "tag: ${tag}"
  echo "prompt: ${prompt_file}"
  echo "prompt_sha1: $(sha1sum < "${prompt_file}" | cut -d' ' -f1)"
  echo "max_new_tokens: ${max_new}"
  echo "flags: $*"
  echo "wall_seconds: ${elapsed}"
  echo "exit: ${status}"
  grep -E '^generated token ids' "${log}" | head -1
  grep -E '^input: ' "${log}"
  grep -E '^n-gram ' "${log}"
  grep -E '^MTP speculation' "${log}"
  grep -E '^speculative policy' "${log}"
  grep -E '^policy arm ' "${log}"
} > "${LOGDIR}/${tag}.summary"
exit "${status}"
