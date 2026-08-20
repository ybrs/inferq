#!/bin/bash
# One measured gguf_infer run for the n-gram speculation validation.
#
# usage: run.sh <tag> <prompt-file> [extra gguf_infer flags...]
# Environment is fixed here so every row of every table is comparable:
# 6 physical cores, INFERQ default threading (no thread env vars), fully
# resident experts, greedy decoding, 256 new tokens.
set -u

MODEL_ROOT=/models/Qwen3.6-35B-A3B
BIN=/workspace/target-native/release/gguf_infer
LOGDIR=/workspace/ngram-run-logs

tag="$1"
prompt_file="$2"
shift 2

log="${LOGDIR}/${tag}.log"
taskset -c 0-5 "${BIN}" \
  --model "${MODEL_ROOT}/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  --tokenizer-model "${MODEL_ROOT}" \
  --chat --no-thinking \
  --prompt "$(cat "${prompt_file}")" \
  --max-new-tokens 256 \
  --expert-cache-mib 46000 --warmup-all-experts \
  "$@" > "${log}" 2>&1

status=$?
{
  echo "tag: ${tag}"
  echo "flags: $*"
  grep -E '^generated token ids' "${log}" | head -1
  grep -E '^input: ' "${log}"
  grep -E '^n-gram ' "${log}"
  grep -E '^MTP speculation' "${log}"
} > "${LOGDIR}/${tag}.summary"
exit "${status}"
