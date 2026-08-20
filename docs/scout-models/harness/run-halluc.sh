#!/bin/bash
# Faithfulness suite (h1-h3) across the models that survived the task suite.
# LFM2.5 is excluded: it ignores enable_thinking:false and never emits an answer.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS="
Qwen3-1.7B-Q4_K_M.gguf
Qwen3.5-2B-UD-Q4_K_XL.gguf
LFM2-2.6B-Q4_K_M.gguf
SmolLM3-Q4_K_M.gguf
granite-4.1-3b-Q4_K_M.gguf
granite-4.0-h-micro-Q4_K_M.gguf
Qwen3-4B-Instruct-2507-Q4_K_M.gguf
Phi-4-mini-instruct-Q4_K_M.gguf
"

for m in $MODELS; do
  echo "--- faithfulness suite: $m"
  timeout 900 bash "$HERE/quality.sh" "$m" faith || echo "HALLUC-FAILED $m"
  pkill -x llama-server 2>/dev/null
  sleep 2
done
echo HALLUC-ALL-FINISHED
