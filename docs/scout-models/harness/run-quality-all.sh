#!/bin/bash
# Task suite (t1-t3) across every candidate.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS="
Qwen3.5-2B-UD-Q4_K_XL.gguf
LFM2.5-2.6B-QAD-Q4_0.gguf
granite-4.0-h-micro-Q4_K_M.gguf
granite-4.1-3b-Q4_K_M.gguf
Qwen3-4B-Instruct-2507-Q4_K_M.gguf
Qwen3.5-4B-Q3_K_M.gguf
Qwen3-1.7B-Q4_K_M.gguf
LFM2-2.6B-Q4_K_M.gguf
SmolLM3-Q4_K_M.gguf
gemma-3-4b-it-qat-Q4_0.gguf
Phi-4-mini-instruct-Q4_K_M.gguf
"

for m in $MODELS; do
  echo "--- task suite: $m"
  timeout 1500 bash "$HERE/quality.sh" "$m" task || echo "QUALITY-FAILED $m"
  pkill -x llama-server 2>/dev/null
  sleep 2
done
echo QUALITY-ALL-FINISHED
