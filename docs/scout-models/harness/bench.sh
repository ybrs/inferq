#!/bin/bash
# llama-bench sweep: pp512 prefill and tg128 decode, 6 threads pinned to cores 0-5.
# Writes bench-results.md next to this script.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_BIN="${LLAMA_BIN:-/models/llamacpp-main/build/bin}"
MODEL_DIR="${MODEL_DIR:-/models/small-models}"
OUT="$HERE/bench-results.md"

MODELS="
Qwen3-1.7B-Q4_K_M.gguf
Qwen3.5-2B-UD-Q4_K_XL.gguf
LFM2-2.6B-Q4_K_M.gguf
LFM2.5-2.6B-QAD-Q4_0.gguf
SmolLM3-Q4_K_M.gguf
granite-4.1-3b-Q4_K_M.gguf
granite-4.0-h-micro-Q4_K_M.gguf
Qwen3-4B-Instruct-2507-Q4_K_M.gguf
gemma-3-4b-it-qat-Q4_0.gguf
Phi-4-mini-instruct-Q4_K_M.gguf
Qwen3.5-4B-Q3_K_M.gguf
"

: > "$OUT"
for f in $MODELS; do
  [ -s "$MODEL_DIR/$f" ] || { echo "MISSING $f" | tee -a "$OUT"; continue; }
  echo "=== $f" | tee -a "$OUT"
  taskset -c 0-5 "$LLAMA_BIN/llama-bench" -m "$MODEL_DIR/$f" \
    -t 6 -p 512 -n 128 -r 2 -o md 2>/dev/null | tee -a "$OUT"
done
echo BENCH-FINISHED | tee -a "$OUT"
