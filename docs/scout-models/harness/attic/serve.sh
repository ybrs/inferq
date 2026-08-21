#!/bin/bash
# Start a llama-server for interactive prompt iteration and leave it running.
# Pair with ask.sh, which posts to the same port.
#
# Usage: ./serve.sh [model.gguf]        (default: Qwen3-0.6B-Q4_K_M.gguf)
#        ./serve.sh --stop
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_BIN="${LLAMA_BIN:-/models/llamacpp-main/build/bin}"
MODEL_DIR="${MODEL_DIR:-/models/small-models}"
PORT="${PORT:-8099}"

if [ "${1:-}" = "--stop" ]; then
  pkill -x llama-server && echo "stopped" || echo "nothing running"
  exit 0
fi

MODEL="${1:-Qwen3-0.6B-Q4_K_M.gguf}"
pkill -x llama-server 2>/dev/null
sleep 1

taskset -c 0-5 "$LLAMA_BIN/llama-server" -m "$MODEL_DIR/$MODEL" \
  -t 6 -c 8192 --jinja --host 127.0.0.1 --port "$PORT" \
  > "$HERE/serve.log" 2>&1 &

for _ in $(seq 1 120); do
  curl -s --max-time 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && {
    echo "serving $MODEL on :$PORT — now run ./ask.sh <prompt-file>"; exit 0; }
  sleep 1
done
echo "server failed to come up; see $HERE/serve.log" >&2
exit 1
