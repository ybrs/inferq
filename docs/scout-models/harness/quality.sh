#!/bin/bash
# Run one prompt suite against one model, via llama-server's chat endpoint so the
# model's own chat template applies. Writes one file per prompt to outputs/.
#
# Usage: quality.sh <model-file.gguf> [suite]
#   suite: "task" (t1-t3, default) or "faith" (h1-h3)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_BIN="${LLAMA_BIN:-/models/llamacpp-main/build/bin}"
MODEL_DIR="${MODEL_DIR:-/models/small-models}"
TESTS="$HERE/tests"
OUTDIR="$HERE/outputs"
PORT="${PORT:-8099}"

MODEL="$1"
SUITE="${2:-task}"
NAME="${MODEL%.gguf}"

case "$SUITE" in
  task)  PROMPTS="t1-task-extract t2-ticket-summary t3-python-script" ;;
  faith) PROMPTS="h1-absent-fact h2-conflict h3-ambiguous-tasks" ;;
  *)     echo "unknown suite: $SUITE" >&2; exit 2 ;;
esac

mkdir -p "$OUTDIR"

taskset -c 0-5 "$LLAMA_BIN/llama-server" -m "$MODEL_DIR/$MODEL" \
  -t 6 -c 8192 --jinja --host 127.0.0.1 --port "$PORT" \
  > "$OUTDIR/$NAME.server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for _ in $(seq 1 120); do
  curl -s --max-time 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break
  sleep 2
done
curl -s --max-time 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' || {
  echo "SERVER-FAILED $MODEL"; exit 1; }

for t in $PROMPTS; do
  PROMPT=$(python3 -c "import json;print(json.dumps(open('$TESTS/$t.txt').read()))")
  START=$(python3 -c "import time;print(time.time())")
  # enable_thinking:false is load-bearing — see README. max_tokens caps runaway think loops.
  RESP=$(curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":$PROMPT}],\"temperature\":0,\"max_tokens\":1400,\"stream\":false,\"chat_template_kwargs\":{\"enable_thinking\":false,\"thinking\":false}}")
  END=$(python3 -c "import time;print(time.time())")
  python3 - "$RESP" "$START" "$END" "$OUTDIR/$NAME.$t.txt" <<'PY'
import json, sys
resp, start, end, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
try:
    d = json.loads(resp)
    msg = d["choices"][0]["message"]
    reasoning = msg.get("reasoning_content")
    txt = (f"<think>{reasoning}</think>\n" if reasoning else "") + (msg.get("content") or "")
    u = d.get("usage", {})
    hdr = (f"[elapsed {end-start:.1f}s, completion_tokens {u.get('completion_tokens','?')}, "
           f"prompt_tokens {u.get('prompt_tokens','?')}]\n")
except Exception as e:
    txt = f"PARSE-ERROR {e}\n{resp[:500]}"
    hdr = f"[elapsed {end-start:.1f}s]\n"
open(out, "w").write(hdr + txt)
print(out.split("/")[-1], hdr.strip())
PY
done

kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "QUALITY-DONE $MODEL"
