#!/bin/bash
# Run one prompt suite against one model, via llama-server's chat endpoint so the
# model's own chat template applies. Writes one file per prompt to outputs/.
#
# Usage: quality.sh <model-file.gguf> [suite]
#   suite: "task" (t1-t3, default), "faith" (h1-h3), "control" (c1-c2),
#          "cap" (translation, code reading, DAG/script, diagram),
#          or "tools" (picking taskq MCP tools for a request)
#
# Env:
#   SYSTEM=<file>  prepend this file as a system message. The user prompts stay
#                  byte-identical to the unguarded run, so scores are directly
#                  comparable. Used for the prompt-guard experiment.
#   OUTDIR=<dir>   where to write answers (default outputs/)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_BIN="${LLAMA_BIN:-/models/llamacpp-main/build/bin}"
MODEL_DIR="${MODEL_DIR:-/models/small-models}"
TESTS_ENV="${TESTS:-}"
TESTS="${TESTS_ENV:-$HERE/tests}"
OUTDIR="${OUTDIR:-$HERE/outputs}"
PORT="${PORT:-8099}"

MODEL="$1"
SUITE="${2:-task}"
NAME="${MODEL%.gguf}"

case "$SUITE" in
  task)  PROMPTS="t1-task-extract t2-ticket-summary t3-python-script" ;;
  faith) PROMPTS="h1-absent-fact h2-conflict h3-ambiguous-tasks" ;;
  control) PROMPTS="c1-answerable-fact c2-owned-tasks" ;;
  cap)   TESTS="${TESTS_ENV:-$HERE/tests-capability}"
         PROMPTS="x1-nl-ticket x2-nl-technical x3-nl-falsefriends
                  y1-rust-summary y2-python-summary y3-file-routing y4-file-routing-2
                  z1-dag-json z2-python-script z3-mermaid z4-matplotlib" ;;
  tools) TESTS="${TESTS_ENV:-$HERE/tests-capability}"
         PROMPTS="w1-tool-update w2-tool-search w3-tool-create" ;;
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
  # enable_thinking:false is load-bearing — see README. max_tokens caps runaway think loops.
  BODY=$(SYSTEM="${SYSTEM:-}" python3 -c "
import json, os, sys
msgs = []
sysf = os.environ.get('SYSTEM')
if sysf:
    msgs.append({'role': 'system', 'content': open(sysf).read()})
msgs.append({'role': 'user', 'content': open(sys.argv[1]).read()})
print(json.dumps({'messages': msgs, 'temperature': 0, 'max_tokens': 1400, 'stream': False,
                  'chat_template_kwargs': {'enable_thinking': False, 'thinking': False}}))
" "$TESTS/$t.txt")
  START=$(python3 -c "import time;print(time.time())")
  RESP=$(curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$BODY")
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
