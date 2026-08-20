#!/bin/bash
# Send one prompt file to the server started by serve.sh and print the answer.
# The loop for testing a prompt guard is: edit the file, run this, read the answer.
#
# Usage: ./ask.sh tests/h1-absent-fact.txt
#        ./ask.sh my-guarded-prompt.txt
#        echo "some prompt" | ./ask.sh -
set -u

PORT="${PORT:-8099}"
SRC="${1:?usage: ask.sh <prompt-file|->}"

if [ "$SRC" = "-" ]; then
  PROMPT=$(python3 -c "import json,sys;print(json.dumps(sys.stdin.read()))")
else
  PROMPT=$(python3 -c "import json;print(json.dumps(open('$SRC').read()))")
fi

curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":$PROMPT}],\"temperature\":0,\"max_tokens\":${MAX_TOKENS:-600},\"stream\":false,\"chat_template_kwargs\":{\"enable_thinking\":false,\"thinking\":false}}" \
| python3 -c '
import json, sys
d = json.load(sys.stdin)
m = d["choices"][0]["message"]
if m.get("reasoning_content"):
    print("<think>" + m["reasoning_content"] + "</think>")
print(m.get("content") or "")
u = d.get("usage", {})
sys.stderr.write("\n--- %s tokens\n" % u.get("completion_tokens", "?"))
'
