#!/usr/bin/env python3
"""Pull the live taskq MCP tool schemas and write them as an OpenAI tools array.

The point of doing this over the wire instead of typing a list into a text file
is that the eval then tests the tools the scout would actually be handed, with
the argument schemas it would actually have to fill in. A hand-written listing
of names can only test whether a model can copy a name out of a list.

Usage: fetch_tools.py [out.json]
"""
import json, os, sys, urllib.request

URL = os.environ.get("TASKQ_MCP_URL", "http://172.17.0.1:7777/mcp")
TOK = os.environ.get("TASKQ_MCP_TOKEN", "TASKQ_MCP_TOKEN_2398329r0qa")
SID = {"v": None}


def rpc(method, params=None, notify=False):
    body = {"jsonrpc": "2.0", "method": method}
    if not notify:
        body["id"] = 1
    if params is not None:
        body["params"] = params
    h = {"Authorization": "Bearer " + TOK, "Content-Type": "application/json",
         "Accept": "application/json, text/event-stream"}
    if SID["v"]:
        h["Mcp-Session-Id"] = SID["v"]
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), h), timeout=60) as r:
        if r.headers.get("Mcp-Session-Id"):
            SID["v"] = r.headers["Mcp-Session-Id"]
        raw = r.read().decode()
    if notify:
        return None
    for line in raw.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    return json.loads(raw)


def main():
    rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "scout-eval", "version": "2"}})
    rpc("notifications/initialized", {}, notify=True)
    tools = rpc("tools/list", {})["result"]["tools"]
    oai = [{"type": "function",
            "function": {"name": t["name"],
                         "description": t.get("description", ""),
                         "parameters": t["inputSchema"]}}
           for t in sorted(tools, key=lambda x: x["name"])]
    out = sys.argv[1] if len(sys.argv) > 1 else "taskq-tools.json"
    json.dump(oai, open(out, "w"), indent=1)
    print(f"{len(oai)} tools -> {out} ({len(json.dumps(oai))} chars)")


if __name__ == "__main__":
    main()
