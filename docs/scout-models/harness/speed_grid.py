#!/usr/bin/env python3
"""Every speed number, at every depth, cold and warm. No condition is chosen for you.

Two independent measurements, because they answer different questions and the
old report conflated them:

  A. llama-bench  - clean, repeatable, no HTTP or chat template in the path.
     Comparable to any published benchmark. Run by bench_grid.sh.
  B. llama-server - what a scout actually experiences: the model's own chat
     template, the tool schemas, HTTP, and llama.cpp's prompt cache. This file.

Depths measured (B):
  128    - the depth llama-bench tg128 reports, and what most benchmarks quote
  901    - what the previous evaluation's tool test actually sent (a stripped
           name+one-line listing of the 46 tools, measured from its own output)
  7427   - what the 46 real taskq MCP schemas actually cost
Each depth is measured twice with plain filler and, at 7427, also with the real
tool block, so content type is separated from depth.

Passes:
  cold - the prompt has never been seen; llama.cpp prefills all of it
  warm - the identical prompt again; the prefix is served from the KV cache

Nothing here decides which number matters. All of them are recorded.
"""
import json, os, sys, time
import serverctl as sc
import hostcheck as hc

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA_BIN = os.environ.get("LLAMA_BIN", "/models/llamacpp-main/build/bin")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/small-models")
PORT = int(os.environ.get("PORT", "8099"))
CPUS, THREADS, CTX = "0-5", 6, 16384
FA = os.environ.get("EVAL_FA", "off")
GEN = int(os.environ.get("GRID_GEN", "64"))      # decode tokens per measurement
DEPTHS = [128, 901, 7427]
WORD = "The quick brown fox jumps over the lazy dog. "


def calibrate(srv, target, nonce):
    """Filler whose rendered prompt lands within a few tokens of `target`.

    Built by measuring, not estimating: tokenizers differ per model, so a
    character-count heuristic would make the same 'depth' mean different depths
    on different models and silently un-fair the comparison.
    """
    lo, hi = 1, 4000
    best = None
    for _ in range(14):
        mid = (lo + hi) // 2
        msgs = [{"role": "user", "content": nonce + " " + WORD * mid + "\nSay OK."}]
        p = srv.post("/apply-template", {"messages": msgs})["prompt"]
        n = len(srv.post("/tokenize", {"content": p, "add_special": True})["tokens"])
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (msgs, n)
        if n < target:
            lo = mid + 1
        elif n > target:
            hi = mid - 1
        else:
            return msgs, n
        if lo > hi:
            break
    return best


def measure(srv, msgs, tools, ctk, label):
    body = {"messages": msgs, "temperature": 0, "top_k": 1, "top_p": 1.0, "seed": 42,
            "repeat_penalty": 1.0, "max_tokens": GEN, "stream": False}
    if tools:
        body.update({"tools": tools, "tool_choice": "auto"})
    if ctk:
        body["chat_template_kwargs"] = ctk
    with sc.step(label, timeout=300):
        d, wall = srv.chat(body, timeout=280)
    t = d.get("timings") or {}
    return {
        "prompt_n": t.get("prompt_n"),
        "cache_n": t.get("cache_n"),
        "prefill_tps": round(t.get("prompt_per_second") or 0, 1),
        "decode_n": t.get("predicted_n"),
        "decode_tps": round(t.get("predicted_per_second") or 0, 2),
        "depth_total": (t.get("prompt_n") or 0) + (t.get("cache_n") or 0),
        "wall_s": round(wall, 2),
        "finish_reason": d["choices"][0].get("finish_reason"),
    }


def grid_for(entry, tools, tool_system):
    name = entry["file"]
    path = os.path.join(MODEL_DIR, name)
    kwarg = entry.get("thinking_kwarg") if entry.get("thinking_kwarg_effective") else None
    ctk = {kwarg: False} if kwarg else {}
    out = {"file": name, "arch": entry["arch"], "size_bytes": entry["size_bytes"],
           "fa": FA, "gen_tokens": GEN, "thinking": "off" if kwarg else "n/a",
           "conditions": {}, "host": hc.snapshot(CPUS)}

    srv = sc.LlamaServer(LLAMA_BIN, path, PORT, CPUS, THREADS, CTX,
                         os.path.join(HERE, "speed-grid.log"), fa=FA)
    try:
        with sc.step(f"start[{name}]", timeout=300) as s:
            srv.start()
            srv.wait_healthy(280)
        out["load_seconds"] = round(s.elapsed, 1)
        srv.start_liveness_watch()
        srv.assert_props(CTX)
        srv.assert_not_mmapped()
        srv.assert_serving(path)
        with sc.step(f"warmup[{name}]", timeout=280):
            srv.chat({"messages": [{"role": "user", "content": "Say OK."}],
                      "max_tokens": 8, "temperature": 0, "stream": False}, timeout=280)

        conds = []
        for d in DEPTHS:
            conds.append((f"d{d}-filler", d, None))
        conds.append(("d7427-tools", None, "tools"))

        for label, target, kind in conds:
            try:
                if kind == "tools":
                    msgs = [{"role": "system", "content": tool_system},
                            {"role": "user", "content":
                             "Ticket 405 is finished. Mark it done and record a note "
                             "saying the benchmark is committed on branch scout-models."}]
                    use_tools, n = tools, None
                else:
                    # a nonce unique per condition, so no condition warms another
                    msgs, n = calibrate(srv, target, f"[probe {label} {name}]")
                    use_tools = None
                cold = measure(srv, msgs, use_tools, ctk, f"{name}/{label}/cold")
                warm = measure(srv, msgs, use_tools, ctk, f"{name}/{label}/warm")
                out["conditions"][label] = {"target_depth": target, "cold": cold, "warm": warm}
                print(f"    {label:16} cold: prefill {cold['prompt_n']:>5}@{cold['prefill_tps']:>7.1f}"
                      f"  decode {cold['decode_tps']:>6.2f} | "
                      f"warm: cached {warm['cache_n']:>5} prefill {warm['prompt_n']:>4}"
                      f"  decode {warm['decode_tps']:>6.2f}", flush=True)
            except sc.StepTimeout as e:
                out["conditions"][label] = {"state": "TOO_SLOW", "why": str(e)}
                print(f"    {label:16} TOO SLOW ({e})", flush=True)
                srv.stop()
                with sc.step(f"restart[{name}]", timeout=300):
                    srv.start(); srv.wait_healthy(280)
                srv.start_liveness_watch()
            except sc.FatalRunError as e:
                out["conditions"][label] = {"state": "ERROR", "why": str(e)[:300]}
                print(f"    {label:16} ERROR {str(e)[:120]}", flush=True)
                if not srv.alive():
                    out["aborted_after"] = label
                    break
    except sc.FatalRunError as e:
        out["fatal"] = str(e)
        print(f"    FATAL {e}", flush=True)
    finally:
        try:
            srv.stop()
        except Exception as e:
            out["stop_problem"] = str(e)
    return out


def main():
    vpath = os.path.join(HERE, "roster-verified.json")
    src = json.load(open(vpath))["models"] if os.path.exists(vpath) \
        else json.load(open(os.path.join(HERE, "roster.json")))
    models = [m for m in src if "fatal" not in m]
    if len(sys.argv) > 1:
        want = sys.argv[1].split(",")
        models = [m for m in models if any(w in m["file"] for w in want)]
    tools = json.load(open(os.path.join(HERE, "taskq-tools.json")))
    tool_system = json.load(open(os.path.join(HERE, "scenarios-tools.json")))["system"]

    probs, notes = sc.preflight(LLAMA_BIN, PORT, CPUS, THREADS)
    if probs:
        for p in probs:
            print("BLOCKED:", p)
        sys.exit(1)
    print(f"speed grid: {len(models)} models, depths {DEPTHS} + real tool block, "
          f"cold and warm, fa={FA}, {GEN} decode tokens\n")

    results, t0 = [], time.monotonic()
    for i, m in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {m['file']}  ({(time.monotonic()-t0)/60:.0f} min)", flush=True)
        results.append(grid_for(m, tools, tool_system))
        json.dump(results, open(os.path.join(HERE, "speed-grid.json"), "w"), indent=1)
    print(f"\nGRID-DONE in {(time.monotonic()-t0)/60:.1f} min -> speed-grid.json")


if __name__ == "__main__":
    main()
