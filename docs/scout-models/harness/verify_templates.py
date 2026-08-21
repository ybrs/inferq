#!/usr/bin/env python3
"""Check every model one by one before any sweep, and rebuild the roster from
what is measured rather than from what a template's text looks like.

Guards FAILURE-MODES 9, 12, 15, 16, 17, 18, 19, 20, 21.

The motivating bug: roster.json said Phi-4-mini supports tools because the
string "tools" appears in its chat template. It appears inside
`message['tools']` - a key on a system message - and the top-level OpenAI tools
array is never rendered. Scoring that model's "tool calling" would have
measured llama.cpp's generic fallback instead. The only reliable test is to
render a prompt through the real template and look for the tool in it.

Every model gets: an md5 re-check, a /props read-back, a not-mmapped check, a
tool-rendering probe, a thinking-kwarg probe, a render of every scenario, a
double-BOS check, and one real generation plus one real tool call.

Writes roster-verified.json and template-report.md. Nothing else may run
against a roster this script has not stamped.
"""
import hashlib, json, os, sys, time
import serverctl as sc
import hostcheck as hc
from gguf_meta import read_kv

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA_BIN = os.environ.get("LLAMA_BIN", "/models/llamacpp-main/build/bin")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/small-models")
PORT = int(os.environ.get("PORT", "8099"))
CPUS, THREADS = "0-5", 6
FA = os.environ.get("EVAL_FA", "off")     # measured 2.7x faster than on, at depth

PROBE_TOOL = [{"type": "function", "function": {
    "name": "zzz_probe_tool_xyz",
    "description": "A probe tool that exists only to see whether the chat template renders tools.",
    "parameters": {"type": "object", "properties": {"probe_arg": {"type": "string"}},
                   "required": ["probe_arg"]}}}]


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def rendered(srv, messages, tools=None, ctk=None):
    """Render through the model's own template. Returns text, or an error string."""
    body = {"messages": messages}
    if tools:
        body["tools"] = tools
    if ctk:
        body["chat_template_kwargs"] = ctk
    try:
        r = srv.post("/apply-template", body)
    except sc.FatalRunError as e:
        return None, str(e)[:200]
    return r.get("prompt", ""), None


def check_model(entry, tool_spec, qual_spec):
    name = entry["file"]
    path = os.path.join(MODEL_DIR, name)
    r = {"file": name, "arch": entry["arch"], "size_bytes": entry["size_bytes"]}

    # 21: the file must still be the one we hashed into the roster
    r["md5_ok"] = (md5(path) == entry["md5"])
    if not r["md5_ok"]:
        r["fatal"] = "md5 changed since roster was built"
        return r

    # 12: the model's own trained context, so we never ask for more
    kv = read_kv(path, {f"{entry['arch']}.context_length", "general.architecture"})
    r["context_length"] = kv.get(f"{entry['arch']}.context_length")
    ctx = min(int(r["context_length"] or 16384), 16384)
    r["ctx_used"] = ctx

    srv = sc.LlamaServer(LLAMA_BIN, path, PORT, CPUS, THREADS, ctx,
                         os.path.join(HERE, "verify-templates.log"), fa=FA)
    try:
        with sc.step(f"start[{name}]", timeout=300) as st:
            srv.start()
            srv.wait_healthy(timeout=280)
        r["load_seconds"] = round(st.elapsed, 1)
        srv.start_liveness_watch()

        with sc.step(f"props[{name}]", timeout=60):
            p = srv.assert_props(ctx)                      # 9
            srv.assert_not_mmapped()                       # 9
            srv.assert_serving(path)                       # stale-server guard
        r["props_total_slots"] = p.get("total_slots")
        r["mmapped"] = False

        msgs = [{"role": "user", "content": "Say OK."}]

        # 15: does the template actually render a tools array?
        with sc.step(f"tools-render[{name}]", timeout=60):
            with_t, err_t = rendered(srv, msgs, tools=PROBE_TOOL)
            without_t, _ = rendered(srv, msgs)
        if err_t:
            r["template_renders_tools"] = False
            r["tools_render_error"] = err_t
        else:
            r["template_renders_tools"] = ("zzz_probe_tool_xyz" in (with_t or ""))
            r["tools_change_prompt"] = (with_t != without_t)

        # 16: does the thinking kwarg actually do anything?
        kwarg = entry.get("thinking_kwarg")
        if kwarg:
            with sc.step(f"think-render[{name}]", timeout=60):
                on, _ = rendered(srv, msgs, ctk={kwarg: True})
                off, _ = rendered(srv, msgs, ctk={kwarg: False})
            r["thinking_kwarg"] = kwarg
            r["thinking_kwarg_effective"] = (on is not None and on != off)
        else:
            r["thinking_kwarg"] = None
            r["thinking_kwarg_effective"] = False

        # 18: every scenario must survive this template, s08's tool turn included
        bad = {}
        with sc.step(f"scenario-render[{name}]", timeout=200):
            for sc_ in tool_spec["scenarios"]:
                m = [{"role": "system", "content": tool_spec["system"]}] + sc_["messages"]
                txt, err = rendered(srv, m, tools=PROBE_TOOL)
                if err:
                    bad[sc_["id"]] = f"render error: {err}"
                elif sc_["id"] == "s08-multiturn-followup":
                    # the id and the tool name must reach the model, or the
                    # scenario is testing the template, not the model
                    miss = [s for s in ("388", "search_taskq") if s not in txt]
                    if miss:
                        bad[sc_["id"]] = f"tool-result turn dropped {miss}"
            for sc_ in qual_spec["scenarios"]:
                m = [{"role": "system", "content": qual_spec["system"]}] + sc_["messages"]
                _, err = rendered(srv, m)
                if err:
                    bad[sc_["id"]] = f"render error: {err}"
        r["unrenderable_scenarios"] = bad

        # What the real 46-tool block costs this tokenizer. Rendering and
        # tokenizing needs no inference, so the number is free; actually
        # prefilling it is what took minutes per model and is measured elsewhere.
        with sc.step(f"toolblock-size[{name}]", timeout=60):
            m2 = [{"role": "system", "content": tool_spec["system"]},
                  {"role": "user", "content": "Mark ticket 405 done."}]
            full_p, _ = rendered(srv, m2, tools=tools)
            bare_p, _ = rendered(srv, m2)
            if full_p is not None and bare_p is not None:
                nf = len(srv.post("/tokenize", {"content": full_p, "add_special": True})["tokens"])
                nb = len(srv.post("/tokenize", {"content": bare_p, "add_special": True})["tokens"])
                r["prompt_tokens_46_tools"] = nf
                r["prompt_tokens_no_tools"] = nb
                r["tool_block_tokens"] = nf - nb

        # 20: a BOS added by both template and tokenizer shifts every position
        with sc.step(f"bos[{name}]", timeout=60):
            txt, _ = rendered(srv, msgs)
            toks = srv.post("/tokenize", {"content": txt or "", "add_special": True}).get("tokens", [])
        r["first_tokens"] = toks[:3]
        r["double_bos"] = len(toks) >= 2 and toks[0] == toks[1]

        # does it actually generate, and actually call a tool?
        with sc.step(f"generate[{name}]", timeout=280):
            d, _ = srv.chat({"messages": msgs, "temperature": 0, "max_tokens": 16,
                             "stream": False})
        r["generates"] = bool((d["choices"][0]["message"].get("content") or "").strip())

        tools = json.load(open(os.path.join(HERE, "taskq-tools.json")))
        probe_tools = [t for t in tools if t["function"]["name"] in
                       ("set_task_status", "add_comment", "create_task",
                        "search_taskq", "get_task")]
        ctk = {kwarg: False} if kwarg else {}
        with sc.step(f"toolcall[{name}]", timeout=120):
            body = {"messages": [{"role": "system", "content": tool_spec["system"]},
                                 {"role": "user", "content":
                                  "Ticket 405 is finished. Mark it done and record a note "
                                  "saying the benchmark is committed on branch scout-models."}],
                    "tools": probe_tools, "tool_choice": "auto", "temperature": 0,
                    "top_k": 1, "seed": 42, "max_tokens": 200, "stream": False}
            if ctk:
                body["chat_template_kwargs"] = ctk
            d2, _ = srv.chat(body)
        tc = d2["choices"][0]["message"].get("tool_calls") or []
        t = d2.get("timings", {})
        r["emits_tool_call"] = bool(tc)
        r["first_tool"] = tc[0]["function"]["name"] if tc else None
        r["finish_reason"] = d2["choices"][0].get("finish_reason")
        # Rates from the SMALL probe only. Not comparable across models and not
        # to be reported as speed: bench_grid.py and speed_grid.py do that.
        r["probe_prefill_n"] = t.get("prompt_n")
        r["probe_prefill_tps"] = round(t.get("prompt_per_second", 0), 1)
        r["probe_decode_tps"] = round(t.get("predicted_per_second", 0), 2)
        r["chat_format"] = srv.chat_format()                # 13/17
        # 19: reasoning must not leak into content
        c = d2["choices"][0]["message"].get("content") or ""
        r["reasoning_leak"] = c.lstrip().startswith(("<think", "<|thinking", "<seed:think"))
    except sc.StepTimeout as e:
        # Not a crash: this model is too slow for the cap. Recorded as such, with
        # whatever rates were gathered before it ran out of time.
        r["too_slow"] = str(e)
    except sc.FatalRunError as e:
        r["fatal"] = str(e)
    except Exception as e:
        r["fatal"] = f"unexpected {type(e).__name__}: {e}"
    finally:
        try:
            srv.stop()
        except Exception as e:
            r["stop_problem"] = str(e)
    return r


def main():
    roster = json.load(open(os.path.join(HERE, "roster.json")))
    tool_spec = json.load(open(os.path.join(HERE, "scenarios-tools.json")))
    qual_spec = json.load(open(os.path.join(HERE, "scenarios-quality.json")))

    print("PRE-FLIGHT")
    probs, notes = sc.preflight(LLAMA_BIN, PORT, CPUS, THREADS)
    for k, v in notes.items():
        print(f"  {k:26} {v}")
    if probs:
        for p in probs:
            print(f"  BLOCKED: {p}")
        sys.exit(1)
    host = hc.snapshot(CPUS)
    print(f"  governor {host['governors']['cpu0']['scaling_governor']}, "
          f"turbo {'on' if host['governors']['no_turbo'] == '0' else 'OFF'}, "
          f"{host['cpu_mhz_mean']} MHz, {host['coretemp_c']}C")
    print(f"  flash-attn = {FA}\n")

    out = []
    for i, e in enumerate(roster, 1):
        print(f"[{i}/{len(roster)}] {e['file']}", flush=True)
        r = check_model(e, tool_spec, qual_spec)
        out.append(r)
        if "fatal" in r:
            print(f"    FATAL: {r['fatal']}", flush=True)
        elif "too_slow" in r:
            print(f"    TOO SLOW: {r['too_slow']}", flush=True)
        else:
            print(f"    renders_tools={r['template_renders_tools']} "
                  f"think={r['thinking_kwarg']}/{r['thinking_kwarg_effective']} "
                  f"tool_call={r['first_tool']} "
                  f"46-tool prompt={r.get('prompt_tokens_46_tools')} tok "
                  f"load={r.get('load_seconds')}s", flush=True)
            if r["unrenderable_scenarios"]:
                print(f"    UNRENDERABLE: {r['unrenderable_scenarios']}", flush=True)
            if r["double_bos"]:
                print(f"    DOUBLE BOS: {r['first_tokens']}", flush=True)
            if r["reasoning_leak"]:
                print("    REASONING LEAKS INTO content", flush=True)

    stamp = {"verified_at": time.time(), "llama_bin": LLAMA_BIN, "fa": FA,
             "host": host, "models": out}
    json.dump(stamp, open(os.path.join(HERE, "roster-verified.json"), "w"), indent=1)
    print(f"\nwrote roster-verified.json ({len(out)} models)")


if __name__ == "__main__":
    main()
