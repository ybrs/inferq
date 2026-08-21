#!/usr/bin/env python3
"""Read GGUF header KV pairs without numpy/gguf-py, which are not installed here.

Used to answer two questions per model before a single token is generated:
does its chat template know about tools at all, and can it be told not to think?
Both turned out to matter: gemma-3-4b's template never mentions tools, and the
answer for a scout is "this model cannot hold the role", not a low score.
"""
import struct

_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
        7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_kv(path, want):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", f"{path} is not a GGUF file"
    struct.unpack("<I", f.read(4))                      # version
    _, n_kv = struct.unpack("<QQ", f.read(16))

    def s():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def val(t):
        if t == 8:
            return s()
        if t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            return [val(et) for _ in range(n)]
        return struct.unpack(_FMT[t], f.read(_SZ[t]))[0]

    out = {}
    for _ in range(n_kv):
        k = s()
        t, = struct.unpack("<I", f.read(4))
        v = val(t)
        if k in want:
            out[k] = v
    f.close()
    return out


def probe(path):
    kv = read_kv(path, {"tokenizer.chat_template", "general.architecture"})
    t = kv.get("tokenizer.chat_template") or ""
    return {
        "arch": kv.get("general.architecture", "?"),
        "template_has_tools": "tools" in t,
        "template_has_tool_call": "tool_call" in t,
        # Which chat_template_kwargs key, if any, turns thinking off. The name is
        # not standard: Qwen3/Qwen3.5/SmolLM3 read enable_thinking, LFM2.5 reads
        # thinking, and Granite 4.x and Phi-4-mini have no switch at all. Sending
        # the wrong key is silently ignored, which is how the previous run
        # concluded LFM2.5 "ignores enable_thinking=false" - it does, that is not
        # its key.
        "thinking_kwarg": ("enable_thinking" if "enable_thinking" in t
                           else "thinking" if "thinking" in t else None),
        "template_chars": len(t),
    }
