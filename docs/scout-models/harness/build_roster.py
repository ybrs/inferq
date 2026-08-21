#!/usr/bin/env python3
"""Write roster.json: every distinct GGUF in MODEL_DIR with what its template supports.

Byte-identical files are collapsed to one entry. The old report counted
SmolLM3-3B-Q4_K_M.gguf and SmolLM3-Q4_K_M.gguf as two separate models; they
have the same md5.
"""
import glob, hashlib, json, os
from gguf_meta import probe

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/small-models")


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


seen, roster = {}, []
for p in sorted(glob.glob(os.path.join(MODEL_DIR, "*.gguf"))):
    name = os.path.basename(p)
    digest = md5(p)
    if digest in seen:
        print(f"skip {name}: byte-identical to {seen[digest]}")
        continue
    seen[digest] = name
    e = {"file": name, "size_bytes": os.path.getsize(p), "md5": digest}
    e.update(probe(p))
    roster.append(e)

roster.sort(key=lambda e: e["size_bytes"])
json.dump(roster, open("roster.json", "w"), indent=1)
print(f"\n{len(roster)} distinct models")
print(f"{'file':40} {'arch':14} {'GiB':>5} tools think-kwarg")
for e in roster:
    print(f"{e['file']:40} {e['arch']:14} {e['size_bytes']/2**30:5.2f} "
          f"{str(e['template_has_tools']):5} {e['thinking_kwarg']}")
