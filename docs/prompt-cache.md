# Prompt caching

An agent opens every task with the same preamble — system prompt, tool
definitions, project context — and prefill is the expensive half of a request
here. Attention scores each new token against every earlier one, so a preamble
costs time that grows with its square: on the four-core benchmark host a
~3,300-token preamble takes over ten minutes before the first token appears,
and it is the same ten minutes every time an agent starts a task.

The prompt cache turns the second and every later occurrence of that preamble
into a file read.

```bash
./target/release/serve \
  --model /models/qwen3.6/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --tokenizer-model /models/qwen3.6 \
  --prompt-cache-dir ~/.cache/inferq/prompts \
  --prompt-cache-mib 20480
```

Nothing is written to disk without `--prompt-cache-dir`. Reuse within a running
process needs no configuration and is always on; `--no-prefix-reuse` turns off
both tiers, which is the behaviour reuse is measured against.

## Measured

Qwen3.6-35B-A3B Q4_K_M (`fnv1a64:7ae605bb0922e4ef`), four threads, no expert
residency (`--expert-cache-mib 0`), `--prompt-cache-block 128
--prompt-cache-min-tokens 256`. A 652-token agent-style prompt: a stable
~630-token preamble of tool descriptions plus one short question, 16 generated
tokens.

| Run | Prefill | Wall time |
| --- | --- | --- |
| First request, empty cache | 652 tokens | 241.4 s |
| Second request, **new process**, different question | 140 tokens (512 restored) | 60.0 s |

The entry was 82.8 MiB and took 0.047 s to write. The second run is a separate
`serve` process with nothing in memory: the 512-token prefix came off disk.
This host has no expert residency configured, so both numbers are dominated by
expert reads; the ratio is what the cache changes, not the absolute rate.

## What is cached

A sequence's state is the whole of what prefill produces, and in this engine it
is plain floats:

| Part | Size for Qwen3.6-35B-A3B |
| --- | --- |
| KV rows, 10 full-attention layers | ~40 KiB per token |
| Conv and recurrent state, 30 linear layers | ~62 MiB, fixed |
| The MTP predictor's own KV rows | ~4 KiB per token |
| Target hidden carry, so the MTP arm resumes | 8 KiB |

The predictor's cache is the reason a boundary prefill also synchronises it.
During generation the MTP block is caught up lazily, only when its arm is about
to draft — the right trade there, since the arm may never draft. At a boundary
that would mean the image could not carry the predictor at all, and every
restored request would decode with that arm silently sitting out. An entry that
cannot supply it is not written, so a session that lost its predictor state
leaves the key free for a later request that can fill it.

So an entry costs roughly `62 MiB + 44 KiB × tokens`: about 240 MiB at 4k
tokens, 420 MiB at 8k. That is why entries are budgeted and evicted, and why
short prefixes are not worth storing (`--prompt-cache-min-tokens`, 512 by
default).

## Where reuse comes from

A request starts from the longest state that is provably a prefix of its own
tokens. Two tiers, checked in that order:

1. **The live session.** When a request extends the conversation the previous
   one left behind — the same tokens, plus more — the state is already correct
   and nothing is read or written. This is the multi-turn case.
2. **The prompt cache.** Otherwise the cache is asked for the longest stored
   prefix above whatever the live session already covers. This is the case that
   survives a restart, and the case where two different tasks share a preamble.

Anything else prefills from empty. In particular, a request whose history
diverges from the live session — an edited conversation, or a reply the client
did not send back verbatim — cannot rewind the live state, because DeltaNet
recurrence is destructive: state can be extended but never un-applied. That is
the reason the cache stores *images* captured at a boundary rather than
trimming a longer state back.

## Boundaries

Entries are stored at token boundaries that are multiples of
`--prompt-cache-block` (256 by default). Quantising the boundary is what lets
two requests that share a preamble but diverge later land on the same key: had
each request stored an entry at its own length, the second would never match.
The cost is at most one block of tokens re-prefilled on a hit.

Choosing *which* boundary matters more than it looks. Storing at the highest
boundary the prompt reaches is wrong: a conversation's final message is the
part that changes, so an entry placed above where two requests diverge is keyed
on tokens nothing else ever sends, and the next request misses everything. The
server therefore tells the engine how much of the prompt is expected to recur —
the conversation rendered without its final message — and the boundary is
capped inside that. The hint is re-encoded and checked to be a genuine token
prefix of the prompt before it is used; a hint that is not is ignored rather
than trusted.

Per request the engine stores at most one entry, at the highest boundary inside
that stable prefix and beyond what the request already reused, and only when
that boundary is not already on disk. So the first task with a given preamble
writes one entry, and later tasks that share it write nothing until the
conversation grows past the next boundary.

Writing happens on its own thread, after the response has been delivered. A
write already in flight means the next one is skipped rather than queued, since
a queued image is a whole state copy held in memory.

## Keys, and what makes a hit safe

The file name carries a fingerprint of the model, the prefix length, and a hash
of the token ids. A hash is never trusted on its own: the entry stores the token
ids themselves, and they are compared exactly before any state is read. A
collision therefore costs one header read, not a wrong answer.

The fingerprint is the GGUF's layout hash — file size, mtime, and every
tensor's name, dtype, shape and offset — folded together with a hash of the
whole `config.json`. The GGUF alone is not enough: norm epsilon, RoPE base, the
layer pattern and every head dimension come from the configuration, and two
revisions of it can name the same weights while producing different state.

An entry is rejected — and deleted — when its format version, its fingerprint,
its layer sequence, or any tensor's length disagrees with what is loaded. Each
full-attention layer's rows are checked against the width this model's KV cache
actually uses, so an entry from a differently shaped model is refused rather
than restored into a kernel that would read it as another shape. Entries belonging to another checkpoint are left alone but still count
against the budget, so a shared directory stays bounded; they are the first
thing eviction reclaims.

Entries are written to a temporary name and renamed into place, so a reader
sees either the whole entry or none of it, and a process killed mid-write
leaves nothing a later run will read.

## Correctness

Restoring is exact. `tests/prompt_cache.rs` asserts, against the real
checkpoint, that a session rebuilt from an image — including one round-tripped
through disk into a fresh runtime — decodes token for token like the same
session that was never interrupted, with speculation both off and on. The n-gram
index is reseeded from the restored prefix and the MTP predictor's own cache is
restored with it, so a restored request speculates as well as a prefilled one
rather than silently decoding slower.

One caveat worth stating plainly: prefilling `[0..b]` and then `[b..n]` is not
bit-for-bit the same computation as prefilling `[0..n]` in one pass — the
batched reductions differ in their last bits, as they do for any chunked
prefill. Reuse reproduces the chunked result exactly. Both are the model; the
difference is the same one a multi-turn conversation already has.

## Privacy

An entry contains the token ids of the prefix it describes, which is
recoverable prompt text, alongside state derived from it. The directory is
created owner-only (`0700`) and nothing is written unless `--prompt-cache-dir`
is given. Point it at an encrypted volume if that matters, or leave it off and
keep the live-session tier only.

## Observing it

Every request logs where it started from:

```
INFO request complete prompt_tokens=743 completion_tokens=16
     reused_tokens=640 reuse=cache finish=Stop seconds=8.9
```

`reuse` is `none`, `live`, or `cache`. `GET /health` reports the cache's
counters — entries, bytes against the budget, hits, misses, reused tokens,
writes, skipped writes, evictions, and rejected entries:

```json
{"prompt_cache": {"entries": 3, "bytes": 754974720, "budget_bytes": 21474836480,
                  "hits": 12, "misses": 1, "reused_tokens": 7680, "writes": 3,
                  "write_skips": 0, "evictions": 0, "failures": 0}}
```

## Inspecting an entry

Entries are safetensors files, so a cached state can be opened without this
engine:

```python
from safetensors import safe_open
with safe_open("~/.cache/inferq/prompts/7ae6...-00004096-....inferq-prompt", "pt") as f:
    print(f.metadata())            # format, checkpoint fingerprint, position
    print(f.get_slice("tokens").get_shape())
```

## Limits

- One entry per request, at one boundary. A prompt that diverges below the
  highest boundary re-prefills from the next boundary down, not from the exact
  divergence point.
- A turn that ends early — a stop string, a client disconnect — can leave the
  live session without its predictor state until the next reset, and requests
  reusing that session decode without the MTP arm. Nothing is stored from such
  a session, so the cache itself never inherits the problem.
- No compression: state is dense floats and compresses poorly.
- No sharing between models — entries are per checkpoint fingerprint.
- Nothing is deduplicated between entries, so two prefixes that share their
  first 4k tokens store those tokens' state twice.
