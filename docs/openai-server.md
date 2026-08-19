# OpenAI-compatible server

`serve` puts an HTTP surface in front of the quantized runtime, so anything
that speaks the OpenAI chat-completions API — the `openai` SDK, Aider,
Continue, Open WebUI, `curl` — can drive this engine the way it would drive
`llama-server`.

Build it the way every other benchmark in this repository is built — for the
host's own instruction set. A stock `cargo build --release` targets baseline
x86-64, which on an AVX2 host halves decode and costs more than that on
prefill; measured on an i7-8700 at six threads, the same 218-token turn ran at
4.00 tok/s from `target/` and 8.25 tok/s from `target-native/`, token for
token identical.

```bash
CARGO_TARGET_DIR=target-native RUSTFLAGS='-C target-cpu=native' \
  cargo build --release --bin serve

./target-native/release/serve \
  --model /models/qwen3.6/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --tokenizer-model /models/qwen3.6 \
  --host 127.0.0.1 --port 8080 \
  --api-key "$(openssl rand -hex 24)" \
  --expert-cache-mib 46000 --warmup-all-experts
```

The process loads the model before it binds the port, so a bad checkpoint
fails at startup rather than on the first request. `Ctrl-C` shuts down
gracefully, letting in-flight responses finish.

Every request logs a `timings` line beside its `request complete` line, with
prefill and decode reported separately — they are paid for differently, and one
figure over both cannot say which half a slow request spent its time in:

```
timings prefill_tokens=11 prefill_seconds=0.661 prefill_tokens_per_second=16.6
        decode_tokens=218 decode_seconds=26.4 decode_tokens_per_second=8.25
        time_to_first_token_seconds=0.661 drafted_tokens=209
        accepted_draft_tokens=142 draft_acceptance=0.679
```

`draft_acceptance` is the share of speculative draft tokens the target kept; a
run that reports `drafted_tokens=0` is decoding without either arm.

A turn that ends early reports the same fields from the same measurements. An
agentic client ends every turn early — the engine stops at the first closed
`</tool_call>` rather than letting the model write call after call — and those
turns are the ones whose numbers matter most, so their prefill is the prefill
they actually paid for rather than the boundary pass alone, and the prompt is
not folded into the decode figure.

## Decode against context depth

Every throughput figure elsewhere in this repository is measured on a short
prompt — the qualified sustained case reaches 151 context tokens — and decode
is not the same operation at agent depths. The full-attention layers scan a KV
cache that grows with the conversation; the linear and MoE layers cost the same
at any depth. Measured with `gguf_decode_depth` on an i7-8700, six threads,
Qwen3.6-35B-A3B Q4_K_M, native build, speculation off, sixteen decode passes
per depth:

| context | decode tok/s | scan | prefill tok/s |
| ---: | ---: | ---: | ---: |
| 64 | 7.95 | 0.02 s | 13.1 |
| 512 | 7.44 | 0.11 s | 12.4 |
| 1024 | 7.40 | 0.19 s | 13.4 |
| 2048 | 6.72 | 0.39 s | 12.4 |
| 3072 | 6.14 | 0.61 s | 11.7 |

(The scan column is the wall time of the whole parallel region over sixteen
decode passes; `scores`, `softmax` and `weighted_sum` beside it in the profile
are summed across threads and are a share of each other, not of it.)

The scan was until recently the whole story at depth: it ran on one core, and
at 3072 context it was 3.38 s of a 5.36 s pass — 63% of decode, against 2.6%
at 64 tokens. Every other stage was flat across the range, so a long request
was not slow in general, it was slow in one loop. Two changes fixed that, in
the order the measurements asked for.

First the loop was split across the heads, which are independent. That is an
arithmetic identity — every reduction still runs in its original order — so it
cost nothing in output and took decode at 3072 from 2.97 to 6.14 tok/s.

Splitting the scan into its three operations then said where the rest was: the
QK dot was 73% of the scan's CPU time, the weighted sum 26%, the softmax 1%.
The dot was a chain of dependent FMAs, since `f32` addition is not associative
and a compiler may not reassociate the reduction; measured 0.501 FLOP/cycle
per core against 0.500 for exactly that shape. Giving it eight independent
lane accumulators broke the chain and let it vectorise, for 2.5x on the dot
and decode at 3072 from 6.14 to 7.06 tok/s.

| context | original | +threads | +lanes | overall |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 5.53 | 7.40 | 7.80 | 1.41x |
| 3072 | 2.97 | 6.14 | 7.06 | **2.38x** |

Prefill gained twice over, and for a different reason the second time. The
threading and the lane accumulators took it from 6.02 to 13.13 tok/s at 3072,
a prefill pass being the same scan over more rows. Then it turned out a pass
was also computing logits for every position while sampling only the last —
28% of it, over a 248,320-wide vocabulary — which is unused work rather than a
trade, since rows are independent. Dropping it took prefill to 17.55 tok/s at
3072 and left the LM head at 0.1% of a pass. Then the dispatch turned out to
be handing wide passes to the per-row kernel — it only used the fused one for
two to sixteen rows — so a 256-row prefill was decoding each weight block once
per row. Tiling those passes into sixteen-row groups took the DeltaNet
projections from 3.76 s to 1.48 s and prefill to 21.48 tok/s at 3072, or 24.54
measured on the pass alone. That one is bit-identical: on a build with FMA the
per-row and multi-row kernels agree exactly, which is also why the engine now
refuses to open a checkpoint without it. Decode is
untouched: a one-token pass is its own last row.

That left a prefill pass at width 256 spending 47.6% in the MoE and 42.8% in
the linear layers, 90% of it between them, with the MoE the larger of the two
again. Then the MoE turned out to be running its experts one at a time while
each expert's matmul split its own few hundred output rows across the pool —
20,480 fork/joins in a forty-layer pass, each dividing work smaller than one
core's share — and reaching Candle's per-row kernel rather than the fused one,
because expert matrices are under the 4 MiB dispatch threshold. Inverting that,
so the 256 experts are the parallel unit and each matmul runs whole on one
thread through the fused kernel, halved MoE compute:

| width | prefill tok/s | one pass ms/token | MoE | MoE compute | linear |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 19.83 → 23.83 | 59.76 → 45.33 | 34.80 → 20.33 | 28.37 → 14.52 | 19.92 → 19.80 |
| 64 | 25.34 → 31.69 | 42.07 → 31.86 | 21.59 → 11.56 | 16.93 → 8.01 | 16.88 → 16.81 |
| 256 | 26.37 → **32.86** | 35.71 → 28.63 | 16.13 → 9.24 | 12.21 → 6.12 | 16.11 → 15.97 |
| 512 | 27.54 → 32.96 | 36.29 → 29.40 | 16.00 → 8.96 | 11.92 → 5.88 | 16.58 → 16.59 |

Bit-identical, and asserted rather than argued: which thread computes an
output row cannot change it, the weighted accumulation is still a serial pass
in token-then-route order, and the grouped path is tested equal to the
token-major reference exactly rather than to a tolerance. Decode is untouched —
one row is still token-major and still on Candle's matvec.

So the MoE has stopped being the larger half. At width 256 the linear layers
are now 55.8% of a pass against the MoE's 32.3%. Prefill spends itself on
weights; decode increasingly on the cache. The two halves want different work.

Attributing those linear layers found a third of them in one loop: the gated
delta rule, 5.16 ms of a 27.54 ms token at width 256. It is serial over tokens
by definition, so a wide pass ran it on one core with the other five idle — but
it is not serial over *heads*. Each of the 32 value heads owns its own
`128 x 128` block of the recurrent state and its own slice of every input, so
spreading the heads is an arithmetic identity, asserted rather than argued by
`spreading_the_heads_over_cores_does_not_move_a_bit`. One pass, milliseconds
per token:

| width | prefill tok/s | one pass | recurrence | linear |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 24.80 → 25.46 | 39.89 → 37.15 | 5.51 → 3.91 | 17.14 → 15.09 |
| 64 | 31.91 → 35.05 | 29.55 → 27.33 | 5.27 → 2.86 | 15.59 → 13.25 |
| 256 | 33.49 → **36.18** | 27.54 → 26.19 | 5.16 → 2.80 | 15.52 → 13.50 |
| 512 | 35.30 → 36.61 | 28.40 → 26.52 | 5.14 → 2.72 | 16.18 → 13.96 |

1.85x rather than six: the loop is bound by the state it walks, not by the
arithmetic over it, so five more cores buy less than five times the throughput.

The step runs once per row, so a pass of `n` rows wakes the pool `n` times per
layer: the waking is paid per row, not per pass, with candle's own matvec pool
holding the cores in between. Spreading it unconditionally made decode worse at
every depth — recurrence over sixteen passes went 0.126 → 0.161 s at 64
context, 0.128 → 0.191 at 1024, 0.136 → 0.197 at 3072 — so the spread is a
property of the pass, not of the build, and which one runs cannot change a
value.

Where it turns is a measurement and not a guess, and the first guess was wrong.
Recurrence, milliseconds per token, over 96 tokens prefilled at each width:

| rows | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calling thread | 19.93 | 8.15 | 6.95 | 6.65 | 6.13 | 5.99 | 5.71 | 5.67 | 5.37 |
| thread pool | — | 11.77 | 8.01 | 6.15 | 5.53 | 4.97 | 4.13 | 3.96 | 3.36 |

Four rows. What made that worth chasing is that four is also the width of a
speculative verification pass: an MTP draft accepted about four times in five
leaves most verifications two or three rows wide, so a threshold of two put the
whole decode path on the wrong side of the crossover. The end-to-end agent turn
said so before the microbenchmark did — the same 1807-token first turn decoded
its same 128 tokens 4% slower, in two runs — which is the case for running it.
With the threshold at four, recurrence at decode measures 0.132 / 0.127 /
0.127 s against a 0.126 / 0.128 / 0.136 baseline: the same loop it always was.

That leaves the KV scan as the largest term at depth, and the lever the
DeltaNet work already named: the 8 query heads sharing a kv head each re-read
that head's whole cache. Blocking them together reads it once — the traffic
falls by the group size — and it can be done without moving a bit, because
every score is still the same dot of the same two vectors and every output
element still sums its positions in increasing order. The scan was built that
way and measured, and the measurement said something the traffic argument had
missed.

The eight re-reads were never eight trips to memory. They are eight work items
running at the same time, on the same last-level cache, asking for the same
lines. While a layer's live KV fits in that cache they are nearly free, and the
per-head scan's single parallel region and single write per output row are the
cheaper shape; the blocked scan needs three regions, a score buffer, and a
transpose, and loses. Past the cache they become real misses and the blocking
starts to pay. Scan wall seconds over sixteen decode passes, and what a layer's
live KV weighs at each depth against this host's 12 MiB L3:

| context | layer KV | per-head | blocked |
| ---: | ---: | ---: | ---: |
| 1024 | 4.2 MB | 0.154 | 0.241 |
| 3072 | 12.6 MB | 0.436 | 0.458 |
| 6144 | 25.2 MB | 0.915 | 0.821 |
| 8192 | 33.6 MB | 1.335 | 1.089 |

So both scans stay and a pass picks one by what its cache weighs, at 16 MiB —
the first power of two past this host's L3, and past the depth where the two
measured equal. They are bit-identical, so the choice is a scheduling one:
`the_two_scans_agree_at_the_checkpoint_geometry_that_switches_them` asserts
them equal at 16 query heads over 2 KV heads at head_dim 256, which is what
this checkpoint has, at the first depth that switches them.

Against the branch point, with the DeltaNet change already in:

| context | decode tok/s | scan | prefill at that depth |
| ---: | ---: | ---: | ---: |
| 1024 | 7.964 → 7.916 | 0.161 → 0.158 | 33.51 → 35.07 |
| 3072 | 6.494 → 6.754 | 0.457 → 0.451 | 29.81 → 30.88 |
| 6144 | 5.695 → 5.903 | 0.915 → 0.849 | 21.52 → **24.75** |
| 8192 | 4.872 → **5.423** | 1.335 → 1.104 | 17.82 → **21.93** |

Prefill gains more than decode past the threshold, and for the reason the
blocking predicts: a 512-row pass has 512 rows each re-reading the cache eight
times per KV head, where a one-row decode has one. Below 16 MiB nothing
changes, because nothing there was worth changing.

The other thing that changed is how wide a pass gets to be. Prefill ran one
pass over the whole prompt, and a pass costs more per token the wider it goes
past a few hundred rows — the full-attention layers score quadratically many
row pairs, and the MoE has taken all the weight reuse a wide pass has to give
by about sixty-four. Prefilling 3072 tokens:

| pass width | 256 | 512 | 1024 | 3072 |
| --- | ---: | ---: | ---: | ---: |
| prefill tok/s | 28.07 | **28.60** | 28.56 | 26.41 |
| one pass, ms/token | 29.39 | 29.74 | 31.44 | 39.14 |

so passes are now capped at 512 rows, worth 1.08x on a 3k-token tool result.
That one is a numerical change and not an identity — each pass reduces over its
own rows — but it is the change the prompt cache has always made on a hit, and
`a_chunked_prefill_decodes_like_a_single_pass` asserts the decoded tokens
across a boundary either way. See [prompt-cache.md](prompt-cache.md).

The lane accumulators reorder the summation, so this one is a numerical change
rather than an identity: the last bits move and a token could follow. On this
host it did not. Plain decode and the speculative policy agree token for token,
a two-thread run agrees with a six-thread one, and 128 tokens at 3072 context
are identical to what the serial reduction produced. The property that must
hold is the agreement between paths, and it does.

The scan is now 15% of a pass at 3072 rather than 63%, and the dot has stopped
being latency-bound: at 1.13 FLOP/cycle per core it is close to the weighted
sum's 1.30, which suggests both are now waiting on cache bandwidth for the KV
rows. The next lever is therefore the one this section previously dismissed —
each of the 8 query heads sharing a kv head re-reads that head's whole cache,
so blocking them together would read it once. That was not worth doing while
the dot was latency-bound; now it is the largest term left.

## Ending a turn

A turn ends on the tokens the checkpoint calls end-of-sequence. Which those are
is not a settled question: this checkpoint's `config.json` names
`<|endoftext|>` while its chat template closes every assistant turn with
`<|im_end|>`, so a server that reads only the config never stops — it runs to
the token budget, emits the marker, and then writes both sides of the
conversation itself. The server therefore stops on the union of
`config.json`'s `eos_token_id` and `tokenizer_config.json`'s `eos_token`, and
refuses to start if neither names a token the tokenizer knows.

## One sequence at a time

`QuantizedRuntime` holds a single sequence: recurrent DeltaNet state, the KV
cache, and the MTP block's synchronisation point all describe one conversation,
and generation takes `&mut self`. The server does not change that. It is a
queue in front of one inference slot:

- a dedicated engine thread owns the checkpoint and the runtime for the
  process's lifetime, so neither ever crosses a thread boundary;
- requests are served strictly first-in, first-out, one at a time;
- `--max-queue N` (default 8) bounds queued plus running requests. Beyond it
  the server answers `503` with `Retry-After: 1` rather than growing an
  unbounded backlog;
- a client that disconnects cancels its request: if it had not started, the
  worker skips it; if it had, the worker abandons it at the next token.

Requests are **stateless** in meaning: every request is decoded against exactly
the conversation it sent, and nothing carries over implicitly. They are not
stateless in cost. A request starts from the longest state that provably
describes a prefix of its own tokens — the previous request's session when this
one continues it, or a prompt cache entry when `--prompt-cache-dir` is set —
and prefills only the remainder. See [prompt-cache.md](prompt-cache.md); the
per-request log line reports where the state came from.

`GET /health` reports the queue depth and what is loaded, and is the one route
that never requires the API key.

## Authentication

`--api-key KEY` requires `Authorization: Bearer KEY` (or `X-Api-Key: KEY`) on
every `/v1` route; the comparison does not exit early on the first differing
byte. `--api-key-file PATH` reads the key from a file instead, keeping it out
of the process list. With neither flag the API is open, and binding a
non-loopback `--host` without a key prints a warning.

## Supported requests

`POST /v1/chat/completions`, `GET /v1/models`, `GET /v1/models/{model}`,
`GET /health`. Streaming (`"stream": true`) is server-sent events in OpenAI's
chunk format, terminated by `data: [DONE]`; `stream_options.include_usage`
adds the trailing usage-only chunk.

Honoured request fields:

| Field | Notes |
| --- | --- |
| `messages` | `system`/`developer`, `user`, `assistant`, `tool`. Text content or an array of `{"type":"text"}` parts |
| `tools`, `tool_choice` | Function calling; `tool_choice` accepts `auto` and `none` |
| `max_tokens`, `max_completion_tokens` | The newer name wins. Capped at 32768 |
| `temperature`, `top_p`, `top_k`, `min_p`, `seed` | `top_k`/`min_p` are extensions; `top_p: 1` means unrestricted |
| `stop` | A string or an array. Matched on decoded text across token boundaries |
| `stream`, `stream_options.include_usage` | |
| `reasoning_effort` | `none`/`minimal`/`low`/`medium`/`high`/`xhigh`, mapped to a token budget |
| `chat_template_kwargs.enable_thinking`, `enable_thinking` | Qwen's thinking prefix, overriding `--no-thinking` |
| `thinking_budget` | Extension: force-close `<think>` after exactly N committed tokens |

Anything the request omits inherits the server's own flags, so
`--temperature`, `--max-new-tokens` and the speculative settings are the
operator's policy for clients that do not care. Unknown fields (`user`,
`presence_penalty`, …) are ignored.

Rejected with `400` rather than silently ignored, because ignoring them would
return something other than what was asked for: `n` above 1, the deprecated
`functions` field, `logprobs`, `response_format` other than `text`, and a
`tool_choice` that forces a call (this engine does not constrain decoding, so
it cannot promise one).

Errors use the OpenAI envelope — `{"error": {"message", "type", "param",
"code"}}` — including when a stream that has already started fails, where the
envelope arrives as the final SSE payload because the status line is spent.

## Tool calling

Tools are rendered into the prompt by the checkpoint's own template, and this
checkpoint asks for XML rather than the JSON some Qwen releases use:

```text
<tool_call>
<function=read_file>
<parameter=path>
src/lib.rs
</parameter>
</function>
</tool_call>
```

The server renders `tools` into the leading system block, hides that markup
from the response text, and reports it as OpenAI `tool_calls` with
`finish_reason: "tool_calls"`. Parameter values arrive as raw text, so the
declared JSON type in the tool's schema is what decides whether `42` is the
number or the string; a value that should be JSON but will not parse is
returned as the string the model wrote rather than dropped.

Send results back as `role: "tool"` messages — consecutive ones are folded into
a single `<tool_response>` turn, as the template does — and echo the
assistant's `tool_calls` back with them so the model sees its own call.

### The round trip preserves bytes

An assistant turn sent back on the next request is rendered into the prompt
again, and the markup that comes out is byte for byte what the model generated.
That is a performance contract, not a cosmetic one: the engine continues the
live session only while the new prompt is a token prefix continuation of the
last one, so a single changed space re-prefills the whole conversation from a
cache boundary.

The bytes cannot simply be carried across. OpenAI's `function.arguments` is a
JSON *string*, and a client is entitled to parse it and re-emit it — pi 0.84.2
does exactly that, with `JSON.stringify`, which compacts away every space before
the call ever comes back. What makes the round trip exact anyway is that the
model's formatting is not arbitrary. The template writes a structured parameter
with Jinja's `tojson`; transformers configures `tojson` with Python's default
separators; so the template — and therefore the model trained on it — writes
`[{"oldText": "a", "newText": "b"}]`, with a space after every colon and comma.
Re-rendering the parsed value in that same form lands on the model's own bytes
whatever the client did in between.

Both sides of the field are held to it:

- **Out.** The `arguments` object is assembled from the parameter text rather
  than re-serialised from parsed values, so a client that *does* echo the string
  verbatim never had a chance to lose anything. A parameter whose text is
  already JSON of a non-string type is copied in as it stands; anything else —
  including every parameter the schema declares a `string` — becomes a JSON
  string of exactly that text.
- **Back.** A JSON string is written as its contents. An object or an array is
  written with the template's own separators. A number, a boolean or `null`
  keeps its exact text, which no formatting choice affects.

The residual is the inverse case: a parameter whose JSON the model wrote in some
*other* formatting — indented across lines, or compact — is normalised to the
template's form, because that is the one formatting both ends can agree on
without the bytes. On this checkpoint the model writes the template's form, so
this has not been observed in an agent run. Leading or trailing whitespace
around a JSON-valued parameter is trimmed for the same reason.

A turn ends at the closing `</tool_call>` tag, so it carries at most one call.
That is a deliberate limit, not an oversight: the template tells the model to
reply with a call and no suffix, and this checkpoint does not honour it — left
to run, an agent request emitted 31 calls in a row and stopped only at the
token budget, 19 minutes later. Ending the turn where the call closes is what
makes tool use terminate.

Streaming reports the call in one chunk rather than assembling it across
chunks: the whole turn is decoded before the engine knows a call was made at
all.

When the prompt leaves a `<think>` block open — the template's default — the
response's first delta is the opening `<think>` tag. The model's own output
begins mid-thought, so without it a client parsing Qwen's thinking tags sees
text that ends with an unmatched `</think>`.

The renderer is checked against the real thing: `src/tokenizer.rs`'s tests
compare its output byte for byte with the checkpoint's `chat_template` rendered
through Jinja2 with the filter overrides transformers applies, for a tools
prompt, a full tool round trip, and a plain conversation. `src/tool_calls.rs`
checks the byte-preservation contract on its own — parse, hand out `arguments`,
render back, and compare with the markup that went in — over nested JSON with
the model's spacing, multi-line values, quotes, unicode and untyped parameters,
and over arguments a client compacted or pretty-printed on the way back.
`tests/openai_server.rs` closes the loop on the real tokenizer:
the prompt built from an echoed tool call must be a token prefix continuation
of the previous prompt plus the tokens the model generated, which is the
comparison the engine itself makes.

## Driving it with an agent

Verified end to end with the pi coding agent (`@earendil-works/pi-coding-agent`
0.84.2) against Qwen3.6-35B-A3B Q4_K_M. `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "inferq": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "local-key",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "qwen3.6-35b",
          "reasoning": true,
          "input": ["text"],
          "compat": { "maxTokensField": "max_tokens", "thinkingFormat": "qwen-chat-template" },
          "contextWindow": 65536,
          "maxTokens": 2048,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```

`supportsReasoningEffort` can be `true` against this server: it honours
`reasoning_effort` and maps it to a token budget. The model `id` must match
what the server serves — `--served-model-name`, or
the GGUF file stem by default. `GET /v1/models/{id}` answers 404 for any other
name, which is the first thing a misconfigured client trips on.

```bash
pi -p "Read notes.txt and tell me the magic number." \
   --provider inferq --model qwen3.6-35b -t read
```

Two turns, eight threads, 24 GiB expert cache, with the prompt cache warm:

```
accepted a chat completion  messages=2 tools=1 stream=true max_tokens=2048
request complete  prompt_tokens=806 completion_tokens=59 reused_tokens=768
                  reuse=cache tool_calls=1 finish=ToolCalls seconds=47.6
accepted a chat completion  messages=4 tools=1 stream=true max_tokens=2048
request complete  prompt_tokens=906 completion_tokens=37 reused_tokens=865
                  reuse=live tool_calls=0 finish=Stop seconds=35.0
```

The first turn restored pi's preamble from disk and answered with a tool call;
the second continued the live session — pi had echoed the assistant's
`tool_calls` and the tool result back verbatim — and answered from the file's
contents. The same first turn against an empty cache took 19 minutes.

A longer whole-task run is the measurement that keeps the microbenchmarks
honest, because it is the only one that pays for every path at once — a cold
boundary prefill, live-reuse continuations, a cache fallback, and a speculative
decode behind all of them. Asking pi to identify the model a repository uses and
update its `CLAUDE.md` and `README.md` produces seven turns. On an i7-8700 at
six threads with the experts resident, against the same task from a clean
checkout and an empty prompt cache:

| turn | prefill tokens | prefill tok/s | decode tok/s |
| ---: | ---: | ---: | ---: |
| 1 (cold) | 1807 | 30.44 → **33.91** | 9.11 → 9.20 |
| 2 | 197 | 28.40 → 30.47 | 10.13 → 9.86 |
| 3 | 171 | 26.19 → 28.14 | 10.98 → 10.14 |
| 4 | 116 | 23.38 → 24.86 | 10.58 → 10.42 |
| 5 | 132 | 23.29 → 25.49 | 8.39 → 8.98 |
| 6 (fallback) | ~1820 | 24.53 → 25.88 | 9.13 → 9.32 |
| 7 | ~590 | 19.55 → **21.01** | 7.66 → 7.52 |
| whole task | | 330 s → **310 s** | |

Prefill is up on every turn; decode is flat, which is what the depth table
predicts for a conversation that never passes the blocking threshold. The
decode columns are not directly comparable turn by turn — the model does not
generate the same number of tokens twice, and a longer generation ends deeper —
except on turn 1, which produces the same 128 tokens at the same acceptance
both times. Run the two sides in the same session: the same turn measured
9.59 tok/s in an earlier session and 9.11 in this one, which is larger than
anything measured here.

Turns 6 and 7 in that table were not throughput measurements at all. They were
the tool-call round trip losing the model's bytes: both fell off the live
session and re-prefilled from the boundary. They are also the first two turns to
call `edit`, whose `edits` parameter is the only JSON-valued one this task uses
— see [the round trip](#the-round-trip-preserves-bytes).

Re-measured in one later session, before and after the render side writes that
parameter the way the template does. Both sides ran back to back on the same
host with the same warm cache, so the two columns are comparable with each other
rather than with the table above:

| turn | prompt tokens | prefill tokens | reuse | seconds |
| ---: | ---: | ---: | --- | ---: |
| 1 | 1807 | 271 | cache | 22.9 |
| 2 | 2132 | 197 | live | 15.8 |
| 3 | 2399 | 171 | live | 19.6 |
| 4 | 2643 | 116 | live | 13.6 |
| 5 | 2865 | 132 | live | 46.0 |
| 6 | 3323 | 1780 → **84** | cache → **live** | 92.5 → **30.5** |
| 7 | 3635 | 554 → **82** | cache → **live** | 50.2 → **25.5** |
| whole task | | | | 265 s → **175 s** |

Every turn after the first now continues the live session, the divergence log is
silent, and the agent's edits to both files are the same edits. Turn 1 stays on
the cache because it is the first request of the process and has no live session
to continue; that is the tier working, not failing.

Turns 1 to 5 reused the same way in both runs, so they carry one figure. The
prompt lengths are the second run's, and they move by a few tokens between runs
for a reason worth knowing before reading anything into them: the agent's first
tool call is `ls -la`, whose output carries file timestamps, so no two runs of
this task decode from quite the same prompt.

## Thinking

OpenAI's API has no thinking budget. It has `reasoning_effort` — a categorical
knob, `none` | `minimal` | `low` | `medium` | `high` (and `xhigh` on newer
models) — with `max_completion_tokens` bounding reasoning and answer together,
and `usage.completion_tokens_details.reasoning_tokens` reporting what the
reasoning cost. Anthropic and Google are the ones that take a token count
(`thinking.budget_tokens`, `thinkingConfig.thinkingBudget`).

This runtime bounds thinking by token count, so the two are bridged: the
operator decides what each effort level can afford, because on a CPU host that
is a property of the machine rather than of the request.

```bash
serve ... \
  --thinking-budget 512 \          # for requests that ask for neither
  --max-thinking-budget 4096 \     # ceiling on anything a request asks for
  --reasoning-budget high=8192      # repeatable; overrides one level
```

Level defaults: `minimal` 64, `low` 256, `medium` 1024, `high` 4096, `xhigh`
16384. `none` is not a budget — it renders the thinking block already closed,
the same thing `chat_template_kwargs.enable_thinking: false` and the server's
`--no-thinking` do.

What a request may send, most specific first:

| Field | Meaning |
| --- | --- |
| `thinking_budget: N` | Extension: exactly N tokens, the way Anthropic and Google express it |
| `reasoning_effort` | The level's configured budget |
| `chat_template_kwargs.enable_thinking`, `enable_thinking` | Whether to think at all |

Anything a request asks for is clamped by `--max-thinking-budget`, so a client
cannot pin the single inference slot in a thinking loop. An unrecognised effort
level falls back to the server's default rather than failing the request.

The budget is a *forced closure*: at the limit the runtime evaluates the
tokenizer's own `</think>\n\n` through both target and MTP state and the
answer continues from there. It is not a truncation of the response.

```bash
curl ... -d '{"messages":[...],"reasoning_effort":"minimal","max_tokens":120}'
# usage: {"prompt_tokens":19,"completion_tokens":49,"total_tokens":68,
#         "completion_tokens_details":{"reasoning_tokens":17}}

curl ... -d '{"messages":[...],"reasoning_effort":"none","max_tokens":120}'
# usage: {"prompt_tokens":21,"completion_tokens":35,"total_tokens":56}
```

`completion_tokens_details` is absent when the turn had no thinking section at
all, which is not the same as having spent nothing on one.

When the prompt leaves a block open, an assistant message in the request's
history has its reasoning section stripped before rendering — except inside the
current query, where the template keeps it, which is what a multi-step tool
exchange needs.

## Sampling and speculation

Speculative decoding verifies drafts against the target's argmax, so it is
defined only for greedy decoding. A request with `temperature > 0` therefore
runs unspeculated; it is not an error, because most OpenAI clients send a
non-zero temperature by default. Requests that leave temperature at the
server's default of 0 use whatever `--speculative` selected — `auto` by
default here, unlike `gguf_infer`, since a server has no interactive operator
to turn it on.

All of `gguf_infer`'s policy flags (`--mtp-depth-cap`, `--ngram-draft-cap`,
`--ngram-suspend-below`, `--backoff-tokens`, `--no-adaptive-length`, …) exist
on `serve` with the same meanings; see
[speculative decoding](speculative-decoding.md). Routing traces and censuses
are not exposed: they are single-file sinks written per turn and are
incompatible with speculation.

Per-request throughput is logged at `info`:

```
INFO qwen_engine::server::engine: request complete prompt_tokens=25
     completion_tokens=64 finish=Stop tokens_per_second=5.98 seconds=10.7
```

The model's own per-layer spans are off by default because there are tens of
them per token; `RUST_LOG=info` (or `debug`) turns them back on.

## Testing

`tests/openai_server.rs` boots the engine on an ephemeral port and drives the
real routes: authentication, rejected bodies, stop-string truncation, and a
check that a streamed response reassembles into exactly the buffered one. It
needs a checkpoint, so it is opt-in:

```bash
INFERQ_TEST_GGUF=/models/qwen3.6/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
INFERQ_TEST_MODEL_DIR=/models/qwen3.6 \
cargo test --release --test openai_server -- --test-threads=1
```

Without those variables it skips with a message. The request parsing, option
merging, stop-sequence buffering, and chat rendering are covered by ordinary
offline unit tests.

## Not implemented

`logprobs`, `n > 1`, embeddings, `/v1/completions`, images, forced tool choice,
and concurrent decoding. None of these are faked: each is either rejected with
a 400 or absent.
