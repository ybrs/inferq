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

## Decode against context depth

Every throughput figure elsewhere in this repository is measured on a short
prompt — the qualified sustained case reaches 151 context tokens — and decode
is not the same operation at agent depths. The full-attention layers scan a KV
cache that grows with the conversation; the linear and MoE layers cost the same
at any depth. Measured with `gguf_decode_depth` on an i7-8700, six threads,
Qwen3.6-35B-A3B Q4_K_M, native build, speculation off, sixteen decode passes
per depth (seconds are totals across those passes):

| context | decode tok/s | attention scan | linear | MoE | lm_head |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 6.72 | 0.06 s | 0.81 s | 1.04 s | 0.29 s |
| 512 | 6.56 | 0.43 s | 0.80 s | 0.73 s | 0.30 s |
| 1024 | 5.53 | 0.89 s | 0.79 s | 0.73 s | 0.30 s |
| 2048 | 3.92 | 2.08 s | 0.79 s | 0.72 s | 0.30 s |
| 3072 | 2.97 | 3.38 s | 0.79 s | 0.72 s | 0.31 s |

Every column except the attention scan is flat. The scan grows in proportion to
the depth — 48 times longer over a 48-fold context — and goes from 2.6% of a
decode pass at 64 tokens to 63% at 3072. So a request that decodes at 3 tok/s
against a long conversation is not decoding slowly for the same reason a short
one would; it is spending most of its time in one operation, and that operation
is the one to optimise for agent workloads. Nothing in the decode path is
suspect until that column is accounted for.

The practical consequence for a client is that the reasoning budget is worth
more at depth: `--max-thinking-budget`, or `reasoning_effort`, bounds the part
of a turn that is paid for at the current context depth.

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
prompt, a full tool round trip, and a plain conversation.

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
