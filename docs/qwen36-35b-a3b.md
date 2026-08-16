# Qwen3.6-35B-A3B GGUF comparison

This branch adds text-only Qwen3.6-35B-A3B execution to the optimized GGUF
runtime and compares Bartowski's Q4_K_M and Q8_0 artifacts. The implementation
uses the model's 40 trunk layers. Its one auxiliary MTP predictor layer remains
inactive during ordinary autoregressive decoding and can be enabled explicitly
for greedy speculative decoding.

## Qualified artifacts

- Config and tokenizer: `Qwen/Qwen3.6-35B-A3B`, revision
  `7da1103448ba36029c34ce1a9a741dfe93ee0c50`.
- GGUF repository: `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`, revision prefix
  `6dd29f0`.
- Q4_K_M: `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`, 22,285,080,192 bytes.
- Q8_0: `Qwen_Qwen3.6-35B-A3B-Q8_0.gguf`, 37,812,647,552 bytes.

The authoritative architecture inputs are the
[Qwen config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json),
the [Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py),
and llama.cpp's [Qwen3.5/3.6 MoE implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35moe.cpp).

Download with the Hugging Face CLI:

```bash
MODEL_ROOT=/data/models/Qwen3.6-35B-A3B
mkdir -p "${MODEL_ROOT}"

hf download Qwen/Qwen3.6-35B-A3B \
  config.json tokenizer.json tokenizer_config.json \
  --revision 7da1103448ba36029c34ce1a9a741dfe93ee0c50 \
  --local-dir "${MODEL_ROOT}"

hf download bartowski/Qwen_Qwen3.6-35B-A3B-GGUF \
  Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
  Qwen_Qwen3.6-35B-A3B-Q8_0.gguf \
  --revision 6dd29f0 \
  --local-dir "${MODEL_ROOT}"
```

## Architecture delta

Both models retain hidden width 2,048, expert width 512, 16 attention heads,
2 KV heads, and the three-linear/one-full-attention pattern. Qwen3.6 reduces
the expensive sparse structure:

| Property | Qwen3-Coder-Next | Qwen3.6-35B-A3B | Change |
| --- | ---: | ---: | ---: |
| Trunk layers | 48 | 40 | -16.7% |
| Linear / full-attention layers | 36 / 12 | 30 / 10 | -16.7% |
| Experts per layer | 512 | 256 | -50.0% |
| Selected experts per token | 10 | 8 | -20.0% |
| Routed expert evaluations per token | 480 | 320 | -33.3% |
| Total layer-qualified experts | 24,576 | 10,240 | -58.3% |
| Vocabulary | 151,936 | 248,320 | +63.4% |

The smaller layer and active-expert counts explain the inference gain. The
larger vocabulary partly offsets it because every token still evaluates the
full LM head.

Qwen3.6 GGUF also differs structurally from the older artifact:

- config values live under `text_config`, and expert width is named
  `moe_intermediate_size`;
- DeltaNet alpha and beta are separate matrices;
- the GGUF converter tiles value-head rows to match ggml broadcast order;
- DeltaNet QKV and gate projections may use different quantization types;
- the chat generation prompt opens a `<think>` block; and
- GGUF `block_count` includes one MTP predictor block in addition to the 40
  trunk blocks.

The runtime handles each difference explicitly and preserves the original
Qwen3-Coder-Next paths. An eight-token Q4 and Q8 smoke sequence matched
llama.cpp exactly: token IDs `[8160, 579, 264, 7047, 1817, 25, 271, 16]`,
which decode to `Here's a thinking process:\n\n1`.

## Reproduce the benchmark

Build on the benchmark host so AVX2/FMA code is specialized for that CPU:

```bash
CARGO_TARGET_DIR=target-native \
RUSTFLAGS='-C target-cpu=native' \
cargo build --release --bin gguf_infer --bin gguf_bench --bin inspect
```

Then substitute `Q4_K_M` and `Q8_0` in this command:

```bash
MODEL_ROOT=/data/models/Qwen3.6-35B-A3B
QUANT=Q4_K_M

CANDLE_NUM_THREADS=4 \
RAYON_NUM_THREADS=4 \
./target-native/release/gguf_bench \
  --model "${MODEL_ROOT}/Qwen_Qwen3.6-35B-A3B-${QUANT}.gguf" \
  --tokenizer-model "${MODEL_ROOT}" \
  --prompts benchmarks/qwen36-gguf-prompts.json \
  --only chat-sustained-128 \
  --repetitions 3 \
  --expert-cache-mib 46000 \
  --warmup-all-experts true \
  --output "/tmp/qwen36-${QUANT}.jsonl"
```

The qualified host is an Intel i7-6700 with four physical cores, 62.6 GiB of
RAM, Linux 6.8, Rust 1.95, Candle/Rayon thread counts 4/4, and a native release
build. The workload is a 25-token rendered chat prompt followed by 128 greedy
tokens. All three repetitions had 100% expert-cache hits, no evictions, zero
physical inference reads, and identical tokens within each quantization.

## Results

| Metric | Qwen3.6 Q4_K_M | Qwen3.6 Q8_0 | Prior Qwen3-Coder Q4 |
| --- | ---: | ---: | ---: |
| GGUF file size | 20.75 GiB | 35.22 GiB | 45.92 GiB |
| Resident expert bytes | 18.16 GiB | 31.88 GiB | 43.50 GiB |
| Mean process RSS | 20.26 GiB | 34.72 GiB | about 47.30 GiB |
| Mean TTFT | 2.364 s | 3.461 s | about 3.186 s |
| Mean decode | **8.118 tok/s** | **6.275 tok/s** | 6.021 tok/s |
| Decode range | 8.116-8.120 | 6.271-6.282 | 5.998-6.041 |
| Mean 127-pass decode time | 15.644 s | 20.240 s | 21.094 s |

Q4 is 34.8% faster than the prior Qwen3-Coder Q4 baseline while using 57.2%
less RSS. Q8 is still 4.2% faster than that baseline and uses 26.6% less RSS.
Within Qwen3.6, Q8 costs 14.46 GiB more RSS and is 22.7% slower than Q4 on
this AVX2 host. Q4 and Q8 agree for the first 27 sustained-output tokens and
then take different but coherent greedy paths, which is normal quantization
sensitivity.

## Auxiliary MTP predictor

Pass `--speculative-mtp N` to use block 40 as an in-model draft predictor for
up to `N` tokens before each target-model verification pass. This mode is
currently restricted to greedy decoding and cannot be combined with routing
traces or expert censuses. It is an opt-in correctness
baseline rather than a recommended performance mode: the current generic
multi-row verifier does not amortize the target weights enough to overcome MTP
execution, rejected rows, and state repair.

The fully resident Q4 benchmark produced exactly the same 128 greedy token IDs
with draft lengths 0, 1, 2, and 3, including runs that exercised rejection and
rollback. Draft length 1 accepted 59 of 67 proposals (88.1%), but decoded at
6.31 token/s versus 8.10 token/s without speculation. See
[Speculative decoding](speculative-decoding.md) for the command, complete
results, state semantics, and next critical optimization.

`--warmup-all-experts` includes the auxiliary block's experts when the model
declares an MTP layer. This adds about 0.84 GiB of Q4 compressed weights to the
18.16 GiB trunk-only expert set. The results table above predates MTP support
and therefore reports trunk-only residency; the speculative comparison reports
the current all-block warmup behavior separately.

Warmup time is intentionally excluded from the architectural comparison. The
observed Q4 warmup was 14.6 seconds from a hot page cache; Q8 took 163.6
seconds with colder storage. Startup is page-cache and disk dependent, whereas
the measured inference phase was fully resident for both.

## Where time remains

Mean Q4 decode attribution over 127 passes was:

| Stage | Seconds | Share of decode |
| --- | ---: | ---: |
| DeltaNet | 6.103 | 39.0% |
| MoE | 5.283 | 33.8% |
| Full attention | 1.934 | 12.4% |
| LM head | 2.145 | 13.7% |

For Q8, DeltaNet took 7.436 seconds and MoE took 7.566 seconds. Q4 is the
better deployment choice on this host: it is both substantially faster and
smaller. Further Qwen3.6 work should prioritize the DeltaNet projection/output
path, then expert compute; optimizing startup or cache lookup cannot materially
improve fully resident decode.
