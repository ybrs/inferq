# Small-model scout evaluation

llama.cpp `version: 0.1.2-dev (build 1, commit a3b1eff)`, harness `4ce68ac`, `-np 1 -fa off -lm none`, 6 threads pinned to CPUs 0-5, temperature 0, seed 42.

Host at start: governor `powersave`, turbo on, 4302.1 MHz, 56.0&deg;C.


## Speed

Prefill is the cold ingest of the 46-tool block (>=5000 tokens, cache empty). Decode is measured on one fixed scenario so every cell is at the same depth; that depth is printed. **These are not comparable to `tg128`**, which is decode at 128 tokens of context - a condition a scout with tools loaded is never in.

| run | GiB | load s | cold prefill t/s | decode t/s | depth | canary t/s | noisy |
|---|---:|---:|---:|---:|---:|---:|:--:|
| Qwen3-0.6B-Q4_K_M__think-off | 0.37 | 2.0 | 135.5 | 14.86 | 7463 | 46.22 |  |

## Capability

_Not graded yet - run grade_llm.py._


## Provenance

| file | sha256 |
|---|---|
| `taskq-tools.json` | `87199d4facf70be8...` |
| `scenarios-tools.json` | `869ba4a6f7144e8e...` |
| `scenarios-quality.json` | `75a3950902b852a0...` |
| `roster.json` | `b39be6a683fce6d3...` |
| `roster-verified.json` | `bcf6e8eddfeca2a8...` |
| `run_eval.py` | `ecf28d0370339aa6...` |
| `serverctl.py` | `28fa4a2087c382ba...` |
