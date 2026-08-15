#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MODEL_DIR CASE CACHE_STATE OUTPUT_PREFIX" >&2
  echo "case: one-token-smoke | twelve-token-prefill | sixteen-token-decode" >&2
  echo "cache state: cold | warm | persistent | unknown" >&2
  exit 2
fi

model_dir=$1
case_name=$2
cache_state=$3
output_prefix=$4
json_output="${output_prefix}.jsonl"
perf_output="${output_prefix}.perf.csv"

if [[ -e "$json_output" || -e "$perf_output" ]]; then
  echo "refusing to overwrite an existing profiling artifact" >&2
  exit 1
fi

if [[ ! -x ./target/release/bench ]]; then
  echo "build the benchmark first: cargo build --release --bin bench" >&2
  exit 1
fi

perf stat \
  -x ';' \
  -o "$perf_output" \
  -e cycles,instructions,cache-misses,page-faults,context-switches \
  -- \
  ./target/release/bench \
  --model "$model_dir" \
  --prompts benchmarks/profile-prompts.json \
  --only "$case_name" \
  --cache-state "$cache_state" \
  --output "$json_output"
