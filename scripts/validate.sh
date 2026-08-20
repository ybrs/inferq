#!/usr/bin/env bash
set -euo pipefail

# Built for this host, not for baseline x86-64. The one-row and multi-row
# quantized kernels only reach the same summation order with FMA available,
# and without it speculative decoding stops being output-preserving — so the
# engine refuses to open a checkpoint at all. See docs/speculative-decoding.md.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target-native}"
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"

cargo fmt --check
cargo check --all-targets
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo build --release --bins

if [[ -n "${QWEN_MODEL_DIR:-}" && -n "${QWEN_REFERENCE_DIR:-}" ]]; then
  cargo run --release --bin compare -- \
    --model "${QWEN_MODEL_DIR}" \
    --tokens "${QWEN_REFERENCE_DIR}/tokens.json" \
    --reference-logits "${QWEN_REFERENCE_DIR}/logits.json"
fi
