#!/usr/bin/env bash
set -euo pipefail

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
