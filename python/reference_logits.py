#!/usr/bin/env python3
"""Emit Transformers reference tokens/logits for the Rust comparison CLI."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--tokens", type=Path)
    parser.add_argument("--tokens-out", type=Path, required=True)
    parser.add_argument("--logits-out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.prompt is not None:
        token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    else:
        token_ids = json.loads(args.tokens.read_text())
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu"
    ).eval()
    with torch.no_grad():
        logits = model(torch.tensor([token_ids]), use_cache=True).logits[0, -1].float()
    args.tokens_out.write_text(json.dumps(token_ids) + "\n")
    args.logits_out.write_text(json.dumps(logits.tolist()) + "\n")


if __name__ == "__main__":
    main()
