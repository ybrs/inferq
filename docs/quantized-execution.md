# Direct GGUF execution boundary

Stage 2B now has an initial executable boundary. `GgufCheckpoint` parses the
GGUF once and loads supported matrices in their stored Q4_K, Q5_K, Q6_K, Q8_0,
or F32 representation. `QuantizedMatrix::forward` accepts F32 activations and
calls the block matmul directly; its public API intentionally has no
whole-matrix dequantization method.

Fused expert tensors use the GGUF shape `[experts, rows, columns]`.
`load_expert_matrix` seeks directly to one expert and reads only that matrix's
compressed range. On the local Q4_K_M checkpoint this reduces a gate/up expert
read from the full 288 MiB tensor to 576 KiB, and a down expert read from
352 MiB to 704 KiB.

Compile for the deployment CPU so Candle includes its AVX2 ggml kernels. The
resulting binary is host-specific:

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin gguf_projection

./target/release/gguf_projection \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --tensor blk.0.attn_qkv.weight \
  --repetitions 20

./target/release/gguf_projection \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --tensor blk.0.ffn_gate_exps.weight \
  --expert 0 \
  --repetitions 100
```

Warm measurements on the i7-6700 were about 5.8 ms for the 8,192×2,048 Q8_0
projection and 0.18 ms for one 512×2,048 Q4_K expert projection. These timings
exercise the direct kernel boundary only. They do not include routing, the rest
of a layer, or all 48 layers and are not full-model throughput claims.

The current loader owns a compact copy of each loaded matrix. It does not yet
mmap the tensor bytes, cache resident non-expert matrices, or reuse selected
expert buffers. The next integration step is one complete quantized layer:

1. keep router, norms, shared expert, attention/DeltaNet projections resident;
2. route with the real F32 gate and load only the ten selected expert ranges;
3. handle GGUF DeltaNet's split `attn_qkv`/`attn_gate` representation;
4. compare layer output and selected expert IDs with the BF16 oracle;
5. promote the path to a complete runtime only after that comparison passes.
