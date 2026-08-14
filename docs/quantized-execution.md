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

## Real routed MoE checkpoint

`QuantizedMoeLayer` is the next integrated slice. It keeps the real F32 router
and quantized shared-expert weights resident, runs learned top-k routing, and
loads only the selected gate/up/down expert matrices. Compare it with the BF16
oracle using the same deterministic hidden vector:

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin gguf_moe

./target/release/gguf_moe \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --reference-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --layer 0 \
  --top-k 10
```

On the local checkpoints, all ten selected expert IDs matched. Quantized versus
BF16 MoE output had maximum absolute error 0.00303 and RMSE 0.00077. With the
selected expert pages warm, the quantized MoE sublayer took 6.4 ms; a prior
cold-page run took roughly 250 ms, of which 244 ms was expert loading and only
3.2 ms expert compute. This confirms both the direct-kernel path and the need
for a persistent page-cache/residency strategy.

## Complete linear-attention layer

The quantized DeltaNet path handles GGUF's optimized global Q/K/V projection,
separate Z projection, converted `-exp(A_log)` state scale, convolution state,
recurrent Gated Delta Rule state, gated normalization, and output projection.
The complete layer adds converted GGUF RMSNorm weights, residuals, and the
routed MoE:

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin gguf_layer

./target/release/gguf_layer \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --reference-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --layer 0
```

For the deterministic comparison vector, the DeltaNet mixer alone had maximum
absolute error `1.78e-5` and RMSE `2.90e-6`. The complete layer selected the
same ten experts as BF16 and had maximum error `0.0180`, RMSE `0.00193`, and
reference output L2 norm `25.56`. A warm complete layer took `11.0 ms`: about
`5.97 ms` in DeltaNet and `4.99 ms` in MoE. The DeltaNet scalar recurrent update
was about `2.01 ms`, making it a later vectorization target.

This closes the one-linear-layer correctness gate.

## Complete full-attention layer

The second layer implementation covers the joint query/gate projection, Q/K
norms, partial RoPE, persistent grouped-query KV state, causal attention,
sigmoid output gate, output projection, residuals, and routed MoE. Run its
complete comparison with:

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin gguf_full_layer

./target/release/gguf_full_layer \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --reference-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --layer 3
```

The complete layer selected the same ten experts as BF16 and produced maximum
absolute error `0.0801`, RMSE `0.0107`, and reference output L2 norm `40.59`.
Warm time was `7.79 ms`: `1.85 ms` attention and `5.87 ms` MoE. A cold-page run
took `586 ms`, dominated by `575 ms` of selected-expert reads.

Persistent KV decode is compared independently with two sequential steps:

```bash
./target/release/gguf_attention \
  --model /data/projects/localllm/models/Qwen3-Coder-Next-UD-Q4_K_M.gguf \
  --reference-model /data/projects/localllm/models/Qwen3-Coder-Next-SafeTensors \
  --layer 3 \
  --steps 2
```

At the second cached position the attention mixer RMSE was `0.00439`; the
quantized and BF16 sessions both reused their first-step keys and values. Both
decoder layer types have now crossed their isolated correctness gate. The next
work is a 48-layer runtime with quantized embedding, final norm, and LM head.
