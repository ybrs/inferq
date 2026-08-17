# Fused multi-row quantized GEMM

Host `702d043633e0`: Intel i7-8700, 6 physical cores / 12 threads, 12 MiB L3,
125 GiB RAM, STREAM 18.28 GB/s. Model `Qwen3.6-35B-A3B` Q4_K_M GGUF. Branch
`multirow-kernel`, based on `qwen36-35b-a3b-comparison` at 28485ef.

All measurements: `taskset -c 0-5`, INFERQ default threading (6 threads, no
thread environment variables), one run at a time on an otherwise idle machine,
native release builds (`-C target-cpu=native`).

## Verdict

Part A's kernel works, is exact, and removes the >8-row cliff decisively. Two
of Part A's four gates fail, so **Part B was not started**, per the task's
instruction to stop at the analysis when a Part A gate fails.

| gate | required | measured | verdict |
| --- | --- | --- | --- |
| C2a | fused ≥ 1.5x small-M per-row at M=8, Q4K dense | **1.58x** (116010 → 73293 ns/row) | **pass** |
| C2b | per-row at M=16 ≤ per-row at M=8 | 8–10% worse at M=16, all three dtypes | **fail** |
| C3 | per-row monotonically non-increasing through K=16 | monotone: 122.6 → 112.1 → 76.7 → 62.5 → 55.3 → 49.9 ms/row | **pass** |
| C3 | per-row at K=4 ≤ 0.75x before | **0.97x** (78.7 → 77.1 ms/row) | **fail** |
| C3 | K=1 total within 2% of before | ±15% run-to-run on an unchanged code path | **indeterminate** |
| A1 | max abs diff explainable as reordering (≤ 1e-3) | **0.0** vs Candle's AVX2 kernel; 1.311e-6 vs its scalar one | **pass** |
| — | greedy output unchanged | 256-token W1 run token-identical to the pre-kernel baseline | **pass** |

The two failures are not the same kind of thing. C2b is a real, reproducible
8–10% effect with a named cause and an identified fix. The C3 K=4 miss is a
scope finding: the kernel does what it was asked to do on the matrices it
covers, but those are not what dominates a verification pass.

## Part 0 — baseline, and the dtype census

### Census

750 tensors, 20.73 GiB total:

| dtype | tensors | GiB | % of bytes |
| --- | ---: | ---: | ---: |
| Q4K | 152 | 14.58 | **70.3%** |
| Q6K | 114 | 4.96 | **23.9%** |
| Q8_0 | 105 | 1.03 | 5.0% |
| F32 | 367 | 0.10 | 0.5% |
| Q5K | 10 | 0.05 | **0.3%** |
| BF16 | 2 | 0.00 | 0.0% |

Q4K and Q6K carry 94.2% of the bytes, confirming both as mandatory. The LM head
(`output.weight`, 397.9 MiB) is Q6K — the single largest non-embedding matrix.
Q5K appears only as `attn_output` on the ten full-attention layers.

### The compute-bound premise, confirmed

The decisive test: a bandwidth-bound kernel's per-row time would fall roughly
8x from M=2 to M=16, because the same weight bytes serve 8x the rows. Measured
per-row nanoseconds for the existing `small_m_forward`:

| tensor | dtype | MiB | M=2 | M=4 | M=8 | M=12 | M=16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attn_q | Q4K | 9.0 | 139204 | 125750 | 129783 | 122238 | 122062 |
| attn_qkv | Q6K | 13.1 | 199605 | 163578 | 139697 | 138647 | 158083 |
| attn_output | Q5K | 5.5 | 89404 | 76012 | 82147 | 80532 | 79718 |
| ssm_out | Q8_0 | 8.5 | 126671 | 107460 | 95793 | 101274 | 90527 |

Flat to within 0–20%, and at M=16 each pass runs 3.0–4.0x above its
weight-traffic floor (matrix bytes ÷ 18.28 GB/s). Compute-bound, as assumed.

### Two findings the brief did not anticipate

**Q4K was getting almost nothing from the small-M path.** Speedup of
`small_m_forward` over Candle before any change:

| tensor | dtype | M=2 | M=4 | M=8 | M=12 | M=16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| attn_q | **Q4K** | 0.98x | 1.08x | **1.07x** | 1.04x | 1.03x |
| attn_qkv | Q6K | 1.93x | 1.90x | 1.76x | 2.56x | 2.32x |
| attn_output | Q5K | 1.04x | 1.04x | 1.24x | 0.88x | 0.89x |
| ssm_out | Q8_0 | 1.30x | 1.68x | 1.63x | 1.28x | 1.46x |

On Q4K — 70% of the model's bytes — the repeated decode dominated so completely
that the cache win was invisible. That is exactly what the fused kernel targets,
and it is why Q4K shows the largest improvement below.

**Expert shapes are actively hurt by the small-M path.** At M=8 the Q6K expert
matrix measures Candle 9789 vs small-M 14608 ns/row — small-M is 49% *worse*.
They escape only because they are 576–840 KiB, under the 4 MiB threshold. This
inverted the risk assessment for the A2 expert exception: the baseline to beat
there is Candle, not small-M.

## Part A — the fused kernel

`src/qgemm.rs`. For each weight block the scale unpack, nibble split and scale
broadcasts run **once**, then feed every input row in the tile. Layout mirrors
of Candle's block types re-declare the GGUF on-disk formats, since Candle keeps
its fields `pub(crate)`; each is pinned by a compile-time size assertion against
Candle's own type.

### Exactness

The per-block integer sums are associative and the f32 accumulators advance once
per block in Candle's order, so the kernel reproduces **Candle's AVX2 `vec_dot`
bit for bit** — max abs diff exactly 0.0 across Q4K/Q6K/Q8_0 at M ∈
{2,3,4,7,8,9,15,16}.

One subtlety worth recording: Candle compiles that AVX2 kernel only when
`target_feature = "avx2"` is set for the build. In a default `cargo test` build
Candle uses a scalar `vec_dot` with a different accumulation order, and the
comparison then agrees only to **1.311e-6** — three orders inside the 1e-3
bound, and pure reordering. The release builds used for every measurement set
`-C target-cpu=native` and take the exact branch. The unit test asserts
exactness on that branch and the reordering bound otherwise.

End-to-end, a 256-token greedy W1 run with the fused kernel is token-identical
to the pre-kernel baseline.

### Tile size, measured

Per-row ns at M=8 / M=16, Q4K `attn_q`:

| TILE | M=8 | M=16 |
| ---: | ---: | ---: |
| 2 | 98705 | 110612 |
| 4 | 88404 | 94691 |
| **8** | **77057** | **86461** |
| 12 | 77877 | 89501 |
| 16 | 87612 | 95455 |

TILE=8 wins. Above it the accumulators no longer fit the 16 architectural ymm
registers alongside the decoded weight state and the compiler spills. Block-major
repacking of the quantized inputs was also tried and measured neutral; it was
kept because it slightly helps Q6K at M=8 and costs nothing.

### Results (best-of-60: one kernel per process, 2 processes x 30 reps)

| tensor | dtype | M | small-M before | fused after | speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| attn_q | Q4K | 2 | 146420 | 112357 | 1.30x |
| attn_q | Q4K | 4 | 121890 | 86476 | 1.41x |
| attn_q | Q4K | **8** | 116010 | **73293** | **1.58x** |
| attn_q | Q4K | 12 | 125757 | 79620 | 1.58x |
| attn_q | Q4K | 16 | 126028 | 80973 | 1.56x |
| attn_qkv | Q6K | 8 | 134700 | 89184 | 1.51x |
| attn_qkv | Q6K | 16 | 138226 | 96704 | 1.43x |
| ssm_out | Q8_0 | 8 | 99916 | 77516 | 1.29x |
| ssm_out | Q8_0 | 16 | 90245 | 83750 | 1.08x |

**C2a passes at 1.58x.**

**C2b fails.** M=16 is 8–10% worse per row than M=8 on all three dtypes
(80973 vs 73293; 96704 vs 89184; 83750 vs 77516). The cause is direct: with
TILE=8, a 16-row pass runs two tiles and therefore decodes every block twice.
Five tile sizes and an input-layout change were measured; none removes it,
because holding 16 accumulators plus the decoded weight state exceeds the
register file. The identified fix — decode once into a 256-byte L1 scratch
buffer and let both tiles read it — was not attempted.

This is a drift, not a cliff. The old behaviour at M=16 was 126028 ns/row; the
new one is 80973. What the gate was written to catch — falling off dispatch at
9 rows — is gone.

### A2 dispatch

Widened to **2..=16** rows for matrices ≥ 4 MiB. One row still takes Candle's
matvec, untouched, and there is nothing to fuse there: no weight byte is reused.
Above 16 rows Candle's generic loop takes over.

**The expert-path exception was measured and rejected.** Fused vs Candle on real
expert matrices:

| expert matrix | dtype | shape | KiB | M=2 | M=3 | M=4 | M=6 | M=8 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ffn_down_exps | Q6K | [2048,512] | 840 | 0.51x | 0.56x | 0.58x | 0.85x | 0.95x |
| ffn_gate_exps | Q4K | [512,2048] | 576 | 0.59x | 0.75x | 0.99x | 1.14x | 1.27x |

The task anticipated input-quantize overhead at group size 2 and suggested a ≥3
gate. The data says that is not enough: Q6K experts lose at *every* group size
up to 8, and Q4K experts only break even at 4. The decisive fact is the group
size distribution — measured expert reuse for this model gives **2.03 rows per
selected expert at K=8** — so the common case sits squarely in the 0.51–0.75x
region. The fixed per-call cost (input quantization to Q8K, block-major repack,
output transpose, rayon dispatch over up to 2048 output rows) cannot be
amortized by a 576–840 KiB matrix at 2–3 rows. Exception not added.

### A3 plumbing

No changes were required. `runtime.rs` carries no `2..=8`-derived assumption,
and the snapshot arena already sizes itself per pass via `begin_pass(state,
rows)`, so it allocates to the configured draft length rather than a maximum.

## Part C3 — whole verification pass, before vs after

`gguf_verify_bench --batch-sizes 1,2,4,8,12,16 --repetitions 3`, fully resident
experts, 16 verification tokens.

| K | before ms | after ms | before ms/row | after ms/row | per-row change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 133.2 | 122.6 | 133.2 | 122.6 | -8.0% |
| 2 | 211.6 | 224.2 | 105.8 | 112.1 | +5.9% |
| 4 | 316.6 | 306.9 | 79.2 | 76.7 | -3.1% |
| 8 | 506.8 | 500.3 | 63.4 | 62.5 | -1.3% |
| 12 | 1022.5 | 663.5 | 85.2 | 55.3 | **-35.1%** |
| 16 | 1435.5 | 798.1 | 89.7 | 49.9 | **-44.4%** |

**The cliff is gone.** Before, per-row cost turned upward past 8 rows (63.4 →
85.2 → 89.7) because the pass fell off dispatch into Candle's generic loop.
After, it decreases monotonically all the way to K=16 (62.5 → 55.3 → 49.9).
That gate passes.

**The K=4 gate fails at 0.97x**, and the per-stage breakdown at K=8 says
precisely why:

| stage | before ms | after ms | change |
| --- | ---: | ---: | ---: |
| dense projections outside MoE | 138.9 | 119.2 | **-14.2%** |
| deltanet projections | 83.1 | 70.8 | **-14.7%** |
| lm_head | 42.6 | 32.7 | **-23.2%** |
| full attention | 40.9 | 38.3 | -6.4% |
| moe routed gate/up | 112.5 | 119.9 | +6.6% |
| moe routed down | 65.6 | 71.3 | +8.7% |
| moe router | 8.5 | 12.8 | +50.5% |
| deltanet recurrence | 49.7 | 51.5 | +3.5% |
| **wall** | **506.8** | **500.3** | **-1.3%** |

Every stage the fused kernel covers improves by 14–23%. The routed experts —
191 ms of the 500 ms pass, 38% — are excluded by the 4 MiB threshold and did not
improve. That is the whole explanation for a 1.58x kernel producing a 1.3%
whole-pass change at K=8.

The K=1 gate is **indeterminate at this harness's noise level**: two runs of the
same unchanged code path gave -8.0% and +14.7%. One row does not enter the
dispatch range, so `apply_op1_no_bwd` is reached by identical code before and
after; the variance is the benchmark's, not the change's.

### Does the removed cliff unlock longer drafts?

The practical question the cliff removal was supposed to answer. W1, 256 greedy
tokens, min-match 4, all token-identical to target-only:

| draft | tok/s | match rate | acceptance | tokens/pass |
| ---: | ---: | ---: | ---: | ---: |
| **7** | **8.51** | 25.2% | 80.4% | 6.48 |
| 11 | 7.50 | 22.3% | 62.8% | 7.61 |
| 15 | 7.76 | 18.4% | 62.5% | 9.72 |

No. Draft 7 remains optimal even with 9–16 row passes now cheap, because
acceptance falls (80.4% → 62.5%) faster than tokens-per-pass rises. This
retires the hypothesis from the previous task's report that widening the row
range was the highest-value follow-up — it was worth doing for the K=12/16 cost
curve, but it does not move end-to-end decoding.

## Where the remaining time actually goes

Combining Part 0 and C3: a verification pass at K=8 is 38% routed experts, and
those are the one major consumer the fused kernel cannot reach — not because of
an oversight but because the fused approach measurably loses at the 2–3 row
group sizes MoE routing produces. Any further work on multi-row verification
cost has to start there, and it needs a different technique than this one:
something with near-zero per-call setup, since the matrices are small and the
groups are tiny.

## Deviations from the specification

| # | Spec | Built | Why |
| ---: | --- | --- | --- |
| 1 | Q5K mandatory if present in either model file | Left on the existing per-row path | Q5K is 0.3% of this model's bytes (10 tensors, `attn_output` on full-attention layers only). The initial Q5K implementation was written from inference and I could not validate its high-bit handling against Candle's kernel with confidence; shipping an unvalidated quantization kernel to reach 0.3% of bytes is the wrong trade. It is dispatched to `small_m_forward` exactly as before. |
| 2 | Expert-path exception routing per-expert groups ≥ 2 (or ≥ 3) through the fused kernel | Not added | Measured regression at every group size that occurs in practice; see the A2 table. The task's own instruction was to revert and report if it regresses. |
| 3 | Tile of 4 suggested | TILE = 8 | Measured across 2/4/8/12/16; 8 is the optimum on this register file. |
| 4 | — | Added `QuantizedMatrix::forward_via` and `MultiRowPath` | Timing a kernel change requires pinning the kernel independently of dispatch. Production `forward` is unaffected. |

## Reproducing

```bash
CARGO_TARGET_DIR=target-native RUSTFLAGS='-C target-cpu=native' \
  cargo build --release --bin gguf_matmul_bench --bin gguf_verify_bench --bin gguf_infer

# Part 0 / C2 microbench: one kernel per process is the fair comparison
for p in candle smallm fused; do
  taskset -c 0-5 ./target-native/release/gguf_matmul_bench \
    --model /models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --repetitions 30 --warmup 8 --rows 2,4,8,12,16 --paths $p \
    --tensor blk.11.attn_q.weight --tensor blk.1.attn_qkv.weight \
    --tensor blk.0.ssm_out.weight --output kernel-run-logs/iso-$p.json
done

# C3 whole-pass, before and after (16 tokens so K=16 is expressible)
taskset -c 0-5 ./target-native/release/gguf_verify_bench \
  --model /models/Qwen3.6-35B-A3B/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --tokenizer-model /models/Qwen3.6-35B-A3B \
  --batch-sizes 1,2,4,8,12,16 --repetitions 3 --expert-cache-mib 46000 \
  --verification-tokens 8160,579,264,7047,1817,25,271,16,8160,579,264,7047,1817,25,271,16 \
  --output kernel-run-logs/verify-after.json
```

Timing methodology note that changed conclusions: measuring all repetitions of
one kernel before starting the next lets clock and thermal drift bias whichever
ran later — enough to move a single ratio between 1.46x and 1.70x. Every number
in this report uses one kernel per process, best-of-N.
