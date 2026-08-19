use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};
use rayon::prelude::*;

use crate::{GgufCheckpoint, QuantizedMatrix, Qwen3NextConfig};

/// Lanes the score reduction carries independently.
///
/// Eight f32 is one AVX2 register, and eight separate chains is also what
/// lets the fused multiply-adds pipeline instead of each waiting on the one
/// before it. Both effects want the same number, so there is only one to pick.
const DOT_LANES: usize = 8;

/// Largest grouped-query group the blocked scan carries in registers.
///
/// The softmax holds one maximum and one denominator per query head sharing a
/// KV head, on the stack, so the group size has to have a compile-time bound.
/// Thirty-two is four times what any Qwen3-Next checkpoint uses; a checkpoint
/// beyond it takes the per-head scan rather than being scanned wrongly.
const MAX_GROUP: usize = 32;

/// Live KV bytes, per attention layer, above which the blocked scan wins.
///
/// The per-head scan reads each cache row once per query head in the group,
/// but those reads happen at the same time on the same last-level cache, which
/// hands them the same lines. So while a layer's live KV fits in that cache
/// the re-reads are nearly free and the per-head scan's single parallel region
/// is the cheaper shape; past it they become real misses and reading each row
/// once starts to matter. Measured on the qualified host (i7-8700, 12 MiB L3,
/// six threads, Qwen3.6-35B-A3B, sixteen decode passes) — scan wall seconds,
/// per-head against blocked:
///
/// | context | layer KV | per-head | blocked |
/// | ---: | ---: | ---: | ---: |
/// | 1024 | 4.2 MB | 0.154 | 0.241 |
/// | 3072 | 12.6 MB | 0.436 | 0.458 |
/// | 6144 | 25.2 MB | 0.915 | 0.821 |
/// | 8192 | 33.6 MB | 1.335 | 1.089 |
///
/// Sixteen mebibytes is the first power of two past this host's last-level
/// cache, and past the depth where the two measured equal. It is a tuning
/// constant for a class of host, not a property of the checkpoint, and it
/// cannot change a result: the two scans are bit-identical.
const KV_BLOCKING_BYTES: usize = 16 << 20;

/// Score scratch the scan keeps in flight, in floats.
///
/// One plane costs `attended positions * group size` floats, and a wide pass
/// at depth has hundreds of planes, so the pass is walked in row chunks that
/// keep this bounded rather than materialising every token's scores at once.
/// Four mebibytes is large enough that a chunk still fills the cores and small
/// enough to stay out of the way of the weights.
const SCORE_BUDGET: usize = 1 << 20;

/// Fewest positions a score block covers, and fewest columns a weighted-sum
/// block covers. Splitting finer than this buys parallelism with per-block
/// overhead and, for the columns, with partial cache lines.
const MIN_PAST_BLOCK: usize = 128;
const MIN_COLUMN_BLOCK: usize = 16;

/// How wide a block of `total` should be so that `planes` of them keep
/// `threads` cores busy, never finer than `grain`.
fn split(total: usize, planes: usize, threads: usize, grain: usize) -> usize {
    let parts = (2 * threads).div_ceil(planes.max(1)).max(1);
    total
        .div_ceil(parts)
        .next_multiple_of(grain)
        .clamp(grain, total.max(grain))
}

/// One column block of one plane's weighted sum, over every attended position.
///
/// The accumulator is transposed — `[column][group]` — so the `G` weights a
/// position contributes are one vector held in a register for the whole column
/// block, and each column is one broadcast and one multiply-add. `G` is a
/// constant for exactly that reason: at a runtime length the innermost loop
/// keeps its trip count and its prologue.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn weighted_sum_group<const G: usize>(
    out: &mut [f32],
    scores: &[f32],
    values: &[f32],
    kv_head: usize,
    kv_heads: usize,
    head_dim: usize,
    first_column: usize,
) {
    let width = out.len() / G;
    for (past, weights) in scores.chunks_exact(G).enumerate() {
        let base = (past * kv_heads + kv_head) * head_dim + first_column;
        let values = &values[base..base + width];
        for (out, value) in out.chunks_exact_mut(G).zip(values) {
            for (out, weight) in out.iter_mut().zip(weights) {
                *out += *weight * *value;
            }
        }
    }
}

/// Dispatch the weighted sum on the group size once per column block, rather
/// than testing it once per attended position.
#[allow(clippy::too_many_arguments)]
fn weighted_sum(
    groups: usize,
    out: &mut [f32],
    scores: &[f32],
    values: &[f32],
    kv_head: usize,
    kv_heads: usize,
    head_dim: usize,
    first_column: usize,
) {
    macro_rules! dispatch {
        ($($group:literal),*) => {
            match groups {
                $($group => weighted_sum_group::<$group>(
                    out, scores, values, kv_head, kv_heads, head_dim, first_column,
                ),)*
                _ => {
                    let width = out.len() / groups;
                    for (past, weights) in scores.chunks_exact(groups).enumerate() {
                        let base = (past * kv_heads + kv_head) * head_dim + first_column;
                        let values = &values[base..base + width];
                        for (out, value) in out.chunks_exact_mut(groups).zip(values) {
                            for (out, weight) in out.iter_mut().zip(weights) {
                                *out += *weight * *value;
                            }
                        }
                    }
                }
            }
        };
    }
    dispatch!(1, 2, 4, 8, 16);
}

/// The score of one key against one query.
///
/// Written as independent lane accumulators rather than a single running sum.
/// A single sum is a chain of dependent FMAs — each add waits four cycles for
/// the one before it, which measured 0.501 FLOP/cycle/core against a 0.500
/// theoretical ceiling for that shape — and the compiler may not break the
/// chain itself, because f32 addition is not associative and reassociating it
/// is not a transformation it is allowed to make.
///
/// Doing it by hand is therefore a deliberate numerical change: the lanes sum
/// their own subsequences and are combined at the end, so the same values are
/// added in a different order and the last bits differ from the serial form.
/// The order here is fixed and does not depend on the length, the thread, or
/// how many rows a pass evaluates, which is what keeps every path that has to
/// agree with another one agreeing.
fn dot(query: &[f32], key: &[f32]) -> f32 {
    debug_assert_eq!(query.len(), key.len());
    let mut lanes = [0f32; DOT_LANES];
    let mut queries = query.chunks_exact(DOT_LANES);
    let mut keys = key.chunks_exact(DOT_LANES);
    for (query, key) in queries.by_ref().zip(keys.by_ref()) {
        for ((lane, query), key) in lanes.iter_mut().zip(query).zip(key) {
            *lane += query * key;
        }
    }
    // Head dimensions are powers of two in every checkpoint this engine
    // loads, so the tail is normally empty; it is here so the function is
    // correct for a width that is not a multiple of the lane count.
    let mut total = queries
        .remainder()
        .iter()
        .zip(keys.remainder())
        .map(|(query, key)| query * key)
        .sum::<f32>();
    for lane in lanes {
        total += lane;
    }
    total
}

#[derive(Debug, Clone, Default)]
pub struct QuantizedAttentionState {
    keys: Vec<f32>,
    values: Vec<f32>,
    pub positions: usize,
}

/// Every byte of one attention layer's KV cache.
///
/// This is what distinguishes an image from a rollback checkpoint: a
/// checkpoint records only a position, because rolling back can truncate rows
/// that are still in memory. Restoring state that was produced by an earlier
/// process has no such rows to truncate, so the image carries them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QuantizedAttentionImage {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub positions: usize,
}

impl QuantizedAttentionImage {
    pub fn bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * std::mem::size_of::<f32>()
    }

    /// Reject a stored image whose row stride does not divide its position, so
    /// a truncated or hand-edited file fails here rather than in a kernel.
    pub fn validate(&self) -> Result<()> {
        if self.positions == 0 {
            ensure!(
                self.keys.is_empty() && self.values.is_empty(),
                "attention image holds rows for zero positions"
            );
            return Ok(());
        }
        ensure!(
            self.keys.len().is_multiple_of(self.positions)
                && self.values.len().is_multiple_of(self.positions),
            "attention image storage is inconsistent with its {} positions",
            self.positions
        );
        Ok(())
    }
}

impl QuantizedAttentionState {
    pub fn image(&self) -> QuantizedAttentionImage {
        QuantizedAttentionImage {
            keys: self.keys.clone(),
            values: self.values.clone(),
            positions: self.positions,
        }
    }

    /// Replace this cache with an image, which may hold more rows than the
    /// live state does. The caller owns the check that the image belongs to
    /// this model; only self-consistency is checked here.
    pub fn restore_image(&mut self, image: &QuantizedAttentionImage) -> Result<()> {
        image.validate()?;
        if self.positions > 0 && image.positions > 0 {
            let live_stride = self.keys.len() / self.positions;
            let image_stride = image.keys.len() / image.positions;
            ensure!(
                live_stride == image_stride,
                "attention image row stride {image_stride} does not match the live stride {live_stride}"
            );
        }
        self.keys.clear();
        self.keys.extend_from_slice(&image.keys);
        self.values.clear();
        self.values.extend_from_slice(&image.values);
        self.positions = image.positions;
        Ok(())
    }

    pub fn truncate(&mut self, positions: usize) -> Result<()> {
        ensure!(
            positions <= self.positions,
            "cannot extend attention state from {} to {positions} positions",
            self.positions
        );
        if positions == self.positions {
            return Ok(());
        }
        if positions == 0 {
            self.keys.clear();
            self.values.clear();
            self.positions = 0;
            return Ok(());
        }
        ensure!(
            self.keys.len().is_multiple_of(self.positions)
                && self.values.len().is_multiple_of(self.positions),
            "attention state storage is inconsistent with its position"
        );
        let keys_per_position = self.keys.len() / self.positions;
        let values_per_position = self.values.len() / self.positions;
        self.keys.truncate(positions * keys_per_position);
        self.values.truncate(positions * values_per_position);
        self.positions = positions;
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct QuantizedAttentionTimings {
    pub wall: Duration,
    pub projections: Duration,
    pub norm_rope: Duration,
    pub attention: Duration,
    /// The three parts of the scan, so the one that grows with the
    /// conversation can be attributed to an operation rather than a stage.
    /// Summed across threads, so they exceed `attention` by the thread count.
    /// Both scan plans report them the same way.
    pub scores: Duration,
    pub softmax: Duration,
    pub weighted_sum: Duration,
    pub output_projection: Duration,
}

impl QuantizedAttentionTimings {
    pub fn accumulate(&mut self, other: &Self) {
        self.wall += other.wall;
        self.projections += other.projections;
        self.norm_rope += other.norm_rope;
        self.attention += other.attention;
        self.scores += other.scores;
        self.softmax += other.softmax;
        self.weighted_sum += other.weighted_sum;
        self.output_projection += other.output_projection;
    }
}

pub struct QuantizedAttentionLayer {
    layer: usize,
    hidden_size: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f64,
    eps: f64,
    query: QuantizedMatrix,
    key: QuantizedMatrix,
    value: QuantizedMatrix,
    output: QuantizedMatrix,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
}

impl std::fmt::Debug for QuantizedAttentionLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedAttentionLayer")
            .field("layer", &self.layer)
            .field("query_heads", &self.query_heads)
            .field("kv_heads", &self.kv_heads)
            .field("head_dim", &self.head_dim)
            .finish_non_exhaustive()
    }
}

impl QuantizedAttentionLayer {
    pub fn load(
        checkpoint: &GgufCheckpoint,
        config: &Qwen3NextConfig,
        layer: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer}");
        let query = checkpoint.load_matrix(&format!("{prefix}.attn_q.weight"))?;
        let key = checkpoint.load_matrix(&format!("{prefix}.attn_k.weight"))?;
        let value = checkpoint.load_matrix(&format!("{prefix}.attn_v.weight"))?;
        let output = checkpoint.load_matrix(&format!("{prefix}.attn_output.weight"))?;
        ensure!(
            query.shape()
                == [
                    config.num_attention_heads * config.head_dim * 2,
                    config.hidden_size
                ],
            "invalid GGUF query/gate shape"
        );
        ensure!(
            key.shape()
                == [
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size
                ]
                && value.shape() == key.shape(),
            "invalid GGUF key/value shapes"
        );
        ensure!(
            output.shape()
                == [
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim
                ],
            "invalid GGUF attention output shape"
        );
        let query_norm = checkpoint
            .load_f32_vector(&format!("{prefix}.attn_q_norm.weight"))?
            .to_vec1::<f32>()?;
        let key_norm = checkpoint
            .load_f32_vector(&format!("{prefix}.attn_k_norm.weight"))?
            .to_vec1::<f32>()?;
        ensure!(
            query_norm.len() == config.head_dim && key_norm.len() == config.head_dim,
            "invalid attention Q/K norm dimensions"
        );
        Ok(Self {
            layer,
            hidden_size: config.hidden_size,
            query_heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rotary_dim: config.rotary_dim(),
            rope_theta: config.rope_theta,
            eps: config.rms_norm_eps,
            query,
            key,
            value,
            output,
            query_norm,
            key_norm,
        })
    }

    pub fn new_state(&self) -> QuantizedAttentionState {
        QuantizedAttentionState::default()
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        position: usize,
        state: &mut QuantizedAttentionState,
    ) -> Result<(Tensor, QuantizedAttentionTimings)> {
        let wall_started = Instant::now();
        ensure!(
            state.positions == position,
            "attention cache position mismatch"
        );
        ensure!(
            xs.elem_count().is_multiple_of(self.hidden_size),
            "attention input is not divisible by hidden size"
        );
        let seq = xs.elem_count() / self.hidden_size;
        let flat = xs.to_dtype(DType::F32)?.reshape((seq, self.hidden_size))?;
        let mut timings = QuantizedAttentionTimings::default();
        let projection_started = Instant::now();
        let projected_query = self
            .query
            .forward(&flat)?
            .reshape((seq, self.query_heads, self.head_dim * 2))?
            .to_vec3::<f32>()?;
        let mut keys = self
            .key
            .forward(&flat)?
            .reshape((seq, self.kv_heads, self.head_dim))?
            .to_vec3::<f32>()?;
        let values = self
            .value
            .forward(&flat)?
            .reshape((seq, self.kv_heads, self.head_dim))?
            .to_vec3::<f32>()?;
        timings.projections = projection_started.elapsed();
        let norm_started = Instant::now();
        let mut queries = vec![vec![vec![0.; self.head_dim]; self.query_heads]; seq];
        let mut gates = vec![vec![vec![0.; self.head_dim]; self.query_heads]; seq];
        for token in 0..seq {
            for head in 0..self.query_heads {
                queries[token][head]
                    .copy_from_slice(&projected_query[token][head][..self.head_dim]);
                gates[token][head].copy_from_slice(&projected_query[token][head][self.head_dim..]);
                converted_norm(&mut queries[token][head], &self.query_norm, self.eps);
                apply_rope(
                    &mut queries[token][head],
                    position + token,
                    self.rotary_dim,
                    self.rope_theta,
                );
            }
            for key in &mut keys[token] {
                converted_norm(key, &self.key_norm, self.eps);
                apply_rope(key, position + token, self.rotary_dim, self.rope_theta);
            }
        }
        timings.norm_rope = norm_started.elapsed();
        let existing = state.positions;
        for token in 0..seq {
            for head in 0..self.kv_heads {
                state.keys.extend_from_slice(&keys[token][head]);
                state.values.extend_from_slice(&values[token][head]);
            }
        }
        state.positions += seq;
        let attention_started = Instant::now();
        let shape = ScanShape {
            seq,
            existing,
            query_heads: self.query_heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            scale: (self.head_dim as f32).sqrt().recip(),
        };
        let mut result = vec![0.; seq * self.query_heads * self.head_dim];
        let scan = match shape.plan() {
            ScanPlan::PerHead => per_head_scan,
            ScanPlan::Blocked => blocked_scan,
        };
        let (scores_time, softmax_time, weighted_time) = scan(
            &shape,
            &queries,
            &gates,
            &state.keys,
            &state.values,
            &mut result,
        );
        timings.attention = attention_started.elapsed();
        timings.scores = scores_time;
        timings.softmax = softmax_time;
        timings.weighted_sum = weighted_time;
        let output_started = Instant::now();
        let result =
            Tensor::from_vec(result, (seq, self.query_heads * self.head_dim), xs.device())?;
        let result = self.output.forward(&result)?.reshape(xs.shape())?;
        timings.output_projection = output_started.elapsed();
        timings.wall = wall_started.elapsed();
        Ok((result, timings))
    }
}

/// The KV cache scan, one work item per `(token, query head)`.
///
/// Each item reads the shared cache and writes only its own `head_dim` floats.
/// Splitting there rather than inside a row is what keeps this exact: every
/// reduction still runs in the order it ran in serially, so the heads occupy
/// separate cores without moving a single last bit.
///
/// The `groups` query heads sharing a KV head each walk that head's whole
/// cache, so the rows are read `groups` times — but they are read by items
/// running at the same time on the same last-level cache, which hands them the
/// same lines. While a layer's live KV fits there that costs nothing, and this
/// plan's one parallel region and one output write per row are cheaper than
/// anything blocked. Past that it stops being free; see `blocked_scan`.
///
/// Returns the three operations' time summed across threads.
fn per_head_scan(
    shape: &ScanShape,
    queries: &[Vec<Vec<f32>>],
    gates: &[Vec<Vec<f32>>],
    keys: &[f32],
    values: &[f32],
    result: &mut [f32],
) -> (Duration, Duration, Duration) {
    let &ScanShape {
        existing,
        query_heads,
        kv_heads,
        head_dim,
        scale,
        ..
    } = shape;
    let groups = shape.groups();
    result
        .par_chunks_mut(head_dim)
        .enumerate()
        .map(|(row, out)| {
            let (token, head) = (row / query_heads, row % query_heads);
            let kv_head = head / groups;
            let attend_to = existing + token + 1;
            let query = &queries[token][head];
            let scores_started = Instant::now();
            let mut scores = Vec::with_capacity(attend_to);
            for past in 0..attend_to {
                let base = (past * kv_heads + kv_head) * head_dim;
                scores.push(dot(query, &keys[base..base + head_dim]) * scale);
            }
            let scores_elapsed = scores_started.elapsed();
            let softmax_started = Instant::now();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // Every score's exponential is needed twice, once for the
            // denominator and once for its own weight. Keeping the first one
            // halves the transcendental calls and returns the same float,
            // where recomputing it merely spends the call again.
            let mut denominator = 0.;
            for score in scores.iter_mut() {
                *score = (*score - max).exp();
                denominator += *score;
            }
            let softmax_elapsed = softmax_started.elapsed();
            let weighted_started = Instant::now();
            out.fill(0.);
            for (past, weight) in scores.iter().enumerate() {
                // A division, not a multiply by the reciprocal: those two do
                // not round alike, and this path is the exact one.
                let probability = *weight / denominator;
                let base = (past * kv_heads + kv_head) * head_dim;
                for (out, value) in out.iter_mut().zip(&values[base..base + head_dim]) {
                    *out += probability * value;
                }
            }
            for (out, gate) in out.iter_mut().zip(&gates[token][head]) {
                *out *= sigmoid(*gate);
            }
            (scores_elapsed, softmax_elapsed, weighted_started.elapsed())
        })
        .reduce(
            || (Duration::ZERO, Duration::ZERO, Duration::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        )
}

/// The KV cache scan, blocked so the query heads that share a KV head read it
/// together.
///
/// Every one of the `groups` query heads sharing a KV head attends over the
/// same key and value rows. Scanning them one at a time — which is what one
/// work item per `(token, head)` did — reads each of those rows once per query
/// head, and at agent context depths that traffic, not the arithmetic, is what
/// the scan waits on. Here a *plane* is one `(token, kv head)` pair carrying
/// its whole group, so each row is read once and dotted or accumulated against
/// all `groups` heads while it is still in L1.
///
/// The work is exactly the work the per-head scan did, in exactly the order it
/// did it: each score is the same dot of the same two vectors, each softmax
/// reduces over its own positions in increasing order, and each output element
/// accumulates its positions in increasing order. The result is bit-identical,
/// which `blocking_the_group_does_not_move_a_bit` asserts against a per-head
/// reference.
///
/// Returns the wall time of the three phases, which are disjoint.
fn blocked_scan(
    shape: &ScanShape,
    queries: &[Vec<Vec<f32>>],
    gates: &[Vec<Vec<f32>>],
    keys: &[f32],
    values: &[f32],
    result: &mut [f32],
) -> (Duration, Duration, Duration) {
    let &ScanShape {
        seq,
        existing,
        query_heads,
        kv_heads,
        head_dim,
        scale,
    } = shape;
    let groups = shape.groups();
    let (mut scores_time, mut softmax_time, mut weighted_time) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let threads = rayon::current_num_threads();
    // The plane is the unit the cache is read for: one token against one
    // KV head, carrying all `groups` query heads that share it. Its score
    // buffer is `[past][group]`, group-minor, so a past block is one
    // contiguous run and the eight weights one position contributes are
    // one vector.
    let plane_stride = (existing + seq) * groups;
    let rows_per_chunk = (SCORE_BUDGET / (kv_heads * plane_stride)).clamp(1, seq);
    let mut probabilities = vec![0f32; rows_per_chunk * kv_heads * plane_stride];
    let mut accumulator = vec![0f32; rows_per_chunk * kv_heads * head_dim * groups];
    for first_row in (0..seq).step_by(rows_per_chunk) {
        let rows = rows_per_chunk.min(seq - first_row);
        let planes = rows * kv_heads;
        // A prefill pass has planes to spare; a one-row decode has as many
        // planes as there are KV heads, which is two, so the split has to
        // come from inside a plane or four cores stand idle.
        let past_block = split(existing + first_row + rows, planes, threads, MIN_PAST_BLOCK);
        let column_block = split(head_dim, planes, threads, MIN_COLUMN_BLOCK);
        let probabilities = &mut probabilities[..planes * plane_stride];
        let accumulator = &mut accumulator[..planes * head_dim * groups];

        // Scores. Each key row is loaded once and dotted against every
        // query head in its group, which is the whole point: the cache
        // traffic falls by the group size while each dot keeps its own
        // operands and its own order.
        scores_time += probabilities
            .par_chunks_mut(plane_stride)
            .enumerate()
            .map(|(plane, buffer)| {
                let (row, kv_head) = (plane / kv_heads, plane % kv_heads);
                let token = first_row + row;
                let attend_to = existing + token + 1;
                let heads = &queries[token][kv_head * groups..(kv_head + 1) * groups];
                buffer[..attend_to * groups]
                    .par_chunks_mut(past_block * groups)
                    .enumerate()
                    .map(|(block, buffer)| {
                        let started = Instant::now();
                        let first_past = block * past_block;
                        for (offset, slot) in buffer.chunks_mut(groups).enumerate() {
                            let base = ((first_past + offset) * kv_heads + kv_head) * head_dim;
                            let key = &keys[base..base + head_dim];
                            for (slot, query) in slot.iter_mut().zip(heads) {
                                *slot = dot(query, key) * scale;
                            }
                        }
                        started.elapsed()
                    })
                    .sum::<Duration>()
            })
            .sum::<Duration>();

        // Softmax, in place, ending in probabilities rather than weights.
        // The lanes are the group's query heads, so each head's maximum
        // and denominator are still reduced over the positions in the
        // order a single head would have taken them.
        softmax_time += probabilities
            .par_chunks_mut(plane_stride)
            .enumerate()
            .map(|(plane, buffer)| {
                let started = Instant::now();
                let attend_to = existing + first_row + plane / kv_heads + 1;
                let buffer = &mut buffer[..attend_to * groups];
                let mut max = [f32::NEG_INFINITY; MAX_GROUP];
                for slot in buffer.chunks(groups) {
                    for (max, score) in max.iter_mut().zip(slot) {
                        *max = f32::max(*max, *score);
                    }
                }
                // Every score's exponential is needed twice, once for the
                // denominator and once for its own weight. Keeping the
                // first one halves the transcendental calls and returns
                // the same float, where recomputing it merely spends the
                // call again.
                let mut denominator = [0f32; MAX_GROUP];
                for slot in buffer.chunks_mut(groups) {
                    for ((slot, max), denominator) in
                        slot.iter_mut().zip(&max).zip(&mut denominator)
                    {
                        *slot = (*slot - *max).exp();
                        *denominator += *slot;
                    }
                }
                for slot in buffer.chunks_mut(groups) {
                    for (slot, denominator) in slot.iter_mut().zip(&denominator) {
                        // A division, not a multiply by the reciprocal:
                        // those two do not round alike, and this path is
                        // the exact one.
                        *slot /= *denominator;
                    }
                }
                started.elapsed()
            })
            .sum::<Duration>();

        // Weighted sum, accumulated transposed as `[column][group]`. That
        // is what lets a column block be a contiguous chunk one core can
        // own, so the value rows are also read once per group rather than
        // once per query head, and every output element still sums its
        // positions in increasing order.
        let probabilities = &probabilities[..];
        accumulator.fill(0.);
        weighted_time += accumulator
            .par_chunks_mut(head_dim * groups)
            .enumerate()
            .map(|(plane, out)| {
                let (row, kv_head) = (plane / kv_heads, plane % kv_heads);
                let attend_to = existing + first_row + row + 1;
                let scores =
                    &probabilities[plane * plane_stride..plane * plane_stride + attend_to * groups];
                out.par_chunks_mut(column_block * groups)
                    .enumerate()
                    .map(|(block, out)| {
                        let started = Instant::now();
                        weighted_sum(
                            groups,
                            out,
                            scores,
                            values,
                            kv_head,
                            kv_heads,
                            head_dim,
                            block * column_block,
                        );
                        started.elapsed()
                    })
                    .sum::<Duration>()
            })
            .sum::<Duration>();
        // Back into `[token][head][column]` order, gated on the way.
        let accumulator = &accumulator[..];
        let rows_first = first_row * query_heads * head_dim;
        weighted_time += result[rows_first..rows_first + rows * query_heads * head_dim]
            .par_chunks_mut(query_heads * head_dim)
            .enumerate()
            .map(|(row, out)| {
                let started = Instant::now();
                let token = first_row + row;
                for (head, out) in out.chunks_mut(head_dim).enumerate() {
                    let (kv_head, lane) = (head / groups, head % groups);
                    let plane = (row * kv_heads + kv_head) * head_dim * groups;
                    let accumulated = &accumulator[plane..plane + head_dim * groups];
                    for (column, (out, gate)) in out.iter_mut().zip(&gates[token][head]).enumerate()
                    {
                        *out = accumulated[column * groups + lane] * sigmoid(*gate);
                    }
                }
                started.elapsed()
            })
            .sum::<Duration>();
    }
    (scores_time, softmax_time, weighted_time)
}

/// What one scan is over: a pass of `seq` rows against `existing` cached
/// positions, in a checkpoint's head geometry.
struct ScanShape {
    seq: usize,
    existing: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl ScanShape {
    /// Query heads per KV head: the width of the block this scan reads for.
    fn groups(&self) -> usize {
        self.query_heads / self.kv_heads
    }

    /// Bytes of key and value this scan's cache holds once it is appended to.
    fn cache_bytes(&self) -> usize {
        (self.existing + self.seq) * self.kv_heads * self.head_dim * 2 * size_of::<f32>()
    }

    /// Which scan this shape should take.
    fn plan(&self) -> ScanPlan {
        if self.groups() > 1
            && self.groups() <= MAX_GROUP
            && self.cache_bytes() >= KV_BLOCKING_BYTES
        {
            ScanPlan::Blocked
        } else {
            ScanPlan::PerHead
        }
    }
}

/// Which of the two exact scans a pass takes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ScanPlan {
    /// One work item per `(token, query head)`, each walking the whole cache
    /// for its own KV head. See `per_head_scan`.
    PerHead,
    /// One work item per `(token, KV head)`, reading each cache row once for
    /// the whole group. See `blocked_scan`.
    Blocked,
}

fn sigmoid(value: f32) -> f32 {
    1. / (1. + (-value).exp())
}

fn converted_norm(values: &mut [f32], weight: &[f32], eps: f64) {
    let variance = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let scale = (variance + eps as f32).sqrt().recip();
    for (value, weight) in values.iter_mut().zip(weight) {
        *value = *value * scale * weight;
    }
}

fn apply_rope(values: &mut [f32], position: usize, rotary_dim: usize, theta: f64) {
    let half = rotary_dim / 2;
    let original = values[..rotary_dim].to_vec();
    for index in 0..half {
        let frequency = 1. / theta.powf((2 * index) as f64 / rotary_dim as f64);
        let angle = position as f64 * frequency;
        let (sin, cos) = angle.sin_cos();
        values[index] = original[index] * cos as f32 - original[half + index] * sin as f32;
        values[half + index] = original[half + index] * cos as f32 + original[index] * sin as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converted_qk_norm_uses_weight_without_second_offset() {
        let mut values = vec![3., 4.];
        converted_norm(&mut values, &[2., 3.], 0.);
        let rms = (12.5f32).sqrt();
        assert!((values[0] - 6. / rms).abs() < 1e-6);
        assert!((values[1] - 12. / rms).abs() < 1e-6);
    }

    #[test]
    fn attention_state_truncation_preserves_position_stride() {
        let mut state = QuantizedAttentionState {
            keys: (0..12).map(|value| value as f32).collect(),
            values: (0..18).map(|value| value as f32).collect(),
            positions: 3,
        };
        state.truncate(2).unwrap();
        assert_eq!(state.positions, 2);
        assert_eq!(state.keys.len(), 8);
        assert_eq!(state.values.len(), 12);
        state.truncate(0).unwrap();
        assert!(state.keys.is_empty());
        assert!(state.values.is_empty());
    }

    #[test]
    fn the_dot_sums_its_lanes_in_a_fixed_order() {
        // A serial sum and a lane sum agree exactly whenever the arithmetic is
        // exact, which is what makes the difference elsewhere a rounding one
        // rather than a mistake.
        let query: Vec<f32> = (0..256).map(|value| value as f32).collect();
        let key: Vec<f32> = (0..256).map(|value| (value % 7) as f32).collect();
        let serial: f32 = query.iter().zip(&key).map(|(q, k)| q * k).sum();
        assert_eq!(dot(&query, &key), serial);

        // A width that is not a multiple of the lane count still consumes
        // every element.
        let query = vec![1.; 11];
        let key: Vec<f32> = (0..11).map(|value| value as f32).collect();
        assert_eq!(dot(&query, &key), (0..11).map(|v| v as f32).sum::<f32>());
        assert_eq!(dot(&[], &[]), 0.);

        // The order is the property the callers depend on: the same inputs
        // give the same bits every time, whatever the length.
        for length in [8usize, 64, 256, 300] {
            let query: Vec<f32> = (0..length).map(|v| (v as f32).sin()).collect();
            let key: Vec<f32> = (0..length).map(|v| (v as f32).cos()).collect();
            assert_eq!(dot(&query, &key).to_bits(), dot(&query, &key).to_bits());
        }
    }

    #[test]
    fn an_image_restores_every_row_into_a_fresh_state() {
        let state = QuantizedAttentionState {
            keys: (0..12).map(|value| value as f32).collect(),
            values: (0..18).map(|value| value as f32).collect(),
            positions: 3,
        };
        let image = state.image();
        // A checkpoint of the same state could only ever shorten it; the image
        // rebuilds it from nothing, which is what a restart needs.
        let mut restored = QuantizedAttentionState::default();
        restored.restore_image(&image).unwrap();
        assert_eq!(restored.image(), image);
        assert_eq!(restored.positions, 3);

        // Restoring over a longer state replaces it rather than appending.
        let mut longer = QuantizedAttentionState {
            keys: vec![9.; 40],
            values: vec![9.; 60],
            positions: 10,
        };
        longer.restore_image(&image).unwrap();
        assert_eq!(longer.image(), image);
    }

    /// Per-token per-head queries and gates, plus a filled key and value cache.
    struct ScanFixture {
        queries: Vec<Vec<Vec<f32>>>,
        gates: Vec<Vec<Vec<f32>>>,
        keys: Vec<f32>,
        values: Vec<f32>,
    }

    fn scan_fixture(shape: &ScanShape) -> ScanFixture {
        // Cheap rather than pretty: the property under test is bit-identity
        // between two scans of the same numbers, and the deepest fixture fills
        // eight million cells, so a transcendental per cell is the test's whole
        // runtime for nothing.
        let wave = |seed: f32, index: usize| {
            let mixed = (index as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.to_bits().rotate_left(11));
            ((mixed >> 9) % 4096) as f32 / 2048. - 1.
        };
        let rows = |heads: usize, seed: f32| -> Vec<Vec<f32>> {
            (0..heads)
                .map(|head| {
                    (0..shape.head_dim)
                        .map(|column| wave(seed + head as f32 * 0.11, column))
                        .collect()
                })
                .collect()
        };
        let queries = (0..shape.seq)
            .map(|token| rows(shape.query_heads, 0.37 + token as f32 * 0.07))
            .collect();
        let gates = (0..shape.seq)
            .map(|token| rows(shape.query_heads, 0.91 + token as f32 * 0.05))
            .collect();
        let cached = shape.existing + shape.seq;
        let cells = cached * shape.kv_heads * shape.head_dim;
        let keys = (0..cells).map(|index| wave(0.013, index)).collect();
        let values = (0..cells).map(|index| wave(0.029, index)).collect();
        ScanFixture {
            queries,
            gates,
            keys,
            values,
        }
    }

    #[test]
    fn blocking_the_group_does_not_move_a_bit() {
        // Both shapes that matter: a wide pass, where the planes alone fill
        // the cores, and a one-row decode at depth, where they cannot and the
        // split has to come from inside a plane.
        for (seq, existing) in [(6usize, 5usize), (1, 613), (1, 3), (4, 0)] {
            let shape = ScanShape {
                seq,
                existing,
                query_heads: 8,
                kv_heads: 2,
                head_dim: 24,
                scale: (24f32).sqrt().recip(),
            };
            let fixture = scan_fixture(&shape);
            let ScanFixture {
                queries,
                gates,
                keys,
                values,
            } = &fixture;
            let mut blocked = vec![0.; seq * shape.query_heads * shape.head_dim];
            let mut reference = blocked.clone();
            blocked_scan(&shape, queries, gates, keys, values, &mut blocked);
            per_head_scan(&shape, queries, gates, keys, values, &mut reference);
            assert_eq!(blocked, reference, "seq {seq}, existing {existing}");
        }
    }

    #[test]
    fn the_two_scans_agree_at_the_checkpoint_geometry_that_switches_them() {
        // The checkpoint-gated suites never reach a context deep enough to
        // take the blocked plan, so the equivalence at the geometry and the
        // depth where the switch actually happens is asserted here: 16 query
        // heads over 2 KV heads at head_dim 256, one row against 4096 cached
        // positions, which is the first depth past `KV_BLOCKING_BYTES`.
        let shape = ScanShape {
            seq: 1,
            existing: 4096,
            query_heads: 16,
            kv_heads: 2,
            head_dim: 256,
            scale: (256f32).sqrt().recip(),
        };
        assert_eq!(shape.plan(), ScanPlan::Blocked);
        let fixture = scan_fixture(&shape);
        let ScanFixture {
            queries,
            gates,
            keys,
            values,
        } = &fixture;
        let mut blocked = vec![0.; shape.query_heads * shape.head_dim];
        let mut reference = blocked.clone();
        blocked_scan(&shape, queries, gates, keys, values, &mut blocked);
        per_head_scan(&shape, queries, gates, keys, values, &mut reference);
        assert_eq!(blocked, reference);
    }

    #[test]
    fn the_blocked_scan_waits_for_a_cache_that_does_not_fit() {
        let shape = |existing: usize, seq: usize, query_heads: usize, kv_heads: usize| ScanShape {
            seq,
            existing,
            query_heads,
            kv_heads,
            head_dim: 256,
            scale: 1.,
        };
        // Qwen3.6-35B-A3B's geometry: 4.2 MB at 1024 context, 33.6 MB at 8192.
        assert_eq!(shape(1024, 1, 16, 2).plan(), ScanPlan::PerHead);
        assert_eq!(shape(3072, 1, 16, 2).plan(), ScanPlan::PerHead);
        assert_eq!(shape(8192, 1, 16, 2).plan(), ScanPlan::Blocked);
        // A wide pass counts the rows it is about to append.
        assert_eq!(shape(3584, 512, 16, 2).plan(), ScanPlan::Blocked);
        // Nothing to block when every query head has its own KV head, however
        // deep the conversation gets.
        assert_eq!(shape(8192, 1, 16, 16).plan(), ScanPlan::PerHead);
    }

    #[test]
    fn a_block_never_splits_finer_than_its_grain() {
        // Enough planes to fill the cores: one block each.
        assert_eq!(split(3072, 64, 6, 128), 3072);
        // A one-row decode has two planes, so the positions carry the split.
        let block = split(3072, 2, 6, 128);
        assert!(block <= 3072 / 3 && block.is_multiple_of(128), "{block}");
        // Never finer than the grain, even when the total is smaller than it.
        assert_eq!(split(8, 1, 64, 16), 16);
    }

    #[test]
    fn an_image_from_another_shape_is_rejected() {
        let mut state = QuantizedAttentionState {
            keys: vec![1.; 12],
            values: vec![1.; 18],
            positions: 3,
        };
        let wider = QuantizedAttentionImage {
            keys: vec![1.; 24],
            values: vec![1.; 18],
            positions: 3,
        };
        assert!(state.restore_image(&wider).is_err());
        let ragged = QuantizedAttentionImage {
            keys: vec![1.; 13],
            values: vec![1.; 18],
            positions: 3,
        };
        assert!(ragged.validate().is_err());
        assert!(state.restore_image(&ragged).is_err());
    }
}
