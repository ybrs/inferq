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
        let groups = self.query_heads / self.kv_heads;
        let scale = (self.head_dim as f32).sqrt().recip();
        let (head_dim, query_heads, kv_heads) = (self.head_dim, self.query_heads, self.kv_heads);
        let mut result = vec![0.; seq * query_heads * head_dim];
        // One `(token, head)` output row is an independent piece of work: it
        // reads the shared cache and writes only its own `head_dim` floats.
        // Splitting there rather than inside a row is what keeps this exact:
        // every reduction still runs in the order it ran in serially, so the
        // heads occupy separate cores without moving a single last bit.
        //
        // The scan is the one part of a decode pass that grows with the
        // conversation — at three thousand context tokens it is most of the
        // pass — so it is also the one part worth the threads.
        let (keys, values) = (&state.keys, &state.values);
        let (scores_time, softmax_time, weighted_time) = result
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
                // denominator and once for its own weight. Keeping the first
                // one halves the transcendental calls and returns the same
                // float, where recomputing it merely spends the call again.
                let mut denominator = 0.;
                for score in scores.iter_mut() {
                    *score = (*score - max).exp();
                    denominator += *score;
                }
                let softmax_elapsed = softmax_started.elapsed();
                let weighted_started = Instant::now();
                for (past, weight) in scores.iter().enumerate() {
                    // A division, not a multiply by the reciprocal: those two
                    // do not round alike, and this path is the exact one.
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
