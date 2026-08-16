use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

use crate::{GgufCheckpoint, QuantizedMatrix, Qwen3NextConfig};

#[derive(Debug, Clone, Default)]
pub struct QuantizedAttentionState {
    keys: Vec<f32>,
    values: Vec<f32>,
    pub positions: usize,
}

impl QuantizedAttentionState {
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
    pub output_projection: Duration,
}

impl QuantizedAttentionTimings {
    pub fn accumulate(&mut self, other: &Self) {
        self.wall += other.wall;
        self.projections += other.projections;
        self.norm_rope += other.norm_rope;
        self.attention += other.attention;
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
        let mut result = vec![0.; seq * self.query_heads * self.head_dim];
        for token in 0..seq {
            let attend_to = existing + token + 1;
            for head in 0..self.query_heads {
                let kv_head = head / groups;
                let mut scores = Vec::with_capacity(attend_to);
                for past in 0..attend_to {
                    let base = (past * self.kv_heads + kv_head) * self.head_dim;
                    scores.push(
                        queries[token][head]
                            .iter()
                            .zip(&state.keys[base..base + self.head_dim])
                            .map(|(query, key)| query * key)
                            .sum::<f32>()
                            * scale,
                    );
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator: f32 = scores.iter().map(|score| (*score - max).exp()).sum();
                let output_base = (token * self.query_heads + head) * self.head_dim;
                for (past, score) in scores.into_iter().enumerate() {
                    let probability = (score - max).exp() / denominator;
                    let value_base = (past * self.kv_heads + kv_head) * self.head_dim;
                    for dimension in 0..self.head_dim {
                        result[output_base + dimension] +=
                            probability * state.values[value_base + dimension];
                    }
                }
                for dimension in 0..self.head_dim {
                    result[output_base + dimension] *= sigmoid(gates[token][head][dimension]);
                }
            }
        }
        timings.attention = attention_started.elapsed();
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
}
