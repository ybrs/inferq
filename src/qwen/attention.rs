use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

use crate::{Checkpoint, Qwen3NextConfig};

use super::{linear_profiled, load_f32_profiled, load_profiled, model::ForwardTimings};

#[derive(Debug, Clone, Default)]
pub(crate) struct AttentionState {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub positions: usize,
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

fn qwen_norm(values: &mut [f32], weight: &[f32], eps: f64) {
    let variance = values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32;
    let scale = (variance + eps as f32).sqrt().recip();
    for (value, weight) in values.iter_mut().zip(weight) {
        *value = *value * scale * (1. + weight);
    }
}

fn apply_rope(values: &mut [f32], position: usize, rotary_dim: usize, theta: f64) {
    let half = rotary_dim / 2;
    let original = values[..rotary_dim].to_vec();
    for i in 0..half {
        let frequency = 1. / theta.powf((2 * i) as f64 / rotary_dim as f64);
        let angle = position as f64 * frequency;
        let (sin, cos) = angle.sin_cos();
        values[i] = original[i] * cos as f32 - original[half + i] * sin as f32;
        values[half + i] = original[half + i] * cos as f32 + original[i] * sin as f32;
    }
}

pub(crate) fn forward(
    checkpoint: &Checkpoint,
    config: &Qwen3NextConfig,
    layer: usize,
    xs: &Tensor,
    position: usize,
    state: &mut AttentionState,
    timings: &mut ForwardTimings,
) -> Result<Tensor> {
    let dev = xs.device();
    let p = format!("model.layers.{layer}.self_attn");
    let seq = xs.elem_count() / config.hidden_size;
    let nh = config.num_attention_heads;
    let nkh = config.num_key_value_heads;
    let hd = config.head_dim;
    let groups = nh / nkh;
    ensure!(
        state.positions == position,
        "attention cache has {} positions but decode begins at {position}",
        state.positions
    );

    let q_weight = load_profiled(checkpoint, &format!("{p}.q_proj.weight"), dev, timings)?;
    let k_weight = load_profiled(checkpoint, &format!("{p}.k_proj.weight"), dev, timings)?;
    let v_weight = load_profiled(checkpoint, &format!("{p}.v_proj.weight"), dev, timings)?;
    let q_projected = linear_profiled(xs, &q_weight, timings)?
        .reshape((seq, nh, hd * 2))?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let mut keys = linear_profiled(xs, &k_weight, timings)?
        .reshape((seq, nkh, hd))?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let values = linear_profiled(xs, &v_weight, timings)?
        .reshape((seq, nkh, hd))?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let q_norm = load_f32_profiled(checkpoint, &format!("{p}.q_norm.weight"), dev, timings)?
        .to_vec1::<f32>()?;
    let k_norm = load_f32_profiled(checkpoint, &format!("{p}.k_norm.weight"), dev, timings)?
        .to_vec1::<f32>()?;
    let mut query = vec![vec![vec![0.; hd]; nh]; seq];
    let mut gate = vec![vec![vec![0.; hd]; nh]; seq];
    for t in 0..seq {
        for h in 0..nh {
            query[t][h].copy_from_slice(&q_projected[t][h][..hd]);
            gate[t][h].copy_from_slice(&q_projected[t][h][hd..]);
            qwen_norm(&mut query[t][h], &q_norm, config.rms_norm_eps);
            apply_rope(
                &mut query[t][h],
                position + t,
                config.rotary_dim(),
                config.rope_theta,
            );
        }
        for key in keys[t].iter_mut().take(nkh) {
            qwen_norm(key, &k_norm, config.rms_norm_eps);
            apply_rope(key, position + t, config.rotary_dim(), config.rope_theta);
        }
    }

    let existing = state.positions;
    for t in 0..seq {
        for h in 0..nkh {
            state.keys.extend_from_slice(&keys[t][h]);
            state.values.extend_from_slice(&values[t][h]);
        }
    }
    state.positions += seq;
    let scale = (hd as f32).sqrt().recip();
    let mut output = vec![0f32; seq * nh * hd];
    for t in 0..seq {
        let attend_to = existing + t + 1;
        for h in 0..nh {
            let kvh = h / groups;
            let mut scores = Vec::with_capacity(attend_to);
            for past in 0..attend_to {
                let key_base = (past * nkh + kvh) * hd;
                let score = query[t][h]
                    .iter()
                    .zip(&state.keys[key_base..key_base + hd])
                    .map(|(q, k)| q * k)
                    .sum::<f32>()
                    * scale;
                scores.push(score);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = scores.iter().map(|x| (*x - max).exp()).sum();
            let out_base = (t * nh + h) * hd;
            for (past, score) in scores.into_iter().enumerate() {
                let probability = (score - max).exp() / denominator;
                let value_base = (past * nkh + kvh) * hd;
                for d in 0..hd {
                    output[out_base + d] += probability * state.values[value_base + d];
                }
            }
            for d in 0..hd {
                output[out_base + d] *= sigmoid(gate[t][h][d]);
            }
        }
    }
    let output = Tensor::from_vec(output, (seq, nh * hd), dev)?;
    let o_weight = load_profiled(checkpoint, &format!("{p}.o_proj.weight"), dev, timings)?;
    Ok(linear_profiled(&output, &o_weight, timings)?.reshape(xs.shape())?)
}

pub fn reference_attention(
    checkpoint: &Checkpoint,
    config: &Qwen3NextConfig,
    layer: usize,
    xs: &Tensor,
    position: usize,
) -> Result<Tensor> {
    let mut state = ReferenceAttentionState::default();
    reference_attention_step(checkpoint, config, layer, xs, position, &mut state)
}

#[derive(Debug, Clone, Default)]
pub struct ReferenceAttentionState {
    inner: AttentionState,
}

pub fn reference_attention_step(
    checkpoint: &Checkpoint,
    config: &Qwen3NextConfig,
    layer: usize,
    xs: &Tensor,
    position: usize,
    state: &mut ReferenceAttentionState,
) -> Result<Tensor> {
    let mut timings = ForwardTimings::default();
    forward(
        checkpoint,
        config,
        layer,
        xs,
        position,
        &mut state.inner,
        &mut timings,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_rope_preserves_unrotated_suffix() {
        let mut x = vec![1., 2., 3., 4., 5., 6.];
        apply_rope(&mut x, 3, 4, 10_000.);
        assert_eq!(&x[4..], &[5., 6.]);
    }
}
