use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

use crate::{Checkpoint, Qwen3NextConfig};

use super::{
    linear_profiled, load_f32_profiled, load_profiled, model::ForwardTimings, norm::rms_norm_gated,
};

#[derive(Debug, Clone)]
pub(crate) struct DeltaState {
    pub conv: Vec<f32>,
    pub recurrent: Vec<f32>,
}

impl DeltaState {
    pub fn new(config: &Qwen3NextConfig) -> Self {
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        Self {
            conv: vec![0.; conv_dim * (config.linear_conv_kernel_dim - 1)],
            recurrent: vec![
                0.;
                config.linear_num_value_heads
                    * config.linear_key_head_dim
                    * config.linear_value_head_dim
            ],
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    if x > 20. {
        x
    } else if x < -20. {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

fn l2_normalize_slice(x: &mut [f32]) {
    let scale = (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt().recip();
    for value in x {
        *value *= scale;
    }
}

/// Scalar recurrent Gated Delta Rule. State layout is `[head, key, value]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn recurrent_delta_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    heads: usize,
    key_dim: usize,
    value_dim: usize,
    state: &mut [f32],
    out: &mut [f32],
) {
    let q_scale = (key_dim as f32).sqrt().recip();
    for h in 0..heads {
        let state_base = h * key_dim * value_dim;
        let q_base = h * key_dim;
        let v_base = h * value_dim;
        let decay = g[h].exp();
        for cell in &mut state[state_base..state_base + key_dim * value_dim] {
            *cell *= decay;
        }
        for j in 0..value_dim {
            let mut prediction = 0.;
            for i in 0..key_dim {
                prediction += state[state_base + i * value_dim + j] * k[q_base + i];
            }
            let delta = (v[v_base + j] - prediction) * beta[h];
            for i in 0..key_dim {
                state[state_base + i * value_dim + j] += k[q_base + i] * delta;
            }
        }
        for j in 0..value_dim {
            let mut value = 0.;
            for i in 0..key_dim {
                value += state[state_base + i * value_dim + j] * q[q_base + i] * q_scale;
            }
            out[v_base + j] = value;
        }
    }
}

pub(crate) fn forward(
    checkpoint: &Checkpoint,
    config: &Qwen3NextConfig,
    layer: usize,
    xs: &Tensor,
    state: &mut DeltaState,
    timings: &mut ForwardTimings,
) -> Result<Tensor> {
    let dev = xs.device();
    let p = format!("model.layers.{layer}.linear_attn");
    let seq = xs.elem_count() / config.hidden_size;
    let nk = config.linear_num_key_heads;
    let nv = config.linear_num_value_heads;
    let kd = config.linear_key_head_dim;
    let vd = config.linear_value_head_dim;
    let ratio = nv / nk;
    let key_dim = nk * kd;
    let value_dim = nv * vd;
    let conv_dim = key_dim * 2 + value_dim;
    let kernel = config.linear_conv_kernel_dim;

    let qkvz_weight = load_profiled(
        checkpoint,
        &format!("{p}.in_proj_qkvz.weight"),
        dev,
        timings,
    )?;
    let ba_weight = load_profiled(checkpoint, &format!("{p}.in_proj_ba.weight"), dev, timings)?;
    let projected = linear_profiled(xs, &qkvz_weight, timings)?
        .reshape((seq, nk, kd * 2 + ratio * vd * 2))?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let projected_ba = linear_profiled(xs, &ba_weight, timings)?
        .reshape((seq, nk, ratio * 2))?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let conv_weight = load_f32_profiled(checkpoint, &format!("{p}.conv1d.weight"), dev, timings)?
        .reshape((conv_dim, kernel))?
        .to_vec2::<f32>()?;
    let a_log =
        load_f32_profiled(checkpoint, &format!("{p}.A_log"), dev, timings)?.to_vec1::<f32>()?;
    let dt_bias =
        load_f32_profiled(checkpoint, &format!("{p}.dt_bias"), dev, timings)?.to_vec1::<f32>()?;
    ensure!(
        state.conv.len() == conv_dim * (kernel - 1),
        "invalid DeltaNet convolution state"
    );
    ensure!(
        state.recurrent.len() == nv * kd * vd,
        "invalid DeltaNet recurrent state"
    );

    let mut output = vec![0f32; seq * value_dim];
    let mut gates = vec![0f32; seq * value_dim];
    for t in 0..seq {
        let mut q = vec![0.; key_dim];
        let mut k = vec![0.; key_dim];
        let mut v = vec![0.; value_dim];
        let mut z = vec![0.; value_dim];
        let mut beta = vec![0.; nv];
        let mut g = vec![0.; nv];
        for kh in 0..nk {
            let row = &projected[t][kh];
            q[kh * kd..(kh + 1) * kd].copy_from_slice(&row[..kd]);
            k[kh * kd..(kh + 1) * kd].copy_from_slice(&row[kd..kd * 2]);
            let value_start = kd * 2;
            let value_end = value_start + ratio * vd;
            v[kh * ratio * vd..(kh + 1) * ratio * vd].copy_from_slice(&row[value_start..value_end]);
            z[kh * ratio * vd..(kh + 1) * ratio * vd].copy_from_slice(&row[value_end..]);
            for r in 0..ratio {
                let vh = kh * ratio + r;
                beta[vh] = sigmoid(projected_ba[t][kh][r]);
                let a = projected_ba[t][kh][ratio + r];
                g[vh] = -a_log[vh].exp() * softplus(a + dt_bias[vh]);
            }
        }

        let mut mixed = Vec::with_capacity(conv_dim);
        mixed.extend_from_slice(&q);
        mixed.extend_from_slice(&k);
        mixed.extend_from_slice(&v);
        let mut convolved = vec![0.; conv_dim];
        for channel in 0..conv_dim {
            let hist = &state.conv[channel * (kernel - 1)..(channel + 1) * (kernel - 1)];
            let mut sum = conv_weight[channel][kernel - 1] * mixed[channel];
            for i in 0..kernel - 1 {
                sum += conv_weight[channel][i] * hist[i];
            }
            convolved[channel] = silu(sum);
        }
        if kernel > 1 {
            for (channel, &current) in mixed.iter().enumerate() {
                let hist = &mut state.conv[channel * (kernel - 1)..(channel + 1) * (kernel - 1)];
                hist.rotate_left(1);
                hist[kernel - 2] = current;
            }
        }
        q.copy_from_slice(&convolved[..key_dim]);
        k.copy_from_slice(&convolved[key_dim..key_dim * 2]);
        v.copy_from_slice(&convolved[key_dim * 2..]);

        let mut q_repeated = vec![0.; nv * kd];
        let mut k_repeated = vec![0.; nv * kd];
        for kh in 0..nk {
            l2_normalize_slice(&mut q[kh * kd..(kh + 1) * kd]);
            l2_normalize_slice(&mut k[kh * kd..(kh + 1) * kd]);
            for r in 0..ratio {
                let vh = kh * ratio + r;
                q_repeated[vh * kd..(vh + 1) * kd].copy_from_slice(&q[kh * kd..(kh + 1) * kd]);
                k_repeated[vh * kd..(vh + 1) * kd].copy_from_slice(&k[kh * kd..(kh + 1) * kd]);
            }
        }
        let out = &mut output[t * value_dim..(t + 1) * value_dim];
        recurrent_delta_step(
            &q_repeated,
            &k_repeated,
            &v,
            &g,
            &beta,
            nv,
            kd,
            vd,
            &mut state.recurrent,
            out,
        );
        gates[t * value_dim..(t + 1) * value_dim].copy_from_slice(&z);
    }
    // The reference kernel returns to the projection dtype before the gated
    // normalization, which matters for BF16 checkpoint comparisons.
    let output = Tensor::from_vec(output, (seq * nv, vd), dev)?.to_dtype(xs.dtype())?;
    let gates = Tensor::from_vec(gates, (seq * nv, vd), dev)?;
    let norm_weight = load_profiled(checkpoint, &format!("{p}.norm.weight"), dev, timings)?;
    let output = rms_norm_gated(&output, &norm_weight, &gates, config.rms_norm_eps)?
        .reshape((seq, value_dim))?;
    let out_weight = load_profiled(checkpoint, &format!("{p}.out_proj.weight"), dev, timings)?;
    Ok(linear_profiled(&output, &out_weight, timings)?.reshape(xs.shape())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_rule_updates_and_reuses_state() {
        let mut state = vec![0.; 4];
        let mut out = vec![0.; 2];
        recurrent_delta_step(
            &[1., 0.],
            &[1., 0.],
            &[2., 3.],
            &[0.],
            &[1.],
            1,
            2,
            2,
            &mut state,
            &mut out,
        );
        assert!((out[0] - 2. / 2f32.sqrt()).abs() < 1e-6);
        assert!((out[1] - 3. / 2f32.sqrt()).abs() < 1e-6);
        recurrent_delta_step(
            &[1., 0.],
            &[1., 0.],
            &[2., 3.],
            &[0.],
            &[1.],
            1,
            2,
            2,
            &mut state,
            &mut out,
        );
        assert!((out[0] - 2. / 2f32.sqrt()).abs() < 1e-6);
        assert!((out[1] - 3. / 2f32.sqrt()).abs() < 1e-6);
    }
}
