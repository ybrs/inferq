use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

use crate::{GgufCheckpoint, QuantizedMatrix, Qwen3NextConfig};

use super::{deltanet::recurrent_delta_step, norm::rms_norm_gated};

#[derive(Debug, Clone)]
pub struct QuantizedDeltaState {
    conv: Vec<f32>,
    recurrent: Vec<f32>,
}

impl QuantizedDeltaState {
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

#[derive(Debug, Clone, Default)]
pub struct QuantizedDeltaTimings {
    pub wall: Duration,
    pub projections: Duration,
    pub convolution: Duration,
    pub recurrence: Duration,
    pub gated_norm: Duration,
    pub output_projection: Duration,
}

impl QuantizedDeltaTimings {
    pub fn accumulate(&mut self, other: &Self) {
        self.wall += other.wall;
        self.projections += other.projections;
        self.convolution += other.convolution;
        self.recurrence += other.recurrence;
        self.gated_norm += other.gated_norm;
        self.output_projection += other.output_projection;
    }
}

pub struct QuantizedDeltaLayer {
    layer: usize,
    hidden_size: usize,
    key_heads: usize,
    value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_kernel: usize,
    eps: f64,
    qkv: QuantizedMatrix,
    z: QuantizedMatrix,
    beta_alpha: QuantizedMatrix,
    output: QuantizedMatrix,
    conv_weight: Vec<Vec<f32>>,
    state_scale: Vec<f32>,
    dt_bias: Vec<f32>,
    norm_weight: Tensor,
}

impl std::fmt::Debug for QuantizedDeltaLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedDeltaLayer")
            .field("layer", &self.layer)
            .field("hidden_size", &self.hidden_size)
            .field("key_heads", &self.key_heads)
            .field("value_heads", &self.value_heads)
            .finish_non_exhaustive()
    }
}

impl QuantizedDeltaLayer {
    pub fn load(
        checkpoint: &GgufCheckpoint,
        config: &Qwen3NextConfig,
        layer: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer}");
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let qkv = checkpoint.load_matrix(&format!("{prefix}.attn_qkv.weight"))?;
        let z = checkpoint.load_matrix(&format!("{prefix}.attn_gate.weight"))?;
        let beta_alpha = checkpoint.load_matrix(&format!("{prefix}.ssm_ba.weight"))?;
        let output = checkpoint.load_matrix(&format!("{prefix}.ssm_out.weight"))?;
        ensure!(
            qkv.shape() == [conv_dim, config.hidden_size],
            "invalid GGUF qkv shape"
        );
        ensure!(
            z.shape() == [value_dim, config.hidden_size],
            "invalid GGUF z shape"
        );
        ensure!(
            beta_alpha.shape() == [config.linear_num_value_heads * 2, config.hidden_size],
            "invalid GGUF beta/alpha shape"
        );
        ensure!(
            output.shape() == [config.hidden_size, value_dim],
            "invalid GGUF DeltaNet output shape"
        );
        let conv_weight = checkpoint
            .load_f32_tensor(&format!("{prefix}.ssm_conv1d.weight"))?
            .reshape((conv_dim, config.linear_conv_kernel_dim))?
            .to_vec2::<f32>()?;
        let state_scale = checkpoint
            .load_f32_vector(&format!("{prefix}.ssm_a"))?
            .to_vec1::<f32>()?;
        let dt_bias = checkpoint
            .load_f32_vector(&format!("{prefix}.ssm_dt.bias"))?
            .to_vec1::<f32>()?;
        let norm_weight = checkpoint.load_f32_vector(&format!("{prefix}.ssm_norm.weight"))?;
        ensure!(
            state_scale.len() == config.linear_num_value_heads,
            "invalid ssm_a length"
        );
        ensure!(
            dt_bias.len() == config.linear_num_value_heads,
            "invalid ssm_dt length"
        );
        ensure!(
            norm_weight.elem_count() == config.linear_value_head_dim,
            "invalid ssm norm length"
        );
        Ok(Self {
            layer,
            hidden_size: config.hidden_size,
            key_heads: config.linear_num_key_heads,
            value_heads: config.linear_num_value_heads,
            key_head_dim: config.linear_key_head_dim,
            value_head_dim: config.linear_value_head_dim,
            conv_kernel: config.linear_conv_kernel_dim,
            eps: config.rms_norm_eps,
            qkv,
            z,
            beta_alpha,
            output,
            conv_weight,
            state_scale,
            dt_bias,
            norm_weight,
        })
    }

    pub fn new_state(&self) -> QuantizedDeltaState {
        let key_dim = self.key_heads * self.key_head_dim;
        let value_dim = self.value_heads * self.value_head_dim;
        QuantizedDeltaState {
            conv: vec![0.; (key_dim * 2 + value_dim) * (self.conv_kernel - 1)],
            recurrent: vec![0.; self.value_heads * self.key_head_dim * self.value_head_dim],
        }
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        state: &mut QuantizedDeltaState,
    ) -> Result<(Tensor, QuantizedDeltaTimings)> {
        let wall_started = Instant::now();
        ensure!(
            xs.elem_count().is_multiple_of(self.hidden_size),
            "DeltaNet input is not divisible by hidden size"
        );
        let seq = xs.elem_count() / self.hidden_size;
        let flat = xs.to_dtype(DType::F32)?.reshape((seq, self.hidden_size))?;
        let mut timings = QuantizedDeltaTimings::default();
        let projection_started = Instant::now();
        let projected = self.qkv.forward(&flat)?.to_vec2::<f32>()?;
        let gates = self.z.forward(&flat)?.to_vec2::<f32>()?;
        let projected_ba = self
            .beta_alpha
            .forward(&flat)?
            .reshape((seq, self.key_heads, 2 * self.value_heads / self.key_heads))?
            .to_vec3::<f32>()?;
        timings.projections = projection_started.elapsed();

        let key_dim = self.key_heads * self.key_head_dim;
        let value_dim = self.value_heads * self.value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let ratio = self.value_heads / self.key_heads;
        ensure!(
            state.conv.len() == conv_dim * (self.conv_kernel - 1),
            "invalid convolution state"
        );
        ensure!(
            state.recurrent.len() == self.value_heads * self.key_head_dim * self.value_head_dim,
            "invalid recurrent state"
        );
        let mut recurrent_output = vec![0.; seq * value_dim];
        for token in 0..seq {
            let mut q = projected[token][..key_dim].to_vec();
            let mut k = projected[token][key_dim..key_dim * 2].to_vec();
            let mut v = projected[token][key_dim * 2..].to_vec();
            let convolution_started = Instant::now();
            let mut mixed = Vec::with_capacity(conv_dim);
            mixed.extend_from_slice(&q);
            mixed.extend_from_slice(&k);
            mixed.extend_from_slice(&v);
            let mut convolved = vec![0.; conv_dim];
            for channel in 0..conv_dim {
                let history = &state.conv
                    [channel * (self.conv_kernel - 1)..(channel + 1) * (self.conv_kernel - 1)];
                let mut sum = self.conv_weight[channel][self.conv_kernel - 1] * mixed[channel];
                for (tap, previous) in history.iter().enumerate() {
                    sum += self.conv_weight[channel][tap] * previous;
                }
                convolved[channel] = silu(sum);
            }
            if self.conv_kernel > 1 {
                for (channel, &current) in mixed.iter().enumerate() {
                    let history = &mut state.conv
                        [channel * (self.conv_kernel - 1)..(channel + 1) * (self.conv_kernel - 1)];
                    history.rotate_left(1);
                    history[self.conv_kernel - 2] = current;
                }
            }
            timings.convolution += convolution_started.elapsed();
            q.copy_from_slice(&convolved[..key_dim]);
            k.copy_from_slice(&convolved[key_dim..key_dim * 2]);
            v.copy_from_slice(&convolved[key_dim * 2..]);

            let recurrence_started = Instant::now();
            let mut q_repeated = vec![0.; self.value_heads * self.key_head_dim];
            let mut k_repeated = vec![0.; self.value_heads * self.key_head_dim];
            for key_head in 0..self.key_heads {
                normalize(&mut q[key_head * self.key_head_dim..(key_head + 1) * self.key_head_dim]);
                normalize(&mut k[key_head * self.key_head_dim..(key_head + 1) * self.key_head_dim]);
                for repeat in 0..ratio {
                    let value_head = key_head * ratio + repeat;
                    q_repeated
                        [value_head * self.key_head_dim..(value_head + 1) * self.key_head_dim]
                        .copy_from_slice(
                            &q[key_head * self.key_head_dim..(key_head + 1) * self.key_head_dim],
                        );
                    k_repeated
                        [value_head * self.key_head_dim..(value_head + 1) * self.key_head_dim]
                        .copy_from_slice(
                            &k[key_head * self.key_head_dim..(key_head + 1) * self.key_head_dim],
                        );
                }
            }
            let mut beta = vec![0.; self.value_heads];
            let mut decay = vec![0.; self.value_heads];
            for (key_head, parameters) in
                projected_ba[token].iter().enumerate().take(self.key_heads)
            {
                for repeat in 0..ratio {
                    let value_head = key_head * ratio + repeat;
                    beta[value_head] = sigmoid(parameters[repeat]);
                    let alpha = parameters[ratio + repeat];
                    decay[value_head] =
                        self.state_scale[value_head] * softplus(alpha + self.dt_bias[value_head]);
                }
            }
            recurrent_delta_step(
                &q_repeated,
                &k_repeated,
                &v,
                &decay,
                &beta,
                self.value_heads,
                self.key_head_dim,
                self.value_head_dim,
                &mut state.recurrent,
                &mut recurrent_output[token * value_dim..(token + 1) * value_dim],
            );
            timings.recurrence += recurrence_started.elapsed();
        }
        let norm_started = Instant::now();
        let output = Tensor::from_vec(
            recurrent_output,
            (seq * self.value_heads, self.value_head_dim),
            xs.device(),
        )?;
        let gates = Tensor::from_vec(
            gates.into_iter().flatten().collect(),
            (seq * self.value_heads, self.value_head_dim),
            xs.device(),
        )?;
        let output = rms_norm_gated(&output, &self.norm_weight, &gates, self.eps)?
            .reshape((seq, value_dim))?;
        timings.gated_norm = norm_started.elapsed();
        let output_started = Instant::now();
        let output = self.output.forward(&output)?.reshape(xs.shape())?;
        timings.output_projection = output_started.elapsed();
        timings.wall = wall_started.elapsed();
        Ok((output, timings))
    }
}

fn sigmoid(value: f32) -> f32 {
    1. / (1. + (-value).exp())
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn softplus(value: f32) -> f32 {
    if value > 20. {
        value
    } else if value < -20. {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

fn normalize(values: &mut [f32]) {
    let scale = (values.iter().map(|value| value * value).sum::<f32>() + 1e-6)
        .sqrt()
        .recip();
    for value in values {
        *value *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_state_scale_is_already_negative_exp_a_log() {
        let a_log = 1.5f32;
        let gguf_value = -a_log.exp();
        let alpha = 0.25;
        let bias = -0.1;
        let reference = -a_log.exp() * softplus(alpha + bias);
        assert!((gguf_value * softplus(alpha + bias) - reference).abs() < 1e-6);
    }
}
