use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

use crate::{GgufCheckpoint, QuantizedMatrix, Qwen3NextConfig};

use super::{deltanet::recurrent_delta_step, norm::rms_norm_gated};

#[derive(Debug, Clone)]
pub struct QuantizedDeltaState {
    conv: Vec<f32>,
    recurrent: Vec<f32>,
    scratch: Box<QuantizedDeltaScratch>,
}

#[derive(Debug, Clone)]
struct QuantizedDeltaScratch {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    convolved: Vec<f32>,
    q_repeated: Vec<f32>,
    k_repeated: Vec<f32>,
    beta: Vec<f32>,
    decay: Vec<f32>,
    recurrent_output: Vec<f32>,
    gates: Vec<f32>,
}

impl QuantizedDeltaScratch {
    fn new(
        key_heads: usize,
        value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Self {
        let key_dim = key_heads * key_head_dim;
        let value_dim = value_heads * value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        Self {
            q: vec![0.; key_dim],
            k: vec![0.; key_dim],
            v: vec![0.; value_dim],
            convolved: vec![0.; conv_dim],
            q_repeated: vec![0.; value_heads * key_head_dim],
            k_repeated: vec![0.; value_heads * key_head_dim],
            beta: vec![0.; value_heads],
            decay: vec![0.; value_heads],
            recurrent_output: Vec::new(),
            gates: Vec::new(),
        }
    }

    fn prepare_output(&mut self, len: usize) {
        self.recurrent_output.resize(len, 0.);
        self.recurrent_output.fill(0.);
        self.gates.resize(len, 0.);
    }
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
            scratch: Box::new(QuantizedDeltaScratch::new(
                config.linear_num_key_heads,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            )),
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
    qkvz: QuantizedMatrix,
    beta_alpha: QuantizedMatrix,
    output: QuantizedMatrix,
    conv_weight: Vec<f32>,
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
        let qkvz = qkv.concatenate_rows(&z)?;
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
            .flatten_all()?
            .to_vec1::<f32>()?;
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
            qkvz,
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
            scratch: Box::new(QuantizedDeltaScratch::new(
                self.key_heads,
                self.value_heads,
                self.key_head_dim,
                self.value_head_dim,
            )),
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
        let key_dim = self.key_heads * self.key_head_dim;
        let value_dim = self.value_heads * self.value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let projected_width = conv_dim + value_dim;
        let ratio = self.value_heads / self.key_heads;
        let mut timings = QuantizedDeltaTimings::default();
        let projection_started = Instant::now();
        let projected = self.qkvz.forward(&flat)?.flatten_all()?.to_vec1::<f32>()?;
        state.scratch.prepare_output(seq * value_dim);
        for token in 0..seq {
            let token_projection =
                &projected[token * projected_width..(token + 1) * projected_width];
            state.scratch.gates[token * value_dim..(token + 1) * value_dim]
                .copy_from_slice(&token_projection[conv_dim..]);
        }
        let projected_ba = self
            .beta_alpha
            .forward(&flat)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        timings.projections = projection_started.elapsed();

        ensure!(
            state.conv.len() == conv_dim * (self.conv_kernel - 1),
            "invalid convolution state"
        );
        ensure!(
            state.recurrent.len() == self.value_heads * self.key_head_dim * self.value_head_dim,
            "invalid recurrent state"
        );
        let QuantizedDeltaState {
            conv,
            recurrent,
            scratch,
        } = state;
        for token in 0..seq {
            let projected = &projected[token * projected_width..token * projected_width + conv_dim];
            let convolution_started = Instant::now();
            causal_depthwise_conv_step(
                projected,
                &self.conv_weight,
                self.conv_kernel,
                conv,
                &mut scratch.convolved,
            );
            timings.convolution += convolution_started.elapsed();
            scratch.q.copy_from_slice(&scratch.convolved[..key_dim]);
            scratch
                .k
                .copy_from_slice(&scratch.convolved[key_dim..key_dim * 2]);
            scratch.v.copy_from_slice(&scratch.convolved[key_dim * 2..]);

            let recurrence_started = Instant::now();
            for key_head in 0..self.key_heads {
                normalize(
                    &mut scratch.q
                        [key_head * self.key_head_dim..(key_head + 1) * self.key_head_dim],
                );
                normalize(
                    &mut scratch.k
                        [key_head * self.key_head_dim..(key_head + 1) * self.key_head_dim],
                );
                for repeat in 0..ratio {
                    let value_head = key_head * ratio + repeat;
                    scratch.q_repeated
                        [value_head * self.key_head_dim..(value_head + 1) * self.key_head_dim]
                        .copy_from_slice(
                            &scratch.q
                                [key_head * self.key_head_dim..(key_head + 1) * self.key_head_dim],
                        );
                    scratch.k_repeated
                        [value_head * self.key_head_dim..(value_head + 1) * self.key_head_dim]
                        .copy_from_slice(
                            &scratch.k
                                [key_head * self.key_head_dim..(key_head + 1) * self.key_head_dim],
                        );
                }
            }
            let parameters_per_key_head = 2 * ratio;
            let token_parameters =
                &projected_ba[token * self.value_heads * 2..(token + 1) * self.value_heads * 2];
            for key_head in 0..self.key_heads {
                let parameters = &token_parameters
                    [key_head * parameters_per_key_head..(key_head + 1) * parameters_per_key_head];
                for repeat in 0..ratio {
                    let value_head = key_head * ratio + repeat;
                    scratch.beta[value_head] = sigmoid(parameters[repeat]);
                    let alpha = parameters[ratio + repeat];
                    scratch.decay[value_head] =
                        self.state_scale[value_head] * softplus(alpha + self.dt_bias[value_head]);
                }
            }
            recurrent_delta_step(
                &scratch.q_repeated,
                &scratch.k_repeated,
                &scratch.v,
                &scratch.decay,
                &scratch.beta,
                self.value_heads,
                self.key_head_dim,
                self.value_head_dim,
                recurrent,
                &mut scratch.recurrent_output[token * value_dim..(token + 1) * value_dim],
            );
            timings.recurrence += recurrence_started.elapsed();
        }
        let norm_started = Instant::now();
        let output = Tensor::from_slice(
            &scratch.recurrent_output,
            (seq * self.value_heads, self.value_head_dim),
            xs.device(),
        )?;
        let gates = Tensor::from_slice(
            &scratch.gates,
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

fn causal_depthwise_conv_step(
    input: &[f32],
    weights: &[f32],
    kernel: usize,
    state: &mut [f32],
    output: &mut [f32],
) {
    debug_assert!(kernel > 0);
    debug_assert_eq!(weights.len(), input.len() * kernel);
    debug_assert_eq!(state.len(), input.len() * (kernel - 1));
    debug_assert_eq!(output.len(), input.len());
    let history_len = kernel - 1;
    for channel in 0..input.len() {
        let weight_base = channel * kernel;
        let history_base = channel * history_len;
        let mut sum = weights[weight_base + history_len] * input[channel];
        let history = &mut state[history_base..history_base + history_len];
        for (tap, &previous) in history.iter().enumerate() {
            sum += weights[weight_base + tap] * previous;
        }
        output[channel] = silu(sum);
        if history_len > 0 {
            history.rotate_left(1);
            history[history_len - 1] = input[channel];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_output_reuses_capacity_and_clears_values() {
        let mut scratch = QuantizedDeltaScratch::new(2, 4, 8, 8);
        scratch.prepare_output(64);
        let capacity = scratch.recurrent_output.capacity();
        scratch.recurrent_output.fill(1.);

        scratch.prepare_output(32);

        assert_eq!(scratch.recurrent_output.len(), 32);
        assert_eq!(scratch.recurrent_output.capacity(), capacity);
        assert!(scratch.recurrent_output.iter().all(|&value| value == 0.));
    }

    #[test]
    fn flat_causal_convolution_matches_row_reference() {
        let input = [0.25, -0.5, 1.25];
        let weights = [0.1, 0.2, -0.3, 0.4, -0.2, 0.3, 0.5, -0.7, 0.6];
        let mut state = [0.75, -0.25, 0.5, 1.0, -1.5, 0.2];
        let mut expected_state = state;
        let mut expected_output = [0.; 3];
        for channel in 0..input.len() {
            let history = &expected_state[channel * 2..(channel + 1) * 2];
            let row = &weights[channel * 3..(channel + 1) * 3];
            let mut sum = row[2] * input[channel];
            for (tap, &previous) in history.iter().enumerate() {
                sum += row[tap] * previous;
            }
            expected_output[channel] = silu(sum);
        }
        for (channel, &current) in input.iter().enumerate() {
            let history = &mut expected_state[channel * 2..(channel + 1) * 2];
            history.rotate_left(1);
            history[1] = current;
        }

        let mut output = [0.; 3];
        causal_depthwise_conv_step(&input, &weights, 3, &mut state, &mut output);

        assert_eq!(output, expected_output);
        assert_eq!(state, expected_state);
    }

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
