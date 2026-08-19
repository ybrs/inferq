use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};
use rayon::prelude::*;

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

/// Fewest rows in a pass for the recurrence to be worth the thread pool.
///
/// The step runs once per row, so a pass of `n` rows wakes the pool `n` times
/// per layer and the waking is paid per row, not per pass. What falls with the
/// row count is the cost of each waking: consecutive rows call back into the
/// pool immediately, so only the first finds the workers asleep. Measured on
/// the qualified host — DeltaNet recurrence, milliseconds per token, over
/// 96 tokens prefilled at each width:
///
/// | rows | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 32 |
/// | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
/// | calling thread | 19.93 | 8.15 | 6.95 | 6.65 | 6.13 | 5.99 | 5.71 | 5.67 | 5.37 |
/// | thread pool | — | 11.77 | 8.01 | 6.15 | 5.53 | 4.97 | 4.13 | 3.96 | 3.36 |
///
/// Four is where it turns. That matters beyond prefill because it is also the
/// width of a speculative verification pass: with an MTP draft accepted about
/// four times in five, most verifications are two or three rows, and spreading
/// those cost 4% of decode on the end-to-end agent turn — twice, before this
/// threshold was measured rather than assumed.
const HEAD_SPREAD_MIN_ROWS: usize = 4;

/// Where one recurrence step's heads are computed.
///
/// Measured on the qualified host, DeltaNet recurrence over sixteen decode
/// passes at 64 context: 0.126 s on the calling thread, 0.161 s through the
/// pool. Over a 256-row prefill pass, per token: 5.16 ms on the calling
/// thread, 2.79 ms through the pool. See `HEAD_SPREAD_MIN_ROWS` for where
/// between those two the choice turns.
///
/// Which one runs cannot change a value — see `recurrent_delta_step` — so this
/// is a scheduling choice and nothing else.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HeadSpread {
    /// Spread the heads across the global rayon pool. For a prefill pass.
    Pool,
    /// Run every head on the calling thread. For a one-row decode pass and for
    /// the narrow verification pass speculation runs behind it.
    Caller,
}

impl HeadSpread {
    /// What a pass of `rows` rows should use.
    pub(crate) fn for_rows(rows: usize) -> Self {
        if rows >= HEAD_SPREAD_MIN_ROWS {
            Self::Pool
        } else {
            Self::Caller
        }
    }
}

/// Scalar recurrent Gated Delta Rule. State layout is `[head, key, value]`.
///
/// The heads share nothing. Head `h` reads its own `key_dim` slice of the
/// query and key, its own `value_dim` slice of the value, one decay and one
/// beta, and it reads and writes only its own `key_dim * value_dim` block of
/// the state and its own `value_dim` output row. Running the heads on
/// separate cores therefore leaves every reduction in exactly the order it
/// had when they ran one after another, which is what lets a wide
/// verification pass and a one-row decode still agree bit for bit.
///
/// This is the one part of a DeltaNet layer that is serial over tokens, so
/// during a wide pass it was also the one part running on a single core while
/// the other five sat idle.
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
    spread: HeadSpread,
    state: &mut [f32],
    out: &mut [f32],
) {
    debug_assert_eq!(state.len(), heads * key_dim * value_dim);
    debug_assert_eq!(out.len(), heads * value_dim);
    let q_scale = (key_dim as f32).sqrt().recip();
    let head = |h: usize, state: &mut [f32], output: &mut [f32]| {
        let q_base = h * key_dim;
        let v_base = h * value_dim;
        delta_head_step(
            &q[q_base..q_base + key_dim],
            &k[q_base..q_base + key_dim],
            &v[v_base..v_base + value_dim],
            g[h],
            beta[h],
            key_dim,
            value_dim,
            q_scale,
            state,
            output,
        );
    };
    match spread {
        HeadSpread::Pool => state
            .par_chunks_mut(key_dim * value_dim)
            .zip(out.par_chunks_mut(value_dim))
            .enumerate()
            .for_each(|(h, (state, output))| head(h, state, output)),
        HeadSpread::Caller => state
            .chunks_mut(key_dim * value_dim)
            .zip(out.chunks_mut(value_dim))
            .enumerate()
            .for_each(|(h, (state, output))| head(h, state, output)),
    }
}

/// One head of the gated delta rule, against that head's own state block.
#[allow(clippy::too_many_arguments)]
fn delta_head_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
    key_dim: usize,
    value_dim: usize,
    q_scale: f32,
    state: &mut [f32],
    output: &mut [f32],
) {
    let decay = g.exp();
    for cell in state.iter_mut() {
        *cell *= decay;
    }

    // State is row-major `[key, value]`. Traverse complete value rows so
    // the hot recurrence is contiguous and can be vectorized on CPU.
    output.fill(0.);
    for i in 0..key_dim {
        let row = &state[i * value_dim..(i + 1) * value_dim];
        let key = k[i];
        for (prediction, &cell) in output.iter_mut().zip(row) {
            *prediction += cell * key;
        }
    }
    for (delta, &target) in output.iter_mut().zip(v) {
        *delta = (target - *delta) * beta;
    }
    for i in 0..key_dim {
        let row = &mut state[i * value_dim..(i + 1) * value_dim];
        let key = k[i];
        for (cell, &delta) in row.iter_mut().zip(output.iter()) {
            *cell += key * delta;
        }
    }
    output.fill(0.);
    for i in 0..key_dim {
        let row = &state[i * value_dim..(i + 1) * value_dim];
        for (value, &cell) in output.iter_mut().zip(row) {
            // Preserve the scalar reference's multiplication order. The
            // alternate `cell * (query * scale)` changes greedy output on
            // a near-tied logit despite being algebraically equivalent.
            *value += cell * q[i] * q_scale;
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
            HeadSpread::for_rows(seq),
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

pub fn reference_deltanet(
    checkpoint: &Checkpoint,
    config: &Qwen3NextConfig,
    layer: usize,
    xs: &Tensor,
) -> Result<Tensor> {
    let mut state = DeltaState::new(config);
    let mut timings = ForwardTimings::default();
    forward(checkpoint, config, layer, xs, &mut state, &mut timings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn strided_recurrence_reference(
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
            HeadSpread::Pool,
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
            HeadSpread::Pool,
            &mut state,
            &mut out,
        );
        assert!((out[0] - 2. / 2f32.sqrt()).abs() < 1e-6);
        assert!((out[1] - 3. / 2f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn contiguous_recurrence_matches_strided_reference() {
        let heads = 2;
        let key_dim = 4;
        let value_dim = 3;
        let q: Vec<_> = (0..heads * key_dim)
            .map(|index| (index as f32 - 3.) / 7.)
            .collect();
        let k: Vec<_> = (0..heads * key_dim)
            .map(|index| (5. - index as f32) / 9.)
            .collect();
        let v: Vec<_> = (0..heads * value_dim)
            .map(|index| (index as f32 + 1.) / 5.)
            .collect();
        let mut expected_state: Vec<_> = (0..heads * key_dim * value_dim)
            .map(|index| (index as f32 - 7.) / 31.)
            .collect();
        let mut actual_state = expected_state.clone();
        let mut expected = vec![0.; heads * value_dim];
        let mut actual = expected.clone();
        strided_recurrence_reference(
            &q,
            &k,
            &v,
            &[-0.2, -0.4],
            &[0.3, 0.7],
            heads,
            key_dim,
            value_dim,
            &mut expected_state,
            &mut expected,
        );
        recurrent_delta_step(
            &q,
            &k,
            &v,
            &[-0.2, -0.4],
            &[0.3, 0.7],
            heads,
            key_dim,
            value_dim,
            HeadSpread::Pool,
            &mut actual_state,
            &mut actual,
        );
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in actual_state.iter().zip(&expected_state) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn spreading_the_heads_over_cores_does_not_move_a_bit() {
        // The heads are the unit of parallelism, so the property that has to
        // hold is stronger than "close": running them concurrently must
        // reproduce the serial result exactly, or a wide verification pass
        // could commit a token a one-row decode would not have.
        let heads = 7;
        let key_dim = 16;
        let value_dim = 12;
        let series = |count: usize, seed: f32| -> Vec<f32> {
            (0..count)
                .map(|index| ((index as f32 * seed).sin() * 1.7 + 0.3) / 2.3)
                .collect()
        };
        let q = series(heads * key_dim, 0.31);
        let k = series(heads * key_dim, 0.77);
        let v = series(heads * value_dim, 1.13);
        let g = series(heads, 0.41);
        let beta = series(heads, 0.93);
        let initial = series(heads * key_dim * value_dim, 0.017);

        let mut states = [initial.clone(), initial];
        let mut outputs = [vec![0.; heads * value_dim], vec![0.; heads * value_dim]];
        for (spread, (state, out)) in [HeadSpread::Caller, HeadSpread::Pool]
            .into_iter()
            .zip(states.iter_mut().zip(outputs.iter_mut()))
        {
            recurrent_delta_step(
                &q, &k, &v, &g, &beta, heads, key_dim, value_dim, spread, state, out,
            );
        }

        assert_eq!(outputs[1], outputs[0]);
        assert_eq!(states[1], states[0]);
    }

    #[test]
    fn only_a_wider_pass_pays_for_the_pool() {
        // A one-row decode and the two- or three-row verification pass behind
        // a speculative decode both stay on the calling thread.
        assert_eq!(HeadSpread::for_rows(1), HeadSpread::Caller);
        assert_eq!(HeadSpread::for_rows(3), HeadSpread::Caller);
        assert_eq!(HeadSpread::for_rows(4), HeadSpread::Pool);
        assert_eq!(HeadSpread::for_rows(512), HeadSpread::Pool);
    }
}
