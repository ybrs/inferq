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

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedDeltaCheckpoint {
    conv: Vec<f32>,
    recurrent: Vec<f32>,
}

impl QuantizedDeltaCheckpoint {
    pub fn from_parts(conv: Vec<f32>, recurrent: Vec<f32>) -> Self {
        Self { conv, recurrent }
    }

    pub fn conv(&self) -> &[f32] {
        &self.conv
    }

    pub fn recurrent(&self) -> &[f32] {
        &self.recurrent
    }

    pub fn bytes(&self) -> usize {
        (self.conv.len() + self.recurrent.len()) * std::mem::size_of::<f32>()
    }
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

/// Preallocated per-row snapshots of one linear layer's recurrent state.
///
/// A multi-row verification pass advances the DeltaNet recurrence one row at a
/// time. Slot `r` holds `{conv, recurrent}` as they were *before* row `r` was
/// consumed, so slot 0 is the pre-pass checkpoint and slot `r` restores the
/// state a sequential decode would hold after committing rows `0..r`. Rolling
/// back a partially accepted draft is then a copy rather than a replayed
/// forward pass.
///
/// Buffers are sized once and reused across passes; `store` never allocates.
#[derive(Debug, Clone, Default)]
pub struct QuantizedDeltaSnapshots {
    conv: Vec<f32>,
    recurrent: Vec<f32>,
    conv_len: usize,
    recurrent_len: usize,
    rows: usize,
    stored_rows: usize,
}

impl QuantizedDeltaSnapshots {
    /// Size the arena for `rows` snapshot slots, reusing existing capacity.
    pub fn reserve(&mut self, rows: usize, conv_len: usize, recurrent_len: usize) {
        if self.rows != rows || self.conv_len != conv_len || self.recurrent_len != recurrent_len {
            self.conv.resize(rows * conv_len, 0.);
            self.recurrent.resize(rows * recurrent_len, 0.);
            self.rows = rows;
            self.conv_len = conv_len;
            self.recurrent_len = recurrent_len;
        }
        self.stored_rows = 0;
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn stored_rows(&self) -> usize {
        self.stored_rows
    }

    /// Bytes of live state copied per stored row.
    pub fn bytes_per_row(&self) -> usize {
        (self.conv_len + self.recurrent_len) * std::mem::size_of::<f32>()
    }

    fn store(&mut self, row: usize, conv: &[f32], recurrent: &[f32], nontemporal: bool) {
        debug_assert_eq!(conv.len(), self.conv_len);
        debug_assert_eq!(recurrent.len(), self.recurrent_len);
        debug_assert!(row < self.rows);
        copy_state(
            &mut self.conv[row * self.conv_len..(row + 1) * self.conv_len],
            conv,
            nontemporal,
        );
        copy_state(
            &mut self.recurrent[row * self.recurrent_len..(row + 1) * self.recurrent_len],
            recurrent,
            nontemporal,
        );
        self.stored_rows = self.stored_rows.max(row + 1);
    }

    /// Restore `state` to snapshot slot `row`.
    ///
    /// A layer that did not run during the interrupted pass has no stored rows
    /// and is already at the pre-pass state, so restoring it is a no-op.
    pub fn restore_into(&self, row: usize, state: &mut QuantizedDeltaState) -> Result<()> {
        if self.stored_rows == 0 {
            return Ok(());
        }
        ensure!(
            row < self.stored_rows,
            "DeltaNet snapshot row {row} was not stored ({} rows available)",
            self.stored_rows
        );
        ensure!(
            state.conv.len() == self.conv_len && state.recurrent.len() == self.recurrent_len,
            "DeltaNet snapshot dimensions do not match the live state"
        );
        state
            .conv
            .copy_from_slice(&self.conv[row * self.conv_len..(row + 1) * self.conv_len]);
        state.recurrent.copy_from_slice(
            &self.recurrent[row * self.recurrent_len..(row + 1) * self.recurrent_len],
        );
        Ok(())
    }
}

/// Copy live recurrent state into a snapshot slot.
///
/// The recurrent state is 2 MiB per layer on Qwen3.6-35B-A3B, so an ordinary
/// `copy_from_slice` both costs full read-for-ownership traffic on the
/// destination and evicts the live state from L3 between rows. Streaming
/// stores avoid both; they measured 2.78 ms/row against 4.84 ms/row for
/// `copy_from_slice` on the qualified host. The safe copy stays available as
/// a fallback and as the reference the equivalence test compares against.
fn copy_state(dst: &mut [f32], src: &[f32], nontemporal: bool) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if nontemporal && dst.len() >= NONTEMPORAL_MIN_ELEMENTS && has_avx() {
        // SAFETY: `has_avx` confirms AVX is available on this CPU, and `dst`
        // and `src` are same-length slices of `f32` the caller owns
        // exclusively. `copy_state_nontemporal` only reads `src` and writes
        // `dst` inside those bounds.
        unsafe { copy_state_nontemporal(dst, src) };
        return;
    }
    let _ = nontemporal;
    dst.copy_from_slice(src);
}

#[cfg(target_arch = "x86_64")]
const NONTEMPORAL_MIN_ELEMENTS: usize = 4096;

#[cfg(target_arch = "x86_64")]
fn has_avx() -> bool {
    use std::sync::OnceLock;
    static AVX: OnceLock<bool> = OnceLock::new();
    *AVX.get_or_init(|| std::arch::is_x86_feature_detected!("avx"))
}

/// # Safety
///
/// The caller must guarantee AVX is available and that `dst` and `src` have
/// equal lengths.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn copy_state_nontemporal(dst: &mut [f32], src: &[f32]) {
    use std::arch::x86_64::{_mm_sfence, _mm256_loadu_ps, _mm256_stream_ps};

    const LANES: usize = 8;
    let len = dst.len();
    let dst_ptr = dst.as_mut_ptr();
    // Streaming stores require a 32-byte aligned destination; copy the
    // unaligned head and tail with ordinary stores.
    let head = (dst_ptr.align_offset(32)).min(len);
    dst[..head].copy_from_slice(&src[..head]);
    let body = (len - head) / LANES * LANES;
    // SAFETY: `head + body <= len`, so every load and store below stays inside
    // both slices, and `dst_ptr.add(head)` is 32-byte aligned by construction.
    unsafe {
        let src_ptr = src.as_ptr();
        let mut offset = head;
        while offset < head + body {
            _mm256_stream_ps(dst_ptr.add(offset), _mm256_loadu_ps(src_ptr.add(offset)));
            offset += LANES;
        }
        _mm_sfence();
    }
    dst[head + body..].copy_from_slice(&src[head + body..]);
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

    pub fn checkpoint(&self) -> QuantizedDeltaCheckpoint {
        QuantizedDeltaCheckpoint {
            conv: self.conv.clone(),
            recurrent: self.recurrent.clone(),
        }
    }

    pub fn conv_len(&self) -> usize {
        self.conv.len()
    }

    pub fn recurrent_len(&self) -> usize {
        self.recurrent.len()
    }

    /// The whole linear-layer state, which for a recurrent layer is the same
    /// thing a rollback checkpoint holds: there is no append-only cache to
    /// truncate, so a checkpoint is already a complete image.
    pub fn image(&self) -> QuantizedDeltaCheckpoint {
        self.checkpoint()
    }

    pub fn restore(&mut self, checkpoint: &QuantizedDeltaCheckpoint) -> Result<()> {
        ensure!(
            self.conv.len() == checkpoint.conv.len()
                && self.recurrent.len() == checkpoint.recurrent.len(),
            "DeltaNet checkpoint dimensions do not match the live state"
        );
        self.conv.copy_from_slice(&checkpoint.conv);
        self.recurrent.copy_from_slice(&checkpoint.recurrent);
        Ok(())
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
    /// Time spent copying recurrent state into per-row rollback snapshots.
    pub snapshot: Duration,
}

impl QuantizedDeltaTimings {
    pub fn accumulate(&mut self, other: &Self) {
        self.wall += other.wall;
        self.projections += other.projections;
        self.convolution += other.convolution;
        self.recurrence += other.recurrence;
        self.gated_norm += other.gated_norm;
        self.output_projection += other.output_projection;
        self.snapshot += other.snapshot;
    }
}

pub struct QuantizedDeltaLayer {
    layer: usize,
    hidden_size: usize,
    key_heads: usize,
    value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    value_head_layout: ValueHeadLayout,
    conv_kernel: usize,
    eps: f64,
    qkvz: QuantizedDeltaProjection,
    beta_alpha: QuantizedMatrix,
    beta_alpha_layout: BetaAlphaLayout,
    output: QuantizedMatrix,
    conv_weight: Vec<f32>,
    state_scale: Vec<f32>,
    dt_bias: Vec<f32>,
    norm_weight: Tensor,
}

enum QuantizedDeltaProjection {
    Fused(QuantizedMatrix),
    Separate {
        qkv: QuantizedMatrix,
        z: QuantizedMatrix,
    },
}

#[derive(Clone, Copy)]
enum BetaAlphaLayout {
    InterleavedByKeyHead,
    GroupedByProjection,
}

#[derive(Clone, Copy)]
enum ValueHeadLayout {
    GroupedByKeyHead,
    TiledByRepeat,
}

impl ValueHeadLayout {
    fn index(
        self,
        key_head: usize,
        repeat: usize,
        key_heads: usize,
        repeats_per_key: usize,
    ) -> usize {
        match self {
            Self::GroupedByKeyHead => key_head * repeats_per_key + repeat,
            Self::TiledByRepeat => repeat * key_heads + key_head,
        }
    }
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
        // Qwen3-Next exports beta and alpha as one projection. Qwen3.5/3.6
        // GGUF keeps the two projections separate; concatenate them in the
        // original beta-then-alpha order so the hot path stays shared.
        let fused_ba_name = format!("{prefix}.ssm_ba.weight");
        let (beta_alpha, beta_alpha_layout) = if checkpoint.tensor_info(&fused_ba_name).is_some() {
            (
                checkpoint.load_matrix(&fused_ba_name)?,
                BetaAlphaLayout::InterleavedByKeyHead,
            )
        } else {
            let beta = checkpoint.load_matrix(&format!("{prefix}.ssm_beta.weight"))?;
            let alpha = checkpoint.load_matrix(&format!("{prefix}.ssm_alpha.weight"))?;
            (
                beta.concatenate_rows(&alpha)?,
                BetaAlphaLayout::GroupedByProjection,
            )
        };
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
        let qkvz = if qkv.dtype() == z.dtype() {
            QuantizedDeltaProjection::Fused(qkv.concatenate_rows(&z)?)
        } else {
            QuantizedDeltaProjection::Separate { qkv, z }
        };
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
            value_head_layout: if config.model_type == "qwen3_5_moe_text" {
                ValueHeadLayout::TiledByRepeat
            } else {
                ValueHeadLayout::GroupedByKeyHead
            },
            conv_kernel: config.linear_conv_kernel_dim,
            eps: config.rms_norm_eps,
            qkvz,
            beta_alpha,
            beta_alpha_layout,
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
        self.forward_with_snapshots(xs, state, None, false)
    }

    /// Run the layer, optionally recording a rollback snapshot at every row
    /// boundary the recurrence crosses.
    ///
    /// Slot `r` is written *before* row `r` is consumed, so the final row is
    /// never snapshotted: a fully accepted draft needs no rollback.
    pub fn forward_with_snapshots(
        &self,
        xs: &Tensor,
        state: &mut QuantizedDeltaState,
        mut snapshots: Option<&mut QuantizedDeltaSnapshots>,
        nontemporal: bool,
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
        let ratio = self.value_heads / self.key_heads;
        let mut timings = QuantizedDeltaTimings::default();
        let projection_started = Instant::now();
        let (projected, separate_gates) = match &self.qkvz {
            QuantizedDeltaProjection::Fused(qkvz) => {
                (qkvz.forward(&flat)?.flatten_all()?.to_vec1::<f32>()?, None)
            }
            QuantizedDeltaProjection::Separate { qkv, z } => (
                qkv.forward(&flat)?.flatten_all()?.to_vec1::<f32>()?,
                Some(z.forward(&flat)?.flatten_all()?.to_vec1::<f32>()?),
            ),
        };
        let projected_width = if separate_gates.is_some() {
            conv_dim
        } else {
            conv_dim + value_dim
        };
        state.scratch.prepare_output(seq * value_dim);
        for token in 0..seq {
            let gate = separate_gates.as_ref().map_or_else(
                || &projected[token * projected_width + conv_dim..(token + 1) * projected_width],
                |gates| &gates[token * value_dim..(token + 1) * value_dim],
            );
            state.scratch.gates[token * value_dim..(token + 1) * value_dim].copy_from_slice(gate);
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
        if let Some(sink) = snapshots.as_deref_mut() {
            ensure!(
                sink.rows() >= seq,
                "DeltaNet snapshot arena holds {} rows, need {seq}",
                sink.rows()
            );
            let snapshot_started = Instant::now();
            sink.store(0, conv, recurrent, nontemporal);
            timings.snapshot += snapshot_started.elapsed();
        }
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
                    let value_head =
                        self.value_head_layout
                            .index(key_head, repeat, self.key_heads, ratio);
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
            let token_parameters =
                &projected_ba[token * self.value_heads * 2..(token + 1) * self.value_heads * 2];
            for key_head in 0..self.key_heads {
                for repeat in 0..ratio {
                    let value_head =
                        self.value_head_layout
                            .index(key_head, repeat, self.key_heads, ratio);
                    let (beta, alpha) = match self.beta_alpha_layout {
                        BetaAlphaLayout::InterleavedByKeyHead => {
                            let base = key_head * 2 * ratio;
                            (
                                token_parameters[base + repeat],
                                token_parameters[base + ratio + repeat],
                            )
                        }
                        BetaAlphaLayout::GroupedByProjection => (
                            token_parameters[value_head],
                            token_parameters[self.value_heads + value_head],
                        ),
                    };
                    scratch.beta[value_head] = sigmoid(beta);
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
            // The state now reflects rows `0..=token`; record it as the entry
            // state of the next row so a rollback that commits `token + 1`
            // rows becomes a copy. The last row needs no slot.
            if let Some(sink) = snapshots.as_deref_mut()
                && token + 1 < seq
            {
                let snapshot_started = Instant::now();
                sink.store(token + 1, conv, recurrent, nontemporal);
                timings.snapshot += snapshot_started.elapsed();
            }
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
impl QuantizedDeltaState {
    fn for_test(conv: Vec<f32>, recurrent: Vec<f32>) -> Self {
        Self {
            conv,
            recurrent,
            scratch: Box::new(QuantizedDeltaScratch::new(1, 1, 2, 2)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_snapshots(
        rows: usize,
        conv_len: usize,
        recurrent_len: usize,
    ) -> QuantizedDeltaSnapshots {
        let mut snapshots = QuantizedDeltaSnapshots::default();
        snapshots.reserve(rows, conv_len, recurrent_len);
        snapshots
    }

    #[test]
    fn snapshots_restore_the_state_recorded_for_each_row() {
        let mut snapshots = state_snapshots(3, 2, 3);
        for row in 0..3 {
            let base = row as f32;
            snapshots.store(
                row,
                &[base, base + 0.5],
                &[base, base + 1., base + 2.],
                false,
            );
        }
        for row in 0..3 {
            let mut state = QuantizedDeltaState::for_test(vec![0.; 2], vec![0.; 3]);
            snapshots.restore_into(row, &mut state).unwrap();
            let base = row as f32;
            assert_eq!(state.conv, [base, base + 0.5]);
            assert_eq!(state.recurrent, [base, base + 1., base + 2.]);
        }
    }

    #[test]
    fn restoring_a_row_the_pass_never_reached_is_rejected() {
        let mut snapshots = state_snapshots(4, 1, 1);
        snapshots.store(0, &[1.], &[2.], false);
        let mut state = QuantizedDeltaState::for_test(vec![0.], vec![0.]);
        assert!(snapshots.restore_into(1, &mut state).is_err());
        assert!(snapshots.restore_into(0, &mut state).is_ok());
    }

    #[test]
    fn a_layer_that_never_ran_is_left_untouched() {
        // An interrupted pass leaves later layers with no stored rows; they
        // still hold the pre-pass state, so restoring them is a no-op.
        let snapshots = state_snapshots(4, 1, 1);
        let mut state = QuantizedDeltaState::for_test(vec![7.], vec![9.]);
        snapshots.restore_into(0, &mut state).unwrap();
        assert_eq!(state.conv, [7.]);
        assert_eq!(state.recurrent, [9.]);
    }

    #[test]
    fn reserve_reuses_buffers_and_clears_the_stored_row_count() {
        let mut snapshots = state_snapshots(4, 8, 16);
        snapshots.store(0, &[1.; 8], &[2.; 16], false);
        assert_eq!(snapshots.stored_rows(), 1);
        let capacity = snapshots.recurrent.capacity();
        snapshots.reserve(4, 8, 16);
        assert_eq!(snapshots.stored_rows(), 0);
        assert_eq!(snapshots.recurrent.capacity(), capacity);
        assert_eq!(snapshots.bytes_per_row(), (8 + 16) * 4);
    }

    #[test]
    fn streaming_and_ordinary_copies_produce_identical_state() {
        // Large enough to take the streaming path, and copied at every offset
        // that changes the aligned head and tail it has to handle.
        let source: Vec<f32> = (0..9_001).map(|value| value as f32 * 0.25).collect();
        for offset in 0..9 {
            let source = &source[offset..];
            let mut streamed = vec![0.; source.len()];
            let mut ordinary = vec![0.; source.len()];
            copy_state(&mut streamed, source, true);
            copy_state(&mut ordinary, source, false);
            assert_eq!(streamed, ordinary, "offset {offset}");
            assert_eq!(streamed, source, "offset {offset}");
        }
    }

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

    #[test]
    fn value_head_layout_matches_grouped_and_tiled_gguf_orders() {
        let grouped = ValueHeadLayout::GroupedByKeyHead;
        let tiled = ValueHeadLayout::TiledByRepeat;
        let grouped_indices: Vec<_> = (0..2)
            .flat_map(|key| (0..2).map(move |repeat| grouped.index(key, repeat, 2, 2)))
            .collect();
        let tiled_indices: Vec<_> = (0..2)
            .flat_map(|key| (0..2).map(move |repeat| tiled.index(key, repeat, 2, 2)))
            .collect();
        assert_eq!(grouped_indices, [0, 1, 2, 3]);
        assert_eq!(tiled_indices, [0, 2, 1, 3]);
    }
}
