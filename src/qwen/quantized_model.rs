use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use serde::Serialize;

use crate::{
    ExpertCacheStats, GgufCheckpoint, LayerType, QuantizedEmbedding, QuantizedMatrix,
    Qwen3NextConfig,
    trace::{RoutingRecord, RoutingTrace},
};

use super::{
    QuantizedAttentionImage, QuantizedAttentionState, QuantizedDeltaCheckpoint,
    QuantizedDeltaSnapshots, QuantizedDeltaState, QuantizedFullLayer, QuantizedLayerTimings,
    QuantizedLinearLayer, QuantizedMtpHead, quantized_layer::gguf_rms_norm,
};

enum DecoderLayer<'a> {
    Full(QuantizedFullLayer<'a>),
    Linear(QuantizedLinearLayer<'a>),
}

#[derive(Debug, Clone)]
enum DecoderState {
    Full(QuantizedAttentionState),
    Linear(QuantizedDeltaState),
}

#[derive(Debug, Clone)]
enum DecoderCheckpoint {
    Full { position: usize },
    Linear(QuantizedDeltaCheckpoint),
}

#[derive(Debug, Clone)]
pub struct QuantizedModelCheckpoint {
    layers: Vec<DecoderCheckpoint>,
    position: usize,
}

/// One layer's complete state, KV rows included.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerStateImage {
    Full(QuantizedAttentionImage),
    Linear(QuantizedDeltaCheckpoint),
}

impl LayerStateImage {
    pub fn bytes(&self) -> usize {
        match self {
            Self::Full(image) => image.bytes(),
            Self::Linear(image) => image.bytes(),
        }
    }
}

/// A self-contained copy of a sequence's model state.
///
/// [`QuantizedModelCheckpoint`] exists to roll a live state back and therefore
/// stores only what truncation cannot recover. An image is the whole thing, so
/// it can be written to disk and restored into a state that never saw the
/// tokens that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedStateImage {
    pub layers: Vec<LayerStateImage>,
    pub position: usize,
}

impl QuantizedStateImage {
    pub fn bytes(&self) -> usize {
        self.layers.iter().map(LayerStateImage::bytes).sum()
    }

    /// Check the image against itself: every full-attention layer must hold
    /// exactly the sequence position the image claims.
    pub fn validate(&self) -> Result<()> {
        for (index, layer) in self.layers.iter().enumerate() {
            if let LayerStateImage::Full(image) = layer {
                image.validate()?;
                ensure!(
                    image.positions == self.position,
                    "layer {index} holds {} positions but the image is at position {}",
                    image.positions,
                    self.position
                );
            }
        }
        Ok(())
    }
}

/// Reusable per-row rollback snapshots for one multi-row verification pass.
///
/// Full-attention layers need no storage: their KV cache is append-only and
/// rolls back with `truncate`. Only the linear layers' recurrent state is
/// copied, one slot per row boundary the pass crosses.
#[derive(Debug, Clone, Default)]
pub struct QuantizedStateSnapshots {
    linear_layers: Vec<QuantizedDeltaSnapshots>,
    /// `state.position` immediately before the pass began.
    position: usize,
    rows: usize,
    nontemporal: bool,
}

impl QuantizedStateSnapshots {
    /// Prepare the arena for a pass of `rows` rows starting at `state`.
    ///
    /// Buffers are allocated on the first pass that needs them and reused
    /// afterwards; a later pass with fewer rows keeps the larger allocation.
    pub fn begin_pass(&mut self, state: &QuantizedModelState, rows: usize) {
        let linear = state
            .layers
            .iter()
            .filter(|layer| matches!(layer, DecoderState::Linear(_)))
            .count();
        if self.linear_layers.len() != linear {
            self.linear_layers
                .resize_with(linear, QuantizedDeltaSnapshots::default);
        }
        let mut index = 0;
        for layer in &state.layers {
            if let DecoderState::Linear(delta) = layer {
                self.linear_layers[index].reserve(rows, delta.conv_len(), delta.recurrent_len());
                index += 1;
            }
        }
        self.position = state.position;
        self.rows = rows;
    }

    /// Use streaming stores for the snapshot copy where the CPU supports them.
    pub fn set_nontemporal(&mut self, nontemporal: bool) {
        self.nontemporal = nontemporal;
    }

    pub fn nontemporal(&self) -> bool {
        self.nontemporal
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Bytes copied per stored row across every linear layer.
    pub fn bytes_per_row(&self) -> usize {
        self.linear_layers
            .iter()
            .map(QuantizedDeltaSnapshots::bytes_per_row)
            .sum()
    }

    fn layer_mut(&mut self, linear_index: usize) -> Option<&mut QuantizedDeltaSnapshots> {
        self.linear_layers.get_mut(linear_index)
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedModelState {
    layers: Vec<DecoderState>,
    pub position: usize,
}

impl QuantizedModelState {
    /// Roll the state back to the boundary after `committed_rows` rows of the
    /// pass that produced `snapshots`.
    ///
    /// This costs one state copy per linear layer plus a KV-cache truncation
    /// per full-attention layer; it never replays a forward pass. Passing
    /// `committed_rows == snapshots.rows()` is rejected because a fully
    /// accepted pass has no snapshot to roll back to and needs none.
    pub fn rollback(
        &mut self,
        snapshots: &QuantizedStateSnapshots,
        committed_rows: usize,
    ) -> Result<()> {
        ensure!(
            committed_rows < snapshots.rows || snapshots.rows == 0,
            "cannot roll back to {committed_rows} of {} committed rows",
            snapshots.rows
        );
        let position = snapshots.position + committed_rows;
        let mut linear = 0;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            match layer {
                DecoderState::Full(state) => state.truncate(position)?,
                DecoderState::Linear(state) => {
                    let saved = snapshots
                        .linear_layers
                        .get(linear)
                        .with_context(|| format!("snapshot arena is missing layer {index}"))?;
                    saved.restore_into(committed_rows, state)?;
                    linear += 1;
                }
            }
        }
        self.position = position;
        Ok(())
    }

    /// Copy the whole state, KV rows included, so it can outlive the session.
    pub fn image(&self) -> QuantizedStateImage {
        QuantizedStateImage {
            layers: self
                .layers
                .iter()
                .map(|layer| match layer {
                    DecoderState::Full(state) => LayerStateImage::Full(state.image()),
                    DecoderState::Linear(state) => LayerStateImage::Linear(state.image()),
                })
                .collect(),
            position: self.position,
        }
    }

    /// Replace this state with an image, which may have been produced by an
    /// earlier process. The layer sequence and every tensor length must match
    /// the loaded model exactly; nothing here is coerced.
    pub fn restore_image(&mut self, image: &QuantizedStateImage) -> Result<()> {
        ensure!(
            self.layers.len() == image.layers.len(),
            "state image has {} layers but the model has {}",
            image.layers.len(),
            self.layers.len()
        );
        image.validate()?;
        for (index, (state, saved)) in self.layers.iter_mut().zip(&image.layers).enumerate() {
            match (state, saved) {
                (DecoderState::Full(state), LayerStateImage::Full(image)) => {
                    state.restore_image(image)?
                }
                (DecoderState::Linear(state), LayerStateImage::Linear(image)) => {
                    state.restore(image)?
                }
                _ => anyhow::bail!("state image layer {index} is the wrong layer type"),
            }
        }
        self.position = image.position;
        Ok(())
    }

    pub fn checkpoint(&self) -> QuantizedModelCheckpoint {
        QuantizedModelCheckpoint {
            layers: self
                .layers
                .iter()
                .map(|layer| match layer {
                    DecoderState::Full(state) => DecoderCheckpoint::Full {
                        position: state.positions,
                    },
                    DecoderState::Linear(state) => DecoderCheckpoint::Linear(state.checkpoint()),
                })
                .collect(),
            position: self.position,
        }
    }

    pub fn restore(&mut self, checkpoint: &QuantizedModelCheckpoint) -> Result<()> {
        ensure!(
            self.layers.len() == checkpoint.layers.len(),
            "model checkpoint belongs to a different model"
        );
        for (index, (state, saved)) in self.layers.iter_mut().zip(&checkpoint.layers).enumerate() {
            match (state, saved) {
                (DecoderState::Full(state), DecoderCheckpoint::Full { position }) => {
                    state.truncate(*position)?;
                }
                (DecoderState::Linear(state), DecoderCheckpoint::Linear(saved)) => {
                    state.restore(saved)?;
                }
                _ => anyhow::bail!("checkpoint state type does not match layer {index}"),
            }
        }
        self.position = checkpoint.position;
        Ok(())
    }
}

/// Which rows of a pass get logits.
///
/// The LM head is the most expensive per-row operation in the model -- a
/// 248,320-wide vocabulary against a 397.9 MiB Q6_K matrix -- and a prefill
/// pass reads only the last row of it, because that is the one it samples
/// from. Rows are independent, so computing fewer changes no value that is
/// read; verification is the case that genuinely needs all of them, since it
/// checks a drafted token against every position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogitRows {
    #[default]
    All,
    Last,
}

pub struct QuantizedForwardOutput {
    pub logits: Tensor,
    pub normalized_hidden: Tensor,
    pub timings: QuantizedForwardTimings,
}

#[derive(Debug, Clone, Default)]
pub struct QuantizedForwardTimings {
    pub wall: Duration,
    pub embedding: Duration,
    pub layers: Vec<Duration>,
    pub layer_details: Vec<QuantizedLayerTimings>,
    pub final_norm: Duration,
    pub lm_head: Duration,
}

impl QuantizedForwardTimings {
    pub fn accumulate(&mut self, other: &Self) {
        self.wall += other.wall;
        self.embedding += other.embedding;
        if self.layers.len() < other.layers.len() {
            self.layers.resize(other.layers.len(), Duration::ZERO);
        }
        for (total, elapsed) in self.layers.iter_mut().zip(&other.layers) {
            *total += *elapsed;
        }
        if self.layer_details.len() < other.layer_details.len() {
            self.layer_details
                .resize(other.layer_details.len(), QuantizedLayerTimings::default());
        }
        for (total, elapsed) in self.layer_details.iter_mut().zip(&other.layer_details) {
            total.accumulate(elapsed);
        }
        self.final_norm += other.final_norm;
        self.lm_head += other.lm_head;
    }

    pub fn report(&self) -> QuantizedForwardTimingReport {
        let layer_wall: Duration = self.layers.iter().sum();
        let top_accounted = self.embedding + layer_wall + self.final_norm + self.lm_head;
        let stage_totals = self.layer_details.iter().fold(
            QuantizedStageTimingReport::default(),
            |mut total, layer| {
                total.normalization_seconds += layer.normalization.as_secs_f64();
                total.attention_seconds += layer.attention.wall.as_secs_f64();
                total.deltanet_seconds += layer.delta.wall.as_secs_f64();
                total.moe_seconds += layer.moe.wall.as_secs_f64();
                total
            },
        );
        let operations = self.layer_details.iter().fold(
            QuantizedOperationTimingReport::default(),
            |mut total, layer| {
                total.attention_projections_seconds += layer.attention.projections.as_secs_f64();
                total.attention_norm_rope_seconds += layer.attention.norm_rope.as_secs_f64();
                total.attention_kernel_seconds += layer.attention.attention.as_secs_f64();
                total.attention_output_projection_seconds +=
                    layer.attention.output_projection.as_secs_f64();
                total.deltanet_projections_seconds += layer.delta.projections.as_secs_f64();
                total.deltanet_convolution_seconds += layer.delta.convolution.as_secs_f64();
                total.deltanet_recurrence_seconds += layer.delta.recurrence.as_secs_f64();
                total.deltanet_gated_norm_seconds += layer.delta.gated_norm.as_secs_f64();
                total.deltanet_output_projection_seconds +=
                    layer.delta.output_projection.as_secs_f64();
                total.deltanet_snapshot_seconds += layer.delta.snapshot.as_secs_f64();
                total.moe_router_seconds += layer.moe.router.as_secs_f64();
                total.moe_top_k_seconds += layer.moe.top_k.as_secs_f64();
                total.moe_expert_lookup_seconds += layer.moe.expert_load.as_secs_f64();
                total.moe_expert_compute_seconds += layer.moe.expert_compute.as_secs_f64();
                total.moe_expert_gate_up_seconds += layer.moe.expert_gate_up.as_secs_f64();
                total.moe_expert_activation_seconds += layer.moe.expert_activation.as_secs_f64();
                total.moe_expert_down_seconds += layer.moe.expert_down.as_secs_f64();
                total.moe_expert_accumulation_seconds +=
                    layer.moe.expert_accumulation.as_secs_f64();
                total.moe_shared_expert_seconds += layer.moe.shared_expert.as_secs_f64();
                total
            },
        );
        let wall = self.wall.as_secs_f64();
        QuantizedForwardTimingReport {
            wall_seconds: wall,
            accounted_seconds: top_accounted.as_secs_f64(),
            unattributed_seconds: self.wall.saturating_sub(top_accounted).as_secs_f64(),
            accounted_fraction: if wall == 0. {
                0.
            } else {
                top_accounted.as_secs_f64() / wall
            },
            stages: QuantizedStageTimingReport {
                embedding_seconds: self.embedding.as_secs_f64(),
                final_norm_seconds: self.final_norm.as_secs_f64(),
                lm_head_seconds: self.lm_head.as_secs_f64(),
                ..stage_totals
            },
            nested_operations: operations,
            layers: self
                .layer_details
                .iter()
                .map(QuantizedLayerTimingReport::from)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantizedForwardTimingReport {
    pub wall_seconds: f64,
    pub accounted_seconds: f64,
    pub unattributed_seconds: f64,
    pub accounted_fraction: f64,
    pub stages: QuantizedStageTimingReport,
    pub nested_operations: QuantizedOperationTimingReport,
    pub layers: Vec<QuantizedLayerTimingReport>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct QuantizedStageTimingReport {
    pub embedding_seconds: f64,
    pub normalization_seconds: f64,
    pub attention_seconds: f64,
    pub deltanet_seconds: f64,
    pub moe_seconds: f64,
    pub final_norm_seconds: f64,
    pub lm_head_seconds: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct QuantizedOperationTimingReport {
    pub attention_projections_seconds: f64,
    pub attention_norm_rope_seconds: f64,
    pub attention_kernel_seconds: f64,
    pub attention_output_projection_seconds: f64,
    pub deltanet_projections_seconds: f64,
    pub deltanet_convolution_seconds: f64,
    pub deltanet_recurrence_seconds: f64,
    pub deltanet_gated_norm_seconds: f64,
    pub deltanet_output_projection_seconds: f64,
    pub deltanet_snapshot_seconds: f64,
    pub moe_router_seconds: f64,
    pub moe_top_k_seconds: f64,
    pub moe_expert_lookup_seconds: f64,
    pub moe_expert_compute_seconds: f64,
    pub moe_expert_gate_up_seconds: f64,
    pub moe_expert_activation_seconds: f64,
    pub moe_expert_down_seconds: f64,
    pub moe_expert_accumulation_seconds: f64,
    pub moe_shared_expert_seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantizedLayerTimingReport {
    pub layer: usize,
    pub layer_type: Option<LayerType>,
    pub wall_seconds: f64,
    pub normalization_seconds: f64,
    pub attention_seconds: f64,
    pub deltanet_seconds: f64,
    pub moe_seconds: f64,
    pub unattributed_seconds: f64,
    pub moe_token_expert_assignments: usize,
    pub moe_unique_experts_selected: usize,
    pub moe_duplicate_assignment_rate: f64,
    pub moe_average_rows_per_selected_expert: f64,
    pub moe_max_rows_per_expert: usize,
}

impl From<&QuantizedLayerTimings> for QuantizedLayerTimingReport {
    fn from(timings: &QuantizedLayerTimings) -> Self {
        let accounted =
            timings.normalization + timings.attention.wall + timings.delta.wall + timings.moe.wall;
        Self {
            layer: timings.layer,
            layer_type: timings.layer_type,
            wall_seconds: timings.wall.as_secs_f64(),
            normalization_seconds: timings.normalization.as_secs_f64(),
            attention_seconds: timings.attention.wall.as_secs_f64(),
            deltanet_seconds: timings.delta.wall.as_secs_f64(),
            moe_seconds: timings.moe.wall.as_secs_f64(),
            unattributed_seconds: timings.wall.saturating_sub(accounted).as_secs_f64(),
            moe_token_expert_assignments: timings.moe.routing.token_expert_assignments,
            moe_unique_experts_selected: timings.moe.routing.unique_experts_selected,
            moe_duplicate_assignment_rate: timings.moe.routing.duplicate_assignment_rate(),
            moe_average_rows_per_selected_expert: timings
                .moe
                .routing
                .average_rows_per_selected_expert(),
            moe_max_rows_per_expert: timings.moe.routing.max_rows_per_expert,
        }
    }
}

pub struct QuantizedModel<'a> {
    checkpoint: &'a GgufCheckpoint,
    config: Qwen3NextConfig,
    embedding: QuantizedEmbedding,
    layers: Vec<DecoderLayer<'a>>,
    final_norm: Tensor,
    lm_head: QuantizedMatrix,
    mtp: Option<QuantizedMtpHead<'a>>,
}

impl std::fmt::Debug for QuantizedModel<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedModel")
            .field("layers", &self.layers.len())
            .field("hidden_size", &self.config.hidden_size)
            .field("vocab_size", &self.config.vocab_size)
            .finish_non_exhaustive()
    }
}

impl<'a> QuantizedModel<'a> {
    pub fn load(checkpoint: &'a GgufCheckpoint, config: Qwen3NextConfig) -> Result<Self> {
        let embedding = checkpoint.load_embedding("token_embd.weight")?;
        ensure!(
            embedding.shape() == [config.vocab_size, config.hidden_size],
            "invalid token embedding shape {:?}",
            embedding.shape()
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            tracing::info!(
                layer,
                layer_type = ?config.layer_type(layer),
                "loading quantized decoder layer"
            );
            layers.push(match config.layer_type(layer) {
                LayerType::FullAttention => {
                    DecoderLayer::Full(QuantizedFullLayer::load(checkpoint, &config, layer)?)
                }
                LayerType::LinearAttention => {
                    DecoderLayer::Linear(QuantizedLinearLayer::load(checkpoint, &config, layer)?)
                }
            });
        }
        let final_norm = checkpoint.load_f32_vector("output_norm.weight")?;
        ensure!(
            final_norm.elem_count() == config.hidden_size,
            "invalid final norm shape"
        );
        let lm_head = checkpoint.load_matrix("output.weight")?;
        ensure!(
            lm_head.shape() == [config.vocab_size, config.hidden_size],
            "invalid LM head shape {:?}",
            lm_head.shape()
        );
        let mtp = if config.mtp_num_hidden_layers == 1 {
            Some(QuantizedMtpHead::load(
                checkpoint,
                &config,
                embedding.clone(),
                lm_head.clone(),
            )?)
        } else {
            None
        };
        Ok(Self {
            checkpoint,
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            mtp,
        })
    }

    pub fn config(&self) -> &Qwen3NextConfig {
        &self.config
    }

    pub fn expert_cache_stats(&self) -> Result<ExpertCacheStats> {
        self.checkpoint.expert_cache_stats()
    }

    pub fn mtp(&self) -> Option<&QuantizedMtpHead<'a>> {
        self.mtp.as_ref()
    }

    pub fn new_state(&self) -> QuantizedModelState {
        let layers = self
            .layers
            .iter()
            .map(|layer| match layer {
                DecoderLayer::Full(layer) => DecoderState::Full(layer.new_state()),
                DecoderLayer::Linear(layer) => DecoderState::Linear(layer.new_state()),
            })
            .collect();
        QuantizedModelState {
            layers,
            position: 0,
        }
    }

    pub fn forward(
        &self,
        token_ids: &[u32],
        state: &mut QuantizedModelState,
    ) -> Result<(Tensor, QuantizedForwardTimings)> {
        let output = self.forward_detailed_with_trace(token_ids, state, None)?;
        Ok((output.logits, output.timings))
    }

    pub fn forward_with_trace(
        &self,
        token_ids: &[u32],
        state: &mut QuantizedModelState,
        trace: Option<&mut dyn RoutingTrace>,
    ) -> Result<(Tensor, QuantizedForwardTimings)> {
        let output = self.forward_detailed_with_trace(token_ids, state, trace)?;
        Ok((output.logits, output.timings))
    }

    pub fn forward_detailed(
        &self,
        token_ids: &[u32],
        state: &mut QuantizedModelState,
    ) -> Result<QuantizedForwardOutput> {
        self.forward_detailed_with_trace(token_ids, state, None)
    }

    /// Run a multi-row pass that records a rollback snapshot at every row
    /// boundary. Rolling back to any of them afterwards costs a state copy
    /// rather than a replayed forward pass.
    pub fn forward_detailed_with_snapshots(
        &self,
        token_ids: &[u32],
        state: &mut QuantizedModelState,
        snapshots: &mut QuantizedStateSnapshots,
    ) -> Result<QuantizedForwardOutput> {
        snapshots.begin_pass(state, token_ids.len());
        self.forward_inner(token_ids, state, None, Some(snapshots), LogitRows::All)
    }

    pub fn forward_detailed_with_trace(
        &self,
        token_ids: &[u32],
        state: &mut QuantizedModelState,
        trace: Option<&mut dyn RoutingTrace>,
    ) -> Result<QuantizedForwardOutput> {
        self.forward_inner(token_ids, state, trace, None, LogitRows::All)
    }

    /// As [`Self::forward_detailed_with_trace`], but computing logits only for
    /// the rows the caller will read. See [`LogitRows`].
    pub fn forward_detailed_logits(
        &self,
        token_ids: &[u32],
        state: &mut QuantizedModelState,
        trace: Option<&mut dyn RoutingTrace>,
        logits: LogitRows,
    ) -> Result<QuantizedForwardOutput> {
        self.forward_inner(token_ids, state, trace, None, logits)
    }

    fn forward_inner(
        &self,
        token_ids: &[u32],
        state: &mut QuantizedModelState,
        mut trace: Option<&mut dyn RoutingTrace>,
        mut snapshots: Option<&mut QuantizedStateSnapshots>,
        logit_rows: LogitRows,
    ) -> Result<QuantizedForwardOutput> {
        ensure!(!token_ids.is_empty(), "forward requires at least one token");
        ensure!(
            state.layers.len() == self.layers.len(),
            "model state belongs to a different model"
        );
        ensure!(
            state.position + token_ids.len() <= self.config.max_position_embeddings,
            "sequence exceeds maximum position embeddings"
        );
        let wall_started = Instant::now();
        let mut timings = QuantizedForwardTimings::default();
        let embedding_started = Instant::now();
        let ids = Tensor::from_slice(token_ids, token_ids.len(), &Device::Cpu)?;
        let mut hidden = self.embedding.forward(&ids)?.to_dtype(DType::F32)?;
        timings.embedding = embedding_started.elapsed();
        let position = state.position;
        let mut linear_index = 0;
        for (index, (layer, layer_state)) in
            self.layers.iter().zip(state.layers.iter_mut()).enumerate()
        {
            let started = Instant::now();
            let output = match (layer, layer_state) {
                (DecoderLayer::Full(layer), DecoderState::Full(layer_state)) => {
                    layer.forward(&hidden, position, layer_state)?
                }
                (DecoderLayer::Linear(layer), DecoderState::Linear(layer_state)) => {
                    let sink = snapshots.as_deref_mut();
                    let nontemporal = sink.as_ref().is_some_and(|sink| sink.nontemporal);
                    let output = layer.forward_with_snapshots(
                        &hidden,
                        layer_state,
                        sink.and_then(|sink| sink.layer_mut(linear_index)),
                        nontemporal,
                    )?;
                    linear_index += 1;
                    output
                }
                _ => anyhow::bail!("state type does not match layer {index}"),
            };
            if let Some(sink) = trace.as_mut() {
                ensure!(
                    output.routes.len() == token_ids.len(),
                    "layer {index} produced {} routes for {} input tokens",
                    output.routes.len(),
                    token_ids.len()
                );
                for (token_offset, (&token_id, route)) in
                    token_ids.iter().zip(&output.routes).enumerate()
                {
                    sink.record(&RoutingRecord {
                        token_index: position + token_offset,
                        token_id,
                        layer: index,
                        selected_expert_ids: route.experts.clone(),
                        router_weights: route.weights.clone(),
                        router_logits: Some(route.logits.clone()),
                    })?;
                }
            }
            hidden = output.hidden;
            let elapsed = started.elapsed();
            timings.layers.push(elapsed);
            timings.layer_details.push(output.timings);
            tracing::info!(
                layer = index,
                elapsed_ms = elapsed.as_secs_f64() * 1_000.,
                "completed quantized decoder layer"
            );
        }
        let norm_started = Instant::now();
        hidden = gguf_rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps)?;
        timings.final_norm = norm_started.elapsed();
        let lm_started = Instant::now();
        // Only the rows that will be read. `hidden` is returned whole either
        // way, so the MTP arm and the hidden-state carry are unaffected.
        let logits = match logit_rows {
            LogitRows::All => self.lm_head.forward(&hidden)?,
            LogitRows::Last => {
                self.lm_head
                    .forward(&hidden.narrow(0, token_ids.len() - 1, 1)?)?
            }
        };
        timings.lm_head = lm_started.elapsed();
        state.position += token_ids.len();
        timings.wall = wall_started.elapsed();
        Ok(QuantizedForwardOutput {
            logits,
            normalized_hidden: hidden,
            timings,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn report_preserves_nested_quantized_stage_timings() {
        let layer = QuantizedLayerTimings {
            layer: 7,
            layer_type: Some(LayerType::LinearAttention),
            wall: Duration::from_millis(80),
            normalization: Duration::from_millis(2),
            delta: super::super::QuantizedDeltaTimings {
                wall: Duration::from_millis(20),
                recurrence: Duration::from_millis(5),
                ..Default::default()
            },
            moe: super::super::QuantizedMoeTimings {
                wall: Duration::from_millis(55),
                expert_load: Duration::from_millis(3),
                expert_compute: Duration::from_millis(40),
                expert_gate_up: Duration::from_millis(17),
                expert_activation: Duration::from_millis(2),
                expert_down: Duration::from_millis(18),
                expert_accumulation: Duration::from_millis(3),
                ..Default::default()
            },
            ..Default::default()
        };
        let timings = QuantizedForwardTimings {
            wall: Duration::from_millis(90),
            embedding: Duration::from_millis(1),
            layers: vec![Duration::from_millis(80)],
            layer_details: vec![layer],
            final_norm: Duration::from_millis(1),
            lm_head: Duration::from_millis(5),
        };

        let report = timings.report();
        assert_eq!(report.layers[0].layer, 7);
        assert_eq!(
            report.layers[0].layer_type,
            Some(LayerType::LinearAttention)
        );
        assert_eq!(report.stages.moe_seconds, 0.055);
        assert_eq!(report.nested_operations.moe_expert_lookup_seconds, 0.003);
        assert_eq!(report.nested_operations.moe_expert_compute_seconds, 0.040);
        assert_eq!(report.nested_operations.moe_expert_gate_up_seconds, 0.017);
        assert_eq!(
            report.nested_operations.moe_expert_activation_seconds,
            0.002
        );
        assert_eq!(report.nested_operations.moe_expert_down_seconds, 0.018);
        assert_eq!(
            report.nested_operations.moe_expert_accumulation_seconds,
            0.003
        );
        assert_eq!(report.nested_operations.deltanet_recurrence_seconds, 0.005);
    }

    #[test]
    fn accumulate_merges_layer_details() {
        let sample = QuantizedForwardTimings {
            wall: Duration::from_millis(10),
            layers: vec![Duration::from_millis(8)],
            layer_details: vec![QuantizedLayerTimings {
                layer: 0,
                layer_type: Some(LayerType::FullAttention),
                wall: Duration::from_millis(8),
                moe: super::super::QuantizedMoeTimings {
                    expert_compute: Duration::from_millis(6),
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut total = QuantizedForwardTimings::default();
        total.accumulate(&sample);
        total.accumulate(&sample);
        assert_eq!(total.wall, Duration::from_millis(20));
        assert_eq!(total.layer_details[0].wall, Duration::from_millis(16));
        assert_eq!(
            total.layer_details[0].moe.expert_compute,
            Duration::from_millis(12)
        );
    }
}
