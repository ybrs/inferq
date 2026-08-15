use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use serde::Serialize;

use crate::{
    ExpertCacheStats, GgufCheckpoint, LayerType, QuantizedEmbedding, QuantizedMatrix,
    Qwen3NextConfig,
    trace::{RoutingRecord, RoutingTrace},
};

use super::{
    QuantizedAttentionState, QuantizedDeltaState, QuantizedFullLayer, QuantizedLayerTimings,
    QuantizedLinearLayer, quantized_layer::gguf_rms_norm,
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
pub struct QuantizedModelState {
    layers: Vec<DecoderState>,
    pub position: usize,
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
                total.moe_router_seconds += layer.moe.router.as_secs_f64();
                total.moe_top_k_seconds += layer.moe.top_k.as_secs_f64();
                total.moe_expert_lookup_seconds += layer.moe.expert_load.as_secs_f64();
                total.moe_expert_compute_seconds += layer.moe.expert_compute.as_secs_f64();
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
    pub moe_router_seconds: f64,
    pub moe_top_k_seconds: f64,
    pub moe_expert_lookup_seconds: f64,
    pub moe_expert_compute_seconds: f64,
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
        Ok(Self {
            checkpoint,
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
        })
    }

    pub fn config(&self) -> &Qwen3NextConfig {
        &self.config
    }

    pub fn expert_cache_stats(&self) -> Result<ExpertCacheStats> {
        self.checkpoint.expert_cache_stats()
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
        self.forward_with_trace(token_ids, state, None)
    }

    pub fn forward_with_trace(
        &self,
        token_ids: &[u32],
        state: &mut QuantizedModelState,
        mut trace: Option<&mut dyn RoutingTrace>,
    ) -> Result<(Tensor, QuantizedForwardTimings)> {
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
        for (index, (layer, layer_state)) in
            self.layers.iter().zip(state.layers.iter_mut()).enumerate()
        {
            let started = Instant::now();
            let output = match (layer, layer_state) {
                (DecoderLayer::Full(layer), DecoderState::Full(layer_state)) => {
                    layer.forward(&hidden, position, layer_state)?
                }
                (DecoderLayer::Linear(layer), DecoderState::Linear(layer_state)) => {
                    layer.forward(&hidden, layer_state)?
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
        let logits = self.lm_head.forward(&hidden)?;
        timings.lm_head = lm_started.elapsed();
        state.position += token_ids.len();
        timings.wall = wall_started.elapsed();
        Ok((logits, timings))
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
