use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};

use serde::Serialize;

use crate::{Checkpoint, LayerType, Qwen3NextConfig, trace::RoutingTrace};

use super::{
    attention::{self, AttentionState},
    deltanet::{self, DeltaState},
    linear_profiled, load_profiled, moe,
    norm::rms_norm,
};

use super::moe::{reference_routes, reference_sparse_moe};

#[derive(Debug, Clone)]
enum LayerState {
    Attention(AttentionState),
    Delta(DeltaState),
}

#[derive(Debug, Clone)]
pub struct ModelState {
    layers: Vec<LayerState>,
    pub position: usize,
}

impl ModelState {
    pub fn new(config: &Qwen3NextConfig) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|layer| match config.layer_type(layer) {
                LayerType::FullAttention => LayerState::Attention(AttentionState::default()),
                LayerType::LinearAttention => LayerState::Delta(DeltaState::new(config)),
            })
            .collect();
        Self {
            layers,
            position: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LayerTimings {
    pub layer: usize,
    pub layer_type: Option<LayerType>,
    pub wall: Duration,
}

/// Timing data for one or more model forward passes.
///
/// Stage timings are disjoint at the top level and can be compared with
/// `wall`. Operation timings are nested within those stages and therefore must
/// not be added to the stage timings.
#[derive(Debug, Clone, Default)]
pub struct ForwardTimings {
    pub wall: Duration,
    pub embedding: Duration,
    pub normalization: Duration,
    pub attention: Duration,
    pub deltanet: Duration,
    pub moe: Duration,
    pub lm_head: Duration,
    pub weight_load: Duration,
    pub dtype_conversion: Duration,
    pub matmul: Duration,
    pub router: Duration,
    pub top_k: Duration,
    pub routed_experts: Duration,
    pub shared_expert: Duration,
    pub layers: Vec<LayerTimings>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ForwardTimingReport {
    pub wall_seconds: f64,
    pub accounted_seconds: f64,
    pub unattributed_seconds: f64,
    pub accounted_fraction: f64,
    pub stages: StageTimingReport,
    pub nested_operations: OperationTimingReport,
    pub layers: Vec<LayerTimingReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageTimingReport {
    pub embedding_seconds: f64,
    pub normalization_seconds: f64,
    pub attention_seconds: f64,
    pub deltanet_seconds: f64,
    pub moe_seconds: f64,
    pub lm_head_seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct OperationTimingReport {
    pub weight_load_seconds: f64,
    pub dtype_conversion_seconds: f64,
    pub matmul_seconds: f64,
    pub router_seconds: f64,
    pub top_k_seconds: f64,
    pub routed_experts_seconds: f64,
    pub shared_expert_seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerTimingReport {
    pub layer: usize,
    pub layer_type: Option<LayerType>,
    pub wall_seconds: f64,
}

impl ForwardTimings {
    pub fn accumulate(&mut self, other: &Self) {
        self.wall += other.wall;
        self.embedding += other.embedding;
        self.normalization += other.normalization;
        self.attention += other.attention;
        self.deltanet += other.deltanet;
        self.moe += other.moe;
        self.lm_head += other.lm_head;
        self.weight_load += other.weight_load;
        self.dtype_conversion += other.dtype_conversion;
        self.matmul += other.matmul;
        self.router += other.router;
        self.top_k += other.top_k;
        self.routed_experts += other.routed_experts;
        self.shared_expert += other.shared_expert;
        if self.layers.is_empty() {
            self.layers = other.layers.clone();
        } else {
            for (total, sample) in self.layers.iter_mut().zip(&other.layers) {
                total.wall += sample.wall;
            }
        }
    }

    pub fn report(&self) -> ForwardTimingReport {
        let accounted = self.embedding
            + self.normalization
            + self.attention
            + self.deltanet
            + self.moe
            + self.lm_head;
        let unattributed = self.wall.saturating_sub(accounted);
        let wall = self.wall.as_secs_f64();
        ForwardTimingReport {
            wall_seconds: wall,
            accounted_seconds: accounted.as_secs_f64(),
            unattributed_seconds: unattributed.as_secs_f64(),
            accounted_fraction: if wall == 0. {
                0.
            } else {
                accounted.as_secs_f64() / wall
            },
            stages: StageTimingReport {
                embedding_seconds: self.embedding.as_secs_f64(),
                normalization_seconds: self.normalization.as_secs_f64(),
                attention_seconds: self.attention.as_secs_f64(),
                deltanet_seconds: self.deltanet.as_secs_f64(),
                moe_seconds: self.moe.as_secs_f64(),
                lm_head_seconds: self.lm_head.as_secs_f64(),
            },
            nested_operations: OperationTimingReport {
                weight_load_seconds: self.weight_load.as_secs_f64(),
                dtype_conversion_seconds: self.dtype_conversion.as_secs_f64(),
                matmul_seconds: self.matmul.as_secs_f64(),
                router_seconds: self.router.as_secs_f64(),
                top_k_seconds: self.top_k.as_secs_f64(),
                routed_experts_seconds: self.routed_experts.as_secs_f64(),
                shared_expert_seconds: self.shared_expert.as_secs_f64(),
            },
            layers: self
                .layers
                .iter()
                .map(|layer| LayerTimingReport {
                    layer: layer.layer,
                    layer_type: layer.layer_type,
                    wall_seconds: layer.wall.as_secs_f64(),
                })
                .collect(),
        }
    }
}

pub struct Model {
    checkpoint: Checkpoint,
    device: Device,
}

impl Model {
    pub fn new(checkpoint: Checkpoint) -> Self {
        Self {
            checkpoint,
            device: Device::Cpu,
        }
    }
    pub fn config(&self) -> &Qwen3NextConfig {
        self.checkpoint.config()
    }
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }
    pub fn new_state(&self) -> ModelState {
        ModelState::new(self.config())
    }

    pub fn forward(
        &self,
        token_ids: &[u32],
        state: &mut ModelState,
        mut trace: Option<&mut dyn RoutingTrace>,
    ) -> Result<(Tensor, ForwardTimings)> {
        ensure!(!token_ids.is_empty(), "forward requires at least one token");
        ensure!(
            state.layers.len() == self.config().num_hidden_layers,
            "model state belongs to a different configuration"
        );
        ensure!(
            state.position + token_ids.len() <= self.config().max_position_embeddings,
            "sequence length {} exceeds max_position_embeddings {}",
            state.position + token_ids.len(),
            self.config().max_position_embeddings
        );
        let c = self.config();
        let mut timings = ForwardTimings::default();
        let forward_started = Instant::now();
        let started = Instant::now();
        let ids = Tensor::from_slice(token_ids, token_ids.len(), &self.device)?;
        let embeddings = load_profiled(
            &self.checkpoint,
            "model.embed_tokens.weight",
            &self.device,
            &mut timings,
        )?;
        let mut hidden = embeddings.index_select(&ids, 0)?;
        timings.embedding = started.elapsed();
        let token_offset = state.position;

        for layer in 0..c.num_hidden_layers {
            let layer_started = Instant::now();
            let p = format!("model.layers.{layer}");
            let norm_started = Instant::now();
            let input_norm = load_profiled(
                &self.checkpoint,
                &format!("{p}.input_layernorm.weight"),
                &self.device,
                &mut timings,
            )?;
            let normalized = rms_norm(&hidden, &input_norm, c.rms_norm_eps)?;
            timings.normalization += norm_started.elapsed();
            let mixer_started = Instant::now();
            let mixed = match (&mut state.layers[layer], c.layer_type(layer)) {
                (LayerState::Attention(cache), LayerType::FullAttention) => {
                    let value = attention::forward(
                        &self.checkpoint,
                        c,
                        layer,
                        &normalized,
                        token_offset,
                        cache,
                        &mut timings,
                    )?;
                    timings.attention += mixer_started.elapsed();
                    value
                }
                (LayerState::Delta(cache), LayerType::LinearAttention) => {
                    let value = deltanet::forward(
                        &self.checkpoint,
                        c,
                        layer,
                        &normalized,
                        cache,
                        &mut timings,
                    )?;
                    timings.deltanet += mixer_started.elapsed();
                    value
                }
                _ => anyhow::bail!("state type does not match layer {layer}"),
            };
            hidden = (hidden + mixed)?;
            let norm_started = Instant::now();
            let post_norm = load_profiled(
                &self.checkpoint,
                &format!("{p}.post_attention_layernorm.weight"),
                &self.device,
                &mut timings,
            )?;
            let normalized = rms_norm(&hidden, &post_norm, c.rms_norm_eps)?;
            timings.normalization += norm_started.elapsed();
            let moe_started = Instant::now();
            let feed_forward = if c.layer_is_moe(layer) {
                let layer_trace = trace
                    .as_deref_mut()
                    .map(|sink| sink as &mut dyn RoutingTrace);
                moe::sparse_moe(
                    &self.checkpoint,
                    c,
                    layer,
                    &normalized,
                    token_ids,
                    token_offset,
                    layer_trace,
                    &mut timings,
                )?
            } else {
                moe::dense_mlp(&self.checkpoint, layer, &normalized, &mut timings)?
            };
            timings.moe += moe_started.elapsed();
            hidden = (hidden + feed_forward)?;
            let layer_elapsed = layer_started.elapsed();
            timings.layers.push(LayerTimings {
                layer,
                layer_type: Some(c.layer_type(layer)),
                wall: layer_elapsed,
            });
            tracing::debug!(
                layer,
                layer_type = ?c.layer_type(layer),
                elapsed_ms = layer_elapsed.as_secs_f64() * 1_000.,
                "completed decoder layer"
            );
        }
        let norm_started = Instant::now();
        let final_norm = load_profiled(
            &self.checkpoint,
            "model.norm.weight",
            &self.device,
            &mut timings,
        )?;
        hidden = rms_norm(&hidden, &final_norm, c.rms_norm_eps)?;
        timings.normalization += norm_started.elapsed();
        let lm_started = Instant::now();
        let lm_head = if c.tie_word_embeddings {
            load_profiled(
                &self.checkpoint,
                "model.embed_tokens.weight",
                &self.device,
                &mut timings,
            )?
        } else {
            load_profiled(
                &self.checkpoint,
                "lm_head.weight",
                &self.device,
                &mut timings,
            )?
        };
        let logits = linear_profiled(&hidden, &lm_head, &mut timings)?.to_dtype(DType::F32)?;
        timings.lm_head = lm_started.elapsed();
        state.position += token_ids.len();
        timings.wall = forward_started.elapsed();
        Ok((logits, timings))
    }
}

#[derive(Debug)]
pub struct ReferenceLayerOutput {
    pub hidden: Tensor,
    pub routes: Vec<moe::Route>,
}

pub fn reference_linear_layer(
    checkpoint: &Checkpoint,
    config: &Qwen3NextConfig,
    layer: usize,
    xs: &Tensor,
) -> Result<ReferenceLayerOutput> {
    ensure!(
        config.layer_type(layer) == LayerType::LinearAttention,
        "layer {layer} is not a linear-attention layer"
    );
    let prefix = format!("model.layers.{layer}");
    let input_norm = checkpoint.load(&format!("{prefix}.input_layernorm.weight"), xs.device())?;
    let normalized = rms_norm(xs, &input_norm, config.rms_norm_eps)?;
    let mixed = deltanet::reference_deltanet(checkpoint, config, layer, &normalized)?;
    let hidden = (xs + mixed)?;
    let post_norm = checkpoint.load(
        &format!("{prefix}.post_attention_layernorm.weight"),
        xs.device(),
    )?;
    let normalized = rms_norm(&hidden, &post_norm, config.rms_norm_eps)?;
    let routes = reference_routes(checkpoint, config, layer, &normalized)?;
    let feed_forward = reference_sparse_moe(checkpoint, config, layer, &normalized)?;
    Ok(ReferenceLayerOutput {
        hidden: (hidden + feed_forward)?,
        routes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_separates_top_level_and_nested_timings() {
        let timings = ForwardTimings {
            wall: Duration::from_secs(10),
            embedding: Duration::from_secs(1),
            normalization: Duration::from_secs(1),
            attention: Duration::from_secs(2),
            moe: Duration::from_secs(5),
            weight_load: Duration::from_secs(4),
            matmul: Duration::from_secs(3),
            ..Default::default()
        };
        let report = timings.report();
        assert_eq!(report.accounted_seconds, 9.);
        assert_eq!(report.unattributed_seconds, 1.);
        assert_eq!(report.accounted_fraction, 0.9);
        assert_eq!(report.nested_operations.weight_load_seconds, 4.);
    }

    #[test]
    fn accumulate_merges_matching_layers() {
        let mut total = ForwardTimings::default();
        let sample = ForwardTimings {
            wall: Duration::from_secs(2),
            layers: vec![LayerTimings {
                layer: 0,
                layer_type: Some(LayerType::LinearAttention),
                wall: Duration::from_secs(1),
            }],
            ..Default::default()
        };
        total.accumulate(&sample);
        total.accumulate(&sample);
        assert_eq!(total.wall, Duration::from_secs(4));
        assert_eq!(total.layers[0].wall, Duration::from_secs(2));
    }
}
