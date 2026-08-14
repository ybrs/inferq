use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};

use crate::{Checkpoint, LayerType, Qwen3NextConfig, trace::RoutingTrace};

use super::{
    attention::{self, AttentionState},
    deltanet::{self, DeltaState},
    linear, moe,
    norm::rms_norm,
};

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
pub struct ForwardTimings {
    pub embedding: Duration,
    pub normalization: Duration,
    pub attention: Duration,
    pub deltanet: Duration,
    pub moe: Duration,
    pub lm_head: Duration,
    pub layers: Vec<Duration>,
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
        let started = Instant::now();
        let ids = Tensor::from_slice(token_ids, token_ids.len(), &self.device)?;
        let embeddings = self
            .checkpoint
            .load("model.embed_tokens.weight", &self.device)?;
        let mut hidden = embeddings.index_select(&ids, 0)?;
        timings.embedding = started.elapsed();
        let token_offset = state.position;

        for layer in 0..c.num_hidden_layers {
            let layer_started = Instant::now();
            let p = format!("model.layers.{layer}");
            let norm_started = Instant::now();
            let input_norm = self
                .checkpoint
                .load(&format!("{p}.input_layernorm.weight"), &self.device)?;
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
                    )?;
                    timings.attention += mixer_started.elapsed();
                    value
                }
                (LayerState::Delta(cache), LayerType::LinearAttention) => {
                    let value = deltanet::forward(&self.checkpoint, c, layer, &normalized, cache)?;
                    timings.deltanet += mixer_started.elapsed();
                    value
                }
                _ => anyhow::bail!("state type does not match layer {layer}"),
            };
            hidden = (hidden + mixed)?;
            let norm_started = Instant::now();
            let post_norm = self.checkpoint.load(
                &format!("{p}.post_attention_layernorm.weight"),
                &self.device,
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
                )?
            } else {
                moe::dense_mlp(&self.checkpoint, layer, &normalized)?
            };
            timings.moe += moe_started.elapsed();
            hidden = (hidden + feed_forward)?;
            let layer_elapsed = layer_started.elapsed();
            timings.layers.push(layer_elapsed);
            tracing::debug!(
                layer,
                layer_type = ?c.layer_type(layer),
                elapsed_ms = layer_elapsed.as_secs_f64() * 1_000.,
                "completed decoder layer"
            );
        }
        let norm_started = Instant::now();
        let final_norm = self.checkpoint.load("model.norm.weight", &self.device)?;
        hidden = rms_norm(&hidden, &final_norm, c.rms_norm_eps)?;
        timings.normalization += norm_started.elapsed();
        let lm_started = Instant::now();
        let lm_head = if c.tie_word_embeddings {
            self.checkpoint
                .load("model.embed_tokens.weight", &self.device)?
        } else {
            self.checkpoint.load("lm_head.weight", &self.device)?
        };
        let logits = linear(&hidden, &lm_head)?.to_dtype(DType::F32)?;
        timings.lm_head = lm_started.elapsed();
        state.position += token_ids.len();
        Ok((logits, timings))
    }
}
