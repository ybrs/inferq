use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};

use crate::{GgufCheckpoint, LayerType, QuantizedEmbedding, QuantizedMatrix, Qwen3NextConfig};

use super::{
    QuantizedAttentionState, QuantizedDeltaState, QuantizedFullLayer, QuantizedLinearLayer,
    quantized_layer::gguf_rms_norm,
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
    pub final_norm: Duration,
    pub lm_head: Duration,
}

pub struct QuantizedModel<'a> {
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
            hidden = match (layer, layer_state) {
                (DecoderLayer::Full(layer), DecoderState::Full(layer_state)) => {
                    layer.forward(&hidden, position, layer_state)?.hidden
                }
                (DecoderLayer::Linear(layer), DecoderState::Linear(layer_state)) => {
                    layer.forward(&hidden, layer_state)?.hidden
                }
                _ => anyhow::bail!("state type does not match layer {index}"),
            };
            let elapsed = started.elapsed();
            timings.layers.push(elapsed);
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
