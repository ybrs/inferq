use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};

use crate::{GgufCheckpoint, QuantizedEmbedding, QuantizedMatrix, Qwen3NextConfig};

use super::{
    QuantizedAttentionState, QuantizedFullLayer, QuantizedLayerTimings,
    quantized_layer::gguf_rms_norm,
};

#[derive(Debug, Clone, Default)]
pub struct QuantizedMtpState {
    attention: QuantizedAttentionState,
}

impl QuantizedMtpState {
    pub fn position(&self) -> usize {
        self.attention.positions
    }

    pub fn truncate(&mut self, position: usize) -> Result<()> {
        self.attention.truncate(position)
    }
}

#[derive(Debug, Clone, Default)]
pub struct QuantizedMtpTimings {
    pub wall: Duration,
    pub input_projection: Duration,
    pub layer: QuantizedLayerTimings,
    pub head_norm: Duration,
    pub lm_head: Duration,
}

impl QuantizedMtpTimings {
    pub fn accumulate(&mut self, other: &Self) {
        self.wall += other.wall;
        self.input_projection += other.input_projection;
        self.layer.accumulate(&other.layer);
        self.head_norm += other.head_norm;
        self.lm_head += other.lm_head;
    }
}

pub struct QuantizedMtpOutput {
    pub logits: Option<Tensor>,
    pub normalized_hidden: Tensor,
    pub timings: QuantizedMtpTimings,
}

/// Qwen3.5/3.6's single learned NextN predictor block.
///
/// The predictor shares the target token embedding and LM head. Its input at
/// position P is the pair `(token[P], normalized_target_hidden[P - 1])` during
/// catch-up, or the preceding MTP hidden row while drafting beyond the target.
pub struct QuantizedMtpHead<'a> {
    hidden_size: usize,
    eps: f64,
    embedding: QuantizedEmbedding,
    embedding_norm: Tensor,
    hidden_norm: Tensor,
    input_projection: QuantizedMatrix,
    layer: QuantizedFullLayer<'a>,
    head_norm: Tensor,
    lm_head: QuantizedMatrix,
}

impl std::fmt::Debug for QuantizedMtpHead<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedMtpHead")
            .field("hidden_size", &self.hidden_size)
            .finish_non_exhaustive()
    }
}

impl<'a> QuantizedMtpHead<'a> {
    pub fn load(
        checkpoint: &'a GgufCheckpoint,
        config: &Qwen3NextConfig,
        embedding: QuantizedEmbedding,
        lm_head: QuantizedMatrix,
    ) -> Result<Self> {
        ensure!(
            config.mtp_num_hidden_layers == 1,
            "Qwen MTP requires exactly one configured predictor layer"
        );
        ensure!(
            !config.mtp_use_dedicated_embeddings,
            "dedicated MTP embeddings are not supported"
        );
        let layer = config.num_hidden_layers;
        let prefix = format!("blk.{layer}.nextn");
        let embedding_norm = checkpoint.load_f32_vector(&format!("{prefix}.enorm.weight"))?;
        let hidden_norm = checkpoint.load_f32_vector(&format!("{prefix}.hnorm.weight"))?;
        let head_norm = checkpoint.load_f32_vector(&format!("{prefix}.shared_head_norm.weight"))?;
        for (name, norm) in [
            ("embedding norm", &embedding_norm),
            ("hidden norm", &hidden_norm),
            ("shared head norm", &head_norm),
        ] {
            ensure!(
                norm.elem_count() == config.hidden_size,
                "invalid MTP {name} shape {:?}; expected [{}]",
                norm.shape(),
                config.hidden_size
            );
        }
        let input_projection = checkpoint.load_matrix(&format!("{prefix}.eh_proj.weight"))?;
        ensure!(
            input_projection.shape() == [config.hidden_size, 2 * config.hidden_size],
            "invalid MTP input projection shape {:?}; expected [{}, {}]",
            input_projection.shape(),
            config.hidden_size,
            2 * config.hidden_size
        );
        Ok(Self {
            hidden_size: config.hidden_size,
            eps: config.rms_norm_eps,
            embedding,
            embedding_norm,
            hidden_norm,
            input_projection,
            layer: QuantizedFullLayer::load(checkpoint, config, layer)?,
            head_norm,
            lm_head,
        })
    }

    /// A draft-only LM head covering the first `vocab` rows of the shared one.
    pub fn draft_head(&self, vocab: usize) -> Result<QuantizedMatrix> {
        self.lm_head.leading_rows(vocab)
    }

    /// Rows in the shared LM head, i.e. the full vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.lm_head.shape()[0]
    }

    pub fn new_state(&self) -> QuantizedMtpState {
        QuantizedMtpState {
            attention: self.layer.new_state(),
        }
    }

    pub fn forward(
        &self,
        token_ids: &[u32],
        hidden_inputs: &Tensor,
        state: &mut QuantizedMtpState,
        produce_logits: bool,
    ) -> Result<QuantizedMtpOutput> {
        self.forward_with_head(token_ids, hidden_inputs, state, produce_logits, None)
    }

    /// As `forward`, but scoring the draft against `head` instead of the full
    /// LM head.
    ///
    /// `head` is expected to be a leading row slice of the shared LM head, so
    /// the returned logits cover a vocabulary prefix rather than the whole
    /// vocabulary. That is admissible **only** on the drafting path: a draft is
    /// a proposal, and one the prefix gets wrong is rejected by the target,
    /// which always scores against the full head.
    pub fn forward_with_head(
        &self,
        token_ids: &[u32],
        hidden_inputs: &Tensor,
        state: &mut QuantizedMtpState,
        produce_logits: bool,
        head: Option<&QuantizedMatrix>,
    ) -> Result<QuantizedMtpOutput> {
        ensure!(
            !token_ids.is_empty(),
            "MTP forward requires at least one token"
        );
        ensure!(
            hidden_inputs.dims() == [token_ids.len(), self.hidden_size],
            "MTP hidden input has shape {:?}, expected [{}, {}]",
            hidden_inputs.shape(),
            token_ids.len(),
            self.hidden_size
        );
        let wall_started = Instant::now();
        let input_started = Instant::now();
        let ids = Tensor::from_slice(token_ids, token_ids.len(), &Device::Cpu)?;
        let embeddings = self.embedding.forward(&ids)?.to_dtype(DType::F32)?;
        let embeddings = gguf_rms_norm(&embeddings, &self.embedding_norm, self.eps)?;
        let hidden_inputs = gguf_rms_norm(hidden_inputs, &self.hidden_norm, self.eps)?;
        let joined = Tensor::cat(&[&embeddings, &hidden_inputs], 1)?;
        let projected = self.input_projection.forward(&joined)?;
        let input_projection = input_started.elapsed();

        let position = state.position();
        let layer = self
            .layer
            .forward(&projected, position, &mut state.attention)?;
        let norm_started = Instant::now();
        let normalized_hidden = gguf_rms_norm(&layer.hidden, &self.head_norm, self.eps)?;
        let head_norm = norm_started.elapsed();
        let lm_started = Instant::now();
        let logits = produce_logits
            .then(|| head.unwrap_or(&self.lm_head).forward(&normalized_hidden))
            .transpose()?;
        let lm_head = lm_started.elapsed();
        Ok(QuantizedMtpOutput {
            logits,
            normalized_hidden,
            timings: QuantizedMtpTimings {
                wall: wall_started.elapsed(),
                input_projection,
                layer: layer.timings,
                head_norm,
                lm_head,
            },
        })
    }
}
