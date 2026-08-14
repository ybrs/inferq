use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use candle_core::{D, DType, Tensor};

use crate::{GgufCheckpoint, LayerType, Qwen3NextConfig};

use super::{
    QuantizedAttentionLayer, QuantizedAttentionState, QuantizedAttentionTimings,
    QuantizedDeltaLayer, QuantizedDeltaState, QuantizedDeltaTimings, QuantizedMoeLayer,
    QuantizedMoeTimings, Route,
};

#[derive(Debug, Clone, Default)]
pub struct QuantizedLayerTimings {
    pub wall: Duration,
    pub normalization: Duration,
    pub attention: QuantizedAttentionTimings,
    pub delta: QuantizedDeltaTimings,
    pub moe: QuantizedMoeTimings,
}

#[derive(Debug)]
pub struct QuantizedLayerOutput {
    pub hidden: Tensor,
    pub routes: Vec<Route>,
    pub timings: QuantizedLayerTimings,
}

pub struct QuantizedLinearLayer<'a> {
    layer: usize,
    eps: f64,
    input_norm: Tensor,
    post_attention_norm: Tensor,
    delta: QuantizedDeltaLayer,
    moe: QuantizedMoeLayer<'a>,
}

impl std::fmt::Debug for QuantizedLinearLayer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedLinearLayer")
            .field("layer", &self.layer)
            .finish_non_exhaustive()
    }
}

impl<'a> QuantizedLinearLayer<'a> {
    pub fn load(
        checkpoint: &'a GgufCheckpoint,
        config: &Qwen3NextConfig,
        layer: usize,
    ) -> Result<Self> {
        ensure!(
            config.layer_type(layer) == LayerType::LinearAttention,
            "layer {layer} is not a linear-attention layer"
        );
        let prefix = format!("blk.{layer}");
        let input_norm = checkpoint.load_f32_vector(&format!("{prefix}.attn_norm.weight"))?;
        let post_attention_norm =
            checkpoint.load_f32_vector(&format!("{prefix}.post_attention_norm.weight"))?;
        ensure!(
            input_norm.elem_count() == config.hidden_size
                && post_attention_norm.elem_count() == config.hidden_size,
            "invalid layer norm dimensions for layer {layer}"
        );
        Ok(Self {
            layer,
            eps: config.rms_norm_eps,
            input_norm,
            post_attention_norm,
            delta: QuantizedDeltaLayer::load(checkpoint, config, layer)?,
            moe: QuantizedMoeLayer::load(
                checkpoint,
                layer,
                config.num_experts_per_tok,
                config.norm_topk_prob,
            )?,
        })
    }

    pub fn new_state(&self) -> QuantizedDeltaState {
        self.delta.new_state()
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        state: &mut QuantizedDeltaState,
    ) -> Result<QuantizedLayerOutput> {
        let wall_started = Instant::now();
        let norm_started = Instant::now();
        let normalized = gguf_rms_norm(xs, &self.input_norm, self.eps)?;
        let mut normalization = norm_started.elapsed();
        let (mixed, delta) = self.delta.forward(&normalized, state)?;
        let hidden = (xs.to_dtype(DType::F32)? + mixed)?;
        let norm_started = Instant::now();
        let normalized = gguf_rms_norm(&hidden, &self.post_attention_norm, self.eps)?;
        normalization += norm_started.elapsed();
        let moe = self.moe.forward(&normalized)?;
        let hidden = (hidden + &moe.hidden)?;
        Ok(QuantizedLayerOutput {
            hidden,
            routes: moe.routes,
            timings: QuantizedLayerTimings {
                wall: wall_started.elapsed(),
                normalization,
                attention: QuantizedAttentionTimings::default(),
                delta,
                moe: moe.timings,
            },
        })
    }
}

pub struct QuantizedFullLayer<'a> {
    eps: f64,
    input_norm: Tensor,
    post_attention_norm: Tensor,
    attention: QuantizedAttentionLayer,
    moe: QuantizedMoeLayer<'a>,
}

impl<'a> QuantizedFullLayer<'a> {
    pub fn load(
        checkpoint: &'a GgufCheckpoint,
        config: &Qwen3NextConfig,
        layer: usize,
    ) -> Result<Self> {
        ensure!(
            config.layer_type(layer) == LayerType::FullAttention,
            "layer {layer} is not a full-attention layer"
        );
        let prefix = format!("blk.{layer}");
        let input_norm = checkpoint.load_f32_vector(&format!("{prefix}.attn_norm.weight"))?;
        let post_attention_norm =
            checkpoint.load_f32_vector(&format!("{prefix}.post_attention_norm.weight"))?;
        ensure!(
            input_norm.elem_count() == config.hidden_size
                && post_attention_norm.elem_count() == config.hidden_size,
            "invalid layer norm dimensions for layer {layer}"
        );
        Ok(Self {
            eps: config.rms_norm_eps,
            input_norm,
            post_attention_norm,
            attention: QuantizedAttentionLayer::load(checkpoint, config, layer)?,
            moe: QuantizedMoeLayer::load(
                checkpoint,
                layer,
                config.num_experts_per_tok,
                config.norm_topk_prob,
            )?,
        })
    }

    pub fn new_state(&self) -> QuantizedAttentionState {
        self.attention.new_state()
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        position: usize,
        state: &mut QuantizedAttentionState,
    ) -> Result<QuantizedLayerOutput> {
        let wall_started = Instant::now();
        let norm_started = Instant::now();
        let normalized = gguf_rms_norm(xs, &self.input_norm, self.eps)?;
        let mut normalization = norm_started.elapsed();
        let (mixed, attention) = self.attention.forward(&normalized, position, state)?;
        let hidden = (xs.to_dtype(DType::F32)? + mixed)?;
        let norm_started = Instant::now();
        let normalized = gguf_rms_norm(&hidden, &self.post_attention_norm, self.eps)?;
        normalization += norm_started.elapsed();
        let moe = self.moe.forward(&normalized)?;
        let hidden = (hidden + &moe.hidden)?;
        Ok(QuantizedLayerOutput {
            hidden,
            routes: moe.routes,
            timings: QuantizedLayerTimings {
                wall: wall_started.elapsed(),
                normalization,
                attention,
                delta: QuantizedDeltaTimings::default(),
                moe: moe.timings,
            },
        })
    }
}

/// GGUF conversion stores ordinary RMSNorm weights with the `1 + weight`
/// adjustment already applied.
pub(super) fn gguf_rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let xs = xs.to_dtype(DType::F32)?;
    let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(xs
        .broadcast_div(&(variance + eps)?.sqrt()?)?
        .broadcast_mul(weight)?)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;
    use crate::qwen::rms_norm;

    #[test]
    fn converted_gguf_norm_matches_one_centered_hf_norm() {
        let xs = Tensor::new(&[[2f32, -3., 4.]], &Device::Cpu).unwrap();
        let hf_weight = Tensor::new(&[0.1f32, -0.2, 0.3], &Device::Cpu).unwrap();
        let gguf_weight = (&hf_weight + 1.).unwrap();
        let reference = rms_norm(&xs, &hf_weight, 1e-6)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let actual = gguf_rms_norm(&xs, &gguf_weight, 1e-6)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(actual, reference);
    }
}
