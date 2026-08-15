use std::{fs, path::Path};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

fn default_model_type() -> String {
    "qwen3_next".into()
}
fn default_hidden_act() -> String {
    "silu".into()
}
fn default_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    5_000_000.0
}
fn default_partial_rotary_factor() -> f64 {
    0.25
}
fn default_true() -> bool {
    true
}
fn default_sparse_step() -> usize {
    1
}
fn default_full_attention_interval() -> usize {
    4
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3NextConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts_per_tok: usize,
    pub num_experts: usize,
    #[serde(default = "default_sparse_step")]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_true")]
    pub attn_output_gate: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    One(u32),
    Many(Vec<u32>),
}

impl EosTokenId {
    pub fn contains(&self, token: u32) -> bool {
        match self {
            Self::One(id) => *id == token,
            Self::Many(ids) => ids.contains(&token),
        }
    }
}

impl Qwen3NextConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read model config {}", path.display()))?;
        Self::from_json(&bytes).with_context(|| format!("invalid model config {}", path.display()))
    }

    fn from_json(bytes: &[u8]) -> Result<Self> {
        let root: Value = serde_json::from_slice(bytes)?;
        let mut text = root.get("text_config").unwrap_or(&root).clone();
        let object = text
            .as_object_mut()
            .context("model config must be a JSON object")?;
        if !object.contains_key("intermediate_size") {
            let moe = object
                .get("moe_intermediate_size")
                .cloned()
                .context("model config has neither intermediate_size nor moe_intermediate_size")?;
            object.insert("intermediate_size".into(), moe);
        }
        if !object.contains_key("rope_theta")
            && let Some(theta) = object
                .get("rope_parameters")
                .and_then(|rope| rope.get("rope_theta"))
                .cloned()
        {
            object.insert("rope_theta".into(), theta);
        }
        let config: Self = serde_json::from_value(text)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.model_type.as_str(), "qwen3_next" | "qwen3_5_moe_text"),
            "unsupported model_type {:?}; expected qwen3_next or qwen3_5_moe_text",
            self.model_type
        );
        ensure!(
            self.hidden_act == "silu",
            "unsupported hidden_act {:?}; expected silu",
            self.hidden_act
        );
        ensure!(
            self.attn_output_gate,
            "this runtime requires the full-attention output gate"
        );
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("num_experts", self.num_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
        ] {
            ensure!(value > 0, "{name} must be greater than zero");
        }
        ensure!(
            self.num_attention_heads
                .is_multiple_of(self.num_key_value_heads),
            "num_attention_heads must be divisible by num_key_value_heads"
        );
        ensure!(
            self.linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads),
            "linear_num_value_heads must be divisible by linear_num_key_heads"
        );
        ensure!(
            self.num_experts_per_tok <= self.num_experts,
            "num_experts_per_tok cannot exceed num_experts"
        );
        ensure!(
            self.decoder_sparse_step > 0,
            "decoder_sparse_step must be greater than zero"
        );
        ensure!(
            self.full_attention_interval > 0,
            "full_attention_interval must be greater than zero"
        );
        ensure!(
            self.linear_conv_kernel_dim > 0,
            "linear_conv_kernel_dim must be greater than zero"
        );
        let rotary_dim = (self.head_dim as f64 * self.partial_rotary_factor) as usize;
        ensure!(
            rotary_dim > 0 && rotary_dim.is_multiple_of(2),
            "partial rotary dimension must be a positive even number"
        );
        if let Some(types) = &self.layer_types {
            ensure!(
                types.len() == self.num_hidden_layers,
                "layer_types has {} entries, expected {}",
                types.len(),
                self.num_hidden_layers
            );
        }
        for &layer in &self.mlp_only_layers {
            if layer >= self.num_hidden_layers {
                bail!("mlp_only_layers contains out-of-range layer {layer}");
            }
        }
        Ok(())
    }

    pub fn layer_type(&self, layer: usize) -> LayerType {
        self.layer_types
            .as_ref()
            .map(|v| v[layer])
            .unwrap_or_else(|| {
                if (layer + 1).is_multiple_of(self.full_attention_interval) {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
    }

    pub fn layer_is_moe(&self, layer: usize) -> bool {
        !self.mlp_only_layers.contains(&layer)
            && self.num_experts > 0
            && (layer + 1).is_multiple_of(self.decoder_sparse_step)
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Qwen3NextConfig {
        Qwen3NextConfig {
            model_type: "qwen3_next".into(),
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            linear_conv_kernel_dim: 2,
            linear_key_head_dim: 2,
            linear_value_head_dim: 2,
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
            moe_intermediate_size: 4,
            shared_expert_intermediate_size: 4,
            num_experts_per_tok: 2,
            num_experts: 4,
            decoder_sparse_step: 1,
            mlp_only_layers: vec![],
            layer_types: None,
            full_attention_interval: 4,
            hidden_act: "silu".into(),
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.,
            partial_rotary_factor: 1.,
            norm_topk_prob: true,
            attn_output_gate: true,
            tie_word_embeddings: false,
            attention_bias: false,
            max_position_embeddings: 128,
            bos_token_id: None,
            eos_token_id: None,
        }
    }

    #[test]
    fn derives_three_to_one_layer_pattern() {
        let c = config();
        assert_eq!(c.layer_type(0), LayerType::LinearAttention);
        assert_eq!(c.layer_type(2), LayerType::LinearAttention);
        assert_eq!(c.layer_type(3), LayerType::FullAttention);
        c.validate().unwrap();
    }

    #[test]
    fn loads_nested_qwen35_moe_text_config() {
        let source = serde_json::json!({
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "vocab_size": 248320,
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
                "num_experts_per_tok": 8,
                "num_experts": 256,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention"
                ],
                "full_attention_interval": 4,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "partial_rotary_factor": 0.25,
                "rope_parameters": { "rope_theta": 10000000 },
                "attn_output_gate": true,
                "max_position_embeddings": 262144
            }
        });
        let config = Qwen3NextConfig::from_json(source.to_string().as_bytes()).unwrap();
        assert_eq!(config.model_type, "qwen3_5_moe_text");
        assert_eq!(config.intermediate_size, 512);
        assert_eq!(config.rope_theta, 10_000_000.);
        assert!(config.norm_topk_prob);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.layer_type(3), LayerType::FullAttention);
    }
}
