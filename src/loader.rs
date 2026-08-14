use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor, safetensors::MmapedSafetensors};
use serde::{Deserialize, Serialize};

use crate::{LayerType, Qwen3NextConfig};

#[derive(Debug, Clone, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSummary {
    pub architecture: String,
    pub layers: usize,
    pub hidden_size: usize,
    pub experts_per_layer: usize,
    pub experts_selected: usize,
    pub vocab_size: usize,
    pub full_attention_layers: usize,
    pub linear_attention_layers: usize,
    pub tensor_count: usize,
    pub dtypes: Vec<String>,
    pub format: String,
}

#[derive(Debug, Deserialize)]
struct SafetensorIndex {
    weight_map: BTreeMap<String, String>,
}

pub struct Checkpoint {
    root: PathBuf,
    config: Qwen3NextConfig,
    tensors: MmapedSafetensors,
    inventory: BTreeMap<String, TensorInfo>,
}

impl std::fmt::Debug for Checkpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Checkpoint")
            .field("root", &self.root)
            .field("config", &self.config)
            .field("tensor_count", &self.inventory.len())
            .finish_non_exhaustive()
    }
}

impl Checkpoint {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let root = model_dir.as_ref().to_path_buf();
        ensure!(
            root.is_dir(),
            "model path {} is not a directory",
            root.display()
        );
        let config = Qwen3NextConfig::from_path(root.join("config.json"))?;
        let paths = discover_safetensors(&root)?;

        // SAFETY: the checkpoint owns the mappings for its entire lifetime. Model
        // files must not be mutated while inference is running.
        let tensors = unsafe { MmapedSafetensors::multi(&paths) }
            .with_context(|| format!("failed to memory-map checkpoint at {}", root.display()))?;
        let mut inventory = BTreeMap::new();
        for (name, view) in tensors.tensors() {
            inventory.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dtype: format!("{:?}", view.dtype()),
                    shape: view.shape().to_vec(),
                },
            );
        }
        let checkpoint = Self {
            root,
            config,
            tensors,
            inventory,
        };
        checkpoint.validate_required_tensors()?;
        Ok(checkpoint)
    }

    pub fn config(&self) -> &Qwen3NextConfig {
        &self.config
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn tensor_infos(&self) -> impl Iterator<Item = &TensorInfo> {
        self.inventory.values()
    }
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.inventory.get(name)
    }

    pub fn load(&self, name: &str, device: &Device) -> Result<Tensor> {
        self.tensors
            .load(name, device)
            .with_context(|| format!("failed to load tensor {name:?}"))
    }

    pub fn load_f32(&self, name: &str, device: &Device) -> Result<Tensor> {
        let tensor = self.load(name, device)?;
        if tensor.dtype() == DType::F32 {
            Ok(tensor)
        } else {
            Ok(tensor.to_dtype(DType::F32)?)
        }
    }

    pub fn summary(&self) -> ModelSummary {
        let full_attention_layers = (0..self.config.num_hidden_layers)
            .filter(|&i| self.config.layer_type(i) == LayerType::FullAttention)
            .count();
        let dtypes = self
            .inventory
            .values()
            .map(|t| t.dtype.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        ModelSummary {
            architecture: self.config.model_type.clone(),
            layers: self.config.num_hidden_layers,
            hidden_size: self.config.hidden_size,
            experts_per_layer: self.config.num_experts,
            experts_selected: self.config.num_experts_per_tok,
            vocab_size: self.config.vocab_size,
            full_attention_layers,
            linear_attention_layers: self.config.num_hidden_layers - full_attention_layers,
            tensor_count: self.inventory.len(),
            dtypes,
            format: "safetensors".into(),
        }
    }

    fn expect(&self, name: &str, expected: &[usize]) -> Result<()> {
        let info = self
            .inventory
            .get(name)
            .with_context(|| format!("checkpoint is missing required tensor {name:?}"))?;
        ensure!(
            info.shape == expected,
            "tensor {name:?} has shape {:?}, expected {:?}",
            info.shape,
            expected
        );
        ensure!(
            matches!(info.dtype.as_str(), "BF16" | "F16" | "F32"),
            "tensor {name:?} has unsupported dtype {}; expected BF16, F16, or F32",
            info.dtype
        );
        Ok(())
    }

    fn validate_required_tensors(&self) -> Result<()> {
        let c = &self.config;
        self.expect("model.embed_tokens.weight", &[c.vocab_size, c.hidden_size])?;
        self.expect("model.norm.weight", &[c.hidden_size])?;
        self.expect("lm_head.weight", &[c.vocab_size, c.hidden_size])?;
        let key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
        let value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
        for layer in 0..c.num_hidden_layers {
            let p = format!("model.layers.{layer}");
            self.expect(&format!("{p}.input_layernorm.weight"), &[c.hidden_size])?;
            self.expect(
                &format!("{p}.post_attention_layernorm.weight"),
                &[c.hidden_size],
            )?;
            match c.layer_type(layer) {
                LayerType::FullAttention => {
                    let a = format!("{p}.self_attn");
                    self.expect(
                        &format!("{a}.q_proj.weight"),
                        &[c.num_attention_heads * c.head_dim * 2, c.hidden_size],
                    )?;
                    self.expect(
                        &format!("{a}.k_proj.weight"),
                        &[c.num_key_value_heads * c.head_dim, c.hidden_size],
                    )?;
                    self.expect(
                        &format!("{a}.v_proj.weight"),
                        &[c.num_key_value_heads * c.head_dim, c.hidden_size],
                    )?;
                    self.expect(
                        &format!("{a}.o_proj.weight"),
                        &[c.hidden_size, c.num_attention_heads * c.head_dim],
                    )?;
                    self.expect(&format!("{a}.q_norm.weight"), &[c.head_dim])?;
                    self.expect(&format!("{a}.k_norm.weight"), &[c.head_dim])?;
                }
                LayerType::LinearAttention => {
                    let a = format!("{p}.linear_attn");
                    self.expect(
                        &format!("{a}.in_proj_qkvz.weight"),
                        &[key_dim * 2 + value_dim * 2, c.hidden_size],
                    )?;
                    self.expect(
                        &format!("{a}.in_proj_ba.weight"),
                        &[c.linear_num_value_heads * 2, c.hidden_size],
                    )?;
                    self.expect(
                        &format!("{a}.conv1d.weight"),
                        &[key_dim * 2 + value_dim, 1, c.linear_conv_kernel_dim],
                    )?;
                    self.expect(&format!("{a}.dt_bias"), &[c.linear_num_value_heads])?;
                    self.expect(&format!("{a}.A_log"), &[c.linear_num_value_heads])?;
                    self.expect(&format!("{a}.norm.weight"), &[c.linear_value_head_dim])?;
                    self.expect(&format!("{a}.out_proj.weight"), &[c.hidden_size, value_dim])?;
                }
            }
            if c.layer_is_moe(layer) {
                let m = format!("{p}.mlp");
                self.expect(&format!("{m}.gate.weight"), &[c.num_experts, c.hidden_size])?;
                for expert in 0..c.num_experts {
                    let e = format!("{m}.experts.{expert}");
                    self.expect(
                        &format!("{e}.gate_proj.weight"),
                        &[c.moe_intermediate_size, c.hidden_size],
                    )?;
                    self.expect(
                        &format!("{e}.up_proj.weight"),
                        &[c.moe_intermediate_size, c.hidden_size],
                    )?;
                    self.expect(
                        &format!("{e}.down_proj.weight"),
                        &[c.hidden_size, c.moe_intermediate_size],
                    )?;
                }
                self.expect(
                    &format!("{m}.shared_expert.gate_proj.weight"),
                    &[c.shared_expert_intermediate_size, c.hidden_size],
                )?;
                self.expect(
                    &format!("{m}.shared_expert.up_proj.weight"),
                    &[c.shared_expert_intermediate_size, c.hidden_size],
                )?;
                self.expect(
                    &format!("{m}.shared_expert.down_proj.weight"),
                    &[c.hidden_size, c.shared_expert_intermediate_size],
                )?;
                self.expect(
                    &format!("{m}.shared_expert_gate.weight"),
                    &[1, c.hidden_size],
                )?;
            } else {
                let m = format!("{p}.mlp");
                self.expect(
                    &format!("{m}.gate_proj.weight"),
                    &[c.intermediate_size, c.hidden_size],
                )?;
                self.expect(
                    &format!("{m}.up_proj.weight"),
                    &[c.intermediate_size, c.hidden_size],
                )?;
                self.expect(
                    &format!("{m}.down_proj.weight"),
                    &[c.hidden_size, c.intermediate_size],
                )?;
            }
        }
        Ok(())
    }
}

fn discover_safetensors(root: &Path) -> Result<Vec<PathBuf>> {
    let index_path = root.join("model.safetensors.index.json");
    if index_path.exists() {
        let data = fs::read(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;
        let index: SafetensorIndex = serde_json::from_slice(&data)
            .with_context(|| format!("invalid {}", index_path.display()))?;
        let files: BTreeSet<_> = index.weight_map.into_values().collect();
        ensure!(!files.is_empty(), "safetensors index contains no shards");
        return files
            .into_iter()
            .map(|file| {
                let path = root.join(&file);
                if !path.is_file() {
                    bail!(
                        "checkpoint shard listed in index is missing: {}",
                        path.display()
                    );
                }
                Ok(path)
            })
            .collect();
    }
    let single = root.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    bail!(
        "no model.safetensors or model.safetensors.index.json found in {}",
        root.display()
    )
}
