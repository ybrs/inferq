use std::{collections::BTreeSet, fs::File, path::Path};

use anyhow::{Context, Result, ensure};
use candle_core::quantized::gguf_file::{Content, Value};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct GgufSummary {
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

fn integer(value: &Value) -> Result<usize> {
    match value {
        Value::U8(v) => Ok(*v as usize),
        Value::I8(v) => Ok(*v as usize),
        Value::U16(v) => Ok(*v as usize),
        Value::I16(v) => Ok(*v as usize),
        Value::U32(v) => Ok(*v as usize),
        Value::I32(v) => Ok(*v as usize),
        Value::U64(v) => Ok(*v as usize),
        Value::I64(v) => Ok(*v as usize),
        _ => anyhow::bail!("expected integer GGUF metadata, found {value:?}"),
    }
}

fn get_usize(content: &Content, key: &str) -> Result<usize> {
    integer(
        content
            .metadata
            .get(key)
            .with_context(|| format!("GGUF is missing {key:?}"))?,
    )
}

pub fn inspect_gguf(path: impl AsRef<Path>) -> Result<GgufSummary> {
    let path = path.as_ref();
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let content = Content::read(&mut file)
        .with_context(|| format!("failed to parse GGUF header {}", path.display()))?;
    let architecture = match content.metadata.get("general.architecture") {
        Some(Value::String(value)) => value.clone(),
        other => anyhow::bail!("invalid general.architecture metadata: {other:?}"),
    };
    ensure!(
        architecture == "qwen3next",
        "unsupported GGUF architecture {architecture:?}"
    );
    let layers = get_usize(&content, "qwen3next.block_count")?;
    let interval = get_usize(&content, "qwen3next.full_attention_interval")?;
    ensure!(interval > 0, "full_attention_interval must be positive");
    let full_attention_layers = layers / interval;
    let vocab_size = match content.metadata.get("tokenizer.ggml.tokens") {
        Some(Value::Array(tokens)) => tokens.len(),
        other => anyhow::bail!("invalid tokenizer.ggml.tokens metadata: {other:?}"),
    };
    let dtypes = content
        .tensor_infos
        .values()
        .map(|tensor| format!("{:?}", tensor.ggml_dtype))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    Ok(GgufSummary {
        architecture,
        layers,
        hidden_size: get_usize(&content, "qwen3next.embedding_length")?,
        experts_per_layer: get_usize(&content, "qwen3next.expert_count")?,
        experts_selected: get_usize(&content, "qwen3next.expert_used_count")?,
        vocab_size,
        full_attention_layers,
        linear_attention_layers: layers - full_attention_layers,
        tensor_count: content.tensor_infos.len(),
        dtypes,
        format: "gguf".into(),
    })
}
