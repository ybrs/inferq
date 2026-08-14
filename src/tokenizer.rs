use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct ModelTokenizer(Tokenizer);

impl ModelTokenizer {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .with_context(|| format!("failed to load tokenizer {}", path.display()))?;
        Ok(Self(tokenizer))
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        Ok(self
            .0
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.0
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    pub fn decode_token(&self, id: u32) -> Result<String> {
        self.decode(&[id], false)
    }
}
