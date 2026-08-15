use std::{fs, path::Path};

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct ModelTokenizer {
    tokenizer: Tokenizer,
    chat_template: Option<String>,
}

impl ModelTokenizer {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .with_context(|| format!("failed to load tokenizer {}", path.display()))?;
        let config_path = model_dir.as_ref().join("tokenizer_config.json");
        let chat_template = if config_path.exists() {
            let bytes = fs::read(&config_path).with_context(|| {
                format!("failed to read tokenizer config {}", config_path.display())
            })?;
            let config: Value = serde_json::from_slice(&bytes)
                .with_context(|| format!("invalid tokenizer config {}", config_path.display()))?;
            config
                .get("chat_template")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        } else {
            None
        };
        Ok(Self {
            tokenizer,
            chat_template,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    pub fn decode_token(&self, id: u32) -> Result<String> {
        self.decode(&[id], false)
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    pub fn decode_stream(
        &self,
        skip_special_tokens: bool,
    ) -> impl FnMut(u32) -> Result<Option<String>> + '_ {
        let mut stream = self.tokenizer.decode_stream(skip_special_tokens);
        move |token| {
            stream
                .step(token)
                .map_err(|error| anyhow::anyhow!(error.to_string()))
        }
    }

    fn ensure_qwen_chat_template(&self) -> Result<()> {
        let template = self
            .chat_template
            .as_deref()
            .context("tokenizer_config.json does not define a chat_template")?;
        ensure!(
            template.contains("<|im_start|>") && template.contains("<|im_end|>"),
            "unsupported chat template: expected Qwen im_start/im_end markers"
        );
        Ok(())
    }

    /// Render the official template's plain-message, no-tools subset.
    pub fn initial_chat_prompt(&self, user: &str, system: Option<&str>) -> Result<String> {
        self.ensure_qwen_chat_template()?;
        Ok(qwen_initial_chat_prompt(user, system))
    }

    /// Append a user turn to an existing assistant generation. If the model
    /// already emitted `<|im_end|>`, only its required trailing newline is
    /// inserted; otherwise the assistant message is closed first.
    pub fn chat_continuation(&self, user: &str, assistant_closed: bool) -> Result<String> {
        self.ensure_qwen_chat_template()?;
        Ok(qwen_chat_continuation(user, assistant_closed))
    }
}

fn qwen_initial_chat_prompt(user: &str, system: Option<&str>) -> String {
    let mut prompt = String::new();
    if let Some(system) = system {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(user);
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
    prompt
}

fn qwen_chat_continuation(user: &str, assistant_closed: bool) -> String {
    let mut prompt = if assistant_closed {
        "\n".to_owned()
    } else {
        "<|im_end|>\n".to_owned()
    };
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(user);
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
    prompt
}

#[cfg(test)]
mod tests {
    use super::{qwen_chat_continuation, qwen_initial_chat_prompt};

    #[test]
    fn renders_plain_qwen_chat_turns() {
        assert_eq!(
            qwen_initial_chat_prompt("hello", Some("be concise")),
            "<|im_start|>system\nbe concise<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            qwen_chat_continuation("next", false),
            "<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            qwen_chat_continuation("next", true),
            "\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
