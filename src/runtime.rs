use std::{
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use candle_core::{IndexOp, Tensor};

use crate::{
    Checkpoint,
    qwen::{ForwardTimings, Model, ModelState},
    sampling::{Sampler, SamplingConfig},
    tokenizer::ModelTokenizer,
    trace::RoutingTrace,
};

#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_tokens: Vec<u32>,
    pub add_special_tokens: bool,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            sampling: SamplingConfig::default(),
            stop_tokens: vec![],
            add_special_tokens: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationMetrics {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_wall_time: Duration,
    pub decode_wall_time: Duration,
    pub time_to_first_token: Duration,
    pub prefill_profile: ForwardTimings,
    pub decode_profile: ForwardTimings,
}

impl GenerationMetrics {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill_wall_time.as_secs_f64()
    }
    pub fn decode_tokens_per_second(&self) -> f64 {
        self.generated_tokens.saturating_sub(1) as f64 / self.decode_wall_time.as_secs_f64()
    }
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub text: String,
    pub metrics: GenerationMetrics,
}

pub struct Runtime {
    model: Model,
    tokenizer: ModelTokenizer,
    state: ModelState,
    trace: Option<Box<dyn RoutingTrace>>,
}

impl Runtime {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let checkpoint = Checkpoint::open(&model_dir)?;
        let tokenizer = ModelTokenizer::from_model_dir(&model_dir)?;
        let model = Model::new(checkpoint);
        let state = model.new_state();
        Ok(Self {
            model,
            tokenizer,
            state,
            trace: None,
        })
    }
    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn tokenizer(&self) -> &ModelTokenizer {
        &self.tokenizer
    }
    pub fn state(&self) -> &ModelState {
        &self.state
    }
    pub fn set_trace(&mut self, trace: Option<Box<dyn RoutingTrace>>) {
        self.trace = trace;
    }
    pub fn reset(&mut self) {
        self.state = self.model.new_state();
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor> {
        self.prefill_profiled(tokens).map(|(logits, _)| logits)
    }

    pub fn prefill_profiled(&mut self, tokens: &[u32]) -> Result<(Tensor, ForwardTimings)> {
        ensure!(!tokens.is_empty(), "prompt token sequence is empty");
        self.reset();
        let result = if let Some(trace) = self.trace.as_mut() {
            self.model
                .forward(tokens, &mut self.state, Some(trace.as_mut()))?
        } else {
            self.model.forward(tokens, &mut self.state, None)?
        };
        Ok(result)
    }

    pub fn decode(&mut self, token: u32) -> Result<Tensor> {
        self.decode_profiled(token).map(|(logits, _)| logits)
    }

    pub fn decode_profiled(&mut self, token: u32) -> Result<(Tensor, ForwardTimings)> {
        ensure!(self.state.position > 0, "decode requires a prior prefill");
        let result = if let Some(trace) = self.trace.as_mut() {
            self.model
                .forward(&[token], &mut self.state, Some(trace.as_mut()))?
        } else {
            self.model.forward(&[token], &mut self.state, None)?
        };
        Ok(result)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<GenerationResult> {
        let prompt_token_ids = self.tokenizer.encode(prompt, options.add_special_tokens)?;
        ensure!(
            !prompt_token_ids.is_empty(),
            "tokenizer produced an empty prompt"
        );
        let mut sampler = Sampler::new(options.sampling.clone())?;
        let prefill_started = Instant::now();
        let (mut logits, prefill_profile) = self.prefill_profiled(&prompt_token_ids)?;
        let prefill_wall_time = prefill_started.elapsed();
        let decode_started = Instant::now();
        let mut decode_profile = ForwardTimings::default();
        let mut time_to_first_token = None;
        let mut generated = Vec::with_capacity(options.max_new_tokens);
        for step in 0..options.max_new_tokens {
            let last = logits.i(logits.dim(0)? - 1)?.to_vec1::<f32>()?;
            let token = sampler.sample(&last)?;
            generated.push(token);
            time_to_first_token.get_or_insert_with(|| prefill_started.elapsed());
            let is_config_eos = self
                .model
                .config()
                .eos_token_id
                .as_ref()
                .is_some_and(|ids| ids.contains(token));
            if is_config_eos
                || options.stop_tokens.contains(&token)
                || step + 1 == options.max_new_tokens
            {
                break;
            }
            let (next_logits, profile) = self.decode_profiled(token)?;
            decode_profile.accumulate(&profile);
            logits = next_logits;
        }
        if let Some(trace) = &mut self.trace {
            trace.flush()?;
        }
        let decode_wall_time = decode_started.elapsed();
        let text = self
            .tokenizer
            .decode(&generated, true)
            .context("failed to decode generated tokens")?;
        Ok(GenerationResult {
            metrics: GenerationMetrics {
                prompt_tokens: prompt_token_ids.len(),
                generated_tokens: generated.len(),
                prefill_wall_time,
                decode_wall_time,
                time_to_first_token: time_to_first_token.unwrap_or(prefill_wall_time),
                prefill_profile,
                decode_profile,
            },
            prompt_token_ids,
            generated_token_ids: generated,
            text,
        })
    }
}
