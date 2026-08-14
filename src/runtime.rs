use std::{
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use candle_core::{IndexOp, Tensor};

use crate::{
    Checkpoint, ExpertCacheStats, GgufCheckpoint, Qwen3NextConfig,
    qwen::{
        ForwardTimings, Model, ModelState, QuantizedForwardTimings, QuantizedModel,
        QuantizedModelState,
    },
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

#[derive(Debug, Clone)]
pub struct QuantizedGenerationMetrics {
    /// Tokens supplied by the caller for this turn.
    pub prompt_tokens: usize,
    /// Tokens actually evaluated before sampling. After the first turn this
    /// includes the previously emitted, pending token.
    pub evaluated_input_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_wall_time: Duration,
    pub decode_wall_time: Duration,
    pub time_to_first_token: Duration,
    pub prefill_profile: QuantizedForwardTimings,
    pub decode_profile: QuantizedForwardTimings,
    pub expert_cache: ExpertCacheStats,
}

impl QuantizedGenerationMetrics {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        self.evaluated_input_tokens as f64 / self.prefill_wall_time.as_secs_f64()
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        self.generated_tokens.saturating_sub(1) as f64 / self.decode_wall_time.as_secs_f64()
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedGenerationResult {
    pub prompt_token_ids: Vec<u32>,
    pub evaluated_input_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub text: String,
    pub metrics: QuantizedGenerationMetrics,
}

/// Persistent quantized runtime for one active sequence.
///
/// Loaded model weights and the checkpoint handle live for the lifetime of
/// this value. Session state is retained between `generate` calls until
/// `reset` is called.
pub struct QuantizedRuntime<'a> {
    model: QuantizedModel<'a>,
    tokenizer: ModelTokenizer,
    state: QuantizedModelState,
    /// The last emitted token is not evaluated unless another decode pass is
    /// required. It must be prepended to the next turn to keep model state
    /// aligned with the visible context.
    pending_token: Option<u32>,
    trace: Option<Box<dyn RoutingTrace>>,
}

impl<'a> QuantizedRuntime<'a> {
    pub fn load(
        checkpoint: &'a GgufCheckpoint,
        tokenizer_model_dir: impl AsRef<Path>,
    ) -> Result<Self> {
        let tokenizer_model_dir = tokenizer_model_dir.as_ref();
        let config = Qwen3NextConfig::from_path(tokenizer_model_dir.join("config.json"))?;
        let tokenizer = ModelTokenizer::from_model_dir(tokenizer_model_dir)?;
        let model = QuantizedModel::load(checkpoint, config)?;
        let state = model.new_state();
        Ok(Self {
            model,
            tokenizer,
            state,
            pending_token: None,
            trace: None,
        })
    }

    pub fn model(&self) -> &QuantizedModel<'a> {
        &self.model
    }

    pub fn tokenizer(&self) -> &ModelTokenizer {
        &self.tokenizer
    }

    /// Number of tokens represented by the state plus the final emitted token
    /// waiting to be evaluated on the next turn.
    pub fn context_tokens(&self) -> usize {
        self.state.position + usize::from(self.pending_token.is_some())
    }

    pub fn set_trace(&mut self, trace: Option<Box<dyn RoutingTrace>>) {
        self.trace = trace;
    }

    pub fn reset(&mut self) {
        self.state = self.model.new_state();
        self.pending_token = None;
    }

    fn forward(&mut self, tokens: &[u32]) -> Result<(Tensor, QuantizedForwardTimings)> {
        let model = &self.model;
        let state = &mut self.state;
        match self.trace.as_mut() {
            Some(trace) => model.forward_with_trace(tokens, state, Some(trace.as_mut())),
            None => model.forward(tokens, state),
        }
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<QuantizedGenerationResult> {
        self.generate_with_token_callback(prompt, options, |_| Ok(()))
    }

    pub fn generate_with_token_callback(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<QuantizedGenerationResult> {
        ensure!(
            options.max_new_tokens > 0,
            "max_new_tokens must be at least one"
        );
        let prompt_token_ids = self.tokenizer.encode(prompt, options.add_special_tokens)?;
        ensure!(
            !prompt_token_ids.is_empty(),
            "tokenizer produced an empty prompt"
        );
        let evaluated_input_token_ids =
            continuation_input(&prompt_token_ids, self.pending_token.take());

        let mut sampler = Sampler::new(options.sampling.clone())?;
        let cache_before = self.model.expert_cache_stats()?;
        let prefill_started = Instant::now();
        let (mut logits, prefill_profile) = self.forward(&evaluated_input_token_ids)?;
        let prefill_wall_time = prefill_started.elapsed();
        let decode_started = Instant::now();
        let mut decode_profile = QuantizedForwardTimings::default();
        let mut generated = Vec::with_capacity(options.max_new_tokens);
        for step in 0..options.max_new_tokens {
            let last = logits.i(logits.dim(0)? - 1)?.to_vec1::<f32>()?;
            let token = sampler.sample(&last)?;
            generated.push(token);
            if let Err(error) = on_token(token) {
                // The current sampled token has not been evaluated yet, so it
                // is the correct pending token if the output sink fails.
                self.pending_token = Some(token);
                if let Some(trace) = &mut self.trace {
                    trace.flush()?;
                }
                return Err(error);
            }
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
            let (next_logits, profile) = self.forward(&[token])?;
            decode_profile.accumulate(&profile);
            logits = next_logits;
        }
        self.pending_token = generated.last().copied();
        if let Some(trace) = &mut self.trace {
            trace.flush()?;
        }
        let decode_wall_time = decode_started.elapsed();
        let expert_cache = self
            .model
            .expert_cache_stats()?
            .activity_since(cache_before);
        let text = self
            .tokenizer
            .decode(&generated, true)
            .context("failed to decode generated tokens")?;
        Ok(QuantizedGenerationResult {
            metrics: QuantizedGenerationMetrics {
                prompt_tokens: prompt_token_ids.len(),
                evaluated_input_tokens: evaluated_input_token_ids.len(),
                generated_tokens: generated.len(),
                prefill_wall_time,
                decode_wall_time,
                time_to_first_token: prefill_wall_time,
                prefill_profile,
                decode_profile,
                expert_cache,
            },
            prompt_token_ids,
            evaluated_input_token_ids,
            generated_token_ids: generated,
            text,
        })
    }
}

fn continuation_input(prompt_token_ids: &[u32], pending_token: Option<u32>) -> Vec<u32> {
    let mut input =
        Vec::with_capacity(prompt_token_ids.len() + usize::from(pending_token.is_some()));
    input.extend(pending_token);
    input.extend_from_slice(prompt_token_ids);
    input
}

#[cfg(test)]
mod quantized_runtime_tests {
    use super::continuation_input;

    #[test]
    fn persistent_turn_evaluates_pending_generated_token_first() {
        assert_eq!(continuation_input(&[20, 21], Some(10)), [10, 20, 21]);
        assert_eq!(continuation_input(&[20, 21], None), [20, 21]);
    }
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
