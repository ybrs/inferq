use std::{
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use candle_core::{Device, IndexOp, Tensor};

use crate::{
    Checkpoint, ExpertCacheStats, GgufCheckpoint, Qwen3NextConfig,
    qwen::{
        ForwardTimings, Model, ModelState, QuantizedForwardTimings, QuantizedModel,
        QuantizedModelState, QuantizedMtpState, QuantizedMtpTimings,
    },
    sampling::{Sampler, SamplingConfig, argmax},
    tokenizer::ModelTokenizer,
    trace::RoutingTrace,
};

#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_tokens: Vec<u32>,
    pub add_special_tokens: bool,
    /// Maximum tokens proposed by Qwen3.5/3.6's embedded MTP head per target
    /// verification pass. Zero keeps ordinary autoregressive decoding.
    pub speculative_mtp_draft_tokens: usize,
    /// Optional raw top-1/top-2 MTP logit-margin gate. Proposals below this
    /// threshold fall back to a one-row authoritative target pass.
    pub speculative_mtp_min_margin: Option<f32>,
    /// Maximum authoritative generated tokens allowed inside the Qwen
    /// `<think>` section. `None` preserves unbounded model-controlled thinking.
    pub thinking_budget: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct QuantizedDraftObservation {
    pub logit_margin: f32,
    pub accepted: bool,
    pub gated: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ThinkingMetrics {
    pub budget: Option<usize>,
    pub committed_thinking_tokens: usize,
    pub forced_closures: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingBoundary {
    Continue,
    NaturalClosure,
    ForceClosure,
}

#[derive(Debug, Clone, Copy)]
struct MtpDraftCandidate {
    token: u32,
    logit_margin: f32,
}

#[derive(Debug, Clone)]
struct ThinkingBudget {
    budget: usize,
    committed: usize,
    close_tokens: Vec<u32>,
    forced_close_tokens: Vec<u32>,
    recent: Vec<u32>,
    active: bool,
    forced: bool,
}

impl ThinkingBudget {
    fn from_tokenizer(tokenizer: &ModelTokenizer, budget: Option<usize>) -> Result<Option<Self>> {
        let Some(budget) = budget else {
            return Ok(None);
        };
        let close_tokens = tokenizer.encode("</think>", false)?;
        let forced_close_tokens = tokenizer.encode("</think>\n\n", false)?;
        ensure!(
            !close_tokens.is_empty() && !forced_close_tokens.is_empty(),
            "tokenizer produced an empty thinking-closure sequence"
        );
        Ok(Some(Self {
            budget,
            committed: 0,
            close_tokens,
            forced_close_tokens,
            recent: Vec::new(),
            active: true,
            forced: false,
        }))
    }

    fn should_force_before_sampling(&mut self) -> bool {
        if self.active && self.budget == 0 {
            self.active = false;
            self.forced = true;
            true
        } else {
            false
        }
    }

    fn observe_committed(&mut self, token: u32, enforce_budget: bool) -> ThinkingBoundary {
        if !self.active {
            return ThinkingBoundary::Continue;
        }
        self.committed += 1;
        self.recent.push(token);
        if self.recent.len() > self.close_tokens.len() {
            self.recent.remove(0);
        }
        if self.recent == self.close_tokens {
            self.active = false;
            return ThinkingBoundary::NaturalClosure;
        }
        if enforce_budget && self.committed >= self.budget {
            self.active = false;
            self.forced = true;
            return ThinkingBoundary::ForceClosure;
        }
        ThinkingBoundary::Continue
    }

    fn remaining(&self) -> Option<usize> {
        self.active
            .then(|| self.budget.saturating_sub(self.committed))
    }

    fn forced_close_tokens(&self) -> &[u32] {
        &self.forced_close_tokens
    }

    fn metrics(&self) -> ThinkingMetrics {
        ThinkingMetrics {
            budget: Some(self.budget),
            committed_thinking_tokens: self.committed,
            forced_closures: usize::from(self.forced),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct QuantizedSpeculativeMetrics {
    pub max_draft_tokens: usize,
    pub drafted_tokens: usize,
    pub accepted_tokens: usize,
    pub verification_passes: usize,
    pub verification_tokens: usize,
    pub rollback_replays: usize,
    pub replayed_tokens: usize,
    pub draft_wall_time: Duration,
    pub verification_wall_time: Duration,
    pub resync_wall_time: Duration,
    pub checkpoint_wall_time: Duration,
    pub restore_wall_time: Duration,
    pub replay_wall_time: Duration,
    pub gated_tokens: usize,
    pub draft_observations: Vec<QuantizedDraftObservation>,
    pub draft_profile: QuantizedMtpTimings,
    pub resync_profile: QuantizedMtpTimings,
}

impl QuantizedSpeculativeMetrics {
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted_tokens == 0 {
            0.
        } else {
            self.accepted_tokens as f64 / self.drafted_tokens as f64
        }
    }
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
    pub speculative: QuantizedSpeculativeMetrics,
    pub thinking: ThinkingMetrics,
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
    mtp_state: Option<QuantizedMtpState>,
    last_target_hidden: Option<Vec<f32>>,
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
        let mtp_state = model.mtp().map(|mtp| mtp.new_state());
        Ok(Self {
            model,
            tokenizer,
            state,
            pending_token: None,
            mtp_state,
            last_target_hidden: None,
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
        self.mtp_state = self.model.mtp().map(|mtp| mtp.new_state());
        self.last_target_hidden = None;
    }

    fn forward(&mut self, tokens: &[u32]) -> Result<(Tensor, QuantizedForwardTimings)> {
        let model = &self.model;
        let state = &mut self.state;
        match self.trace.as_mut() {
            Some(trace) => model.forward_with_trace(tokens, state, Some(trace.as_mut())),
            None => model.forward(tokens, state),
        }
    }

    fn force_close_thinking_target_only(
        &mut self,
        pending_token: Option<u32>,
        forced_close_tokens: &[u32],
        generated: &mut Vec<u32>,
        decode_profile: &mut QuantizedForwardTimings,
        on_token: &mut impl FnMut(u32) -> Result<()>,
    ) -> Result<Tensor> {
        let mut logits = None;
        if let Some(token) = pending_token {
            let (next, profile) = self.forward(&[token])?;
            decode_profile.accumulate(&profile);
            logits = Some(next);
        }
        for &token in forced_close_tokens {
            generated.push(token);
            if let Err(error) = on_token(token) {
                self.pending_token = Some(token);
                return Err(error);
            }
            let (next, profile) = self.forward(&[token])?;
            decode_profile.accumulate(&profile);
            logits = Some(next);
        }
        logits.context("forced thinking closure produced no target logits")
    }

    fn evaluate_authoritative_mtp_token(
        &mut self,
        token: u32,
        decode_profile: &mut QuantizedForwardTimings,
        speculative: &mut QuantizedSpeculativeMetrics,
    ) -> Result<Tensor> {
        let position = self.state.position;
        let previous_hidden = self
            .last_target_hidden
            .clone()
            .context("target hidden-state carry is missing")?;
        let output = self.model.forward_detailed(&[token], &mut self.state)?;
        decode_profile.accumulate(&output.timings);
        let resync_started = Instant::now();
        let resync = self.synchronize_mtp(
            position,
            Some(&previous_hidden),
            &[token],
            &output.normalized_hidden,
        )?;
        speculative.resync_wall_time += resync_started.elapsed();
        speculative.resync_profile.accumulate(&resync);
        self.last_target_hidden = Some(last_hidden_row(&output.normalized_hidden)?);
        Ok(output.logits)
    }

    fn force_close_thinking_speculative(
        &mut self,
        pending_token: Option<u32>,
        forced_close_tokens: &[u32],
        generated: &mut Vec<u32>,
        decode_profile: &mut QuantizedForwardTimings,
        speculative: &mut QuantizedSpeculativeMetrics,
        on_token: &mut impl FnMut(u32) -> Result<()>,
    ) -> Result<Tensor> {
        let mut logits = None;
        if let Some(token) = pending_token {
            logits =
                Some(self.evaluate_authoritative_mtp_token(token, decode_profile, speculative)?);
        }
        for &token in forced_close_tokens {
            generated.push(token);
            if let Err(error) = on_token(token) {
                self.pending_token = Some(token);
                return Err(error);
            }
            logits =
                Some(self.evaluate_authoritative_mtp_token(token, decode_profile, speculative)?);
        }
        logits.context("forced thinking closure produced no target logits")
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
        if options.speculative_mtp_draft_tokens > 0 {
            return self.generate_speculative_mtp(prompt, options, on_token);
        }
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
        let mut pending_token = None;
        let mut thinking =
            ThinkingBudget::from_tokenizer(&self.tokenizer, options.thinking_budget)?;
        if thinking
            .as_mut()
            .is_some_and(ThinkingBudget::should_force_before_sampling)
        {
            let closure = thinking
                .as_ref()
                .expect("thinking budget exists")
                .forced_close_tokens()
                .to_vec();
            logits = self.force_close_thinking_target_only(
                None,
                &closure,
                &mut generated,
                &mut decode_profile,
                &mut on_token,
            )?;
        }
        while generated.len() < options.max_new_tokens {
            let last = logits.i(logits.dim(0)? - 1)?.to_vec1::<f32>()?;
            let token = sampler.sample(&last)?;
            generated.push(token);
            pending_token = Some(token);
            if let Err(error) = on_token(token) {
                // The current sampled token has not been evaluated yet, so it
                // is the correct pending token if the output sink fails.
                self.pending_token = Some(token);
                if let Some(trace) = &mut self.trace {
                    trace.flush()?;
                }
                return Err(error);
            }
            let is_stop = self.is_stop_token(token, options);
            let boundary = thinking
                .as_mut()
                .map_or(ThinkingBoundary::Continue, |budget| {
                    budget.observe_committed(token, !is_stop)
                });
            if is_stop {
                break;
            }
            if boundary == ThinkingBoundary::ForceClosure {
                let closure = thinking
                    .as_ref()
                    .expect("thinking budget exists")
                    .forced_close_tokens()
                    .to_vec();
                logits = self.force_close_thinking_target_only(
                    pending_token.take(),
                    &closure,
                    &mut generated,
                    &mut decode_profile,
                    &mut on_token,
                )?;
                if generated.len() >= options.max_new_tokens {
                    break;
                }
                continue;
            }
            if generated.len() >= options.max_new_tokens {
                break;
            }
            let (next_logits, profile) = self.forward(&[token])?;
            decode_profile.accumulate(&profile);
            logits = next_logits;
            pending_token = None;
        }
        self.pending_token = pending_token;
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
                speculative: QuantizedSpeculativeMetrics::default(),
                thinking: thinking
                    .as_ref()
                    .map(ThinkingBudget::metrics)
                    .unwrap_or_default(),
            },
            prompt_token_ids,
            evaluated_input_token_ids,
            generated_token_ids: generated,
            text,
        })
    }

    fn generate_speculative_mtp(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<QuantizedGenerationResult> {
        ensure!(
            options.max_new_tokens > 0,
            "max_new_tokens must be at least one"
        );
        ensure!(
            options.sampling.temperature == 0.,
            "MTP speculative decoding currently requires temperature=0"
        );
        ensure!(
            self.trace.is_none(),
            "MTP speculative decoding does not yet support routing traces"
        );
        ensure!(
            self.model.mtp().is_some() && self.mtp_state.is_some(),
            "the loaded model does not contain a supported MTP predictor"
        );
        let prompt_token_ids = self.tokenizer.encode(prompt, options.add_special_tokens)?;
        ensure!(
            !prompt_token_ids.is_empty(),
            "tokenizer produced an empty prompt"
        );
        let evaluated_input_token_ids =
            continuation_input(&prompt_token_ids, self.pending_token.take());
        let target_position_before_prefill = self.state.position;
        let previous_hidden = self.last_target_hidden.clone();
        ensure!(
            self.mtp_state
                .as_ref()
                .is_some_and(|state| state.position() == target_position_before_prefill),
            "MTP state is not aligned with the target; reset before enabling speculation"
        );
        ensure!(
            target_position_before_prefill == 0 || previous_hidden.is_some(),
            "target hidden-state carry is missing; reset before enabling speculation"
        );

        let cache_before = self.model.expert_cache_stats()?;
        let prefill_started = Instant::now();
        let prefill = self
            .model
            .forward_detailed(&evaluated_input_token_ids, &mut self.state)?;
        self.synchronize_mtp(
            target_position_before_prefill,
            previous_hidden.as_deref(),
            &evaluated_input_token_ids,
            &prefill.normalized_hidden,
        )?;
        self.last_target_hidden = Some(last_hidden_row(&prefill.normalized_hidden)?);
        let prefill_wall_time = prefill_started.elapsed();
        let prefill_profile = prefill.timings;

        let mut sampler = Sampler::new(options.sampling.clone())?;
        let mut next_logits = prefill.logits;
        let decode_started = Instant::now();
        let mut decode_profile = QuantizedForwardTimings::default();
        let mut speculative = QuantizedSpeculativeMetrics {
            max_draft_tokens: options.speculative_mtp_draft_tokens,
            ..Default::default()
        };
        let mut generated = Vec::with_capacity(options.max_new_tokens);
        let mut thinking =
            ThinkingBudget::from_tokenizer(&self.tokenizer, options.thinking_budget)?;
        let mut pending_token = None;

        if thinking
            .as_mut()
            .is_some_and(ThinkingBudget::should_force_before_sampling)
        {
            let closure = thinking
                .as_ref()
                .expect("thinking budget exists")
                .forced_close_tokens()
                .to_vec();
            next_logits = self.force_close_thinking_speculative(
                None,
                &closure,
                &mut generated,
                &mut decode_profile,
                &mut speculative,
                &mut on_token,
            )?;
        }

        if generated.len() < options.max_new_tokens {
            let logits = next_logits.i(next_logits.dim(0)? - 1)?.to_vec1::<f32>()?;
            let first = sampler.sample(&logits)?;
            generated.push(first);
            pending_token = Some(first);
            if let Err(error) = on_token(first) {
                self.pending_token = Some(first);
                return Err(error);
            }
            let is_stop = self.is_stop_token(first, options);
            let boundary = thinking
                .as_mut()
                .map_or(ThinkingBoundary::Continue, |budget| {
                    budget.observe_committed(first, !is_stop)
                });
            if boundary == ThinkingBoundary::ForceClosure {
                let closure = thinking
                    .as_ref()
                    .expect("thinking budget exists")
                    .forced_close_tokens()
                    .to_vec();
                next_logits = self.force_close_thinking_speculative(
                    pending_token.take(),
                    &closure,
                    &mut generated,
                    &mut decode_profile,
                    &mut speculative,
                    &mut on_token,
                )?;
                if generated.len() < options.max_new_tokens {
                    let logits = next_logits.i(next_logits.dim(0)? - 1)?.to_vec1::<f32>()?;
                    let answer = sampler.sample(&logits)?;
                    generated.push(answer);
                    pending_token = Some(answer);
                    if let Err(error) = on_token(answer) {
                        self.pending_token = Some(answer);
                        return Err(error);
                    }
                }
            }
        }

        'generation: while let Some(seed) = pending_token {
            if generated.len() >= options.max_new_tokens || self.is_stop_token(seed, options) {
                break;
            }
            let remaining = options.max_new_tokens - generated.len();
            let mut draft_limit = options
                .speculative_mtp_draft_tokens
                .min(remaining.saturating_sub(1));
            if let Some(thinking_remaining) = thinking.as_ref().and_then(ThinkingBudget::remaining)
            {
                draft_limit = draft_limit.min(thinking_remaining);
            }
            let target_prefix_position = self.state.position;
            let prior_hidden = self
                .last_target_hidden
                .clone()
                .context("target hidden-state carry is missing")?;
            ensure!(
                self.mtp_state
                    .as_ref()
                    .is_some_and(|state| state.position() == target_prefix_position),
                "MTP state position does not match the target before drafting"
            );

            let draft_started = Instant::now();
            let (draft_candidates, draft_profile) =
                match self.draft_mtp(seed, &prior_hidden, draft_limit, options) {
                    Ok(result) => result,
                    Err(error) => {
                        self.mtp_state
                            .as_mut()
                            .expect("MTP availability validated")
                            .truncate(target_prefix_position)?;
                        return Err(error);
                    }
                };
            speculative.draft_wall_time += draft_started.elapsed();
            speculative.draft_profile.accumulate(&draft_profile);
            speculative.drafted_tokens += draft_candidates.len();
            let verified_draft_count =
                options
                    .speculative_mtp_min_margin
                    .map_or(draft_candidates.len(), |threshold| {
                        draft_candidates
                            .iter()
                            .take_while(|draft| draft.logit_margin >= threshold)
                            .count()
                    });
            speculative.gated_tokens += draft_candidates.len() - verified_draft_count;
            let drafts = draft_candidates[..verified_draft_count]
                .iter()
                .map(|draft| draft.token)
                .collect::<Vec<_>>();

            let mut verification_tokens = Vec::with_capacity(1 + drafts.len());
            verification_tokens.push(seed);
            verification_tokens.extend_from_slice(&drafts);
            let checkpoint = if drafts.is_empty() {
                None
            } else {
                let checkpoint_started = Instant::now();
                let checkpoint = self.state.checkpoint();
                speculative.checkpoint_wall_time += checkpoint_started.elapsed();
                Some(checkpoint)
            };
            let verification_started = Instant::now();
            let verified = match self
                .model
                .forward_detailed(&verification_tokens, &mut self.state)
            {
                Ok(output) => output,
                Err(error) => {
                    if let Some(checkpoint) = &checkpoint {
                        self.state.restore(checkpoint)?;
                    }
                    self.mtp_state
                        .as_mut()
                        .expect("MTP availability validated")
                        .truncate(target_prefix_position)?;
                    return Err(error);
                }
            };
            speculative.verification_wall_time += verification_started.elapsed();
            speculative.verification_passes += 1;
            speculative.verification_tokens += verification_tokens.len();
            decode_profile.accumulate(&verified.timings);
            let verifier_logits = verified.logits.to_vec2::<f32>()?;
            ensure!(
                verifier_logits.len() == verification_tokens.len(),
                "target verifier returned {} logit rows for {} tokens",
                verifier_logits.len(),
                verification_tokens.len()
            );
            let accepted = accepted_draft_prefix(&drafts, &verifier_logits)?;
            speculative
                .draft_observations
                .extend(draft_candidates.iter().enumerate().map(|(index, draft)| {
                    QuantizedDraftObservation {
                        logit_margin: draft.logit_margin,
                        accepted: index < accepted,
                        gated: index >= verified_draft_count,
                    }
                }));
            let authoritative = argmax(&verifier_logits[accepted])? as u32;
            let committed_token_count = 1 + accepted;

            let committed_hidden = if accepted == drafts.len() {
                verified.normalized_hidden
            } else {
                speculative.rollback_replays += 1;
                let restore_started = Instant::now();
                self.state.restore(
                    checkpoint
                        .as_ref()
                        .expect("rejection requires a checkpoint"),
                )?;
                speculative.restore_wall_time += restore_started.elapsed();
                let replay_started = Instant::now();
                let replayed = self.model.forward_detailed(
                    &verification_tokens[..committed_token_count],
                    &mut self.state,
                )?;
                speculative.replay_wall_time += replay_started.elapsed();
                speculative.replayed_tokens += committed_token_count;
                decode_profile.accumulate(&replayed.timings);
                replayed.normalized_hidden
            };

            let resync_started = Instant::now();
            let resync_profile = self.synchronize_mtp(
                target_prefix_position,
                Some(&prior_hidden),
                &verification_tokens[..committed_token_count],
                &committed_hidden,
            )?;
            speculative.resync_wall_time += resync_started.elapsed();
            speculative.resync_profile.accumulate(&resync_profile);
            self.last_target_hidden = Some(last_hidden_row(&committed_hidden)?);

            let outputs = speculative_committed_outputs(&drafts, accepted, authoritative);
            for (output_index, (token, accepted_draft)) in outputs.into_iter().enumerate() {
                generated.push(token);
                if let Err(error) = on_token(token) {
                    // Restore the target and MTP contexts to immediately before
                    // the token whose output callback failed, leaving that
                    // token pending just like ordinary generation.
                    let successful_outputs = output_index;
                    let evaluated = 1 + successful_outputs.min(accepted);
                    if let Some(checkpoint) = &checkpoint {
                        self.state.restore(checkpoint)?;
                        let replayed = self
                            .model
                            .forward_detailed(&verification_tokens[..evaluated], &mut self.state)?;
                        self.synchronize_mtp(
                            target_prefix_position,
                            Some(&prior_hidden),
                            &verification_tokens[..evaluated],
                            &replayed.normalized_hidden,
                        )?;
                        self.last_target_hidden =
                            Some(last_hidden_row(&replayed.normalized_hidden)?);
                    }
                    self.pending_token = Some(token);
                    return Err(error);
                }
                if accepted_draft {
                    speculative.accepted_tokens += 1;
                }
                let is_stop = self.is_stop_token(token, options);
                let boundary = thinking
                    .as_mut()
                    .map_or(ThinkingBoundary::Continue, |budget| {
                        budget.observe_committed(token, !is_stop)
                    });
                pending_token = (!accepted_draft).then_some(token);
                if boundary == ThinkingBoundary::ForceClosure {
                    let closure = thinking
                        .as_ref()
                        .expect("thinking budget exists")
                        .forced_close_tokens()
                        .to_vec();
                    next_logits = self.force_close_thinking_speculative(
                        (!accepted_draft).then_some(token),
                        &closure,
                        &mut generated,
                        &mut decode_profile,
                        &mut speculative,
                        &mut on_token,
                    )?;
                    pending_token = None;
                    if generated.len() < options.max_new_tokens {
                        let logits = next_logits.i(next_logits.dim(0)? - 1)?.to_vec1::<f32>()?;
                        let answer = sampler.sample(&logits)?;
                        generated.push(answer);
                        pending_token = Some(answer);
                        if let Err(error) = on_token(answer) {
                            self.pending_token = Some(answer);
                            return Err(error);
                        }
                    }
                    continue 'generation;
                }
                if generated.len() == options.max_new_tokens || is_stop {
                    break 'generation;
                }
            }
        }

        self.pending_token = pending_token;
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
                speculative,
                thinking: thinking
                    .as_ref()
                    .map(ThinkingBudget::metrics)
                    .unwrap_or_default(),
            },
            prompt_token_ids,
            evaluated_input_token_ids,
            generated_token_ids: generated,
            text,
        })
    }

    fn draft_mtp(
        &mut self,
        seed: u32,
        target_hidden: &[f32],
        max_drafts: usize,
        options: &GenerationOptions,
    ) -> Result<(Vec<MtpDraftCandidate>, QuantizedMtpTimings)> {
        let hidden_size = self.model.config().hidden_size;
        ensure!(
            target_hidden.len() == hidden_size,
            "MTP seed hidden row has {} values, expected {hidden_size}",
            target_hidden.len()
        );
        let mut token = seed;
        let mut hidden = target_hidden.to_vec();
        let mut drafts = Vec::with_capacity(max_drafts);
        let mut timings = QuantizedMtpTimings::default();
        for _ in 0..max_drafts {
            let input = Tensor::from_vec(hidden, (1, hidden_size), &Device::Cpu)?;
            let output = self
                .model
                .mtp()
                .expect("MTP availability validated")
                .forward(
                    &[token],
                    &input,
                    self.mtp_state.as_mut().expect("MTP availability validated"),
                    true,
                )?;
            timings.accumulate(&output.timings);
            let logits = output
                .logits
                .context("MTP draft forward did not produce logits")?
                .i(0)?
                .to_vec1::<f32>()?;
            let (draft, logit_margin) = top1_with_margin(&logits)?;
            let draft = draft as u32;
            hidden = output.normalized_hidden.i(0)?.to_vec1::<f32>()?;
            if self.is_stop_token(draft, options) {
                break;
            }
            drafts.push(MtpDraftCandidate {
                token: draft,
                logit_margin,
            });
            token = draft;
        }
        Ok((drafts, timings))
    }

    fn synchronize_mtp(
        &mut self,
        position: usize,
        previous_hidden: Option<&[f32]>,
        token_ids: &[u32],
        target_hidden: &Tensor,
    ) -> Result<QuantizedMtpTimings> {
        let hidden_size = self.model.config().hidden_size;
        let inputs = shifted_hidden_inputs(
            position,
            previous_hidden,
            target_hidden,
            token_ids.len(),
            hidden_size,
        )?;
        let state = self
            .mtp_state
            .as_mut()
            .context("MTP state is unavailable")?;
        ensure!(
            state.position() >= position,
            "MTP state is behind target position {position}"
        );
        state.truncate(position)?;
        let output = self
            .model
            .mtp()
            .context("MTP head is unavailable")?
            .forward(token_ids, &inputs, state, false)?;
        Ok(output.timings)
    }

    fn is_stop_token(&self, token: u32, options: &GenerationOptions) -> bool {
        self.model
            .config()
            .eos_token_id
            .as_ref()
            .is_some_and(|ids| ids.contains(token))
            || options.stop_tokens.contains(&token)
    }
}

fn last_hidden_row(hidden: &Tensor) -> Result<Vec<f32>> {
    ensure!(
        hidden.rank() == 2,
        "normalized hidden output must be rank two"
    );
    let rows = hidden.dim(0)?;
    ensure!(rows > 0, "normalized hidden output has no rows");
    Ok(hidden.i(rows - 1)?.to_vec1::<f32>()?)
}

fn shifted_hidden_inputs(
    position: usize,
    previous_hidden: Option<&[f32]>,
    target_hidden: &Tensor,
    tokens: usize,
    hidden_size: usize,
) -> Result<Tensor> {
    ensure!(
        target_hidden.dims() == [tokens, hidden_size],
        "target hidden output has shape {:?}, expected [{tokens}, {hidden_size}]",
        target_hidden.shape()
    );
    let rows = target_hidden.to_vec2::<f32>()?;
    let mut shifted = Vec::with_capacity(tokens * hidden_size);
    match previous_hidden {
        Some(previous) => {
            ensure!(
                previous.len() == hidden_size,
                "previous target hidden row has {} values, expected {hidden_size}",
                previous.len()
            );
            shifted.extend_from_slice(previous);
        }
        None => {
            ensure!(
                position == 0,
                "previous target hidden row is required after position zero"
            );
            shifted.resize(hidden_size, 0.);
        }
    }
    for row in rows.iter().take(tokens.saturating_sub(1)) {
        shifted.extend_from_slice(row);
    }
    Ok(Tensor::from_vec(
        shifted,
        (tokens, hidden_size),
        target_hidden.device(),
    )?)
}

fn accepted_draft_prefix(drafts: &[u32], verifier_logits: &[Vec<f32>]) -> Result<usize> {
    ensure!(
        verifier_logits.len() > drafts.len(),
        "target verifier needs one more logit row than the draft length"
    );
    let mut accepted = 0;
    while accepted < drafts.len() && argmax(&verifier_logits[accepted])? as u32 == drafts[accepted]
    {
        accepted += 1;
    }
    Ok(accepted)
}

fn top1_with_margin(logits: &[f32]) -> Result<(usize, f32)> {
    let top = argmax(logits)?;
    let second = logits
        .iter()
        .enumerate()
        .filter(|(index, value)| *index != top && value.is_finite())
        .map(|(_, &value)| value)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .context("MTP logits do not contain a finite runner-up")?;
    Ok((top, logits[top] - second))
}

fn speculative_committed_outputs(
    drafts: &[u32],
    accepted: usize,
    authoritative: u32,
) -> Vec<(u32, bool)> {
    let mut outputs = drafts[..accepted]
        .iter()
        .copied()
        .map(|token| (token, true))
        .collect::<Vec<_>>();
    outputs.push((authoritative, false));
    outputs
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
    use candle_core::{Device, Tensor};

    use super::{
        ThinkingBoundary, ThinkingBudget, accepted_draft_prefix, continuation_input,
        shifted_hidden_inputs, speculative_committed_outputs,
    };

    fn thinking_budget(budget: usize) -> ThinkingBudget {
        ThinkingBudget {
            budget,
            committed: 0,
            close_tokens: vec![90, 91],
            forced_close_tokens: vec![90, 91, 92],
            recent: Vec::new(),
            active: true,
            forced: false,
        }
    }

    #[test]
    fn persistent_turn_evaluates_pending_generated_token_first() {
        assert_eq!(continuation_input(&[20, 21], Some(10)), [10, 20, 21]);
        assert_eq!(continuation_input(&[20, 21], None), [20, 21]);
    }

    #[test]
    fn mtp_hidden_inputs_shift_target_rows_by_one_position() {
        let hidden = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Device::Cpu).unwrap();
        let shifted = shifted_hidden_inputs(7, Some(&[9., 8.]), &hidden, 3, 2)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(shifted, [[9., 8.], [1., 2.], [3., 4.]]);
    }

    #[test]
    fn verifier_accepts_only_the_matching_draft_prefix() {
        let logits = vec![vec![0., 3., 1.], vec![4., 0., 1.], vec![0., 1., 5.]];
        assert_eq!(accepted_draft_prefix(&[1, 2], &logits).unwrap(), 1);
        assert_eq!(accepted_draft_prefix(&[1], &logits).unwrap(), 1);
        assert_eq!(accepted_draft_prefix(&[], &logits).unwrap(), 0);
    }

    #[test]
    fn natural_thinking_close_spanning_tokens_wins_before_budget() {
        let mut thinking = thinking_budget(5);
        assert_eq!(
            thinking.observe_committed(7, true),
            ThinkingBoundary::Continue
        );
        assert_eq!(
            thinking.observe_committed(90, true),
            ThinkingBoundary::Continue
        );
        assert_eq!(
            thinking.observe_committed(91, true),
            ThinkingBoundary::NaturalClosure
        );
        assert_eq!(thinking.metrics().committed_thinking_tokens, 3);
        assert_eq!(thinking.metrics().forced_closures, 0);
    }

    #[test]
    fn thinking_budget_forces_the_complete_tokenized_closure() {
        let mut thinking = thinking_budget(2);
        assert_eq!(
            thinking.observe_committed(7, true),
            ThinkingBoundary::Continue
        );
        assert_eq!(
            thinking.observe_committed(8, true),
            ThinkingBoundary::ForceClosure
        );
        assert_eq!(thinking.forced_close_tokens(), [90, 91, 92]);
        assert_eq!(thinking.metrics().forced_closures, 1);
    }

    #[test]
    fn thinking_budget_state_resets_for_each_repl_turn() {
        for _turn in 0..2 {
            let mut thinking = thinking_budget(1);
            assert_eq!(
                thinking.observe_committed(7, true),
                ThinkingBoundary::ForceClosure
            );
            assert_eq!(thinking.metrics().committed_thinking_tokens, 1);
        }
    }

    #[test]
    fn accepted_speculative_drafts_stop_at_the_budget_boundary() {
        let outputs = speculative_committed_outputs(&[11, 12], 2, 13);
        let mut thinking = thinking_budget(2);
        let mut committed = Vec::new();
        for (token, _) in outputs {
            committed.push(token);
            if thinking.observe_committed(token, true) == ThinkingBoundary::ForceClosure {
                break;
            }
        }
        assert_eq!(committed, [11, 12]);
        assert_eq!(thinking.metrics().forced_closures, 1);
    }

    #[test]
    fn rejected_speculative_drafts_do_not_consume_thinking_budget() {
        let outputs = speculative_committed_outputs(&[11, 77], 1, 12);
        let mut thinking = thinking_budget(2);
        let events = outputs
            .into_iter()
            .map(|(token, _)| (token, thinking.observe_committed(token, true)))
            .collect::<Vec<_>>();
        assert_eq!(
            events,
            [
                (11, ThinkingBoundary::Continue),
                (12, ThinkingBoundary::ForceClosure)
            ]
        );
        assert_eq!(thinking.metrics().committed_thinking_tokens, 2);
    }

    #[test]
    fn stop_token_does_not_trigger_a_forced_closure() {
        let mut thinking = thinking_budget(1);
        assert_eq!(
            thinking.observe_committed(7, false),
            ThinkingBoundary::Continue
        );
        assert_eq!(thinking.metrics().forced_closures, 0);
    }
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            sampling: SamplingConfig::default(),
            stop_tokens: vec![],
            add_special_tokens: false,
            speculative_mtp_draft_tokens: 0,
            speculative_mtp_min_margin: None,
            thinking_budget: None,
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
