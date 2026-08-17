use std::{
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use candle_core::{Device, IndexOp, Tensor};

use crate::{
    Checkpoint, ExpertCacheStats, GgufCheckpoint, Qwen3NextConfig,
    ngram::NgramIndex,
    qwen::{
        ForwardTimings, Model, ModelState, QuantizedForwardTimings, QuantizedModel,
        QuantizedModelState, QuantizedMtpState, QuantizedMtpTimings, QuantizedStateSnapshots,
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
    /// Maximum tokens proposed per step by the n-gram (prompt-lookup) drafter.
    /// Zero keeps ordinary autoregressive decoding.
    pub speculative_ngram_draft_tokens: usize,
    /// Shortest token suffix the n-gram drafter will match on.
    pub ngram_min_match: usize,
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

/// Default shortest suffix the n-gram drafter matches on.
///
/// Four rather than three, from the sweep in `ngram-report-702d043633e0.md`:
/// three-token keys are not selective enough on this model. They roughly
/// double the match rate but the extra proposals are mostly wrong, and a wrong
/// proposal costs a whole verification pass. Requiring four tokens improved
/// every measured workload — the copy-heavy one from 1.058x to 1.228x, and
/// both non-repetitive ones by cutting the wrong proposals that made them
/// regress.
pub const DEFAULT_NGRAM_MIN_MATCH: usize = 4;

/// How the proposals served by one indexed key length fared.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NgramMatchLengthStats {
    pub match_len: usize,
    pub drafts: usize,
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
    pub fully_accepted_drafts: usize,
    pub rejected_immediately: usize,
}

impl NgramMatchLengthStats {
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed_tokens == 0 {
            0.
        } else {
            self.accepted_tokens as f64 / self.proposed_tokens as f64
        }
    }
}

/// Per-run measurements for the n-gram (prompt-lookup) drafter.
#[derive(Debug, Clone, Default)]
pub struct QuantizedNgramMetrics {
    pub max_draft_tokens: usize,
    pub min_match: usize,
    /// Decode steps that consulted the index.
    pub steps: usize,
    pub steps_with_match: usize,
    pub steps_without_match: usize,
    pub drafts_issued: usize,
    pub draft_tokens_proposed: usize,
    pub draft_tokens_accepted: usize,
    /// Proposals made at each draft position, `0..max_draft_tokens`.
    pub proposed_by_position: Vec<usize>,
    /// Proposals accepted at each draft position. Dividing the two gives the
    /// per-position acceptance curve that tunes draft length and match length.
    pub accepted_by_position: Vec<usize>,
    /// Per-match-length breakdown, longest key first. Acceptance is bimodal in
    /// practice — a proposal is usually either right to the end or wrong at
    /// once — so this is what says whether a key length earns its proposals.
    pub matches_by_len: Vec<NgramMatchLengthStats>,
    /// Drafts every proposed token of which verified.
    pub fully_accepted_drafts: usize,
    /// Drafts rejected at their very first proposed token.
    pub rejected_immediately: usize,
    /// Drafts cut short because the stored continuation contained a stop token.
    pub drafts_truncated_at_stop: usize,
    pub verification_passes: usize,
    pub verification_tokens: usize,
    /// Verification passes that committed fewer rows than they evaluated.
    pub rollbacks: usize,
    /// Retained for comparability with the MTP path; the snapshot rollback
    /// never replays a forward pass, so these stay zero.
    pub rollback_replays: usize,
    pub replayed_tokens: usize,
    pub replay_wall_time: Duration,
    pub lookup_wall_time: Duration,
    pub verification_wall_time: Duration,
    /// Copying live recurrent state into per-row snapshots, measured inside
    /// the verification forward. Subsumes what the MTP path called
    /// checkpointing: snapshot slot zero is the pre-pass checkpoint.
    pub snapshot_wall_time: Duration,
    pub rollback_wall_time: Duration,
    /// Plain single-row decode steps taken when the index had no match.
    pub target_only_wall_time: Duration,
    pub snapshot_rows: usize,
    pub snapshot_bytes_per_row: usize,
}

impl QuantizedNgramMetrics {
    pub fn new(max_draft_tokens: usize, min_match: usize) -> Self {
        Self {
            max_draft_tokens,
            min_match,
            proposed_by_position: vec![0; max_draft_tokens],
            accepted_by_position: vec![0; max_draft_tokens],
            ..Default::default()
        }
    }

    pub fn acceptance_rate(&self) -> f64 {
        if self.draft_tokens_proposed == 0 {
            0.
        } else {
            self.draft_tokens_accepted as f64 / self.draft_tokens_proposed as f64
        }
    }

    pub fn match_rate(&self) -> f64 {
        if self.steps == 0 {
            0.
        } else {
            self.steps_with_match as f64 / self.steps as f64
        }
    }

    /// Mean tokens committed per verification pass, counting the pass's own
    /// authoritative token. One means speculation bought nothing.
    pub fn tokens_per_verification(&self) -> f64 {
        if self.verification_passes == 0 {
            0.
        } else {
            (self.draft_tokens_accepted + self.verification_passes) as f64
                / self.verification_passes as f64
        }
    }

    /// Acceptance rate at each draft position, `0..max_draft_tokens`.
    pub fn position_acceptance(&self) -> Vec<f64> {
        self.proposed_by_position
            .iter()
            .zip(&self.accepted_by_position)
            .map(|(&proposed, &accepted)| {
                if proposed == 0 {
                    0.
                } else {
                    accepted as f64 / proposed as f64
                }
            })
            .collect()
    }

    fn record_draft(&mut self, draft: &crate::ngram::NgramDraft, accepted: usize) {
        self.drafts_issued += 1;
        self.draft_tokens_proposed += draft.tokens.len();
        self.draft_tokens_accepted += accepted;
        self.drafts_truncated_at_stop += usize::from(draft.truncated_at_stop);
        if self.proposed_by_position.len() < draft.tokens.len() {
            self.proposed_by_position.resize(draft.tokens.len(), 0);
            self.accepted_by_position.resize(draft.tokens.len(), 0);
        }
        for position in 0..draft.tokens.len() {
            self.proposed_by_position[position] += 1;
            if position < accepted {
                self.accepted_by_position[position] += 1;
            }
        }
        let fully_accepted = usize::from(accepted == draft.tokens.len());
        let rejected_immediately = usize::from(accepted == 0);
        self.fully_accepted_drafts += fully_accepted;
        self.rejected_immediately += rejected_immediately;
        if !self
            .matches_by_len
            .iter()
            .any(|stats| stats.match_len == draft.match_len)
        {
            self.matches_by_len.push(NgramMatchLengthStats {
                match_len: draft.match_len,
                ..Default::default()
            });
            self.matches_by_len
                .sort_by_key(|stats| std::cmp::Reverse(stats.match_len));
        }
        let stats = self
            .matches_by_len
            .iter_mut()
            .find(|stats| stats.match_len == draft.match_len)
            .expect("the entry was just inserted");
        stats.drafts += 1;
        stats.proposed_tokens += draft.tokens.len();
        stats.accepted_tokens += accepted;
        stats.fully_accepted_drafts += fully_accepted;
        stats.rejected_immediately += rejected_immediately;
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
    pub ngram: QuantizedNgramMetrics,
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
    /// Rollback snapshots for multi-row verification, allocated on first use
    /// and reused for the lifetime of the session.
    snapshots: QuantizedStateSnapshots,
    /// n-gram index over the tokens in context. Maintained only while the
    /// n-gram drafter is in use; it never affects decoding correctness.
    ngram: NgramIndex,
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
        let mut snapshots = QuantizedStateSnapshots::default();
        snapshots.set_nontemporal(true);
        Ok(Self {
            model,
            tokenizer,
            state,
            pending_token: None,
            mtp_state,
            last_target_hidden: None,
            trace: None,
            snapshots,
            ngram: NgramIndex::new(),
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
        self.ngram.clear();
    }

    /// Copy recurrent state into rollback snapshots with streaming stores
    /// where the CPU supports them. Disabling this falls back to
    /// `copy_from_slice`; both produce identical state.
    pub fn set_snapshot_nontemporal(&mut self, nontemporal: bool) {
        self.snapshots.set_nontemporal(nontemporal);
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
        ensure!(
            options.speculative_mtp_draft_tokens == 0
                || options.speculative_ngram_draft_tokens == 0,
            "n-gram and MTP speculation cannot be enabled at the same time"
        );
        if options.speculative_mtp_draft_tokens > 0 {
            return self.generate_speculative_mtp(prompt, options, on_token);
        }
        if options.speculative_ngram_draft_tokens > 0 {
            return self.generate_speculative_ngram(prompt, options, on_token);
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
                ngram: QuantizedNgramMetrics::default(),
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
            // A single-row pass has no boundary to roll back to, so it skips
            // the snapshot sink entirely.
            let snapshotted = !drafts.is_empty();
            let verification_started = Instant::now();
            let verified = match verify_rows(
                &self.model,
                &verification_tokens,
                &mut self.state,
                &mut self.snapshots,
                snapshotted,
            ) {
                Ok(output) => output,
                Err(error) => {
                    if snapshotted {
                        self.state.rollback(&self.snapshots, 0)?;
                    }
                    self.mtp_state
                        .as_mut()
                        .expect("MTP availability validated")
                        .truncate(target_prefix_position)?;
                    return Err(error);
                }
            };
            speculative.verification_wall_time += verification_started.elapsed();
            speculative.checkpoint_wall_time += snapshot_time(&verified.timings);
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

            // Rows the verification pass already computed for the committed
            // prefix are authoritative, so a rejection needs no replay: roll
            // the recurrent state back to the row boundary and keep the rows.
            let committed_hidden = if accepted == drafts.len() {
                verified.normalized_hidden
            } else {
                let restore_started = Instant::now();
                self.state
                    .rollback(&self.snapshots, committed_token_count)?;
                speculative.restore_wall_time += restore_started.elapsed();
                verified
                    .normalized_hidden
                    .narrow(0, 0, committed_token_count)?
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
                    if snapshotted && evaluated < verification_tokens.len() {
                        self.state.rollback(&self.snapshots, evaluated)?;
                        let hidden = committed_hidden.narrow(0, 0, evaluated)?;
                        self.synchronize_mtp(
                            target_prefix_position,
                            Some(&prior_hidden),
                            &verification_tokens[..evaluated],
                            &hidden,
                        )?;
                        self.last_target_hidden = Some(last_hidden_row(&hidden)?);
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
                ngram: QuantizedNgramMetrics::default(),
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

    /// Force-close the thinking block through the target model, keeping the
    /// n-gram index aligned with the injected closure tokens.
    fn force_close_thinking_ngram(
        &mut self,
        pending_token: Option<u32>,
        closure: &[u32],
        generated: &mut Vec<u32>,
        decode_profile: &mut QuantizedForwardTimings,
        on_token: &mut impl FnMut(u32) -> Result<()>,
    ) -> Result<Tensor> {
        let already_generated = generated.len();
        let logits = self.force_close_thinking_target_only(
            pending_token,
            closure,
            generated,
            decode_profile,
            on_token,
        )?;
        let injected = generated[already_generated..].to_vec();
        self.ngram.extend(&injected);
        Ok(logits)
    }

    /// Sample the next greedy token, commit it to the transcript and the
    /// index, and hand it to the output sink. Returns the new pending token,
    /// or `None` when the turn's token budget is already spent.
    fn sample_and_emit_ngram(
        &mut self,
        logits: &Tensor,
        options: &GenerationOptions,
        sampler: &mut Sampler,
        generated: &mut Vec<u32>,
        on_token: &mut impl FnMut(u32) -> Result<()>,
    ) -> Result<Option<u32>> {
        if generated.len() >= options.max_new_tokens {
            return Ok(None);
        }
        let row = logits.i(logits.dim(0)? - 1)?.to_vec1::<f32>()?;
        let token = sampler.sample(&row)?;
        generated.push(token);
        self.ngram.push(token);
        if let Err(error) = on_token(token) {
            self.pending_token = Some(token);
            return Err(error);
        }
        Ok(Some(token))
    }

    /// Greedy decoding with n-gram (prompt-lookup) speculation.
    ///
    /// Each step proposes the continuation that followed the most recent
    /// earlier occurrence of the current token tail, then verifies the pending
    /// token and the whole proposal in one multi-row target pass. A step whose
    /// tail has no earlier occurrence issues no proposal and runs exactly the
    /// single-row pass ordinary decoding would run, so a workload without
    /// repetition pays only the index lookup.
    ///
    /// Committed tokens are always the target model's own greedy choices: a
    /// draft token is committed only where the target's argmax at that row
    /// equals it, and the token after the last accepted row comes from the
    /// target. Speculation therefore changes speed, never output.
    fn generate_speculative_ngram(
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
            options.speculative_mtp_draft_tokens == 0,
            "n-gram and MTP speculation cannot be enabled at the same time"
        );
        ensure!(
            options.sampling.temperature == 0.,
            "n-gram speculative decoding currently requires temperature=0"
        );
        ensure!(
            self.trace.is_none(),
            "n-gram speculative decoding does not support routing traces"
        );
        ensure!(
            (crate::ngram::MIN_MATCH_LEN..=crate::ngram::MAX_MATCH_LEN)
                .contains(&options.ngram_min_match),
            "n-gram minimum match length must be between {} and {}",
            crate::ngram::MIN_MATCH_LEN,
            crate::ngram::MAX_MATCH_LEN
        );

        let prompt_token_ids = self.tokenizer.encode(prompt, options.add_special_tokens)?;
        ensure!(
            !prompt_token_ids.is_empty(),
            "tokenizer produced an empty prompt"
        );
        let carried_token = self.pending_token.take();
        let context_before = self.state.position + usize::from(carried_token.is_some());
        let evaluated_input_token_ids = continuation_input(&prompt_token_ids, carried_token);
        // The index only supplies proposals, so a gap in it costs speed and
        // never correctness. Extend it when it already mirrors the sequence
        // the model state represents, and reseed it from this turn otherwise
        // (a first turn, a reset, or a turn that ran another decoding mode).
        if self.ngram.len() == context_before {
            self.ngram.extend(&prompt_token_ids);
        } else {
            self.ngram.clear();
            self.ngram.extend(&evaluated_input_token_ids);
        }

        let cache_before = self.model.expert_cache_stats()?;
        let prefill_started = Instant::now();
        let (mut logits, prefill_profile) = self.forward(&evaluated_input_token_ids)?;
        let prefill_wall_time = prefill_started.elapsed();

        let mut sampler = Sampler::new(options.sampling.clone())?;
        let decode_started = Instant::now();
        let mut decode_profile = QuantizedForwardTimings::default();
        let mut ngram = QuantizedNgramMetrics::new(
            options.speculative_ngram_draft_tokens,
            options.ngram_min_match,
        );
        let mut generated = Vec::with_capacity(options.max_new_tokens);
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
            logits = self.force_close_thinking_ngram(
                None,
                &closure,
                &mut generated,
                &mut decode_profile,
                &mut on_token,
            )?;
        }
        let mut pending_token = self.sample_and_emit_ngram(
            &logits,
            options,
            &mut sampler,
            &mut generated,
            &mut on_token,
        )?;
        if let Some(first) = pending_token {
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
                let closed = self.force_close_thinking_ngram(
                    Some(first),
                    &closure,
                    &mut generated,
                    &mut decode_profile,
                    &mut on_token,
                )?;
                pending_token = self.sample_and_emit_ngram(
                    &closed,
                    options,
                    &mut sampler,
                    &mut generated,
                    &mut on_token,
                )?;
            }
        }

        'generation: while let Some(seed) = pending_token {
            if generated.len() >= options.max_new_tokens || self.is_stop_token(seed, options) {
                break;
            }
            // Both clamps keep the committed rows and the emitted tokens in
            // step. A pass commits `1 + accepted` rows and emits `accepted + 1`
            // tokens, so a boundary that stops the emission loop early would
            // otherwise leave evaluated-but-unemitted tokens in the state.
            // Holding the draft to what the turn and the thinking budget can
            // still absorb means the token-limit and budget boundaries can only
            // land on the last accepted draft or on the authoritative token,
            // both of which end the pass with nothing evaluated left over.
            let remaining = options.max_new_tokens - generated.len();
            let mut draft_limit = options
                .speculative_ngram_draft_tokens
                .min(remaining.saturating_sub(1));
            if let Some(thinking_remaining) = thinking.as_ref().and_then(ThinkingBudget::remaining)
            {
                draft_limit = draft_limit.min(thinking_remaining);
            }

            ngram.steps += 1;
            let lookup_started = Instant::now();
            let proposal = if draft_limit == 0 {
                None
            } else {
                self.ngram
                    .draft(draft_limit, options.ngram_min_match, |token| {
                        self.is_stop_token(token, options)
                    })
            };
            ngram.lookup_wall_time += lookup_started.elapsed();
            if proposal.is_some() {
                ngram.steps_with_match += 1;
            } else {
                ngram.steps_without_match += 1;
            }

            let drafts = proposal
                .as_ref()
                .map(|proposal| proposal.tokens.clone())
                .unwrap_or_default();
            let mut verification_tokens = Vec::with_capacity(1 + drafts.len());
            verification_tokens.push(seed);
            verification_tokens.extend_from_slice(&drafts);
            // With no proposal this is the ordinary one-row decode pass, which
            // has no row boundary to roll back to and takes no snapshots.
            let snapshotted = !drafts.is_empty();
            let pass_started = Instant::now();
            let verified = match verify_rows(
                &self.model,
                &verification_tokens,
                &mut self.state,
                &mut self.snapshots,
                snapshotted,
            ) {
                Ok(output) => output,
                Err(error) => {
                    if snapshotted {
                        self.state.rollback(&self.snapshots, 0)?;
                    }
                    return Err(error);
                }
            };
            let pass_wall_time = pass_started.elapsed();
            if snapshotted {
                ngram.verification_wall_time += pass_wall_time;
                ngram.verification_passes += 1;
                ngram.verification_tokens += verification_tokens.len();
                ngram.snapshot_wall_time += snapshot_time(&verified.timings);
                ngram.snapshot_rows += verification_tokens.len();
                if ngram.snapshot_bytes_per_row == 0 {
                    ngram.snapshot_bytes_per_row = self.snapshots.bytes_per_row();
                }
            } else {
                ngram.target_only_wall_time += pass_wall_time;
            }
            decode_profile.accumulate(&verified.timings);

            let verifier_logits = verified.logits.to_vec2::<f32>()?;
            ensure!(
                verifier_logits.len() == verification_tokens.len(),
                "target verifier returned {} logit rows for {} tokens",
                verifier_logits.len(),
                verification_tokens.len()
            );
            let accepted = accepted_draft_prefix(&drafts, &verifier_logits)?;
            let authoritative = argmax(&verifier_logits[accepted])? as u32;
            let committed_rows = 1 + accepted;
            if let Some(proposal) = &proposal {
                ngram.record_draft(proposal, accepted);
            }
            if accepted < drafts.len() {
                let rollback_started = Instant::now();
                self.state.rollback(&self.snapshots, committed_rows)?;
                ngram.rollback_wall_time += rollback_started.elapsed();
                ngram.rollbacks += 1;
            }

            let outputs = speculative_committed_outputs(&drafts, accepted, authoritative);
            for (output_index, (token, accepted_draft)) in outputs.into_iter().enumerate() {
                generated.push(token);
                self.ngram.push(token);
                if let Err(error) = on_token(token) {
                    // Leave the session at the boundary immediately before the
                    // token whose output callback failed, with that token
                    // pending, exactly as ordinary generation does.
                    let evaluated = 1 + output_index.min(accepted);
                    if snapshotted && evaluated < verification_tokens.len() {
                        self.state.rollback(&self.snapshots, evaluated)?;
                    }
                    self.pending_token = Some(token);
                    return Err(error);
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
                    let closed = self.force_close_thinking_ngram(
                        (!accepted_draft).then_some(token),
                        &closure,
                        &mut generated,
                        &mut decode_profile,
                        &mut on_token,
                    )?;
                    pending_token = self.sample_and_emit_ngram(
                        &closed,
                        options,
                        &mut sampler,
                        &mut generated,
                        &mut on_token,
                    )?;
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
                speculative: QuantizedSpeculativeMetrics::default(),
                ngram,
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

/// Evaluate the pending token plus its drafts in one target pass, recording a
/// rollback snapshot at every row boundary unless the pass is a single row.
fn verify_rows(
    model: &QuantizedModel<'_>,
    token_ids: &[u32],
    state: &mut QuantizedModelState,
    snapshots: &mut QuantizedStateSnapshots,
    snapshotted: bool,
) -> Result<crate::qwen::QuantizedForwardOutput> {
    if snapshotted {
        model.forward_detailed_with_snapshots(token_ids, state, snapshots)
    } else {
        model.forward_detailed(token_ids, state)
    }
}

/// Time the forward pass spent copying recurrent state into snapshots.
fn snapshot_time(timings: &QuantizedForwardTimings) -> Duration {
    timings
        .layer_details
        .iter()
        .map(|layer| layer.delta.snapshot)
        .sum()
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

    use std::time::Duration;

    use super::{
        QuantizedNgramMetrics, ThinkingBoundary, ThinkingBudget, accepted_draft_prefix,
        continuation_input, shifted_hidden_inputs, speculative_committed_outputs,
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
    fn accepted_ngram_drafts_stop_at_the_budget_boundary() {
        // The n-gram path commits through the same outputs and the same
        // budget, so a long accepted draft must still stop the turn's thinking
        // exactly at the budget's token, leaving the rest uncommitted.
        let outputs = speculative_committed_outputs(&[11, 12, 13, 14, 15], 5, 16);
        let mut thinking = thinking_budget(3);
        let mut committed = Vec::new();
        for (token, _) in outputs {
            committed.push(token);
            if thinking.observe_committed(token, true) == ThinkingBoundary::ForceClosure {
                break;
            }
        }
        assert_eq!(committed, [11, 12, 13]);
        assert_eq!(thinking.metrics().committed_thinking_tokens, 3);
        assert_eq!(thinking.metrics().forced_closures, 1);
    }

    #[test]
    fn rejected_ngram_drafts_do_not_consume_thinking_budget() {
        // Two of five proposals verify; only those and the authoritative token
        // are committed, so only three tokens are charged to the budget.
        let outputs = speculative_committed_outputs(&[11, 12, 90, 91, 92], 2, 13);
        let mut thinking = thinking_budget(4);
        let events = outputs
            .into_iter()
            .map(|(token, _)| (token, thinking.observe_committed(token, true)))
            .collect::<Vec<_>>();
        assert_eq!(
            events,
            [
                (11, ThinkingBoundary::Continue),
                (12, ThinkingBoundary::Continue),
                (13, ThinkingBoundary::Continue),
            ]
        );
        assert_eq!(thinking.metrics().committed_thinking_tokens, 3);
        assert_eq!(thinking.metrics().forced_closures, 0);
    }

    #[test]
    fn ngram_metrics_record_acceptance_by_draft_position() {
        let mut metrics = QuantizedNgramMetrics::new(4, 3);
        metrics.record_draft(
            &crate::ngram::NgramDraft {
                tokens: vec![1, 2, 3, 4],
                match_len: 3,
                source_position: 9,
                truncated_at_stop: false,
            },
            2,
        );
        metrics.record_draft(
            &crate::ngram::NgramDraft {
                tokens: vec![5, 6],
                match_len: 4,
                source_position: 11,
                truncated_at_stop: true,
            },
            2,
        );
        assert_eq!(metrics.proposed_by_position, [2, 2, 1, 1]);
        assert_eq!(metrics.accepted_by_position, [2, 2, 0, 0]);
        assert_eq!(metrics.draft_tokens_proposed, 6);
        assert_eq!(metrics.draft_tokens_accepted, 4);
        assert_eq!(metrics.drafts_truncated_at_stop, 1);
        assert_eq!(
            metrics
                .matches_by_len
                .iter()
                .map(|stats| (stats.match_len, stats.drafts, stats.accepted_tokens))
                .collect::<Vec<_>>(),
            [(4, 1, 2), (3, 1, 2)]
        );
        assert_eq!(metrics.fully_accepted_drafts, 1);
        assert_eq!(metrics.rejected_immediately, 0);
        assert_eq!(metrics.position_acceptance(), [1., 1., 0., 0.]);
    }

    #[test]
    fn ngram_metrics_report_no_rollback_replays() {
        // The snapshot rollback never replays a forward pass. These fields are
        // kept only so the two speculation paths report the same shape.
        let metrics = QuantizedNgramMetrics::new(7, 3);
        assert_eq!(metrics.rollback_replays, 0);
        assert_eq!(metrics.replayed_tokens, 0);
        assert_eq!(metrics.replay_wall_time, Duration::ZERO);
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
            speculative_ngram_draft_tokens: 0,
            ngram_min_match: DEFAULT_NGRAM_MIN_MATCH,
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
