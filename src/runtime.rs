use std::{
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use candle_core::{Device, IndexOp, Tensor};

use crate::{
    Checkpoint, ExpertCacheStats, GgufCheckpoint, QuantizedMatrix, Qwen3NextConfig,
    ngram::{NgramDraft, NgramIndex},
    qwen::{
        ForwardTimings, LogitRows, Model, ModelState, QuantizedAttentionImage,
        QuantizedForwardTimings, QuantizedModel, QuantizedModelState, QuantizedMtpState,
        QuantizedMtpTimings, QuantizedStateImage, QuantizedStateSnapshots,
    },
    sampling::{Sampler, SamplingConfig, argmax},
    speculative::{
        ArmConfig, ArmController, DEFAULT_BACKOFF_CAP, DEFAULT_BACKOFF_TOKENS, DEFAULT_EWMA_ALPHA,
        DEFAULT_MTP_DEPTH_CAP, DEFAULT_MTP_DEPTH_FLOOR, DEFAULT_MTP_DEPTH_START,
        DEFAULT_MTP_DRAFT_VOCAB, DEFAULT_MTP_MIN_CONFIDENCE, DEFAULT_MTP_SUSPEND_BELOW,
        DEFAULT_NGRAM_DRAFT_CAP, DEFAULT_NGRAM_DRAFT_FLOOR, DEFAULT_NGRAM_SUSPEND_BELOW,
        PolicyStepRecord, QuantizedPolicyMetrics, SpanCursor, SpeculativeMode, StepArm, chain_span,
    },
    tokenizer::ModelTokenizer,
    trace::RoutingTrace,
};

/// Tuning for the unified speculative policy's two controllers.
///
/// Every field is a knob rather than a constant so the measured constants can
/// be swept without a rebuild, and so a mechanism that does not earn its place
/// can be switched off without removing the loop it lives in.
#[derive(Debug, Clone)]
pub struct PolicyTuning {
    /// Shortest draft the n-gram controller will shrink to.
    pub ngram_draft_floor: usize,
    /// Shallowest depth the MTP controller will shrink to.
    pub mtp_depth_floor: usize,
    /// Depth the MTP controller starts each run at.
    pub mtp_depth_start: usize,
    pub ngram_suspend_below: f64,
    pub mtp_suspend_below: f64,
    pub ewma_alpha: f64,
    pub backoff_tokens: usize,
    pub backoff_cap: usize,
    /// Part B1: continue an accepted n-gram draft's source span without a
    /// fresh key match.
    pub span_continuation: bool,
    /// Part B2: grow and shrink each arm's draft length with acceptance.
    pub adaptive_length: bool,
    /// Part B3: suspend an arm whose acceptance EWMA has collapsed.
    pub ewma_backoff: bool,
    /// Resynchronise the MTP block after every committing pass instead of
    /// only when its arm is about to draft.
    ///
    /// The lazy scheme is the shipped one; this exists so the two can be
    /// compared directly — they must produce identical tokens, and the
    /// difference in resync wall time is what laziness buys.
    pub eager_mtp_resync: bool,
}

impl Default for PolicyTuning {
    fn default() -> Self {
        Self {
            ngram_draft_floor: DEFAULT_NGRAM_DRAFT_FLOOR,
            mtp_depth_floor: DEFAULT_MTP_DEPTH_FLOOR,
            mtp_depth_start: DEFAULT_MTP_DEPTH_START,
            ngram_suspend_below: DEFAULT_NGRAM_SUSPEND_BELOW,
            mtp_suspend_below: DEFAULT_MTP_SUSPEND_BELOW,
            ewma_alpha: DEFAULT_EWMA_ALPHA,
            backoff_tokens: DEFAULT_BACKOFF_TOKENS,
            backoff_cap: DEFAULT_BACKOFF_CAP,
            span_continuation: true,
            adaptive_length: true,
            ewma_backoff: true,
            eager_mtp_resync: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_tokens: Vec<u32>,
    pub add_special_tokens: bool,
    /// Which draft sources the generation loop may use. `Off` with a non-zero
    /// legacy draft-token field below is read as that arm's single-arm mode,
    /// which is what keeps the deprecated flags working.
    pub speculative_mode: SpeculativeMode,
    pub policy: PolicyTuning,
    /// Ceiling on the MTP arm's drafting depth. Zero leaves the arm at its
    /// default cap.
    pub speculative_mtp_draft_tokens: usize,
    /// Optional raw top-1/top-2 MTP logit-margin gate. Proposals below this
    /// threshold fall back to a one-row authoritative target pass. Superseded
    /// by `mtp_min_confidence`, which is on a scale the cost model can be
    /// compared against; retained for comparability with earlier measurements.
    pub speculative_mtp_min_margin: Option<f32>,
    /// Vocabulary prefix the MTP predictor scores drafts against. Zero uses
    /// the full LM head.
    pub mtp_draft_vocab: usize,
    /// Softmax confidence below which a chained MTP draft stops extending.
    ///
    /// Unlike the margin gate this acts *inside* the drafting loop, so the
    /// tokens it declines are never drafted at all and their ~25 ms each is
    /// never paid. Zero disables it.
    pub mtp_min_confidence: f32,
    /// Ceiling on the n-gram arm's draft length. Zero leaves the arm at its
    /// default cap.
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
    /// The MTP head's own top-1 softmax probability for this draft.
    pub probability: f32,
    /// Position of this token within its chained draft, zero-based.
    pub depth: usize,
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
    /// Softmax probability the MTP head assigned to its own top-1 choice.
    ///
    /// Unlike the raw logit margin this is on a scale the cost model can be
    /// compared against directly: a drafted token is worth submitting only if
    /// the probability the target agrees exceeds
    /// `(draft_ms + row_ms) / plain_step_ms`.
    probability: f32,
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
    pub policy: QuantizedPolicyMetrics,
    pub thinking: ThinkingMetrics,
}

/// What a run had measured at the moment it ended, whether or not it produced
/// a [`QuantizedGenerationResult`].
///
/// A turn that ends through its token callback returns the callback's error
/// instead of a result, so everything the run measured would otherwise be lost
/// — and that is exactly how an agentic client ends every one of its turns, at
/// the first closed tool call. These are the fields a caller needs to report
/// such a turn as honestly as one that ran to completion; they carry the same
/// meaning as their namesakes on [`QuantizedGenerationMetrics`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PartialRunMetrics {
    pub prefill_wall_time: Duration,
    pub decode_wall_time: Duration,
    pub time_to_first_token: Duration,
    /// Tokens offered by either arm, and how many of them the target kept.
    pub drafted_tokens: usize,
    pub accepted_draft_tokens: usize,
}

/// The live half of [`PartialRunMetrics`], updated as a run proceeds.
///
/// Only what a halted turn has to report is tracked here. The full metrics are
/// still assembled by whichever generation path produced a result; this exists
/// so that the path which cannot is not left fabricating numbers.
#[derive(Debug, Clone, Default)]
struct RunProgress {
    prefill_wall_time: Duration,
    decode_started: Option<Instant>,
    decode_wall_time: Duration,
    drafted_tokens: usize,
    accepted_draft_tokens: usize,
}

impl RunProgress {
    /// Discard the previous run's measurements. Called once the turn is past
    /// the checks that can still reject it without touching the session.
    fn begin(&mut self) {
        *self = Self::default();
    }

    fn finish_prefill(&mut self, prefill_wall_time: Duration) {
        self.prefill_wall_time = prefill_wall_time;
        self.decode_started = Some(Instant::now());
    }

    /// Freeze the decode clock. Idempotent, and called both where a run ends
    /// normally and at every point a token callback can end one early.
    fn finish_decode(&mut self) {
        self.decode_wall_time = self
            .decode_started
            .map_or(Duration::ZERO, |started| started.elapsed());
    }

    fn metrics(&self) -> PartialRunMetrics {
        PartialRunMetrics {
            prefill_wall_time: self.prefill_wall_time,
            decode_wall_time: self.decode_wall_time,
            // The same convention the completed metrics use: a turn's first
            // token follows its prefill immediately.
            time_to_first_token: self.prefill_wall_time,
            drafted_tokens: self.drafted_tokens,
            accepted_draft_tokens: self.accepted_draft_tokens,
        }
    }
}

impl QuantizedGenerationMetrics {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        self.evaluated_input_tokens as f64 / self.prefill_wall_time.as_secs_f64()
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        self.generated_tokens.saturating_sub(1) as f64 / self.decode_wall_time.as_secs_f64()
    }
}

/// Everything needed to resume a sequence in a different session or process.
///
/// The model state is the bulk of it; the MTP cache and the hidden-state carry
/// are what let a restored session keep speculating instead of decoding the
/// rest of the turn unspeculated.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionImage {
    /// The exact token prefix this state represents.
    pub tokens: Vec<u32>,
    pub model: QuantizedStateImage,
    pub mtp: Option<QuantizedAttentionImage>,
    pub last_target_hidden: Option<Vec<f32>>,
}

impl SessionImage {
    pub fn position(&self) -> usize {
        self.model.position
    }

    /// Bytes of state, which is what the cache budgets against.
    pub fn bytes(&self) -> usize {
        self.model.bytes()
            + self.mtp.as_ref().map_or(0, QuantizedAttentionImage::bytes)
            + self
                .last_target_hidden
                .as_ref()
                .map_or(0, |row| row.len() * std::mem::size_of::<f32>())
            + self.tokens.len() * std::mem::size_of::<u32>()
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
    /// Position the MTP block's own sequence state has been brought up to.
    /// Under the policy this trails the target whenever tokens were committed
    /// by the n-gram arm or by plain decode; see `catch_up_mtp`.
    mtp_synced_position: usize,
    /// Tokens committed since that position, with the authoritative target
    /// hidden rows the committing passes already produced for them. Retaining
    /// the rows is what lets one batched pass close a gap of any length.
    mtp_gap_tokens: Vec<u32>,
    mtp_gap_hidden: Vec<f32>,
    /// Target hidden row for the token immediately before the gap, which is
    /// the MTP block's input at the gap's first position.
    mtp_gap_prior_hidden: Option<Vec<f32>>,
    /// Draft-only LM head covering a vocabulary prefix, built on first use.
    /// The target model always scores against its own full head.
    mtp_draft_head: Option<QuantizedMatrix>,
    /// Whether the retained gap still describes the whole distance between the
    /// synced position and the target. A turn decoded with the MTP arm off
    /// stops retaining rows and clears this.
    mtp_gap_valid: bool,
    trace: Option<Box<dyn RoutingTrace>>,
    /// Rollback snapshots for multi-row verification, allocated on first use
    /// and reused for the lifetime of the session.
    snapshots: QuantizedStateSnapshots,
    /// n-gram index over the tokens in context. Maintained only while the
    /// n-gram drafter is in use; it never affects decoding correctness.
    ngram: NgramIndex,
    /// What the run in progress has measured, so that a turn ended by its
    /// token callback still has real numbers to report. See
    /// [`Self::last_run_metrics`].
    run: RunProgress,
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
            mtp_synced_position: 0,
            mtp_gap_tokens: Vec::new(),
            mtp_gap_hidden: Vec::new(),
            mtp_gap_prior_hidden: None,
            mtp_gap_valid: true,
            mtp_draft_head: None,
            trace: None,
            snapshots,
            ngram: NgramIndex::new(),
            run: RunProgress::default(),
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

    /// What the last run measured, including one that ended in its token
    /// callback and therefore returned an error rather than a result.
    ///
    /// The fields carry the same meaning as their namesakes on
    /// [`QuantizedGenerationMetrics`], so a caller reporting a halted turn
    /// beside a completed one is comparing like with like.
    pub fn last_run_metrics(&self) -> PartialRunMetrics {
        self.run.metrics()
    }

    /// Leave `token` pending and stop the run's decode clock.
    ///
    /// Every path that ends a turn through the output sink goes through here,
    /// which is what makes the timings a halted turn reports its own rather
    /// than something the caller has to reconstruct from outside.
    fn halt_with_pending(&mut self, token: u32) {
        self.pending_token = Some(token);
        self.run.finish_decode();
    }

    pub fn reset(&mut self) {
        self.state = self.model.new_state();
        self.pending_token = None;
        self.mtp_state = self.model.mtp().map(|mtp| mtp.new_state());
        self.last_target_hidden = None;
        self.mtp_synced_position = 0;
        self.mtp_gap_tokens.clear();
        self.mtp_gap_hidden.clear();
        self.mtp_gap_prior_hidden = None;
        self.mtp_gap_valid = true;
        self.ngram.clear();
    }

    /// Evaluate `tokens` through the target without sampling, leaving the
    /// session at the boundary immediately after them.
    ///
    /// This is how a caller reaches a chosen prefix boundary: everything the
    /// generation loop does to keep the n-gram index, the hidden-state carry
    /// and the retained MTP gap aligned happens here too, so a turn continued
    /// afterwards is indistinguishable from one that prefilled in a single
    /// pass. `track_mtp` retains the hidden rows the MTP arm would need; it
    /// costs `hidden_size` floats per token and is pointless when that arm
    /// will not run.
    pub fn prefill_tokens(&mut self, tokens: &[u32], track_mtp: bool) -> Result<()> {
        ensure!(!tokens.is_empty(), "prefill requires at least one token");
        ensure!(
            self.trace.is_none(),
            "prefill_tokens does not emit routing traces"
        );
        let carried_token = self.pending_token.take();
        let context_before = self.state.position + usize::from(carried_token.is_some());
        let evaluated = continuation_input(tokens, carried_token);
        if self.ngram.len() == context_before {
            self.ngram.extend(tokens);
        } else {
            self.ngram.clear();
            self.ngram.extend(&evaluated);
        }
        let output = self
            .model
            .forward_detailed_logits(&evaluated, &mut self.state, None, LogitRows::Last)
            .context("prefill pass failed")?;
        self.mtp_gap_valid = track_mtp && self.mtp_gap_valid;
        if self.mtp_gap_valid {
            let prior = self.last_target_hidden.clone();
            self.retain_mtp_gap(&evaluated, &output.normalized_hidden, prior.as_deref())?;
        } else {
            self.mtp_gap_tokens.clear();
            self.mtp_gap_hidden.clear();
            self.mtp_gap_prior_hidden = None;
        }
        self.last_target_hidden = Some(last_hidden_row(&output.normalized_hidden)?);
        // Bring the predictor up to the same position rather than leaving the
        // gap for a later draft to close. The lazy scheme is right during
        // generation, where the arm may never draft; here it would mean the
        // session cannot be imaged with its predictor state, and every
        // restored request would decode with the MTP arm sitting out.
        if self.mtp_gap_valid && self.mtp_state.is_some() {
            let mut discarded = QuantizedPolicyMetrics::default();
            self.catch_up_mtp(&mut discarded)
                .context("failed to synchronise the MTP predictor after prefill")?;
        }
        Ok(())
    }

    /// Whether the MTP arm can draft from the state this session now holds.
    ///
    /// The arm needs the retained gap to still describe the whole distance
    /// between the predictor's synced position and the target's: without the
    /// committing passes' hidden rows there is nothing to catch up from. A
    /// turn decoded with the arm off stops retaining them, so the arm sits out
    /// until the session is reset or replaced by an image that carried the
    /// predictor's own cache.
    pub fn mtp_arm_ready(&self) -> bool {
        self.mtp_gap_valid
            && self.mtp_synced_position + self.mtp_gap_tokens.len() == self.state.position
            && (self.state.position == 0 || self.last_target_hidden.is_some())
    }

    /// Whether an image taken now would carry the MTP predictor's own cache.
    ///
    /// This is the condition [`Self::session_image`] applies internally, asked
    /// separately so a caller that needs the predictor's state can find out
    /// before it pays for the image: an image is a deep copy of the KV cache
    /// and every layer's recurrent state, which is hundreds of megabytes at
    /// agent depths, and a caller that would discard it should not build it.
    pub fn images_mtp_state(&self) -> bool {
        self.mtp_state.is_some() && self.mtp_synced_position == self.state.position
    }

    /// Copy everything a later session needs to continue this sequence.
    ///
    /// `tokens` is the exact token prefix the state represents; passing a
    /// different length is rejected rather than stored, because a restored
    /// image whose tokens do not describe its state would silently decode
    /// against the wrong context. There must be no unevaluated pending token,
    /// which is why images are taken after [`Self::prefill_tokens`] rather
    /// than after generation.
    pub fn session_image(&self, tokens: Vec<u32>) -> Result<SessionImage> {
        ensure!(
            self.pending_token.is_none(),
            "cannot image a session with an unevaluated pending token"
        );
        ensure!(
            tokens.len() == self.state.position,
            "session holds {} tokens but {} were supplied",
            self.state.position,
            tokens.len()
        );
        let mtp = self
            .mtp_state
            .as_ref()
            .filter(|_| self.images_mtp_state())
            .map(QuantizedMtpState::image);
        Ok(SessionImage {
            tokens,
            model: self.state.image(),
            mtp,
            last_target_hidden: self.last_target_hidden.clone(),
        })
    }

    /// Adopt a previously captured session, discarding whatever this one held.
    ///
    /// The MTP arm resumes only when the image carried the predictor's own
    /// cache at the same position; otherwise the arm sits out this session, as
    /// it does after any pass that stopped retaining its gap.
    pub fn restore_session(&mut self, image: &SessionImage) -> Result<()> {
        ensure!(
            image.tokens.len() == image.model.position,
            "session image holds {} tokens for position {}",
            image.tokens.len(),
            image.model.position
        );
        self.state.restore_image(&image.model)?;
        self.pending_token = None;
        self.mtp_gap_tokens.clear();
        self.mtp_gap_hidden.clear();
        self.mtp_gap_prior_hidden = None;
        self.last_target_hidden = image.last_target_hidden.clone();
        let mtp_restored = match (self.model.mtp(), &image.mtp, self.mtp_state.as_mut()) {
            (Some(_), Some(saved), Some(state)) => {
                state.restore_image(saved)?;
                ensure!(
                    state.position() == image.model.position,
                    "MTP image is at position {} but the target image is at {}",
                    state.position(),
                    image.model.position
                );
                true
            }
            _ => {
                self.mtp_state = self.model.mtp().map(|mtp| mtp.new_state());
                false
            }
        };
        self.mtp_synced_position = if mtp_restored {
            image.model.position
        } else {
            0
        };
        self.mtp_gap_valid = mtp_restored && self.last_target_hidden.is_some();
        // Seeding the index costs one hash per token and keeps the n-gram arm
        // as strong on a restored prefix as it is on a prefilled one.
        self.ngram.clear();
        self.ngram.extend(&image.tokens);
        Ok(())
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
        // Only the last row's logits are ever read here: this serves the plain
        // path's prefill and its one-token decodes, and both sample the tail.
        let output = match self.trace.as_mut() {
            Some(trace) => model.forward_detailed_logits(
                tokens,
                state,
                Some(trace.as_mut()),
                LogitRows::Last,
            )?,
            None => model.forward_detailed_logits(tokens, state, None, LogitRows::Last)?,
        };
        Ok((output.logits, output.timings))
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
                self.halt_with_pending(token);
                return Err(error);
            }
            let (next, profile) = self.forward(&[token])?;
            decode_profile.accumulate(&profile);
            logits = Some(next);
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
        on_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<QuantizedGenerationResult> {
        let prompt_token_ids = self.tokenizer.encode(prompt, options.add_special_tokens)?;
        self.generate_tokens_with_callback(&prompt_token_ids, options, on_token)
    }

    /// Continue the session from already-encoded tokens.
    ///
    /// This is the entry point a caller uses when it decides where the prompt
    /// begins, which is what prefix reuse needs: the tokens passed here are the
    /// ones the state has not seen yet, not the whole conversation.
    pub fn generate_tokens_with_callback(
        &mut self,
        prompt_token_ids: &[u32],
        options: &GenerationOptions,
        mut on_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<QuantizedGenerationResult> {
        ensure!(
            options.speculative_mode != SpeculativeMode::Off
                || options.speculative_mtp_draft_tokens == 0
                || options.speculative_ngram_draft_tokens == 0,
            "the deprecated --speculative-ngram and --speculative-mtp flags select \
             single-arm modes and cannot be combined; use --speculative auto"
        );
        let mode = effective_speculative_mode(options);
        if mode.is_speculative() {
            return self.generate_speculative_policy(prompt_token_ids, options, mode, on_token);
        }
        ensure!(
            options.max_new_tokens > 0,
            "max_new_tokens must be at least one"
        );
        let prompt_token_ids = prompt_token_ids.to_vec();
        ensure!(
            !prompt_token_ids.is_empty(),
            "tokenizer produced an empty prompt"
        );
        let evaluated_input_token_ids =
            continuation_input(&prompt_token_ids, self.pending_token.take());

        let mut sampler = Sampler::new(options.sampling.clone())?;
        let cache_before = self.model.expert_cache_stats()?;
        self.run.begin();
        let prefill_started = Instant::now();
        let (mut logits, prefill_profile) = self.forward(&evaluated_input_token_ids)?;
        let prefill_wall_time = prefill_started.elapsed();
        let decode_started = Instant::now();
        self.run.finish_prefill(prefill_wall_time);
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
                self.halt_with_pending(token);
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
        self.run.finish_decode();
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
                policy: QuantizedPolicyMetrics::default(),
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

    /// Bring the MTP block's sequence state up to the target's position.
    ///
    /// Tokens are committed by three different passes under the policy and
    /// only one of them is an MTP pass, so the block's state lags whenever the
    /// n-gram arm or plain decode is doing the committing. Catching up is
    /// deferred until the arm is actually about to draft: while it is
    /// suspended its state may lag arbitrarily far at no cost at all, which is
    /// the whole point of suspending it.
    ///
    /// The retained gap rows are the authoritative hidden rows the committing
    /// passes already produced, so a gap of any length closes in one batched
    /// pass rather than one pass per token.
    fn catch_up_mtp(&mut self, policy: &mut QuantizedPolicyMetrics) -> Result<usize> {
        let rows = self.mtp_gap_tokens.len();
        if rows == 0 {
            return Ok(0);
        }
        let hidden_size = self.model.config().hidden_size;
        let tokens = std::mem::take(&mut self.mtp_gap_tokens);
        let hidden = std::mem::take(&mut self.mtp_gap_hidden);
        let prior = self.mtp_gap_prior_hidden.take();
        ensure!(
            hidden.len() == rows * hidden_size,
            "retained MTP gap holds {} hidden values for {rows} tokens",
            hidden.len()
        );
        let target_hidden = Tensor::from_vec(hidden, (rows, hidden_size), &Device::Cpu)?;
        let started = Instant::now();
        let timings = self.synchronize_mtp(
            self.mtp_synced_position,
            prior.as_deref(),
            &tokens,
            &target_hidden,
        )?;
        policy.resync_wall_time += started.elapsed();
        policy.resync_passes += 1;
        policy.resync_tokens += rows;
        policy.max_resync_tokens = policy.max_resync_tokens.max(rows);
        policy.resync_profile.accumulate(&timings);
        self.mtp_synced_position += rows;
        Ok(rows)
    }

    /// Drop the retained gap without touching the predictor's own state.
    fn clear_mtp_gap(&mut self) {
        self.mtp_gap_tokens.clear();
        self.mtp_gap_hidden.clear();
        self.mtp_gap_prior_hidden = None;
    }

    /// Bring the retained MTP gap back into step with a target state that has
    /// just been rolled back to `position`.
    ///
    /// The gap describes exactly the tokens between the predictor's synced
    /// position and the target's, so a rollback that drops committed rows has
    /// to drop the gap entries those rows contributed — and nothing else. The
    /// alternative, abandoning the gap, ends the arm for the rest of the
    /// session rather than for the rest of the pass: `mtp_arm_ready` is what
    /// `arm_configs` gates the arm on, an arm gated off retains no rows, and a
    /// gap that is never retained can never be closed. One rollback would cost
    /// every later turn its speculation.
    ///
    /// Nothing here changes a committed token: the retained rows the gap keeps
    /// are the same authoritative rows it held before, in the same order, and
    /// the predictor still reads them through the same catch-up pass.
    fn truncate_mtp_gap(&mut self, position: usize) -> Result<()> {
        match gap_after_rollback(
            self.mtp_synced_position,
            self.mtp_gap_tokens.len(),
            position,
        ) {
            GapAfterRollback::Retain(retained) => {
                let hidden_size = self.model.config().hidden_size;
                self.mtp_gap_tokens.truncate(retained);
                self.mtp_gap_hidden.truncate(retained * hidden_size);
                if retained == 0 {
                    // The prior row describes the token before the gap's first
                    // one, and there is no longer a first one; the next
                    // retained row supplies it again.
                    self.mtp_gap_prior_hidden = None;
                }
            }
            GapAfterRollback::TruncatePredictor => {
                // The predictor synced past the rolled-back point, which only
                // `eager_mtp_resync` arranges: it catches up after a pass has
                // committed its rows, and the rollback then unwinds some of
                // them. Its rows for those positions were computed from hidden
                // rows that no longer describe the sequence, so they are
                // truncated away — the same operation every catch-up performs
                // before it writes, and equally exact, because the block's
                // cache is append-only.
                if let Some(state) = self.mtp_state.as_mut() {
                    state.truncate(position)?;
                }
                self.mtp_synced_position = position;
                self.clear_mtp_gap();
            }
            GapAfterRollback::Unrecoverable => {
                tracing::debug!(
                    position,
                    synced = self.mtp_synced_position,
                    gap = self.mtp_gap_tokens.len(),
                    "the retained MTP gap does not reach the rolled-back position, so the \
                     MTP arm sits out until this session is reset or restored"
                );
                self.mtp_gap_valid = false;
                self.clear_mtp_gap();
            }
        }
        Ok(())
    }

    /// Retain a committing pass's tokens and hidden rows for the next catch-up.
    fn retain_mtp_gap(
        &mut self,
        tokens: &[u32],
        hidden: &Tensor,
        prior_hidden: Option<&[f32]>,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let hidden_size = self.model.config().hidden_size;
        ensure!(
            hidden.dims() == [tokens.len(), hidden_size],
            "committed hidden rows have shape {:?}, expected [{}, {hidden_size}]",
            hidden.shape(),
            tokens.len()
        );
        if self.mtp_gap_tokens.is_empty() {
            self.mtp_gap_prior_hidden = prior_hidden.map(<[f32]>::to_vec);
        }
        self.mtp_gap_tokens.extend_from_slice(tokens);
        for row in hidden.to_vec2::<f32>()? {
            self.mtp_gap_hidden.extend_from_slice(&row);
        }
        Ok(())
    }

    /// Sample the next greedy token, commit it to the transcript and the
    /// index, and hand it to the output sink. Returns the new pending token,
    /// or `None` when the turn's token budget is already spent.
    fn sample_and_emit(
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

    /// Evaluate one authoritative token through the target, keeping the n-gram
    /// index, the hidden-state carry and the retained MTP gap aligned with it.
    fn evaluate_authoritative(
        &mut self,
        token: u32,
        decode_profile: &mut QuantizedForwardTimings,
        track_mtp: bool,
    ) -> Result<Tensor> {
        let prior_hidden = self.last_target_hidden.clone();
        let output = self.model.forward_detailed(&[token], &mut self.state)?;
        decode_profile.accumulate(&output.timings);
        if track_mtp {
            self.retain_mtp_gap(&[token], &output.normalized_hidden, prior_hidden.as_deref())?;
        }
        self.last_target_hidden = Some(last_hidden_row(&output.normalized_hidden)?);
        Ok(output.logits)
    }

    /// Force-close the thinking block through the target model. Injected
    /// tokens travel the same authoritative path as any other committed token,
    /// so index, hidden carry and MTP gap stay in step across the closure.
    fn force_close_thinking_policy(
        &mut self,
        pending_token: Option<u32>,
        closure: &[u32],
        generated: &mut Vec<u32>,
        decode_profile: &mut QuantizedForwardTimings,
        track_mtp: bool,
        on_token: &mut impl FnMut(u32) -> Result<()>,
    ) -> Result<Tensor> {
        let mut logits = None;
        if let Some(token) = pending_token {
            logits = Some(self.evaluate_authoritative(token, decode_profile, track_mtp)?);
        }
        for &token in closure {
            generated.push(token);
            self.ngram.push(token);
            if let Err(error) = on_token(token) {
                self.halt_with_pending(token);
                return Err(error);
            }
            logits = Some(self.evaluate_authoritative(token, decode_profile, track_mtp)?);
        }
        logits.context("forced thinking closure produced no target logits")
    }

    /// Greedy decoding under the unified speculative policy.
    ///
    /// Per step the loop takes the first of three options that applies: a free
    /// n-gram proposal when the index holds literal evidence and that arm is
    /// not suspended; an MTP draft when that arm is not suspended; otherwise
    /// exactly the one-row pass unspeculated decoding would run. The two arms
    /// never both run in one step, and an n-gram draft is never extended with
    /// MTP tokens.
    ///
    /// Committed tokens are always the target model's own greedy choices: a
    /// draft token is committed only where the target's argmax at that row
    /// equals it, and the token after the last accepted row comes from the
    /// target. That is what makes every mode here token-identical to
    /// target-only decoding; speculation changes speed, never output.
    fn generate_speculative_policy(
        &mut self,
        prompt_token_ids: &[u32],
        options: &GenerationOptions,
        mode: SpeculativeMode,
        mut on_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<QuantizedGenerationResult> {
        ensure!(
            options.max_new_tokens > 0,
            "max_new_tokens must be at least one"
        );
        ensure!(
            options.sampling.temperature == 0.,
            "speculative decoding currently requires temperature=0"
        );
        ensure!(
            self.trace.is_none(),
            "speculative decoding does not support routing traces"
        );
        ensure!(
            (crate::ngram::MIN_MATCH_LEN..=crate::ngram::MAX_MATCH_LEN)
                .contains(&options.ngram_min_match),
            "n-gram minimum match length must be between {} and {}",
            crate::ngram::MIN_MATCH_LEN,
            crate::ngram::MAX_MATCH_LEN
        );
        let has_mtp_block = self.model.mtp().is_some() && self.mtp_state.is_some();
        ensure!(
            has_mtp_block || mode != SpeculativeMode::Mtp,
            "the loaded model does not contain a supported MTP predictor"
        );

        let prompt_token_ids = prompt_token_ids.to_vec();
        ensure!(
            !prompt_token_ids.is_empty(),
            "tokenizer produced an empty prompt"
        );
        // Checked before anything is consumed, so a rejected turn leaves the
        // session exactly as it was rather than losing its pending token.
        let mtp_aligned = self.mtp_arm_ready();
        ensure!(
            mtp_aligned || mode != SpeculativeMode::Mtp,
            "MTP state is not aligned with the target; reset before enabling speculation"
        );

        let carried_token = self.pending_token.take();
        let context_before = self.state.position + usize::from(carried_token.is_some());
        let evaluated_input_token_ids = continuation_input(&prompt_token_ids, carried_token);

        let (ngram_config, mtp_config) = arm_configs(options, mode, has_mtp_block && mtp_aligned);
        let track_mtp = mtp_config.enabled;
        // Build the draft-only LM head once per turn, and only when the arm
        // that uses it is live. The target model keeps its own full head; this
        // one is never on a path whose output is committed without
        // verification.
        let draft_vocab = if mtp_config.enabled {
            self.model
                .mtp()
                .map(|mtp| {
                    let full = mtp.vocab_size();
                    match options.mtp_draft_vocab {
                        0 => full,
                        requested => requested.min(full),
                    }
                })
                .unwrap_or(0)
        } else {
            0
        };
        let full_vocab = self.model.mtp().map_or(0, |mtp| mtp.vocab_size());
        if draft_vocab == 0 || draft_vocab >= full_vocab {
            self.mtp_draft_head = None;
        } else if self
            .mtp_draft_head
            .as_ref()
            .is_none_or(|head| head.shape()[0] != draft_vocab)
        {
            let head = self
                .model
                .mtp()
                .context("MTP availability validated")?
                .draft_head(draft_vocab)?;
            self.mtp_draft_head = Some(head);
        }
        self.mtp_gap_valid = track_mtp;
        let mut ngram_arm = ArmController::new(ngram_config);
        let mut mtp_arm = ArmController::new(mtp_config);
        let span_continuation = options.policy.span_continuation;

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
        self.run.begin();
        let prefill_started = Instant::now();
        let prefill = self.model.forward_detailed_logits(
            &evaluated_input_token_ids,
            &mut self.state,
            None,
            LogitRows::Last,
        )?;
        if track_mtp {
            let prior = self.last_target_hidden.clone();
            self.retain_mtp_gap(
                &evaluated_input_token_ids,
                &prefill.normalized_hidden,
                prior.as_deref(),
            )?;
        }
        self.last_target_hidden = Some(last_hidden_row(&prefill.normalized_hidden)?);
        let prefill_wall_time = prefill_started.elapsed();
        let prefill_profile = prefill.timings;
        let mut logits = prefill.logits;

        let mut sampler = Sampler::new(options.sampling.clone())?;
        let decode_started = Instant::now();
        self.run.finish_prefill(prefill_wall_time);
        let mut decode_profile = QuantizedForwardTimings::default();
        // A disabled arm reports a zero cap, which is what keeps its section
        // out of the run report entirely.
        let ngram_cap = usize::from(ngram_arm.config().enabled) * ngram_arm.config().cap;
        let mtp_cap = usize::from(mtp_arm.config().enabled) * mtp_arm.config().cap;
        let mut ngram = QuantizedNgramMetrics::new(ngram_cap, options.ngram_min_match);
        let mut speculative = QuantizedSpeculativeMetrics {
            max_draft_tokens: mtp_cap,
            ..Default::default()
        };
        let mut policy = QuantizedPolicyMetrics {
            mode,
            draft_vocab: self
                .mtp_draft_head
                .as_ref()
                .map_or(full_vocab, |h| h.shape()[0]),
            full_vocab,
            ..Default::default()
        };
        let mut generated = Vec::with_capacity(options.max_new_tokens);
        let mut thinking =
            ThinkingBudget::from_tokenizer(&self.tokenizer, options.thinking_budget)?;
        let mut span: Option<SpanCursor> = None;

        if thinking
            .as_mut()
            .is_some_and(ThinkingBudget::should_force_before_sampling)
        {
            let closure = thinking
                .as_ref()
                .expect("thinking budget exists")
                .forced_close_tokens()
                .to_vec();
            logits = self.force_close_thinking_policy(
                None,
                &closure,
                &mut generated,
                &mut decode_profile,
                track_mtp,
                &mut on_token,
            )?;
        }
        let mut pending_token = self.sample_and_emit(
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
                let closed = self.force_close_thinking_policy(
                    Some(first),
                    &closure,
                    &mut generated,
                    &mut decode_profile,
                    track_mtp,
                    &mut on_token,
                )?;
                pending_token = self.sample_and_emit(
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
            // tokens, so a boundary that stopped the emission loop early would
            // leave evaluated-but-unemitted tokens in the state. Holding every
            // draft — n-gram, span continuation or MTP — to what the turn and
            // the thinking budget can still absorb means those boundaries can
            // only land on the last accepted draft token or on the
            // authoritative one, both of which end a pass cleanly.
            let remaining = options.max_new_tokens - generated.len();
            let mut budget = remaining.saturating_sub(1);
            if let Some(thinking_remaining) = thinking.as_ref().and_then(ThinkingBudget::remaining)
            {
                budget = budget.min(thinking_remaining);
            }
            let committed_before = generated.len();
            policy.steps += 1;
            ngram.steps += 1;

            let ngram_available = ngram_arm.poll(committed_before);
            let mtp_available = mtp_arm.poll(committed_before);

            // Ask the index in every mode, even when the arm cannot use the
            // answer: a lookup is a slice comparison, and recording whether
            // literal evidence existed is what makes MTP acceptance
            // conditioned on that evidence measurable from a single-arm run.
            let lookup_started = Instant::now();
            let requested = if ngram_available {
                ngram_arm.draft_len().min(budget).max(1)
            } else {
                1
            };
            let key_draft = self
                .ngram
                .draft(requested, options.ngram_min_match, |token| {
                    self.is_stop_token(token, options)
                });
            let span_draft = (span_continuation && ngram_available && budget > 0)
                .then(|| {
                    span.and_then(|cursor| {
                        self.ngram.continue_from(
                            cursor.next,
                            ngram_arm.draft_len().min(budget),
                            cursor.match_len,
                            |token| self.is_stop_token(token, options),
                        )
                    })
                })
                .flatten();
            policy.lookup_wall_time += lookup_started.elapsed();
            let ngram_match_len = key_draft.as_ref().map_or(0, |draft| draft.match_len);
            if ngram_match_len > 0 {
                policy.steps_with_ngram_match += 1;
                ngram.steps_with_match += 1;
            } else {
                ngram.steps_without_match += 1;
            }

            // Arm selection. An active span continues without asking for a
            // fresh key; otherwise a key match at or above the floor takes the
            // step outright, and only a step neither of those claims may reach
            // the MTP arm.
            let mut arm = StepArm::Plain;
            let mut proposal: Option<NgramDraft> = None;
            if ngram_available && budget > 0 {
                if let Some(draft) = span_draft {
                    arm = StepArm::NgramSpan;
                    proposal = Some(draft);
                } else if let Some(draft) = key_draft {
                    arm = StepArm::Ngram;
                    proposal = Some(draft);
                }
            }
            let mut drafts = proposal
                .as_ref()
                .map(|draft| draft.tokens.clone())
                .unwrap_or_default();
            let mut mtp_candidates: Vec<MtpDraftCandidate> = Vec::new();
            let mut step_resync = 0;
            if arm == StepArm::Plain && mtp_available && budget > 0 {
                let depth = mtp_arm.draft_len().min(budget);
                if depth > 0 {
                    step_resync = self.catch_up_mtp(&mut policy)?;
                    let target_position = self.state.position;
                    debug_assert_eq!(
                        self.mtp_synced_position, target_position,
                        "the MTP arm drafted from a stale position"
                    );
                    ensure!(
                        self.mtp_state
                            .as_ref()
                            .is_some_and(|state| state.position() == target_position),
                        "MTP state position does not match the target before drafting"
                    );
                    let prior_hidden = self
                        .last_target_hidden
                        .clone()
                        .context("target hidden-state carry is missing")?;
                    let draft_started = Instant::now();
                    // The next catch-up truncates the block back to its synced
                    // position, so a failed draft leaves nothing to unwind.
                    let (candidates, draft_profile, stopped_early) =
                        self.draft_mtp(seed, &prior_hidden, depth, options)?;
                    if stopped_early {
                        policy.confidence_stops += 1;
                    }
                    policy.drafted_tokens += candidates.len() + usize::from(stopped_early);
                    policy.draft_wall_time += draft_started.elapsed();
                    speculative.draft_wall_time += draft_started.elapsed();
                    speculative.draft_profile.accumulate(&draft_profile);
                    speculative.drafted_tokens += candidates.len();
                    self.run.drafted_tokens += candidates.len();
                    let verified_count =
                        options
                            .speculative_mtp_min_margin
                            .map_or(candidates.len(), |threshold| {
                                candidates
                                    .iter()
                                    .take_while(|draft| draft.logit_margin >= threshold)
                                    .count()
                            });
                    speculative.gated_tokens += candidates.len() - verified_count;
                    drafts = candidates[..verified_count]
                        .iter()
                        .map(|draft| draft.token)
                        .collect();
                    mtp_candidates = candidates;
                    if !drafts.is_empty() {
                        arm = StepArm::Mtp;
                    }
                }
            }

            let mut verification_tokens = Vec::with_capacity(1 + drafts.len());
            verification_tokens.push(seed);
            verification_tokens.extend_from_slice(&drafts);
            // With no proposal this is the ordinary one-row decode pass, which
            // has no row boundary to roll back to and takes no snapshots.
            let snapshotted = !drafts.is_empty();
            let prior_hidden = self.last_target_hidden.clone();
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
                policy.verification_passes += 1;
                policy.verification_tokens += verification_tokens.len();
                policy.verification_wall_time += pass_wall_time;
                policy.snapshot_wall_time += snapshot_time(&verified.timings);
                if arm.is_ngram() {
                    ngram.verification_passes += 1;
                    ngram.verification_tokens += verification_tokens.len();
                    ngram.verification_wall_time += pass_wall_time;
                    ngram.snapshot_wall_time += snapshot_time(&verified.timings);
                    ngram.snapshot_rows += verification_tokens.len();
                    if ngram.snapshot_bytes_per_row == 0 {
                        ngram.snapshot_bytes_per_row = self.snapshots.bytes_per_row();
                    }
                } else {
                    speculative.verification_passes += 1;
                    speculative.verification_tokens += verification_tokens.len();
                    speculative.verification_wall_time += pass_wall_time;
                    speculative.checkpoint_wall_time += snapshot_time(&verified.timings);
                }
            } else {
                policy.plain_wall_time += pass_wall_time;
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

            // Both arms' proposals and acceptances are also counted into the
            // session's run progress as they happen, so a turn that never
            // reaches its metrics can still say what it drafted.
            self.run.accepted_draft_tokens += accepted;

            match arm {
                StepArm::Ngram | StepArm::NgramSpan => {
                    policy.ngram_steps += 1;
                    if arm == StepArm::NgramSpan {
                        policy.ngram_span_steps += 1;
                    }
                    self.run.drafted_tokens += drafts.len();
                    ngram.record_draft(
                        proposal
                            .as_ref()
                            .expect("an n-gram arm step has a proposal"),
                        accepted,
                    );
                    ngram_arm.observe(drafts.len(), accepted, committed_before + committed_rows);
                }
                StepArm::Mtp => {
                    policy.mtp_steps += 1;
                    speculative.accepted_tokens += accepted;
                    speculative
                        .draft_observations
                        .extend(mtp_candidates.iter().enumerate().map(|(index, draft)| {
                            QuantizedDraftObservation {
                                logit_margin: draft.logit_margin,
                                probability: draft.probability,
                                depth: index,
                                accepted: index < accepted,
                                gated: index >= drafts.len(),
                            }
                        }));
                    if ngram_match_len > 0 {
                        policy.mtp_proposed_on_ngram_match += drafts.len();
                        policy.mtp_accepted_on_ngram_match += accepted;
                    } else {
                        policy.mtp_proposed_on_ngram_miss += drafts.len();
                        policy.mtp_accepted_on_ngram_miss += accepted;
                    }
                    mtp_arm.observe(drafts.len(), accepted, committed_before + committed_rows);
                }
                StepArm::Plain => policy.plain_steps += 1,
            }

            // The rows a verification pass computed for the committed prefix
            // are authoritative — row r saw exactly the prefix a sequential
            // decode would have fed it — so a rejection rolls the recurrent
            // state back to the row boundary and keeps the rows.
            let committed_hidden = if accepted == drafts.len() {
                verified.normalized_hidden
            } else {
                let rollback_started = Instant::now();
                self.state.rollback(&self.snapshots, committed_rows)?;
                let elapsed = rollback_started.elapsed();
                policy.rollback_wall_time += elapsed;
                policy.rollbacks += 1;
                if arm.is_ngram() {
                    ngram.rollback_wall_time += elapsed;
                    ngram.rollbacks += 1;
                } else {
                    speculative.restore_wall_time += elapsed;
                }
                verified.normalized_hidden.narrow(0, 0, committed_rows)?
            };
            if track_mtp {
                self.retain_mtp_gap(
                    &verification_tokens[..committed_rows],
                    &committed_hidden,
                    prior_hidden.as_deref(),
                )?;
                if options.policy.eager_mtp_resync {
                    self.catch_up_mtp(&mut policy)?;
                }
            }
            self.last_target_hidden = Some(last_hidden_row(&committed_hidden)?);

            // Chain the source span only while it is still describing what the
            // target decoded: every drafted token was accepted, and the
            // authoritative token after them is the one the span predicts next.
            // Anything else — a rejection, a step this arm did not win, or a
            // span that has run off the end of the index — ends the chain.
            span = match arm {
                StepArm::Ngram | StepArm::NgramSpan if span_continuation => {
                    let draft = proposal
                        .as_ref()
                        .expect("an n-gram arm step has a proposal");
                    // The index has not yet been given this pass's committed
                    // tokens, so `follow` can only address the source region:
                    // a span that would continue into the tokens this very
                    // pass appends is left to a fresh key match instead.
                    let follow = draft.source_position + 1 + draft.tokens.len();
                    chain_span(
                        span,
                        draft.source_position,
                        draft.tokens.len(),
                        accepted,
                        draft.match_len,
                        self.ngram.tokens().get(follow).copied(),
                        authoritative,
                    )
                }
                _ => None,
            };

            policy.records.push(PolicyStepRecord {
                step: policy.steps,
                committed_before,
                arm,
                proposed: drafts.len(),
                accepted,
                ngram_len: ngram_arm.draft_len(),
                mtp_depth: mtp_arm.draft_len(),
                ngram_ewma: ngram_arm.ewma() as f32,
                mtp_ewma: mtp_arm.ewma() as f32,
                ngram_suspended: ngram_arm.is_suspended(),
                mtp_suspended: mtp_arm.is_suspended(),
                ngram_match_len,
                resync_tokens: step_resync,
            });

            let outputs = speculative_committed_outputs(&drafts, accepted, authoritative);
            for (output_index, (token, accepted_draft)) in outputs.into_iter().enumerate() {
                generated.push(token);
                self.ngram.push(token);
                if let Err(error) = on_token(token) {
                    // Leave the session at the boundary immediately before the
                    // token whose output callback failed, with that token
                    // pending, exactly as ordinary generation does.
                    //
                    // Only a halt inside the accepted drafts has rows to
                    // unwind. A halt on the pass's authoritative token asks
                    // for the boundary the state already sits at — a fully
                    // accepted pass never left it, and a partially rejected
                    // one was rolled back to it above — and rolling back to
                    // where the state already is copies every linear layer's
                    // recurrent state for nothing.
                    let evaluated = halted_pass_rows(output_index, accepted);
                    if evaluated < committed_rows {
                        self.state.rollback(&self.snapshots, evaluated)?;
                        // The gap entries this pass appended for the unwound
                        // rows go with them, and the rest of the gap stays: the
                        // arm survives a halted turn instead of sitting out
                        // every turn after it.
                        if self.mtp_gap_valid {
                            self.truncate_mtp_gap(self.state.position)?;
                        }
                        // The carry has to describe the row at the position the
                        // session now holds, not the last row the pass
                        // committed before the rollback.
                        self.last_target_hidden =
                            Some(nth_hidden_row(&committed_hidden, evaluated - 1)?);
                    }
                    self.halt_with_pending(token);
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
                    let closed = self.force_close_thinking_policy(
                        (!accepted_draft).then_some(token),
                        &closure,
                        &mut generated,
                        &mut decode_profile,
                        track_mtp,
                        &mut on_token,
                    )?;
                    pending_token = self.sample_and_emit(
                        &closed,
                        options,
                        &mut sampler,
                        &mut generated,
                        &mut on_token,
                    )?;
                    span = None;
                    continue 'generation;
                }
                if generated.len() == options.max_new_tokens || is_stop {
                    break 'generation;
                }
            }
        }

        self.pending_token = pending_token;
        let decode_wall_time = decode_started.elapsed();
        self.run.finish_decode();
        policy.ngram_arm = ngram_arm.stats().clone();
        policy.mtp_arm = mtp_arm.stats().clone();
        speculative.resync_wall_time = policy.resync_wall_time;
        speculative.resync_profile = policy.resync_profile.clone();
        ngram.lookup_wall_time = policy.lookup_wall_time;
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
                ngram,
                policy,
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
    ) -> Result<(Vec<MtpDraftCandidate>, QuantizedMtpTimings, bool)> {
        let hidden_size = self.model.config().hidden_size;
        ensure!(
            target_hidden.len() == hidden_size,
            "MTP seed hidden row has {} values, expected {hidden_size}",
            target_hidden.len()
        );
        let draft_head = self.mtp_draft_head.as_ref();
        let mut token = seed;
        let mut hidden = target_hidden.to_vec();
        let mut drafts = Vec::with_capacity(max_drafts);
        let mut timings = QuantizedMtpTimings::default();
        let mut stopped_early = false;
        for _ in 0..max_drafts {
            let input = Tensor::from_vec(hidden, (1, hidden_size), &Device::Cpu)?;
            let output = self
                .model
                .mtp()
                .expect("MTP availability validated")
                .forward_with_head(
                    &[token],
                    &input,
                    self.mtp_state.as_mut().expect("MTP availability validated"),
                    true,
                    draft_head,
                )?;
            timings.accumulate(&output.timings);
            let logits = output
                .logits
                .context("MTP draft forward did not produce logits")?
                .i(0)?
                .to_vec1::<f32>()?;
            let (draft, logit_margin, probability) = top1_with_scores(&logits)?;
            let draft = draft as u32;
            hidden = output.normalized_hidden.i(0)?.to_vec1::<f32>()?;
            if self.is_stop_token(draft, options) {
                break;
            }
            // Stop the chain rather than submit a token the target is more
            // likely than not to reject. The confidence of this token is
            // exactly why we stop, so drafting it was unavoidable — but every
            // token after it now costs nothing, which is the whole point of
            // gating here instead of after the loop.
            if probability < options.mtp_min_confidence {
                stopped_early = true;
                break;
            }
            drafts.push(MtpDraftCandidate {
                token: draft,
                logit_margin,
                probability,
            });
            token = draft;
        }
        Ok((drafts, timings, stopped_early))
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
    nth_hidden_row(hidden, rows - 1)
}

fn nth_hidden_row(hidden: &Tensor, row: usize) -> Result<Vec<f32>> {
    ensure!(
        hidden.rank() == 2,
        "normalized hidden output must be rank two"
    );
    let rows = hidden.dim(0)?;
    ensure!(
        row < rows,
        "normalized hidden output has {rows} rows, so row {row} does not exist"
    );
    Ok(hidden.i(row)?.to_vec1::<f32>()?)
}

/// How many of a verification pass's rows a session keeps when the output sink
/// fails on the pass's output at `output_index`.
///
/// A pass emits `accepted + 1` tokens but evaluates only `1 + accepted` rows:
/// the accepted drafts were evaluated, the authoritative token that follows
/// them was not. So a halt on accepted draft `j` keeps the rows up to and
/// including it, and a halt on the authoritative token keeps every committed
/// row — which is the boundary the state already sits at, so that case must
/// not roll back at all.
fn halted_pass_rows(output_index: usize, accepted: usize) -> usize {
    1 + output_index.min(accepted)
}

/// What a rollback to `position` leaves of a retained MTP gap of `gap_len`
/// tokens starting at `synced_position`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GapAfterRollback {
    /// Keep this many of the gap's entries, and the arm's invariant —
    /// `synced_position + gap_len == state.position` — holds again.
    Retain(usize),
    /// The predictor is ahead of the rolled-back target, so its own state has
    /// to come back to `position` and the gap goes entirely.
    TruncatePredictor,
    /// The gap does not reach the target position, so the invariant was
    /// already broken before the rollback and truncation cannot restore it.
    Unrecoverable,
}

fn gap_after_rollback(synced_position: usize, gap_len: usize, position: usize) -> GapAfterRollback {
    match position.checked_sub(synced_position) {
        Some(retained) if retained <= gap_len => GapAfterRollback::Retain(retained),
        Some(_) => GapAfterRollback::Unrecoverable,
        None => GapAfterRollback::TruncatePredictor,
    }
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

/// The MTP head's top-1 choice with the two confidence signals worth having:
/// the raw top-1/top-2 logit margin, and the top-1 softmax probability.
///
/// The probability costs one extra pass over a logits vector that has already
/// been materialised, which is nothing beside the LM-head pass that produced
/// it. It is computed with the standard max-subtraction so the exponentials
/// cannot overflow.
fn top1_with_scores(logits: &[f32]) -> Result<(usize, f32, f32)> {
    let top = argmax(logits)?;
    let peak = logits[top];
    let second = logits
        .iter()
        .enumerate()
        .filter(|(index, value)| *index != top && value.is_finite())
        .map(|(_, &value)| value)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .context("MTP logits do not contain a finite runner-up")?;
    let denominator: f32 = logits
        .iter()
        .filter(|value| value.is_finite())
        .map(|&value| (value - peak).exp())
        .sum();
    ensure!(
        denominator > 0.,
        "MTP logits produced a degenerate softmax denominator"
    );
    Ok((top, peak - second, 1. / denominator))
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

/// Which arms a run may use, reading the deprecated single-arm flags as their
/// modes when no mode was given explicitly.
///
/// Callers that need to know what a run will actually do — whether it will
/// speculate at all, or whether it needs the MTP arm — must ask this rather
/// than reading `options.speculative_mode`, which can understate both.
pub fn effective_speculative_mode(options: &GenerationOptions) -> SpeculativeMode {
    match options.speculative_mode {
        SpeculativeMode::Off if options.speculative_mtp_draft_tokens > 0 => SpeculativeMode::Mtp,
        SpeculativeMode::Off if options.speculative_ngram_draft_tokens > 0 => {
            SpeculativeMode::Ngram
        }
        mode => mode,
    }
}

/// Build both arms' controller configurations for one run.
///
/// Each arm starts optimistically — the n-gram arm at its cap, which is the
/// draft length the previous report recommended, and the MTP arm at the depth
/// where measured acceptance is still above 94% — and the controllers move it
/// from there. An arm its mode excludes, or one whose ceiling is zero, or the
/// MTP arm on a model that has no predictor block, is configured disabled and
/// never polls true.
fn arm_configs(
    options: &GenerationOptions,
    mode: SpeculativeMode,
    mtp_usable: bool,
) -> (ArmConfig, ArmConfig) {
    let tuning = &options.policy;
    let ngram_cap = match options.speculative_ngram_draft_tokens {
        0 => DEFAULT_NGRAM_DRAFT_CAP,
        cap => cap,
    };
    let mut ngram = ArmConfig::ngram(ngram_cap);
    ngram.enabled = mode.allows_ngram() && ngram_cap > 0;
    ngram.floor = tuning.ngram_draft_floor.clamp(1, ngram_cap.max(1));
    ngram.start = ngram_cap;
    ngram.adaptive = tuning.adaptive_length;
    ngram.backoff = tuning.ewma_backoff;
    ngram.alpha = tuning.ewma_alpha;
    ngram.suspend_below = tuning.ngram_suspend_below;
    ngram.backoff_tokens = tuning.backoff_tokens;
    ngram.backoff_cap = tuning.backoff_cap;

    let mtp_cap = match options.speculative_mtp_draft_tokens {
        0 => DEFAULT_MTP_DEPTH_CAP,
        cap => cap,
    };
    let mut mtp = ArmConfig::mtp(mtp_cap);
    mtp.enabled = mode.allows_mtp() && mtp_cap > 0 && mtp_usable;
    mtp.floor = tuning.mtp_depth_floor.clamp(1, mtp_cap.max(1));
    mtp.start = tuning.mtp_depth_start;
    mtp.probe_len = Some(mtp.floor);
    mtp.adaptive = tuning.adaptive_length;
    mtp.backoff = tuning.ewma_backoff;
    mtp.alpha = tuning.ewma_alpha;
    mtp.suspend_below = tuning.mtp_suspend_below;
    mtp.backoff_tokens = tuning.backoff_tokens;
    mtp.backoff_cap = tuning.backoff_cap;
    (ngram, mtp)
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
        GapAfterRollback, QuantizedNgramMetrics, ThinkingBoundary, ThinkingBudget,
        accepted_draft_prefix, continuation_input, gap_after_rollback, halted_pass_rows,
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
    fn a_halt_on_the_authoritative_token_keeps_every_committed_row() {
        // Three drafts, two accepted: the pass commits three rows and emits
        // three tokens. Halting on either accepted draft unwinds back to it;
        // halting on the authoritative token asks for the boundary the state
        // already sits at, so that case must not roll back a second time.
        let accepted = 2;
        let committed_rows = 1 + accepted;
        assert_eq!(halted_pass_rows(0, accepted), 1);
        assert_eq!(halted_pass_rows(1, accepted), 2);
        assert_eq!(halted_pass_rows(2, accepted), committed_rows);
        // A pass with nothing accepted emits only its authoritative token.
        assert_eq!(halted_pass_rows(0, 0), 1);
    }

    #[test]
    fn the_retained_gap_follows_the_rollback_rather_than_being_abandoned() {
        // Predictor synced at 100 with ten retained tokens, so the target sat
        // at 110. Rolling back to 106 leaves the first six of them, and the
        // arm's invariant — synced + gap == position — holds again.
        assert_eq!(
            gap_after_rollback(100, 10, 106),
            GapAfterRollback::Retain(6)
        );
        // A rollback all the way to the synced position empties the gap
        // without invalidating it.
        assert_eq!(
            gap_after_rollback(100, 10, 100),
            GapAfterRollback::Retain(0)
        );
        // Nothing was unwound.
        assert_eq!(
            gap_after_rollback(100, 10, 110),
            GapAfterRollback::Retain(10)
        );
    }

    #[test]
    fn a_predictor_ahead_of_the_rollback_is_truncated_instead() {
        // Eager resynchronisation advances the synced position as soon as a
        // pass commits, so a rollback can land behind it. The gap is empty
        // there, and the predictor's own rows for those positions have to go.
        assert_eq!(
            gap_after_rollback(110, 0, 106),
            GapAfterRollback::TruncatePredictor
        );
    }

    #[test]
    fn a_gap_too_short_to_reach_the_target_cannot_be_repaired() {
        // The invariant was already broken before the rollback: no truncation
        // of a four-token gap describes the ten tokens above the synced
        // position, so the arm has to sit out.
        assert_eq!(
            gap_after_rollback(100, 4, 110),
            GapAfterRollback::Unrecoverable
        );
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
            speculative_mode: SpeculativeMode::Off,
            policy: PolicyTuning::default(),
            speculative_mtp_draft_tokens: 0,
            speculative_mtp_min_margin: None,
            mtp_draft_vocab: DEFAULT_MTP_DRAFT_VOCAB,
            mtp_min_confidence: DEFAULT_MTP_MIN_CONFIDENCE,
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
