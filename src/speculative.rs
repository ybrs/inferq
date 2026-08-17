//! The unified speculative policy: one adaptive loop over three draft sources.
//!
//! Neither draft source is a general win on this model. The n-gram drafter
//! fires only on literal token-suffix evidence and is free when it fires, but
//! it sees nothing on a workload that repeats structure without repeating
//! spans. The MTP block sees structural repetition, but it pays a draft
//! forward per drafted token whether or not the draft is going to be accepted,
//! so it collapses on prose. Enabling either one globally loses on some
//! workload.
//!
//! This module holds the decision state that lets both live behind one
//! default-on switch: per step, use the free literal evidence when it exists,
//! use the model-based drafter when it is currently earning its cost, and
//! otherwise decode exactly as an unspeculated run would. Nothing here runs a
//! model or touches session state, so all of it is unit-testable offline.

use std::time::Duration;

use crate::qwen::QuantizedMtpTimings;

/// Which draft sources the generation loop is allowed to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpeculativeMode {
    /// Ordinary autoregressive decoding.
    #[default]
    Off,
    /// The policy: both arms, each under its own controller.
    Auto,
    /// The n-gram arm only, under its controller.
    Ngram,
    /// The MTP arm only, under its controller.
    Mtp,
}

impl SpeculativeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::Ngram => "ngram",
            Self::Mtp => "mtp",
        }
    }

    pub fn is_speculative(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub fn allows_ngram(self) -> bool {
        matches!(self, Self::Auto | Self::Ngram)
    }

    pub fn allows_mtp(self) -> bool {
        matches!(self, Self::Auto | Self::Mtp)
    }
}

/// Which source produced a step's draft, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepArm {
    /// An n-gram key match.
    Ngram,
    /// A continuation of the span an accepted n-gram draft came from.
    NgramSpan,
    /// A chained MTP draft.
    Mtp,
    /// No draft: one authoritative row, exactly as unspeculated decoding.
    Plain,
}

impl StepArm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ngram => "ngram",
            Self::NgramSpan => "ngram-span",
            Self::Mtp => "mtp",
            Self::Plain => "plain",
        }
    }

    pub fn is_ngram(self) -> bool {
        matches!(self, Self::Ngram | Self::NgramSpan)
    }
}

/// Defaults for the n-gram arm's controller.
///
/// The cap is 7 because a verification pass evaluates the pending token plus
/// the drafts, and passes wider than eight rows leave the measured fast path
/// (see `ngram-report-702d043633e0.md`). The floor is the draft length the
/// n-gram report recommended, so the controller only ever shortens below the
/// previously shipped setting when acceptance has actually collapsed.
pub const DEFAULT_NGRAM_DRAFT_FLOOR: usize = 4;
pub const DEFAULT_NGRAM_DRAFT_CAP: usize = 7;
pub const DEFAULT_NGRAM_SUSPEND_BELOW: f64 = 0.4;

/// Defaults for the MTP arm's controller.
///
/// Acceptance by depth measured on this model is 97.7% at depth 1, 94.5% at
/// depth 3 and 55.9% at depth 15, so a deep fixed depth is self-defeating: the
/// arm pays a full draft forward per drafted token. Start at 4, grow one at a
/// time while drafts are fully accepted, halve when they are not.
pub const DEFAULT_MTP_DEPTH_FLOOR: usize = 2;
pub const DEFAULT_MTP_DEPTH_CAP: usize = 7;
pub const DEFAULT_MTP_DEPTH_START: usize = 4;
/// Higher than the n-gram bar: this arm pays its draft cost unconditionally.
pub const DEFAULT_MTP_SUSPEND_BELOW: f64 = 0.5;

/// Confidence below which a chained MTP draft stops extending.
///
/// This is not a taste constant. A drafted token is worth submitting only if
/// the probability the target agrees exceeds the cost of the marginal row plus
/// the draft that produced it, divided by the plain decode step it replaces:
///
/// ```text
/// p* = (draft_ms + row_ms) / plain_step_ms
/// ```
///
/// Measured on this host that is 0.739 (W1), 0.696 (W2) and 0.702 (W3), and an
/// offline sweep over recorded per-token confidences puts the empirical
/// optimum at 0.70 on all three workloads — the derived threshold and the
/// measured one agree. See `draft-report-702d043633e0.md`.
pub const DEFAULT_MTP_MIN_CONFIDENCE: f32 = 0.7;

/// Vocabulary prefix the MTP predictor scores its drafts against.
///
/// The LM head is [248320, 2048] Q6_K = 397.9 MiB, and streaming it is the
/// entire draft cost — 24-26 ms per drafted token on this host, which matches
/// 397.9 MiB at its measured memory bandwidth. Drafting does not need the whole
/// vocabulary: BPE gives frequent tokens low ids, and the target's actual next
/// token lies below id 32768 for 91-98% of decode steps across the measured
/// workloads. A leading row slice is a contiguous byte prefix, so this is a
/// shorter matmul over sequential memory rather than a gather.
///
/// A draft the prefix gets wrong is rejected by the target exactly like any
/// other wrong draft; only speed is at stake, never output.
pub const DEFAULT_MTP_DRAFT_VOCAB: usize = 32_768;

/// Shared backoff constants.
pub const DEFAULT_EWMA_ALPHA: f64 = 0.2;
pub const DEFAULT_BACKOFF_TOKENS: usize = 64;
pub const DEFAULT_BACKOFF_CAP: usize = 512;

/// Static configuration of one arm's controller.
#[derive(Debug, Clone)]
pub struct ArmConfig {
    /// Whether the arm may draft at all in this mode.
    pub enabled: bool,
    pub floor: usize,
    pub cap: usize,
    pub start: usize,
    /// Draft-length increase after a fully accepted proposal.
    pub grow: usize,
    /// Adaptive draft length. Off pins the length at `start`.
    pub adaptive: bool,
    /// EWMA backoff. Off never suspends the arm.
    pub backoff: bool,
    pub alpha: f64,
    pub suspend_below: f64,
    /// Committed tokens the first suspension lasts.
    pub backoff_tokens: usize,
    /// Longest suspension a repeatedly failing probe can reach.
    pub backoff_cap: usize,
    /// Draft length used for the probe that ends a suspension. `None` probes
    /// at whatever length the controller currently holds.
    pub probe_len: Option<usize>,
    /// Optimistic prior the acceptance EWMA starts from.
    ///
    /// Seeding the EWMA from the first proposal instead would suspend an arm
    /// on its first rejected draft, which is wrong: on the copy-heavy workload
    /// the n-gram arm accepts 80% of what it proposes and still rejects whole
    /// drafts occasionally. A prior of one gives each arm a trial whose length
    /// falls out of `alpha` — about five consecutive total rejections at
    /// alpha 0.2 and a 0.4 bar — and decays as `(1 - alpha)^n`, so it cannot
    /// hold a genuinely failing arm open.
    pub initial_ewma: f64,
}

impl ArmConfig {
    /// The n-gram arm's defaults, with `cap` as the draft-length ceiling.
    pub fn ngram(cap: usize) -> Self {
        Self {
            enabled: true,
            floor: DEFAULT_NGRAM_DRAFT_FLOOR,
            cap,
            start: cap,
            grow: 2,
            adaptive: true,
            backoff: true,
            alpha: DEFAULT_EWMA_ALPHA,
            suspend_below: DEFAULT_NGRAM_SUSPEND_BELOW,
            backoff_tokens: DEFAULT_BACKOFF_TOKENS,
            backoff_cap: DEFAULT_BACKOFF_CAP,
            probe_len: None,
            initial_ewma: 1.,
        }
    }

    /// The MTP arm's defaults, with `cap` as the depth ceiling.
    pub fn mtp(cap: usize) -> Self {
        Self {
            enabled: true,
            floor: DEFAULT_MTP_DEPTH_FLOOR,
            cap,
            start: DEFAULT_MTP_DEPTH_START,
            grow: 1,
            adaptive: true,
            backoff: true,
            alpha: DEFAULT_EWMA_ALPHA,
            suspend_below: DEFAULT_MTP_SUSPEND_BELOW,
            backoff_tokens: DEFAULT_BACKOFF_TOKENS,
            backoff_cap: DEFAULT_BACKOFF_CAP,
            probe_len: Some(DEFAULT_MTP_DEPTH_FLOOR),
            initial_ewma: 1.,
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::ngram(0)
        }
    }

    fn effective_start(&self) -> usize {
        self.start.clamp(self.floor.min(self.cap), self.cap)
    }
}

/// Per-arm counters carried into the run report.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArmStats {
    pub proposals: usize,
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
    pub fully_accepted: usize,
    pub rejected_immediately: usize,
    pub suspensions: usize,
    pub probes: usize,
    pub probe_successes: usize,
    /// Steps the arm was unavailable because it was suspended.
    pub suspended_steps: usize,
}

impl ArmStats {
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed_tokens == 0 {
            0.
        } else {
            self.accepted_tokens as f64 / self.proposed_tokens as f64
        }
    }
}

/// One arm's live decision state: draft length, acceptance EWMA, suspension.
///
/// The three mechanisms are independent and individually switchable, so a
/// mechanism that does not earn its place can be turned off without taking the
/// policy loop with it.
#[derive(Debug, Clone)]
pub struct ArmController {
    config: ArmConfig,
    len: usize,
    ewma: f64,
    /// Committed-token count at which the current suspension lifts.
    suspended_until: Option<usize>,
    backoff_window: usize,
    /// The next proposal this arm makes is the probe that ends a suspension.
    probing: bool,
    stats: ArmStats,
}

impl ArmController {
    pub fn new(config: ArmConfig) -> Self {
        let len = config.effective_start();
        let backoff_window = config.backoff_tokens;
        let ewma = config.initial_ewma;
        Self {
            config,
            len,
            ewma,
            suspended_until: None,
            backoff_window,
            probing: false,
            stats: ArmStats::default(),
        }
    }

    pub fn config(&self) -> &ArmConfig {
        &self.config
    }

    pub fn stats(&self) -> &ArmStats {
        &self.stats
    }

    /// Current draft length or depth.
    pub fn draft_len(&self) -> usize {
        self.len
    }

    /// Acceptance EWMA, starting from the configured optimistic prior.
    pub fn ewma(&self) -> f64 {
        self.ewma
    }

    pub fn is_suspended(&self) -> bool {
        self.suspended_until.is_some()
    }

    pub fn is_probing(&self) -> bool {
        self.probing
    }

    pub fn backoff_window(&self) -> usize {
        self.backoff_window
    }

    /// Whether the arm may draft at this point in the run, lifting an expired
    /// suspension. Called once per step per arm; O(1) and allocation-free, so
    /// a run that spends its life with both arms suspended pays nothing beyond
    /// this test.
    pub fn poll(&mut self, committed_tokens: usize) -> bool {
        if !self.config.enabled {
            return false;
        }
        if let Some(until) = self.suspended_until {
            if committed_tokens < until {
                self.stats.suspended_steps += 1;
                return false;
            }
            self.suspended_until = None;
            self.probing = true;
            // A probe asks the cheapest informative question the arm can ask,
            // which for the MTP arm is a two-token draft rather than whatever
            // depth it held when acceptance collapsed.
            if let Some(probe_len) = self.config.probe_len {
                self.len = probe_len.clamp(self.config.floor.min(self.config.cap), self.config.cap);
            }
        }
        true
    }

    /// Record how one proposal of this arm fared and update every mechanism.
    pub fn observe(&mut self, proposed: usize, accepted: usize, committed_tokens: usize) {
        if proposed == 0 {
            return;
        }
        self.stats.proposals += 1;
        self.stats.proposed_tokens += proposed;
        self.stats.accepted_tokens += accepted;
        let fully_accepted = accepted == proposed;
        if fully_accepted {
            self.stats.fully_accepted += 1;
        }
        if accepted == 0 {
            self.stats.rejected_immediately += 1;
        }
        let fraction = accepted as f64 / proposed as f64;

        let was_probing = self.probing;
        if was_probing {
            self.probing = false;
            self.stats.probes += 1;
            // A probe replaces the pre-suspension history rather than blending
            // into it. Blending would carry the very acceptance collapse that
            // caused the suspension, so a workload that has moved on could not
            // resume until several proposals had decayed it away — and the arm
            // only gets one proposal per probe.
            self.ewma = fraction;
        } else {
            self.ewma = self.config.alpha * fraction + (1. - self.config.alpha) * self.ewma;
        }

        if self.config.adaptive {
            if fully_accepted {
                self.len = (self.len + self.config.grow).min(self.config.cap);
            } else if accepted * 2 < proposed {
                self.len = (self.len / 2).max(self.config.floor.min(self.config.cap));
            }
        }

        if !self.config.backoff {
            return;
        }
        if was_probing {
            if fraction >= self.config.suspend_below {
                self.stats.probe_successes += 1;
                self.backoff_window = self.config.backoff_tokens;
            } else {
                self.backoff_window = self
                    .backoff_window
                    .saturating_mul(2)
                    .min(self.config.backoff_cap);
                self.suspend(committed_tokens);
            }
        } else if self.ewma < self.config.suspend_below {
            self.backoff_window = self.config.backoff_tokens;
            self.suspend(committed_tokens);
        }
    }

    fn suspend(&mut self, committed_tokens: usize) {
        self.suspended_until = Some(committed_tokens + self.backoff_window);
        self.stats.suspensions += 1;
    }
}

/// The source span an accepted n-gram draft came from, so the next step can
/// continue it without asking the index for a fresh key match.
///
/// A copied region normally re-matches on its own suffix, but only while the
/// key that covers it survives in the index; a chain keeps drafting across the
/// gaps where it does not. The chain ends the moment a draft is not accepted
/// in full, which is the same evidence the index would have used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanCursor {
    /// Index position of the next token the span would propose.
    pub next: usize,
    /// Key length of the match that started the chain, carried for reporting.
    pub match_len: usize,
    /// Proposals this chain has made so far.
    pub links: usize,
}

impl SpanCursor {
    /// Continue after a fully accepted draft of `len` tokens taken from the
    /// span whose last matched token was at `source_position`.
    pub fn after(source_position: usize, len: usize, match_len: usize, links: usize) -> Self {
        Self {
            next: source_position + 1 + len,
            match_len,
            links,
        }
    }
}

/// Whether an n-gram pass's source span can be chained into the next step.
///
/// A pass commits its accepted drafts *and* the authoritative token the target
/// chose after them, so the span survives only when two things hold: every
/// proposed token verified, and the token the span predicts next is the one
/// the target actually chose. The second condition is the one that matters —
/// a fully accepted draft whose successor diverges has already left the span,
/// and continuing it would propose tokens from a region the text no longer
/// follows.
///
/// `follow_token` is the index token at `source_position + 1 + proposed`, or
/// `None` when the span has run off the end of what the index holds.
pub fn chain_span(
    previous: Option<SpanCursor>,
    source_position: usize,
    proposed: usize,
    accepted: usize,
    match_len: usize,
    follow_token: Option<u32>,
    authoritative: u32,
) -> Option<SpanCursor> {
    if proposed == 0 || accepted != proposed {
        return None;
    }
    if follow_token != Some(authoritative) {
        return None;
    }
    Some(SpanCursor::after(
        source_position,
        // The authoritative token was committed from the span too, so the
        // continuation resumes one position past it.
        proposed + 1,
        match_len,
        previous.map_or(1, |cursor| cursor.links + 1),
    ))
}

/// One decode step's decisions and controller state, for the run report.
#[derive(Debug, Clone, Copy)]
pub struct PolicyStepRecord {
    pub step: usize,
    /// Tokens committed before this step ran.
    pub committed_before: usize,
    pub arm: StepArm,
    pub proposed: usize,
    pub accepted: usize,
    pub ngram_len: usize,
    pub mtp_depth: usize,
    pub ngram_ewma: f32,
    pub mtp_ewma: f32,
    pub ngram_suspended: bool,
    pub mtp_suspended: bool,
    /// Key length the index would have matched on, zero for no match. Recorded
    /// in every mode, which is what makes MTP acceptance conditioned on n-gram
    /// evidence measurable from a single-arm MTP run.
    pub ngram_match_len: usize,
    /// MTP rows the step's lazy catch-up had to resynchronise.
    pub resync_tokens: usize,
}

/// Run-level measurements for the unified policy.
#[derive(Debug, Clone, Default)]
pub struct QuantizedPolicyMetrics {
    pub mode: SpeculativeMode,
    pub steps: usize,
    pub ngram_steps: usize,
    pub ngram_span_steps: usize,
    pub mtp_steps: usize,
    pub plain_steps: usize,
    /// Steps at which the index held a match, whether or not the arm used it.
    pub steps_with_ngram_match: usize,
    pub ngram_arm: ArmStats,
    pub mtp_arm: ArmStats,
    /// MTP proposals split by whether the index also had literal evidence.
    /// Under `auto` the second pair is the only one that can be non-zero; the
    /// split exists so a single-arm MTP run reports what the policy's MTP arm
    /// would have inherited.
    pub mtp_proposed_on_ngram_match: usize,
    pub mtp_accepted_on_ngram_match: usize,
    pub mtp_proposed_on_ngram_miss: usize,
    pub mtp_accepted_on_ngram_miss: usize,
    pub verification_passes: usize,
    pub verification_tokens: usize,
    pub rollbacks: usize,
    pub lookup_wall_time: Duration,
    pub draft_wall_time: Duration,
    pub verification_wall_time: Duration,
    pub snapshot_wall_time: Duration,
    pub rollback_wall_time: Duration,
    /// Single-row passes taken when neither arm drafted.
    pub plain_wall_time: Duration,
    /// Lazy MTP catch-up, kept separate so the scheme's cost is visible.
    pub resync_wall_time: Duration,
    /// Vocabulary the MTP predictor scored its drafts against, and the full
    /// vocabulary it would otherwise have streamed.
    pub draft_vocab: usize,
    pub full_vocab: usize,
    /// Chained drafts the confidence gate cut short.
    pub confidence_stops: usize,
    /// MTP tokens actually drafted, including the one whose confidence ended
    /// a chain. That last one is the gate's unavoidable cost: its confidence
    /// is the reason the chain stopped.
    pub drafted_tokens: usize,
    pub resync_passes: usize,
    pub resync_tokens: usize,
    /// Stage breakdown of the catch-up passes.
    pub resync_profile: QuantizedMtpTimings,
    /// Longest gap one catch-up had to close.
    pub max_resync_tokens: usize,
    pub records: Vec<PolicyStepRecord>,
}

impl QuantizedPolicyMetrics {
    pub fn ngram_match_rate(&self) -> f64 {
        if self.steps == 0 {
            0.
        } else {
            self.steps_with_ngram_match as f64 / self.steps as f64
        }
    }

    pub fn mtp_acceptance_on_ngram_miss(&self) -> f64 {
        if self.mtp_proposed_on_ngram_miss == 0 {
            0.
        } else {
            self.mtp_accepted_on_ngram_miss as f64 / self.mtp_proposed_on_ngram_miss as f64
        }
    }

    pub fn mtp_acceptance_on_ngram_match(&self) -> f64 {
        if self.mtp_proposed_on_ngram_match == 0 {
            0.
        } else {
            self.mtp_accepted_on_ngram_match as f64 / self.mtp_proposed_on_ngram_match as f64
        }
    }

    /// Mean tokens committed per verification pass, counting each pass's own
    /// authoritative token. One means speculation bought nothing.
    pub fn tokens_per_verification(&self) -> f64 {
        if self.verification_passes == 0 {
            0.
        } else {
            (self.ngram_arm.accepted_tokens
                + self.mtp_arm.accepted_tokens
                + self.verification_passes) as f64
                / self.verification_passes as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ngram_controller() -> ArmController {
        ArmController::new(ArmConfig::ngram(DEFAULT_NGRAM_DRAFT_CAP))
    }

    fn mtp_controller() -> ArmController {
        ArmController::new(ArmConfig::mtp(DEFAULT_MTP_DEPTH_CAP))
    }

    #[test]
    fn ngram_controller_starts_at_its_cap_and_cannot_grow_past_it() {
        let mut controller = ngram_controller();
        assert_eq!(controller.draft_len(), 7);
        controller.observe(7, 7, 7);
        assert_eq!(controller.draft_len(), 7, "growth is clamped by the cap");
    }

    #[test]
    fn mtp_controller_grows_one_at_a_time_to_its_cap() {
        let mut controller = mtp_controller();
        assert_eq!(controller.draft_len(), 4);
        let mut committed = 0;
        for expected in [5, 6, 7, 7] {
            let depth = controller.draft_len();
            committed += depth + 1;
            controller.observe(depth, depth, committed);
            assert_eq!(controller.draft_len(), expected);
        }
    }

    #[test]
    fn acceptance_below_half_halves_the_draft_length_down_to_the_floor() {
        let mut controller = mtp_controller();
        // Depth 4 with one accepted token is below half: halve to 2.
        controller.observe(4, 1, 5);
        assert_eq!(controller.draft_len(), 2);
        // Already at the floor, so a second collapse cannot shorten further.
        controller.observe(2, 0, 6);
        assert_eq!(controller.draft_len(), 2);
    }

    #[test]
    fn acceptance_at_exactly_half_neither_grows_nor_shrinks() {
        let mut controller = mtp_controller();
        controller.observe(4, 2, 5);
        assert_eq!(controller.draft_len(), 4);
    }

    #[test]
    fn the_ewma_blends_at_alpha_from_an_optimistic_prior() {
        let mut controller = ngram_controller();
        assert_eq!(controller.ewma(), 1.);
        controller.observe(4, 4, 5);
        assert_eq!(controller.ewma(), 1.);
        controller.observe(4, 0, 6);
        // 0.2 * 0 + 0.8 * 1
        assert!((controller.ewma() - 0.8).abs() < 1e-9);
        controller.observe(4, 2, 7);
        // 0.2 * 0.5 + 0.8 * 0.8
        assert!((controller.ewma() - 0.74).abs() < 1e-9);
    }

    #[test]
    fn one_rejected_draft_does_not_suspend_an_arm_that_is_winning() {
        // The copy-heavy workload accepts about 80% of proposed tokens and
        // still rejects a whole draft now and then; that must not cost it the
        // arm.
        let mut controller = ngram_controller();
        let mut committed = 0;
        for round in 0..20 {
            let accepted = if round % 5 == 0 { 0 } else { 7 };
            committed += accepted + 1;
            controller.observe(7, accepted, committed);
            assert!(controller.poll(committed), "round {round}");
        }
        assert_eq!(controller.stats().suspensions, 0);
    }

    #[test]
    fn five_consecutive_total_rejections_suspend_the_ngram_arm() {
        let mut controller = ngram_controller();
        let mut committed = 0;
        let mut rounds = 0;
        while controller.poll(committed) {
            controller.observe(controller.draft_len(), 0, committed);
            committed += 1;
            rounds += 1;
            assert!(rounds < 20, "the arm never suspended");
        }
        assert_eq!(rounds, 5, "0.8^n crosses 0.4 on the fifth rejection");
        assert_eq!(controller.stats().suspensions, 1);
    }

    #[test]
    fn the_mtp_arm_suspends_a_proposal_sooner_than_the_ngram_arm() {
        // The bar is higher because this arm pays its draft cost whether or
        // not the draft is accepted.
        let mut controller = mtp_controller();
        let mut committed = 0;
        let mut rounds = 0;
        while controller.poll(committed) {
            controller.observe(controller.draft_len(), 0, committed);
            committed += 1;
            rounds += 1;
            assert!(rounds < 20, "the arm never suspended");
        }
        assert_eq!(rounds, 4);
    }

    #[test]
    fn a_collapsed_ewma_suspends_the_arm_for_the_backoff_window() {
        let mut controller = ngram_controller();
        // Drive the EWMA under 0.4 with repeated total rejections, stopping as
        // soon as the arm withdraws, exactly as the loop would.
        let mut committed = 0;
        while controller.poll(committed) {
            committed += 1;
            controller.observe(4, 0, committed);
        }
        assert!(controller.is_suspended(), "ewma {}", controller.ewma());
        assert_eq!(controller.stats().suspensions, 1);
        // Unavailable until 64 further tokens have been committed.
        assert!(!controller.poll(committed));
        assert!(!controller.poll(committed + 63));
        assert!(controller.poll(committed + 64));
        assert!(controller.is_probing());
        assert!(!controller.is_suspended());
    }

    #[test]
    fn a_successful_probe_resumes_the_arm_and_resets_the_backoff() {
        let mut controller = mtp_controller();
        let mut step = 0;
        while controller.poll(step) {
            controller.observe(controller.draft_len(), 0, step);
            step += 1;
        }
        assert!(controller.is_suspended());
        assert!(controller.poll(1_000));
        // The MTP arm probes at its floor, the cheapest informative draft.
        assert_eq!(controller.draft_len(), 2);
        controller.observe(2, 2, 1_002);
        assert!(!controller.is_suspended(), "a good probe resumes the arm");
        assert!(!controller.is_probing());
        assert_eq!(controller.stats().probes, 1);
        assert_eq!(controller.stats().probe_successes, 1);
        assert_eq!(controller.backoff_window(), DEFAULT_BACKOFF_TOKENS);
        // Resumption continues from the probe depth and grows again.
        assert_eq!(controller.draft_len(), 3);
    }

    #[test]
    fn failed_probes_double_the_backoff_up_to_the_cap() {
        let mut controller = ngram_controller();
        let mut step = 0;
        while controller.poll(step) {
            controller.observe(controller.draft_len(), 0, step);
            step += 1;
        }
        assert!(controller.is_suspended());
        assert_eq!(controller.backoff_window(), 64);

        let mut committed = 1_000;
        for expected in [128, 256, 512, 512] {
            assert!(controller.poll(committed), "the suspension has expired");
            assert!(controller.is_probing());
            controller.observe(controller.draft_len(), 0, committed);
            assert!(controller.is_suspended());
            assert_eq!(controller.backoff_window(), expected);
            committed += expected;
        }
        assert_eq!(controller.stats().probes, 4);
        assert_eq!(controller.stats().probe_successes, 0);
    }

    #[test]
    fn a_disabled_arm_never_becomes_available() {
        let mut controller = ArmController::new(ArmConfig::disabled());
        assert!(!controller.poll(0));
        assert!(!controller.poll(1_000_000));
    }

    #[test]
    fn backoff_off_keeps_a_collapsing_arm_available() {
        let mut config = ArmConfig::ngram(7);
        config.backoff = false;
        let mut controller = ArmController::new(config);
        for step in 0..50 {
            controller.observe(controller.draft_len(), 0, step);
            assert!(controller.poll(step), "backoff is disabled");
        }
        assert!(!controller.is_suspended());
        assert_eq!(controller.stats().suspensions, 0);
    }

    #[test]
    fn adaptive_off_pins_the_draft_length_at_the_start_value() {
        let mut config = ArmConfig::mtp(7);
        config.adaptive = false;
        let mut controller = ArmController::new(config);
        controller.observe(4, 4, 5);
        assert_eq!(controller.draft_len(), 4);
        controller.observe(4, 0, 10);
        assert_eq!(controller.draft_len(), 4);
    }

    #[test]
    fn suspended_steps_are_counted_while_the_arm_sits_out() {
        let mut controller = ngram_controller();
        let mut step = 0;
        while controller.poll(step) {
            controller.observe(controller.draft_len(), 0, step);
            step += 1;
        }
        for _ in 0..10 {
            assert!(!controller.poll(step));
        }
        // The poll that ended the loop above was itself a suspended step.
        assert_eq!(controller.stats().suspended_steps, 11);
    }

    #[test]
    fn span_cursor_advances_past_the_accepted_tokens() {
        // Source span ends at index 10 and four tokens were proposed from
        // 11..=14, so the continuation starts at 15.
        let cursor = SpanCursor::after(10, 4, 4, 0);
        assert_eq!(cursor.next, 15);
    }

    #[test]
    fn a_fully_accepted_span_chains_past_the_authoritative_token() {
        // Proposed index positions 11..=14, all accepted; the span's next
        // token is at 15 and the target chose it too, so the chain continues
        // from 16.
        let cursor = chain_span(None, 10, 4, 4, 4, Some(77), 77).expect("the chain continues");
        assert_eq!(cursor.next, 16);
        assert_eq!(cursor.match_len, 4);
        assert_eq!(cursor.links, 1);
        let second = chain_span(Some(cursor), 15, 4, 4, 4, Some(78), 78).expect("still chaining");
        assert_eq!(second.links, 2);
    }

    #[test]
    fn a_rejection_ends_the_chain() {
        assert_eq!(chain_span(None, 10, 4, 3, 4, Some(77), 77), None);
        assert_eq!(chain_span(None, 10, 4, 0, 4, Some(77), 77), None);
    }

    #[test]
    fn a_divergent_successor_ends_the_chain_even_after_full_acceptance() {
        // Every proposed token verified, but the target's own next choice is
        // not what the span predicts: the copy has ended.
        assert_eq!(chain_span(None, 10, 4, 4, 4, Some(77), 99), None);
    }

    #[test]
    fn a_span_that_runs_off_the_end_of_the_index_ends_the_chain() {
        assert_eq!(chain_span(None, 10, 4, 4, 4, None, 77), None);
    }

    #[test]
    fn an_empty_proposal_never_starts_a_chain() {
        assert_eq!(chain_span(None, 10, 0, 0, 4, Some(77), 77), None);
    }

    #[test]
    fn both_arms_suspended_stay_suspended_for_their_whole_windows() {
        let mut ngram = ngram_controller();
        let mut mtp = mtp_controller();
        let mut committed = 0;
        loop {
            let ngram_open = ngram.poll(committed);
            let mtp_open = mtp.poll(committed);
            if !ngram_open && !mtp_open {
                break;
            }
            if ngram_open {
                ngram.observe(ngram.draft_len(), 0, committed);
            }
            if mtp_open {
                mtp.observe(mtp.draft_len(), 0, committed);
            }
            committed += 1;
        }
        assert!(ngram.is_suspended() && mtp.is_suspended());
        // Neither arm needed more than a handful of proposals to withdraw,
        // and from here the run is plain decode: every step is one O(1) test
        // per arm and nothing else. Both windows are at least
        // `DEFAULT_BACKOFF_TOKENS` long counted from where each suspended, so
        // the shared stretch runs to the earlier of the two.
        assert!(committed <= 8, "the trial cost {committed} proposals");
        let shared = DEFAULT_BACKOFF_TOKENS - committed;
        for step in 0..shared {
            assert!(!ngram.poll(committed + step), "n-gram woke at {step}");
            assert!(!mtp.poll(committed + step), "MTP woke at {step}");
        }
        assert_eq!(ngram.stats().proposals + mtp.stats().proposals, 9);
    }
}
