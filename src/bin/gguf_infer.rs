use std::{
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use qwen_engine::{
    GenerationOptions, GgufCheckpoint, GgufModelIdentity, QuantizedGenerationResult,
    QuantizedRuntime, QuantizedSpeculativeMetrics, SpeculativeMode,
    profile::{ProcessDelta, ProcessSnapshot},
    runtime::{DEFAULT_NGRAM_MIN_MATCH, PolicyTuning},
    speculative::{
        DEFAULT_BACKOFF_CAP, DEFAULT_BACKOFF_TOKENS, DEFAULT_EWMA_ALPHA, DEFAULT_MTP_DEPTH_CAP,
        DEFAULT_MTP_DEPTH_FLOOR, DEFAULT_MTP_DEPTH_START, DEFAULT_MTP_DRAFT_VOCAB,
        DEFAULT_MTP_MIN_CONFIDENCE, DEFAULT_MTP_SUSPEND_BELOW, DEFAULT_NGRAM_DRAFT_CAP,
        DEFAULT_NGRAM_DRAFT_FLOOR, DEFAULT_NGRAM_SUSPEND_BELOW, QuantizedPolicyMetrics,
    },
    trace::{JsonRoutingCensus, JsonlRoutingTrace, RoutingCensusArtifact, RoutingTraceSet},
    warm_all_experts,
};

#[derive(Debug, Parser)]
#[command(about = "End-to-end quantized Qwen3-Next-family inference")]
struct Args {
    /// Supported Qwen3-Next or Qwen3.5/3.6 MoE GGUF file.
    #[arg(long)]
    model: PathBuf,
    /// Hugging Face model directory supplying config.json and tokenizer.json.
    #[arg(long)]
    tokenizer_model: PathBuf,
    /// Raw prompt. Optional when --interactive is used.
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value_t = 1)]
    max_new_tokens: usize,
    /// Read turns from standard input while retaining model and sequence state.
    #[arg(long)]
    interactive: bool,
    /// Apply the official Qwen plain-message chat template.
    #[arg(long)]
    chat: bool,
    /// Render the Qwen assistant prefix with its thinking block already closed.
    #[arg(long, requires = "chat", conflicts_with = "thinking_budget")]
    no_thinking: bool,
    /// Force-close Qwen's thinking block after N committed generated tokens.
    #[arg(
        long,
        value_name = "N",
        requires = "chat",
        conflicts_with = "no_thinking"
    )]
    thinking_budget: Option<usize>,
    /// Optional system message for the first chat turn.
    #[arg(long, requires = "chat")]
    system_prompt: Option<String>,
    /// Write every layer-qualified routing decision as JSONL.
    #[arg(long)]
    routing_trace: Option<PathBuf>,
    /// Include all router logits in --routing-trace (large).
    #[arg(long)]
    trace_router_logits: bool,
    /// Write cumulative per-layer expert counts as a versioned JSON sidecar.
    #[arg(long)]
    routing_census: Option<PathBuf>,
    /// Continue an existing --routing-census instead of replacing it.
    #[arg(long, requires = "routing_census")]
    resume_routing_census: bool,
    /// Retain recently used expert matrices in-process, bounded in MiB.
    #[arg(long, default_value_t = 0)]
    expert_cache_mib: usize,
    /// Warm the hottest layer-qualified experts from a prior census.
    #[arg(long)]
    warmup_census: Option<PathBuf>,
    /// Number of hottest experts to warm in each observed layer.
    #[arg(long, default_value_t = 10)]
    warmup_experts_per_layer: usize,
    /// Warm every fused expert tensor, pinning it when an expert cache is configured.
    #[arg(long, conflicts_with = "warmup_census")]
    warmup_all_experts: bool,
    /// Which draft sources decoding may use. `auto` is the unified policy:
    /// free n-gram evidence where it exists, MTP where it is earning, plain
    /// decode otherwise.
    #[arg(long, value_enum, default_value_t = Speculative::Off)]
    speculative: Speculative,
    /// Deprecated alias for `--speculative mtp --mtp-depth-cap N`.
    #[arg(long, default_value_t = 0)]
    speculative_mtp: usize,
    /// Skip MTP proposals whose raw top-1/top-2 logit margin is below this
    /// value. Superseded by --mtp-min-confidence; kept for comparability.
    #[arg(long, value_name = "MARGIN")]
    speculative_mtp_min_margin: Option<f32>,
    /// Score MTP drafts against only the first N rows of the LM head. The LM
    /// head is 398 MiB and streaming it is the entire draft cost; BPE puts
    /// frequent tokens at low ids, so a prefix covers most steps. Zero uses the
    /// full head. Only drafts are affected — the target always scores in full.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MTP_DRAFT_VOCAB)]
    mtp_draft_vocab: usize,
    /// Stop a chained MTP draft at the first token whose own softmax
    /// confidence is below this. Zero disables the gate. The default is the
    /// measured break-even: a drafted token pays for itself only when the
    /// probability the target agrees exceeds (draft + row) / plain step.
    #[arg(long, value_name = "F", default_value_t = DEFAULT_MTP_MIN_CONFIDENCE)]
    mtp_min_confidence: f32,
    /// Deprecated alias for `--speculative ngram --ngram-draft-cap N`.
    #[arg(long, default_value_t = 0, conflicts_with = "speculative_mtp")]
    speculative_ngram: usize,
    /// Shortest token suffix the n-gram drafter will match on.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_NGRAM_MIN_MATCH)]
    ngram_min_match: usize,
    /// Longest draft the n-gram arm may propose. Its controller starts here.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_NGRAM_DRAFT_CAP)]
    ngram_draft_cap: usize,
    /// Shortest draft the n-gram controller will shrink to.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_NGRAM_DRAFT_FLOOR)]
    ngram_draft_floor: usize,
    /// Deepest chained MTP draft.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MTP_DEPTH_CAP)]
    mtp_depth_cap: usize,
    /// Shallowest depth the MTP controller will shrink to, and the depth it
    /// probes at when a suspension expires.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MTP_DEPTH_FLOOR)]
    mtp_depth_floor: usize,
    /// Depth the MTP controller starts each run at.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MTP_DEPTH_START)]
    mtp_depth_start: usize,
    /// Acceptance EWMA below which the n-gram arm suspends itself.
    #[arg(long, value_name = "F", default_value_t = DEFAULT_NGRAM_SUSPEND_BELOW)]
    ngram_suspend_below: f64,
    /// Acceptance EWMA below which the MTP arm suspends itself. Higher than
    /// the n-gram bar because this arm pays its draft cost unconditionally.
    #[arg(long, value_name = "F", default_value_t = DEFAULT_MTP_SUSPEND_BELOW)]
    mtp_suspend_below: f64,
    /// Weight of the newest proposal in each arm's acceptance EWMA.
    #[arg(long, value_name = "F", default_value_t = DEFAULT_EWMA_ALPHA)]
    ewma_alpha: f64,
    /// Committed tokens a first suspension lasts.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_BACKOFF_TOKENS)]
    backoff_tokens: usize,
    /// Longest suspension repeated failed probes can reach.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_BACKOFF_CAP)]
    backoff_cap: usize,
    /// Disable Part B1: continuing an accepted n-gram draft's source span.
    #[arg(long)]
    no_span_continuation: bool,
    /// Disable Part B2: adaptive draft length and depth.
    #[arg(long)]
    no_adaptive_length: bool,
    /// Disable Part B3: EWMA backoff, so neither arm ever suspends.
    #[arg(long)]
    no_ewma_backoff: bool,
    /// Resynchronise the MTP block after every committing pass instead of only
    /// when its arm is about to draft. Same tokens, more resync time; the
    /// comparison is what the lazy scheme is measured against.
    #[arg(long)]
    eager_mtp_resync: bool,
    /// Write one JSON object per decode step describing the arm that fired and
    /// both controllers' state.
    #[arg(long, value_name = "PATH")]
    speculative_trace: Option<PathBuf>,
    /// Write one JSON object per drafted MTP token with its confidence and
    /// whether the target accepted it. This is what calibrates a confidence
    /// gate against measured acceptance rather than against taste.
    #[arg(long, value_name = "PATH")]
    draft_calibration: Option<PathBuf>,
    /// How verification snapshots copy recurrent state. Both produce identical
    /// state; `plain` exists to measure what the streaming stores buy.
    #[arg(long, value_enum, default_value_t = SnapshotCopy::Streaming)]
    snapshot_copy: SnapshotCopy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Speculative {
    /// Ordinary autoregressive decoding.
    Off,
    /// The unified policy: both arms, each under its own controller.
    Auto,
    /// The n-gram arm only.
    Ngram,
    /// The MTP arm only.
    Mtp,
}

impl From<Speculative> for SpeculativeMode {
    fn from(value: Speculative) -> Self {
        match value {
            Speculative::Off => Self::Off,
            Speculative::Auto => Self::Auto,
            Speculative::Ngram => Self::Ngram,
            Speculative::Mtp => Self::Mtp,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum SnapshotCopy {
    /// Non-temporal stores, skipping read-for-ownership and L3 pollution.
    Streaming,
    /// Ordinary `copy_from_slice`.
    Plain,
}

/// Print the unified policy's per-arm summary. Nothing here is printed when
/// decoding ran unspeculated.
fn report_policy(policy: &QuantizedPolicyMetrics) {
    if !policy.mode.is_speculative() || policy.steps == 0 {
        return;
    }
    eprintln!(
        "speculative policy: mode {}; {} steps = {} n-gram ({} span) + {} MTP + {} plain; {} steps had literal evidence ({:.1}%); {:.2} tokens per verification pass; {} verification passes over {} tokens; {} rollbacks; lookup {:.3}s, draft {:.3}s, verify {:.3}s, snapshot {:.3}s, rollback {:.3}s, plain decode {:.3}s, MTP resync {:.3}s over {} passes / {} rows (longest {})",
        policy.mode.as_str(),
        policy.steps,
        policy.ngram_steps,
        policy.ngram_span_steps,
        policy.mtp_steps,
        policy.plain_steps,
        policy.steps_with_ngram_match,
        policy.ngram_match_rate() * 100.,
        policy.tokens_per_verification(),
        policy.verification_passes,
        policy.verification_tokens,
        policy.rollbacks,
        policy.lookup_wall_time.as_secs_f64(),
        policy.draft_wall_time.as_secs_f64(),
        policy.verification_wall_time.as_secs_f64(),
        policy.snapshot_wall_time.as_secs_f64(),
        policy.rollback_wall_time.as_secs_f64(),
        policy.plain_wall_time.as_secs_f64(),
        policy.resync_wall_time.as_secs_f64(),
        policy.resync_passes,
        policy.resync_tokens,
        policy.max_resync_tokens,
    );
    if policy.draft_vocab > 0 && policy.draft_vocab < policy.full_vocab {
        eprintln!(
            "policy mtp draft head: scoring against {} of {} vocabulary rows ({:.1}% of the LM head)",
            policy.draft_vocab,
            policy.full_vocab,
            100. * policy.draft_vocab as f64 / policy.full_vocab as f64,
        );
    }
    if policy.drafted_tokens > 0 {
        eprintln!(
            "policy mtp confidence gate: {} tokens drafted, {} submitted, {} chains stopped early ({:.1}% of drafted tokens declined)",
            policy.drafted_tokens,
            policy.mtp_arm.proposed_tokens,
            policy.confidence_stops,
            100. * (policy.drafted_tokens - policy.mtp_arm.proposed_tokens) as f64
                / policy.drafted_tokens as f64,
        );
    }
    for (name, arm) in [("ngram", &policy.ngram_arm), ("mtp", &policy.mtp_arm)] {
        if arm.proposals == 0 && arm.suspended_steps == 0 {
            continue;
        }
        eprintln!(
            "policy arm {name}: {} proposals, {}/{} tokens accepted ({:.1}%), {} fully accepted, {} rejected at once; {} suspensions over {} steps, {} probes ({} resumed)",
            arm.proposals,
            arm.accepted_tokens,
            arm.proposed_tokens,
            arm.acceptance_rate() * 100.,
            arm.fully_accepted,
            arm.rejected_immediately,
            arm.suspensions,
            arm.suspended_steps,
            arm.probes,
            arm.probe_successes,
        );
    }
    if policy.mtp_proposed_on_ngram_match + policy.mtp_proposed_on_ngram_miss > 0 {
        eprintln!(
            "policy arm mtp by literal evidence: {}/{} accepted on an n-gram match ({:.1}%), {}/{} accepted on a miss ({:.1}%)",
            policy.mtp_accepted_on_ngram_match,
            policy.mtp_proposed_on_ngram_match,
            policy.mtp_acceptance_on_ngram_match() * 100.,
            policy.mtp_accepted_on_ngram_miss,
            policy.mtp_proposed_on_ngram_miss,
            policy.mtp_acceptance_on_ngram_miss() * 100.,
        );
    }
}

/// Optional per-run JSONL artefacts. Grouped so they travel as one argument.
#[derive(Debug, Clone, Copy, Default)]
struct TraceSinks<'a> {
    speculative: Option<&'a Path>,
    draft_calibration: Option<&'a Path>,
}

/// Append this turn's per-drafted-token confidence records to a JSONL file.
fn write_draft_calibration(path: &Path, speculative: &QuantizedSpeculativeMetrics) -> Result<()> {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    for observation in &speculative.draft_observations {
        writeln!(
            file,
            r#"{{"depth":{},"margin":{:.5},"probability":{:.6},"accepted":{},"gated":{}}}"#,
            observation.depth,
            observation.logit_margin,
            observation.probability,
            observation.accepted,
            observation.gated,
        )?;
    }
    Ok(())
}

/// Append this turn's per-step policy records to a JSONL trace.
fn write_speculative_trace(path: &Path, policy: &QuantizedPolicyMetrics) -> Result<()> {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    for record in &policy.records {
        writeln!(
            file,
            r#"{{"step":{},"committed":{},"arm":"{}","proposed":{},"accepted":{},"ngram_len":{},"mtp_depth":{},"ngram_ewma":{:.4},"mtp_ewma":{:.4},"ngram_suspended":{},"mtp_suspended":{},"ngram_match_len":{},"resync_tokens":{}}}"#,
            record.step,
            record.committed_before,
            record.arm.as_str(),
            record.proposed,
            record.accepted,
            record.ngram_len,
            record.mtp_depth,
            record.ngram_ewma,
            record.mtp_ewma,
            record.ngram_suspended,
            record.mtp_suspended,
            record.ngram_match_len,
            record.resync_tokens,
        )?;
    }
    Ok(())
}

fn prepare_all_experts(checkpoint: &GgufCheckpoint, runtime: &QuantizedRuntime<'_>) -> Result<()> {
    eprintln!("starting full expert warmup in GGUF file order; Ctrl-C cancels");
    let process_before = ProcessSnapshot::capture()?;
    let report = warm_all_experts(checkpoint, runtime.model().config(), |progress| {
        if progress.tensors_completed.is_multiple_of(3)
            || progress.tensors_completed == progress.tensors_total
        {
            eprintln!(
                "full warmup: {}/{} tensors, {:.1}/{:.1} GiB",
                progress.tensors_completed,
                progress.tensors_total,
                progress.bytes_loaded as f64 / (1024. * 1024. * 1024.),
                progress.bytes_total as f64 / (1024. * 1024. * 1024.)
            );
        }
    })?;
    let process = process_before.delta(&ProcessSnapshot::capture()?);
    eprintln!(
        "full expert warmup complete ({:?}): {:.1} GiB in {:.3}s ({:.1} MiB/s), {:.1} GiB physical reads; {:.1} GiB resident in {} cache entries",
        report.mode,
        report.bytes_loaded as f64 / (1024. * 1024. * 1024.),
        report.elapsed.as_secs_f64(),
        report.bytes_loaded as f64 / (1024. * 1024.) / report.elapsed.as_secs_f64(),
        process.read_bytes.unwrap_or(0) as f64 / (1024. * 1024. * 1024.),
        report.cache.resident_bytes as f64 / (1024. * 1024. * 1024.),
        report.cache.entries,
    );
    Ok(())
}

fn warm_from_census(
    checkpoint: &GgufCheckpoint,
    runtime: &QuantizedRuntime<'_>,
    identity: &GgufModelIdentity,
    path: &Path,
    experts_per_layer: usize,
) -> Result<()> {
    ensure!(
        experts_per_layer > 0,
        "--warmup-experts-per-layer must be at least one"
    );
    let census = RoutingCensusArtifact::from_path(path)?;
    census.validate_for(
        identity,
        runtime.model().config().num_hidden_layers,
        runtime.model().config().num_experts,
    )?;
    let selected = census.hottest_experts(experts_per_layer);
    ensure!(!selected.is_empty(), "routing census contains no layers");
    let process_before = ProcessSnapshot::capture()?;
    let started = Instant::now();
    let cache_before = checkpoint.expert_cache_stats()?;
    let mut experts = 0usize;
    let mut selected_bytes = 0usize;
    eprintln!(
        "warming up to {experts_per_layer} experts in {} observed layers; Ctrl-C cancels",
        selected.len()
    );
    for (completed, (layer, layer_experts)) in selected.iter().enumerate() {
        for expert in layer_experts {
            selected_bytes = selected_bytes
                .checked_add(checkpoint.warm_expert(*layer, *expert)?)
                .ok_or_else(|| anyhow::anyhow!("expert warmup byte count overflowed"))?;
            experts += 1;
        }
        eprintln!(
            "warmup layer {layer}: {} experts ({}/{})",
            layer_experts.len(),
            completed + 1,
            selected.len()
        );
    }
    let activity = checkpoint
        .expert_cache_stats()?
        .activity_since(cache_before);
    let process = process_before.delta(&ProcessSnapshot::capture()?);
    eprintln!(
        "warmup complete: {experts} experts, {:.1} MiB selected, {:.1} MiB loaded in {:.3}s; {} cache hits; {:.1} MiB physical reads",
        selected_bytes as f64 / (1024. * 1024.),
        activity.bytes_loaded as f64 / (1024. * 1024.),
        started.elapsed().as_secs_f64(),
        activity.hits,
        process.read_bytes.unwrap_or(0) as f64 / (1024. * 1024.),
    );
    Ok(())
}

fn report(
    result: &QuantizedGenerationResult,
    context_tokens: usize,
    process: &ProcessDelta,
) -> Result<()> {
    eprintln!("prompt token ids: {:?}", result.prompt_token_ids);
    if result.evaluated_input_token_ids != result.prompt_token_ids {
        eprintln!(
            "evaluated input token ids (including pending token): {:?}",
            result.evaluated_input_token_ids
        );
    }
    eprintln!("generated token ids: {:?}", result.generated_token_ids);
    let decode_passes = result.metrics.generated_tokens.saturating_sub(1);
    eprintln!(
        "input: {} tokens evaluated in {:.3}s ({:.2} tok/s); decode: {} passes in {:.3}s ({:.2} tok/s); context: {} tokens",
        result.metrics.evaluated_input_tokens,
        result.metrics.prefill_wall_time.as_secs_f64(),
        result.metrics.prefill_tokens_per_second(),
        decode_passes,
        result.metrics.decode_wall_time.as_secs_f64(),
        result.metrics.decode_tokens_per_second(),
        context_tokens,
    );
    let speculative = &result.metrics.speculative;
    if speculative.max_draft_tokens > 0 {
        eprintln!(
            "MTP speculation: {}/{} draft tokens accepted ({:.1}%, {} gated); {} verification passes over {} tokens; {} rollback replays over {} tokens; draft {:.3}s, verify {:.3}s, checkpoint {:.3}s, restore {:.3}s, replay {:.3}s, resync {:.3}s",
            speculative.accepted_tokens,
            speculative.drafted_tokens,
            speculative.acceptance_rate() * 100.,
            speculative.gated_tokens,
            speculative.verification_passes,
            speculative.verification_tokens,
            speculative.rollback_replays,
            speculative.replayed_tokens,
            speculative.draft_wall_time.as_secs_f64(),
            speculative.verification_wall_time.as_secs_f64(),
            speculative.checkpoint_wall_time.as_secs_f64(),
            speculative.restore_wall_time.as_secs_f64(),
            speculative.replay_wall_time.as_secs_f64(),
            speculative.resync_wall_time.as_secs_f64(),
        );
    }
    let ngram = &result.metrics.ngram;
    if ngram.max_draft_tokens > 0 {
        eprintln!(
            "n-gram speculation: draft {} / min match {}; {}/{} steps matched ({:.1}%); {} drafts, {}/{} draft tokens accepted ({:.1}%); {:.2} tokens per verification pass; {} verification passes over {} tokens; {} rollbacks ({} replays over {} tokens); lookup {:.3}s, verify {:.3}s, snapshot {:.3}s, rollback {:.3}s, replay {:.3}s, no-match decode {:.3}s",
            ngram.max_draft_tokens,
            ngram.min_match,
            ngram.steps_with_match,
            ngram.steps,
            ngram.match_rate() * 100.,
            ngram.drafts_issued,
            ngram.draft_tokens_accepted,
            ngram.draft_tokens_proposed,
            ngram.acceptance_rate() * 100.,
            ngram.tokens_per_verification(),
            ngram.verification_passes,
            ngram.verification_tokens,
            ngram.rollbacks,
            ngram.rollback_replays,
            ngram.replayed_tokens,
            ngram.lookup_wall_time.as_secs_f64(),
            ngram.verification_wall_time.as_secs_f64(),
            ngram.snapshot_wall_time.as_secs_f64(),
            ngram.rollback_wall_time.as_secs_f64(),
            ngram.replay_wall_time.as_secs_f64(),
            ngram.target_only_wall_time.as_secs_f64(),
        );
        let histogram = ngram
            .position_acceptance()
            .iter()
            .enumerate()
            .map(|(position, rate)| {
                format!(
                    "{position}:{}/{} ({:.0}%)",
                    ngram.accepted_by_position[position],
                    ngram.proposed_by_position[position],
                    rate * 100.
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("n-gram acceptance by draft position: {histogram}");
        let matches = ngram
            .matches_by_len
            .iter()
            .map(|stats| {
                format!(
                    "len {}: {} drafts, {}/{} tokens ({:.0}%), {} full, {} rejected at once",
                    stats.match_len,
                    stats.drafts,
                    stats.accepted_tokens,
                    stats.proposed_tokens,
                    stats.acceptance_rate() * 100.,
                    stats.fully_accepted_drafts,
                    stats.rejected_immediately,
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        eprintln!("n-gram drafts by match length: {matches}");
        eprintln!(
            "n-gram draft outcomes: {} fully accepted, {} rejected at once, {} truncated at a stop token; snapshots {} rows x {:.1} MiB",
            ngram.fully_accepted_drafts,
            ngram.rejected_immediately,
            ngram.drafts_truncated_at_stop,
            ngram.snapshot_rows,
            ngram.snapshot_bytes_per_row as f64 / (1024. * 1024.),
        );
    }
    report_policy(&result.metrics.policy);
    if let Some(budget) = result.metrics.thinking.budget {
        eprintln!(
            "thinking budget: {}/{} committed thinking tokens; forced closures {}",
            result.metrics.thinking.committed_thinking_tokens,
            budget,
            result.metrics.thinking.forced_closures,
        );
    }
    let cache = result.metrics.expert_cache;
    eprintln!(
        "expert cache: {}/{} hits ({:.1}%); loaded {:.1} MiB of GGUF ranges; resident {:.1}/{:.1} MiB in {} entries (fully resident: {}); {} evictions",
        cache.hits,
        cache.requests,
        cache.hit_rate() * 100.,
        cache.bytes_loaded as f64 / (1024. * 1024.),
        cache.resident_bytes as f64 / (1024. * 1024.),
        cache.capacity_bytes as f64 / (1024. * 1024.),
        cache.entries,
        cache.fully_resident,
        cache.evictions,
    );
    eprintln!(
        "process: physical reads {:.1} MiB; faults {} minor / {} major; RSS {:.1} MiB, peak {:.1} MiB",
        process.read_bytes.unwrap_or(0) as f64 / (1024. * 1024.),
        process.minor_faults.unwrap_or(0),
        process.major_faults.unwrap_or(0),
        process.resident_bytes_after.unwrap_or(0) as f64 / (1024. * 1024.),
        process.peak_resident_bytes.unwrap_or(0) as f64 / (1024. * 1024.),
    );
    Ok(())
}

fn generate_and_report(
    runtime: &mut QuantizedRuntime<'_>,
    prompt: &str,
    options: &GenerationOptions,
    sinks: TraceSinks<'_>,
) -> Result<QuantizedGenerationResult> {
    let before = ProcessSnapshot::capture()?;
    let tokenizer = runtime.tokenizer().clone();
    let mut decoder = tokenizer.decode_stream(true);
    let result = runtime.generate_with_token_callback(prompt, options, |token| {
        if let Some(chunk) = decoder(token)? {
            print!("{chunk}");
            io::stdout().flush()?;
        }
        Ok(())
    })?;
    println!();
    io::stdout().flush()?;
    let process = before.delta(&ProcessSnapshot::capture()?);
    report(&result, runtime.context_tokens(), &process)?;
    if let Some(path) = sinks.speculative {
        write_speculative_trace(path, &result.metrics.policy)?;
    }
    if let Some(path) = sinks.draft_calibration {
        write_draft_calibration(path, &result.metrics.speculative)?;
    }
    Ok(result)
}

fn user_turn_prompt(
    runtime: &QuantizedRuntime<'_>,
    user: &str,
    chat: bool,
    first_turn: bool,
    assistant_closed: bool,
    system_prompt: Option<&str>,
    enable_thinking: bool,
) -> Result<String> {
    if !chat {
        return Ok(user.to_owned());
    }
    if first_turn {
        runtime
            .tokenizer()
            .initial_chat_prompt_with_thinking(user, system_prompt, enable_thinking)
    } else {
        runtime
            .tokenizer()
            .chat_continuation_with_thinking(user, assistant_closed, enable_thinking)
    }
}

fn emitted_im_end(runtime: &QuantizedRuntime<'_>, result: &QuantizedGenerationResult) -> bool {
    runtime
        .tokenizer()
        .token_id("<|im_end|>")
        .zip(result.generated_token_ids.last().copied())
        .is_some_and(|(im_end, last)| im_end == last)
}

fn interactive(
    runtime: &mut QuantizedRuntime<'_>,
    initial_prompt: Option<&str>,
    options: &GenerationOptions,
    chat: bool,
    system_prompt: Option<&str>,
    enable_thinking: bool,
    sinks: TraceSinks<'_>,
) -> Result<()> {
    eprintln!("interactive session ready; /reset clears sequence state, /quit exits");
    let mut first_turn = true;
    let mut assistant_closed = false;
    if let Some(prompt) = initial_prompt {
        let prompt = user_turn_prompt(
            runtime,
            prompt,
            chat,
            first_turn,
            assistant_closed,
            system_prompt,
            enable_thinking,
        )?;
        let result = generate_and_report(runtime, &prompt, options, sinks)?;
        assistant_closed = emitted_im_end(runtime, &result);
        first_turn = false;
    }
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        eprint!("inferq> ");
        io::stderr().flush()?;
        let Some(line) = lines.next() else {
            break;
        };
        let line = line?;
        match line.trim() {
            "" => continue,
            "/quit" | "/exit" => break,
            "/reset" => {
                runtime.reset();
                first_turn = true;
                assistant_closed = false;
                eprintln!("sequence state reset");
            }
            _ => {
                let prompt = user_turn_prompt(
                    runtime,
                    &line,
                    chat,
                    first_turn,
                    assistant_closed,
                    system_prompt,
                    enable_thinking,
                )?;
                let result = generate_and_report(runtime, &prompt, options, sinks)?;
                assistant_closed = emitted_im_end(runtime, &result);
                first_turn = false;
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    qwen_engine::threading::init();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    ensure!(
        args.max_new_tokens > 0,
        "--max-new-tokens must be at least one"
    );
    ensure!(
        args.interactive || args.prompt.is_some(),
        "--prompt is required unless --interactive is used"
    );
    ensure!(
        args.routing_trace.is_some() || !args.trace_router_logits,
        "--trace-router-logits requires --routing-trace"
    );
    // The deprecated single-arm flags now select a mode and that arm's
    // ceiling; everything else about them is unchanged.
    let mut mode = SpeculativeMode::from(args.speculative);
    let mut ngram_cap = args.ngram_draft_cap;
    let mut mtp_cap = args.mtp_depth_cap;
    if args.speculative_mtp > 0 {
        ensure!(
            mode == SpeculativeMode::Off || mode == SpeculativeMode::Mtp,
            "--speculative-mtp is a deprecated alias for --speculative mtp and cannot select another mode"
        );
        eprintln!(
            "warning: --speculative-mtp is deprecated; use --speculative mtp --mtp-depth-cap {}",
            args.speculative_mtp
        );
        mode = SpeculativeMode::Mtp;
        mtp_cap = args.speculative_mtp;
    }
    if args.speculative_ngram > 0 {
        ensure!(
            mode == SpeculativeMode::Off || mode == SpeculativeMode::Ngram,
            "--speculative-ngram is a deprecated alias for --speculative ngram and cannot select another mode"
        );
        eprintln!(
            "warning: --speculative-ngram is deprecated; use --speculative ngram --ngram-draft-cap {}",
            args.speculative_ngram
        );
        mode = SpeculativeMode::Ngram;
        ngram_cap = args.speculative_ngram;
    }
    ensure!(
        args.speculative_mtp_min_margin.is_none() || mode.allows_mtp(),
        "--speculative-mtp-min-margin requires a mode that uses the MTP arm"
    );
    ensure!(
        !mode.is_speculative() || (args.routing_trace.is_none() && args.routing_census.is_none()),
        "speculative decoding does not support routing traces or censuses"
    );
    ensure!(
        args.speculative_trace.is_none() || mode.is_speculative(),
        "--speculative-trace requires a speculative mode"
    );
    ensure!(
        args.draft_calibration.is_none() || mode.allows_mtp(),
        "--draft-calibration requires a mode that uses the MTP arm"
    );
    ensure!(
        ngram_cap > 0 && mtp_cap > 0,
        "draft caps must be at least one"
    );
    // A floor above the cap is clamped rather than rejected, because the
    // deprecated aliases set only a cap: `--speculative-mtp 1` has to keep
    // meaning what it always meant, and one is below the default depth floor.
    if args.ngram_draft_floor > ngram_cap || args.mtp_depth_floor > mtp_cap {
        eprintln!(
            "warning: clamping controller floors to their caps (n-gram {}, MTP {})",
            args.ngram_draft_floor.min(ngram_cap),
            args.mtp_depth_floor.min(mtp_cap)
        );
    }

    let checkpoint = GgufCheckpoint::open(&args.model)?;
    let expert_cache_bytes = args
        .expert_cache_mib
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("--expert-cache-mib is too large"))?;
    checkpoint.configure_expert_cache(expert_cache_bytes)?;
    let identity = checkpoint.identity()?;
    let load_started = Instant::now();
    let mut runtime = QuantizedRuntime::load(&checkpoint, &args.tokenizer_model)?;
    runtime.set_snapshot_nontemporal(args.snapshot_copy == SnapshotCopy::Streaming);
    ensure!(
        (!args.no_thinking && args.thinking_budget.is_none())
            || runtime.tokenizer().supports_thinking_generation(),
        "--no-thinking and --thinking-budget require a Qwen chat template with thinking support"
    );
    let load_time = load_started.elapsed();
    eprintln!(
        "model loaded in {:.3}s ({}, {})",
        load_time.as_secs_f64(),
        identity.layout_fingerprint,
        identity.quantization.join("+")
    );
    if args.warmup_all_experts {
        prepare_all_experts(&checkpoint, &runtime)?;
    } else if let Some(path) = &args.warmup_census {
        warm_from_census(
            &checkpoint,
            &runtime,
            &identity,
            path,
            args.warmup_experts_per_layer,
        )?;
    }

    let mut traces = RoutingTraceSet::default();
    if let Some(path) = args.routing_trace {
        traces.push(Box::new(JsonlRoutingTrace::create(
            path,
            args.trace_router_logits,
        )?));
    }
    if let Some(path) = args.routing_census {
        let census = if args.resume_routing_census {
            JsonRoutingCensus::resume(path, identity)?
        } else {
            JsonRoutingCensus::create(path, identity)
        };
        traces.push(Box::new(census));
    }
    if !traces.is_empty() {
        runtime.set_trace(Some(Box::new(traces)));
    }

    // The caps are the arms' ceilings, not switches; zeroing them when
    // speculation is off keeps `--speculative off` off.
    let (ngram_cap, mtp_cap) = if mode.is_speculative() {
        (ngram_cap, mtp_cap)
    } else {
        (0, 0)
    };
    let options = GenerationOptions {
        max_new_tokens: args.max_new_tokens,
        speculative_mode: mode,
        policy: PolicyTuning {
            ngram_draft_floor: args.ngram_draft_floor.min(ngram_cap.max(1)),
            mtp_depth_floor: args.mtp_depth_floor.min(mtp_cap.max(1)),
            mtp_depth_start: args.mtp_depth_start,
            ngram_suspend_below: args.ngram_suspend_below,
            mtp_suspend_below: args.mtp_suspend_below,
            ewma_alpha: args.ewma_alpha,
            backoff_tokens: args.backoff_tokens,
            backoff_cap: args.backoff_cap,
            span_continuation: !args.no_span_continuation,
            adaptive_length: !args.no_adaptive_length,
            ewma_backoff: !args.no_ewma_backoff,
            eager_mtp_resync: args.eager_mtp_resync,
        },
        speculative_mtp_draft_tokens: mtp_cap,
        speculative_mtp_min_margin: args.speculative_mtp_min_margin,
        mtp_draft_vocab: args.mtp_draft_vocab,
        mtp_min_confidence: args.mtp_min_confidence,
        speculative_ngram_draft_tokens: ngram_cap,
        ngram_min_match: args.ngram_min_match,
        thinking_budget: args.thinking_budget,
        ..GenerationOptions::default()
    };
    let sinks = TraceSinks {
        speculative: args.speculative_trace.as_deref(),
        draft_calibration: args.draft_calibration.as_deref(),
    };
    if args.interactive {
        interactive(
            &mut runtime,
            args.prompt.as_deref(),
            &options,
            args.chat,
            args.system_prompt.as_deref(),
            !args.no_thinking,
            sinks,
        )
    } else {
        let prompt = user_turn_prompt(
            &runtime,
            args.prompt.as_deref().expect("prompt validated above"),
            args.chat,
            true,
            false,
            args.system_prompt.as_deref(),
            !args.no_thinking,
        )?;
        generate_and_report(&mut runtime, &prompt, &options, sinks)?;
        Ok(())
    }
}
