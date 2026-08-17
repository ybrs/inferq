//! Run a whole speculative-configuration matrix in one process.
//!
//! A campaign driven by `gguf_infer` pays the model load and the full expert
//! warmup once per cell. Measured over a 52-run campaign that was 25.5% of the
//! wall clock — 14.8 minutes of 58 — against 59.9% actually spent decoding.
//! This binary loads and warms once, then runs every (configuration, prompt,
//! repetition) from the same resident weights, which is both faster and more
//! comparable: every cell sees an equally warm expert cache instead of the
//! first one seeing a cold one.
//!
//! Repetitions are interleaved — every configuration is run once before any is
//! run twice — so a slow drift over the campaign spreads across configurations
//! rather than landing on whichever was scheduled last.

use std::{fs, path::PathBuf, time::Instant};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use qwen_engine::{
    GenerationOptions, GgufCheckpoint, QuantizedRuntime, SpeculativeMode,
    runtime::PolicyTuning,
    speculative::{
        DEFAULT_BACKOFF_TOKENS, DEFAULT_EWMA_ALPHA, DEFAULT_MTP_DEPTH_START,
        DEFAULT_MTP_DRAFT_VOCAB, DEFAULT_MTP_MIN_CONFIDENCE, DEFAULT_MTP_SUSPEND_BELOW,
    },
    warm_all_experts,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(about = "Run a speculative-configuration matrix from one resident model")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer_model: PathBuf,
    /// JSON array of {name, path, max_new_tokens}.
    #[arg(long)]
    prompts: PathBuf,
    /// JSON array of configurations; see `Cell`.
    #[arg(long)]
    matrix: PathBuf,
    #[arg(long, default_value_t = 1)]
    repetitions: usize,
    #[arg(long, default_value_t = 46_000)]
    expert_cache_mib: usize,
    #[arg(long)]
    output: PathBuf,
    /// Render prompts through the Qwen chat template with thinking closed,
    /// matching the `gguf_infer` harness.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    chat: bool,
}

#[derive(Debug, Deserialize)]
struct Prompt {
    name: String,
    path: PathBuf,
    max_new_tokens: usize,
}

/// One configuration. Every field is optional and falls back to the shipped
/// default, so a matrix file only states what it varies.
#[derive(Debug, Deserialize)]
struct Cell {
    name: String,
    #[serde(default)]
    speculative: Option<String>,
    #[serde(default)]
    mtp_draft_vocab: Option<usize>,
    #[serde(default)]
    mtp_min_confidence: Option<f32>,
    #[serde(default)]
    mtp_depth_start: Option<usize>,
    #[serde(default)]
    mtp_suspend_below: Option<f64>,
    #[serde(default)]
    ewma_alpha: Option<f64>,
    #[serde(default)]
    backoff_tokens: Option<usize>,
    /// Restrict this cell to named prompts. Empty means all of them.
    #[serde(default)]
    prompts: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Record {
    cell: String,
    prompt: String,
    repetition: usize,
    decode_tokens_per_second: f64,
    decode_seconds: f64,
    prefill_seconds: f64,
    generated_tokens: usize,
    token_ids: Vec<u32>,
    mode: String,
    draft_vocab: usize,
    full_vocab: usize,
    drafted_tokens: usize,
    confidence_stops: usize,
    ngram_steps: usize,
    mtp_steps: usize,
    plain_steps: usize,
    mtp_proposed: usize,
    mtp_accepted: usize,
    mtp_suspensions: usize,
    draft_seconds: f64,
    verify_seconds: f64,
    resync_seconds: f64,
}

fn mode_of(name: Option<&str>) -> Result<SpeculativeMode> {
    Ok(match name.unwrap_or("off") {
        "off" => SpeculativeMode::Off,
        "auto" => SpeculativeMode::Auto,
        "ngram" => SpeculativeMode::Ngram,
        "mtp" => SpeculativeMode::Mtp,
        other => anyhow::bail!("unknown speculative mode {other:?}"),
    })
}

fn options_for(cell: &Cell, max_new_tokens: usize) -> Result<GenerationOptions> {
    Ok(GenerationOptions {
        max_new_tokens,
        speculative_mode: mode_of(cell.speculative.as_deref())?,
        mtp_draft_vocab: cell.mtp_draft_vocab.unwrap_or(DEFAULT_MTP_DRAFT_VOCAB),
        mtp_min_confidence: cell
            .mtp_min_confidence
            .unwrap_or(DEFAULT_MTP_MIN_CONFIDENCE),
        policy: PolicyTuning {
            mtp_depth_start: cell.mtp_depth_start.unwrap_or(DEFAULT_MTP_DEPTH_START),
            mtp_suspend_below: cell.mtp_suspend_below.unwrap_or(DEFAULT_MTP_SUSPEND_BELOW),
            ewma_alpha: cell.ewma_alpha.unwrap_or(DEFAULT_EWMA_ALPHA),
            backoff_tokens: cell.backoff_tokens.unwrap_or(DEFAULT_BACKOFF_TOKENS),
            ..PolicyTuning::default()
        },
        ..GenerationOptions::default()
    })
}

fn main() -> Result<()> {
    qwen_engine::threading::init();
    let args = Args::parse();
    ensure!(args.repetitions > 0, "--repetitions must be at least one");

    let prompts: Vec<Prompt> = serde_json::from_slice(&fs::read(&args.prompts)?)
        .with_context(|| format!("invalid prompt list in {}", args.prompts.display()))?;
    let matrix: Vec<Cell> = serde_json::from_slice(&fs::read(&args.matrix)?)
        .with_context(|| format!("invalid matrix in {}", args.matrix.display()))?;
    ensure!(
        !prompts.is_empty() && !matrix.is_empty(),
        "both the prompt list and the matrix must be non-empty"
    );

    let checkpoint = GgufCheckpoint::open(&args.model)?;
    checkpoint.configure_expert_cache(args.expert_cache_mib * 1024 * 1024)?;
    let load_started = Instant::now();
    let mut runtime = QuantizedRuntime::load(&checkpoint, &args.tokenizer_model)?;
    eprintln!(
        "model loaded in {:.1}s",
        load_started.elapsed().as_secs_f64()
    );
    let warm_started = Instant::now();
    warm_all_experts(&checkpoint, runtime.model().config(), |_| {})?;
    eprintln!(
        "experts warmed in {:.1}s — every cell below sees the same resident weights",
        warm_started.elapsed().as_secs_f64()
    );

    // Render once; the rendering is identical for every cell and repetition.
    let rendered = prompts
        .iter()
        .map(|prompt| {
            let raw = fs::read_to_string(&prompt.path)
                .with_context(|| format!("failed to read {}", prompt.path.display()))?;
            if args.chat {
                runtime
                    .tokenizer()
                    .initial_chat_prompt_with_thinking(&raw, None, false)
            } else {
                Ok(raw)
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let mut out = String::new();
    let campaign = Instant::now();
    let total = args.repetitions * matrix.len() * prompts.len();
    let mut done = 0;
    for repetition in 0..args.repetitions {
        for cell in &matrix {
            for (prompt, text) in prompts.iter().zip(&rendered) {
                if !cell.prompts.is_empty() && !cell.prompts.contains(&prompt.name) {
                    continue;
                }
                runtime.reset();
                let options = options_for(cell, prompt.max_new_tokens)?;
                let result = runtime.generate(text, &options)?;
                let m = &result.metrics;
                let p = &m.policy;
                done += 1;
                eprintln!(
                    "[{done}/{total}] {} x {} rep {} -> {:.2} tok/s ({:.0}s elapsed)",
                    cell.name,
                    prompt.name,
                    repetition + 1,
                    m.decode_tokens_per_second(),
                    campaign.elapsed().as_secs_f64()
                );
                let record = Record {
                    cell: cell.name.clone(),
                    prompt: prompt.name.clone(),
                    repetition: repetition + 1,
                    decode_tokens_per_second: m.decode_tokens_per_second(),
                    decode_seconds: m.decode_wall_time.as_secs_f64(),
                    prefill_seconds: m.prefill_wall_time.as_secs_f64(),
                    generated_tokens: m.generated_tokens,
                    token_ids: result.generated_token_ids.clone(),
                    mode: p.mode.as_str().to_owned(),
                    draft_vocab: p.draft_vocab,
                    full_vocab: p.full_vocab,
                    drafted_tokens: p.drafted_tokens,
                    confidence_stops: p.confidence_stops,
                    ngram_steps: p.ngram_steps,
                    mtp_steps: p.mtp_steps,
                    plain_steps: p.plain_steps,
                    mtp_proposed: p.mtp_arm.proposed_tokens,
                    mtp_accepted: p.mtp_arm.accepted_tokens,
                    mtp_suspensions: p.mtp_arm.suspensions,
                    draft_seconds: p.draft_wall_time.as_secs_f64(),
                    verify_seconds: p.verification_wall_time.as_secs_f64(),
                    resync_seconds: p.resync_wall_time.as_secs_f64(),
                };
                out.push_str(&serde_json::to_string(&record)?);
                out.push('\n');
                fs::write(&args.output, &out)?;
            }
        }
    }
    eprintln!(
        "{done} cells in {:.1} min",
        campaign.elapsed().as_secs_f64() / 60.
    );
    Ok(())
}
