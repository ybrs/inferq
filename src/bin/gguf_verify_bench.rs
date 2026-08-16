use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use qwen_engine::{
    GgufCheckpoint, Qwen3NextConfig,
    profile::{BuildInfo, HostInfo, SourceInfo},
    qwen::{QuantizedForwardTimingReport, QuantizedForwardTimings, QuantizedModel},
    tokenizer::ModelTokenizer,
    warm_all_experts,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Measure warmed target verification scaling for quantized Qwen")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer_model: PathBuf,
    /// Deterministic context prefetched before every measured verification.
    #[arg(
        long,
        default_value = "Measure deterministic target verification scaling."
    )]
    prompt: String,
    /// Deterministic Qwen3.6 target input tokens, shared by every K.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "8160,579,264,7047,1817,25,271,16"
    )]
    verification_tokens: Vec<u32>,
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8")]
    batch_sizes: Vec<usize>,
    #[arg(long, default_value_t = 3)]
    repetitions: usize,
    #[arg(long, default_value_t = 1)]
    replay_rows: usize,
    #[arg(long, default_value_t = 46_000)]
    expert_cache_mib: usize,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct StageScaling {
    total_seconds: f64,
    per_row_seconds: f64,
}

impl StageScaling {
    fn new(aggregate_seconds: f64, repetitions: usize, rows: usize) -> Self {
        let total_seconds = aggregate_seconds / repetitions as f64;
        Self {
            total_seconds,
            per_row_seconds: total_seconds / rows as f64,
        }
    }
}

#[derive(Debug, Serialize)]
struct VerificationStages {
    wall: StageScaling,
    deltanet_projections: StageScaling,
    deltanet_recurrence: StageScaling,
    full_attention: StageScaling,
    moe_router: StageScaling,
    moe_top_k: StageScaling,
    moe_routed_gate_up: StageScaling,
    moe_activation: StageScaling,
    moe_routed_down: StageScaling,
    moe_routed_accumulation: StageScaling,
    moe_shared_expert: StageScaling,
    dense_projections_outside_moe: StageScaling,
    final_norm: StageScaling,
    lm_head: StageScaling,
}

impl VerificationStages {
    fn from_report(report: &QuantizedForwardTimingReport, repetitions: usize, rows: usize) -> Self {
        let operation = &report.nested_operations;
        let stage = &report.stages;
        let dense = operation.attention_projections_seconds
            + operation.attention_output_projection_seconds
            + operation.deltanet_projections_seconds
            + operation.deltanet_output_projection_seconds;
        let scaling = |seconds| StageScaling::new(seconds, repetitions, rows);
        Self {
            wall: scaling(report.wall_seconds),
            deltanet_projections: scaling(operation.deltanet_projections_seconds),
            deltanet_recurrence: scaling(operation.deltanet_recurrence_seconds),
            full_attention: scaling(stage.attention_seconds),
            moe_router: scaling(operation.moe_router_seconds),
            moe_top_k: scaling(operation.moe_top_k_seconds),
            moe_routed_gate_up: scaling(operation.moe_expert_gate_up_seconds),
            moe_activation: scaling(operation.moe_expert_activation_seconds),
            moe_routed_down: scaling(operation.moe_expert_down_seconds),
            moe_routed_accumulation: scaling(operation.moe_expert_accumulation_seconds),
            moe_shared_expert: scaling(operation.moe_shared_expert_seconds),
            dense_projections_outside_moe: scaling(dense),
            final_norm: scaling(stage.final_norm_seconds),
            lm_head: scaling(stage.lm_head_seconds),
        }
    }
}

#[derive(Debug, Serialize)]
struct LayerExpertReuse {
    layer: usize,
    token_expert_assignments: f64,
    unique_experts_selected: f64,
    duplicate_expert_assignment_rate: f64,
    average_rows_per_selected_expert: f64,
    maximum_rows_assigned_to_one_expert: usize,
}

#[derive(Debug, Serialize)]
struct StateOperationTimings {
    checkpoint_seconds: f64,
    restore_seconds: f64,
    rejection_replay_seconds: f64,
    rejection_replay_rows: usize,
}

#[derive(Debug, Serialize)]
struct BatchResult {
    rows: usize,
    repetitions: usize,
    stages: VerificationStages,
    state_operations: StateOperationTimings,
    expert_reuse_by_layer: Vec<LayerExpertReuse>,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    source: SourceInfo,
    build: BuildInfo,
    host: HostInfo,
    model: qwen_engine::GgufModelIdentity,
    prompt: String,
    prompt_token_ids: Vec<u32>,
    verification_token_ids: Vec<u32>,
    batch_sizes: Vec<usize>,
    expert_cache_mib: usize,
    results: Vec<BatchResult>,
}

fn duration_seconds(duration: Duration, repetitions: usize) -> f64 {
    duration.as_secs_f64() / repetitions as f64
}

fn run(args: &Args) -> Result<Report> {
    ensure!(args.repetitions > 0, "--repetitions must be at least one");
    ensure!(args.replay_rows > 0, "--replay-rows must be at least one");
    ensure!(!args.batch_sizes.is_empty(), "--batch-sizes is empty");
    ensure!(
        args.batch_sizes.iter().all(|&rows| rows > 0),
        "every verification batch size must be positive"
    );
    let maximum_rows = *args.batch_sizes.iter().max().expect("validated non-empty");
    ensure!(
        maximum_rows <= args.verification_tokens.len(),
        "largest batch K={maximum_rows} exceeds {} verification tokens",
        args.verification_tokens.len()
    );
    ensure!(
        args.replay_rows <= maximum_rows,
        "--replay-rows exceeds the largest verification batch"
    );

    let checkpoint = GgufCheckpoint::open(&args.model)?;
    checkpoint.configure_expert_cache(
        args.expert_cache_mib
            .checked_mul(1024 * 1024)
            .context("--expert-cache-mib is too large")?,
    )?;
    let identity = checkpoint.identity()?;
    let config = Qwen3NextConfig::from_path(args.tokenizer_model.join("config.json"))?;
    let tokenizer = ModelTokenizer::from_model_dir(&args.tokenizer_model)?;
    let model = QuantizedModel::load(&checkpoint, config)?;
    warm_all_experts(&checkpoint, model.config(), |progress| {
        if progress.tensors_completed.is_multiple_of(12)
            || progress.tensors_completed == progress.tensors_total
        {
            eprintln!(
                "expert warmup: {}/{} tensors ({:.1}/{:.1} GiB)",
                progress.tensors_completed,
                progress.tensors_total,
                progress.bytes_loaded as f64 / (1024. * 1024. * 1024.),
                progress.bytes_total as f64 / (1024. * 1024. * 1024.),
            );
        }
    })?;

    let prompt_token_ids = tokenizer.encode(&args.prompt, false)?;
    ensure!(
        !prompt_token_ids.is_empty(),
        "prompt token sequence is empty"
    );
    let mut state = model.new_state();
    model.forward_detailed(&prompt_token_ids, &mut state)?;
    let base = state.checkpoint();

    // One unmeasured pass per K warms dense pages and allocation paths while
    // restoring the exact same recurrent/KV context afterward.
    for &rows in &args.batch_sizes {
        model.forward_detailed(&args.verification_tokens[..rows], &mut state)?;
        state.restore(&base)?;
    }

    let mut results = Vec::with_capacity(args.batch_sizes.len());
    for &rows in &args.batch_sizes {
        let replay_rows = args.replay_rows.min(rows);
        let mut profile = QuantizedForwardTimings::default();
        let mut checkpoint_elapsed = Duration::ZERO;
        let mut restore_elapsed = Duration::ZERO;
        let mut replay_elapsed = Duration::ZERO;
        for _ in 0..args.repetitions {
            let started = Instant::now();
            let before_verification = state.checkpoint();
            checkpoint_elapsed += started.elapsed();

            let output = model.forward_detailed(&args.verification_tokens[..rows], &mut state)?;
            profile.accumulate(&output.timings);

            let started = Instant::now();
            state.restore(&before_verification)?;
            restore_elapsed += started.elapsed();

            let started = Instant::now();
            model.forward_detailed(&args.verification_tokens[..replay_rows], &mut state)?;
            replay_elapsed += started.elapsed();
            state.restore(&before_verification)?;
        }
        let timing_report = profile.report();
        let expert_reuse_by_layer = profile
            .layer_details
            .iter()
            .map(|layer| {
                let assignments =
                    layer.moe.routing.token_expert_assignments as f64 / args.repetitions as f64;
                let unique =
                    layer.moe.routing.unique_experts_selected as f64 / args.repetitions as f64;
                LayerExpertReuse {
                    layer: layer.layer,
                    token_expert_assignments: assignments,
                    unique_experts_selected: unique,
                    duplicate_expert_assignment_rate: if assignments == 0. {
                        0.
                    } else {
                        1. - unique / assignments
                    },
                    average_rows_per_selected_expert: if unique == 0. {
                        0.
                    } else {
                        assignments / unique
                    },
                    maximum_rows_assigned_to_one_expert: layer.moe.routing.max_rows_per_expert,
                }
            })
            .collect();
        results.push(BatchResult {
            rows,
            repetitions: args.repetitions,
            stages: VerificationStages::from_report(&timing_report, args.repetitions, rows),
            state_operations: StateOperationTimings {
                checkpoint_seconds: duration_seconds(checkpoint_elapsed, args.repetitions),
                restore_seconds: duration_seconds(restore_elapsed, args.repetitions),
                rejection_replay_seconds: duration_seconds(replay_elapsed, args.repetitions),
                rejection_replay_rows: replay_rows,
            },
            expert_reuse_by_layer,
        });
    }

    Ok(Report {
        schema_version: 1,
        source: SourceInfo::detect(),
        build: BuildInfo::detect(&HostInfo::detect(&args.model)),
        host: HostInfo::detect(&args.model),
        model: identity,
        prompt: args.prompt.clone(),
        prompt_token_ids,
        verification_token_ids: args.verification_tokens.clone(),
        batch_sizes: args.batch_sizes.clone(),
        expert_cache_mib: args.expert_cache_mib,
        results,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let output_path = args.output.clone();
    let report = run(&args)?;
    let mut output: Box<dyn Write> = match output_path {
        Some(path) => Box::new(BufWriter::new(File::create(path)?)),
        None => Box::new(BufWriter::new(io::stdout().lock())),
    };
    serde_json::to_writer_pretty(&mut output, &report)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}
