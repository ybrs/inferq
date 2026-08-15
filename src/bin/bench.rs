use std::{
    fs::{self, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use qwen_engine::{
    GenerationOptions, Runtime,
    loader::ModelSummary,
    profile::{HostInfo, ProcessDelta, ProcessSnapshot, SourceInfo},
    qwen::ForwardTimingReport,
};
use serde::{Deserialize, Serialize};

const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Parser)]
#[command(about = "Repeatable coding-workload benchmark")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "benchmarks/prompts.json")]
    prompts: PathBuf,
    #[arg(long, default_value_t = 1)]
    repetitions: usize,
    /// Execute matching workloads without recording them before measured runs.
    #[arg(long, default_value_t = 0)]
    warmup_repetitions: usize,
    /// Run only the workload with this exact name.
    #[arg(long)]
    only: Option<String>,
    /// Override each selected workload's generation length.
    #[arg(long)]
    max_new_tokens: Option<usize>,
    /// Cache state established outside the benchmark; this is a label, not an eviction request.
    #[arg(long, value_enum, default_value_t = CacheState::Unknown)]
    cache_state: CacheState,
    /// Create a JSONL artifact instead of writing records to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum CacheState {
    Unknown,
    Cold,
    Warm,
    Persistent,
}

#[derive(Debug, Deserialize)]
struct Prompt {
    name: String,
    category: String,
    prompt: String,
    max_new_tokens: usize,
    #[serde(default)]
    expected_token_prefix: Option<Vec<u32>>,
}

#[derive(Debug, Serialize)]
struct ModelInfo<'a> {
    path: String,
    revision: Option<String>,
    summary: &'a ModelSummary,
    load_seconds: f64,
    load_process: &'a ProcessDelta,
}

#[derive(Debug, Serialize)]
struct RunInfo<'a> {
    name: &'a str,
    category: &'a str,
    repetition: usize,
    cache_state: CacheState,
}

#[derive(Debug, Serialize)]
struct WorkloadInfo<'a> {
    prompt: &'a str,
    max_new_tokens: usize,
}

#[derive(Debug, Serialize)]
struct TokenInfo<'a> {
    prompt_ids: &'a [u32],
    generated_ids: &'a [u32],
}

#[derive(Debug, Serialize)]
struct CorrectnessInfo<'a> {
    status: &'static str,
    expected_token_prefix: Option<&'a [u32]>,
}

#[derive(Debug, Serialize)]
struct PerformanceInfo {
    prompt_tokens: usize,
    generated_tokens: usize,
    time_to_first_token_seconds: f64,
    prefill_seconds: f64,
    prefill_tokens_per_second: f64,
    decode_seconds: f64,
    decode_tokens_per_second: f64,
}

#[derive(Debug, Serialize)]
struct TimingInfo {
    prefill: ForwardTimingReport,
    decode: ForwardTimingReport,
}

#[derive(Debug, Serialize)]
struct Record<'a> {
    schema_version: u32,
    timestamp_unix_ms: u128,
    run: RunInfo<'a>,
    source: &'a SourceInfo,
    host: &'a HostInfo,
    model: ModelInfo<'a>,
    workload: WorkloadInfo<'a>,
    tokens: TokenInfo<'a>,
    correctness: CorrectnessInfo<'a>,
    performance: PerformanceInfo,
    process: ProcessDelta,
    timings: TimingInfo,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.repetitions > 0, "--repetitions must be at least one");
    let prompts: Vec<Prompt> = serde_json::from_slice(
        &fs::read(&args.prompts)
            .with_context(|| format!("failed to read {}", args.prompts.display()))?,
    )?;
    let prompts: Vec<_> = prompts
        .iter()
        .filter(|prompt| args.only.as_ref().is_none_or(|name| &prompt.name == name))
        .collect();
    ensure!(
        !prompts.is_empty(),
        "no benchmark workloads matched {}",
        args.only.as_deref().unwrap_or("the prompt file")
    );

    let mut output: Box<dyn Write> = match &args.output {
        Some(path) => Box::new(BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .with_context(|| format!("failed to create artifact {}", path.display()))?,
        )),
        None => Box::new(BufWriter::new(io::stdout().lock())),
    };
    let host = HostInfo::detect(&args.model);
    let source = SourceInfo::detect();
    let load_before = ProcessSnapshot::capture()?;
    let load_started = Instant::now();
    let mut runtime = Runtime::load(&args.model)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let load_after = ProcessSnapshot::capture()?;
    let load_process = load_before.delta(&load_after);
    let summary = runtime.model().checkpoint().summary();
    let model_path = args
        .model
        .canonicalize()
        .unwrap_or_else(|_| args.model.clone())
        .display()
        .to_string();
    let revision = model_revision(&args.model);

    for prompt in prompts {
        let max_new_tokens = args.max_new_tokens.unwrap_or(prompt.max_new_tokens);
        let options = GenerationOptions {
            max_new_tokens,
            ..Default::default()
        };
        for _ in 0..args.warmup_repetitions {
            runtime.generate(&prompt.prompt, &options)?;
        }
        for repetition in 0..args.repetitions {
            let before = ProcessSnapshot::capture()?;
            let result = runtime.generate(&prompt.prompt, &options)?;
            let after = ProcessSnapshot::capture()?;
            let correctness = correctness(
                prompt.expected_token_prefix.as_deref(),
                &result.generated_token_ids,
            );
            let record = Record {
                schema_version: SCHEMA_VERSION,
                timestamp_unix_ms: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .context("system clock is before the Unix epoch")?
                    .as_millis(),
                run: RunInfo {
                    name: &prompt.name,
                    category: &prompt.category,
                    repetition,
                    cache_state: args.cache_state,
                },
                source: &source,
                host: &host,
                model: ModelInfo {
                    path: model_path.clone(),
                    revision: revision.clone(),
                    summary: &summary,
                    load_seconds,
                    load_process: &load_process,
                },
                workload: WorkloadInfo {
                    prompt: &prompt.prompt,
                    max_new_tokens,
                },
                tokens: TokenInfo {
                    prompt_ids: &result.prompt_token_ids,
                    generated_ids: &result.generated_token_ids,
                },
                correctness,
                performance: PerformanceInfo {
                    prompt_tokens: result.metrics.prompt_tokens,
                    generated_tokens: result.metrics.generated_tokens,
                    time_to_first_token_seconds: result.metrics.time_to_first_token.as_secs_f64(),
                    prefill_seconds: result.metrics.prefill_wall_time.as_secs_f64(),
                    prefill_tokens_per_second: result.metrics.prefill_tokens_per_second(),
                    decode_seconds: result.metrics.decode_wall_time.as_secs_f64(),
                    decode_tokens_per_second: result.metrics.decode_tokens_per_second(),
                },
                process: before.delta(&after),
                timings: TimingInfo {
                    prefill: result.metrics.prefill_profile.report(),
                    decode: result.metrics.decode_profile.report(),
                },
            };
            serde_json::to_writer(&mut output, &record)?;
            writeln!(output)?;
            output.flush()?;
        }
    }
    Ok(())
}

fn correctness<'a>(expected: Option<&'a [u32]>, actual: &[u32]) -> CorrectnessInfo<'a> {
    let status = match expected {
        None => "not_checked",
        Some(expected) if actual.starts_with(expected) => "pass",
        Some(_) => "fail",
    };
    CorrectnessInfo {
        status,
        expected_token_prefix: expected,
    }
}

fn model_revision(model: &Path) -> Option<String> {
    let metadata = model.join(".cache/huggingface/download/config.json.metadata");
    fs::read_to_string(metadata)
        .ok()?
        .lines()
        .next()
        .map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correctness_checks_a_prefix() {
        assert_eq!(correctness(Some(&[1, 2]), &[1, 2, 3]).status, "pass");
        assert_eq!(correctness(Some(&[1, 2]), &[1, 4]).status, "fail");
        assert_eq!(correctness(None, &[1]).status, "not_checked");
    }
}
