use std::{
    fs::{self, OpenOptions},
    io::{self, BufWriter, Write},
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use clap::{ArgAction, Parser};
use qwen_engine::{
    ExpertCacheStats, FullExpertWarmupMode, GenerationOptions, GgufCheckpoint, GgufModelIdentity,
    QuantizedRuntime,
    profile::{BuildInfo, HostInfo, ProcessDelta, ProcessSnapshot, SourceInfo},
    qwen::QuantizedForwardTimingReport,
    tokenizer::ModelTokenizer,
    warm_all_experts,
};
use serde::{Deserialize, Serialize};

const SCHEMA_VERSION: u32 = 3;

#[derive(Debug, Parser)]
#[command(about = "Persistent multi-case benchmark for quantized Qwen inference")]
struct Args {
    /// Supported Qwen3-Next or Qwen3.5/3.6 MoE GGUF file.
    #[arg(long, required_unless_present = "validate_prompts_only")]
    model: Option<PathBuf>,
    /// Hugging Face model directory supplying config.json and tokenizer files.
    #[arg(long)]
    tokenizer_model: PathBuf,
    #[arg(long, default_value = "benchmarks/gguf-prompts.json")]
    prompts: PathBuf,
    #[arg(long, default_value_t = 1)]
    repetitions: usize,
    /// Run only the workload with this exact name.
    #[arg(long)]
    only: Option<String>,
    /// Retain expert matrices in-process, bounded in MiB.
    #[arg(long, default_value_t = 46_000)]
    expert_cache_mib: usize,
    /// Warm every expert once before any measured workload.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    warmup_all_experts: bool,
    /// Validate rendered prompt token counts without opening the GGUF model.
    #[arg(long)]
    validate_prompts_only: bool,
    /// Create a JSONL artifact instead of writing records to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct Prompt {
    name: String,
    category: String,
    prompt: String,
    #[serde(default)]
    chat: bool,
    #[serde(default)]
    system_prompt: Option<String>,
    expected_prompt_tokens: usize,
    max_new_tokens: usize,
    #[serde(default)]
    expected_token_prefix: Option<Vec<u32>>,
}

struct PreparedPrompt {
    definition: Prompt,
    rendered: String,
    token_ids: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct LoadInfo<'a> {
    seconds: f64,
    process: &'a ProcessDelta,
}

#[derive(Debug, Serialize)]
struct WarmupInfo<'a> {
    enabled: bool,
    mode: Option<FullExpertWarmupMode>,
    tensor_count: usize,
    bytes_loaded: usize,
    seconds: f64,
    cache: ExpertCacheStats,
    process: &'a ProcessDelta,
}

#[derive(Debug, Serialize)]
struct RunInfo<'a> {
    name: &'a str,
    category: &'a str,
    repetition: usize,
    residency: &'static str,
}

#[derive(Debug, Serialize)]
struct WorkloadInfo<'a> {
    prompt: &'a str,
    rendered_prompt: &'a str,
    chat: bool,
    system_prompt: Option<&'a str>,
    expected_prompt_tokens: usize,
    max_new_tokens: usize,
}

#[derive(Debug, Serialize)]
struct TokenInfo<'a> {
    prompt_ids: &'a [u32],
    evaluated_input_ids: &'a [u32],
    generated_ids: &'a [u32],
    generated_text: &'a str,
}

#[derive(Debug, Serialize)]
struct CorrectnessInfo<'a> {
    status: &'static str,
    prompt_token_count_matches: bool,
    expected_token_prefix: Option<&'a [u32]>,
}

#[derive(Debug, Serialize)]
struct PerformanceInfo {
    prompt_tokens: usize,
    evaluated_input_tokens: usize,
    generated_tokens: usize,
    decode_passes: usize,
    context_tokens: usize,
    time_to_first_token_seconds: f64,
    prefill_seconds: f64,
    prefill_tokens_per_second: f64,
    decode_seconds: f64,
    decode_tokens_per_second: f64,
}

#[derive(Debug, Serialize)]
struct TimingInfo {
    prefill: QuantizedForwardTimingReport,
    decode: QuantizedForwardTimingReport,
}

#[derive(Debug, Serialize)]
struct Record<'a> {
    schema_version: u32,
    timestamp_unix_ms: u128,
    source: &'a SourceInfo,
    build: &'a BuildInfo,
    host: &'a HostInfo,
    model: &'a GgufModelIdentity,
    load: LoadInfo<'a>,
    warmup: WarmupInfo<'a>,
    run: RunInfo<'a>,
    workload: WorkloadInfo<'a>,
    tokens: TokenInfo<'a>,
    correctness: CorrectnessInfo<'a>,
    performance: PerformanceInfo,
    expert_cache: ExpertCacheStats,
    process: ProcessDelta,
    timings: TimingInfo,
}

fn prepare_prompts(args: &Args, tokenizer: &ModelTokenizer) -> Result<Vec<PreparedPrompt>> {
    let prompts: Vec<Prompt> = serde_json::from_slice(
        &fs::read(&args.prompts)
            .with_context(|| format!("failed to read {}", args.prompts.display()))?,
    )
    .with_context(|| format!("invalid benchmark prompts in {}", args.prompts.display()))?;
    let selected: Vec<_> = prompts
        .into_iter()
        .filter(|prompt| args.only.as_ref().is_none_or(|name| &prompt.name == name))
        .collect();
    ensure!(
        !selected.is_empty(),
        "no benchmark workloads matched {}",
        args.only.as_deref().unwrap_or("the prompt file")
    );

    selected
        .into_iter()
        .map(|definition| {
            ensure!(
                definition.max_new_tokens > 0,
                "workload {:?} must generate at least one token",
                definition.name
            );
            ensure!(
                definition.chat || definition.system_prompt.is_none(),
                "workload {:?} has a system prompt but chat is disabled",
                definition.name
            );
            let rendered = if definition.chat {
                tokenizer
                    .initial_chat_prompt(&definition.prompt, definition.system_prompt.as_deref())?
            } else {
                definition.prompt.clone()
            };
            let token_ids = tokenizer.encode(&rendered, false)?;
            ensure!(
                token_ids.len() == definition.expected_prompt_tokens,
                "workload {:?} rendered to {} prompt tokens, expected {}; token ids: {:?}",
                definition.name,
                token_ids.len(),
                definition.expected_prompt_tokens,
                token_ids
            );
            Ok(PreparedPrompt {
                definition,
                rendered,
                token_ids,
            })
        })
        .collect()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    ensure!(args.repetitions > 0, "--repetitions must be at least one");
    let tokenizer = ModelTokenizer::from_model_dir(&args.tokenizer_model)?;
    let prompts = prepare_prompts(&args, &tokenizer)?;
    for prompt in &prompts {
        eprintln!(
            "validated workload {:?}: {} prompt tokens, {} maximum generated tokens",
            prompt.definition.name,
            prompt.token_ids.len(),
            prompt.definition.max_new_tokens
        );
    }
    if args.validate_prompts_only {
        return Ok(());
    }

    let model_path = args.model.as_ref().context("--model is required")?;
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
    let source = SourceInfo::detect();
    let host = HostInfo::detect(model_path);
    let build = BuildInfo::detect(&host);
    eprintln!(
        "build: {} AVX2={} FMA={}, Candle/Rayon threads {}/{}",
        build.target_arch, build.avx2, build.fma, build.candle_threads, build.rayon_threads
    );
    let load_before = ProcessSnapshot::capture()?;
    let load_started = Instant::now();
    let checkpoint = GgufCheckpoint::open(model_path)?;
    let cache_bytes = args
        .expert_cache_mib
        .checked_mul(1024 * 1024)
        .context("--expert-cache-mib is too large")?;
    checkpoint.configure_expert_cache(cache_bytes)?;
    let identity = checkpoint.identity()?;
    let mut runtime = QuantizedRuntime::load(&checkpoint, &args.tokenizer_model)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let load_process = load_before.delta(&ProcessSnapshot::capture()?);
    eprintln!(
        "model loaded in {load_seconds:.3}s ({}, {})",
        identity.layout_fingerprint,
        identity.quantization.join("+")
    );

    let warmup_before = ProcessSnapshot::capture()?;
    let warmup_report = if args.warmup_all_experts {
        eprintln!("starting full expert warmup in GGUF file order; Ctrl-C cancels");
        Some(warm_all_experts(
            &checkpoint,
            runtime.model().config(),
            |progress| {
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
            },
        )?)
    } else {
        None
    };
    let warmup_process = warmup_before.delta(&ProcessSnapshot::capture()?);
    let cache_after_warmup = checkpoint.expert_cache_stats()?;
    if let Some(report) = &warmup_report {
        eprintln!(
            "full expert warmup complete ({:?}): {:.1} GiB in {:.3}s; {:.1} GiB resident in {} entries",
            report.mode,
            report.bytes_loaded as f64 / (1024. * 1024. * 1024.),
            report.elapsed.as_secs_f64(),
            report.cache.resident_bytes as f64 / (1024. * 1024. * 1024.),
            report.cache.entries,
        );
    }

    let residency = match warmup_report.as_ref().map(|report| report.mode) {
        Some(FullExpertWarmupMode::PinnedExpertCache) => "pinned_expert_cache",
        Some(FullExpertWarmupMode::OsPageCache) => "os_page_cache_warmup",
        None if cache_bytes > 0 => "configured_expert_cache_unwarmed",
        None => "os_page_cache_unwarmed",
    };
    let empty_process = ProcessSnapshot::default().delta(&ProcessSnapshot::default());
    let warmup = WarmupInfo {
        enabled: warmup_report.is_some(),
        mode: warmup_report.as_ref().map(|report| report.mode),
        tensor_count: warmup_report
            .as_ref()
            .map_or(0, |report| report.tensor_count),
        bytes_loaded: warmup_report
            .as_ref()
            .map_or(0, |report| report.bytes_loaded),
        seconds: warmup_report
            .as_ref()
            .map_or(0., |report| report.elapsed.as_secs_f64()),
        cache: cache_after_warmup,
        process: if warmup_report.is_some() {
            &warmup_process
        } else {
            &empty_process
        },
    };

    for prompt in &prompts {
        for repetition in 0..args.repetitions {
            runtime.reset();
            let options = GenerationOptions {
                max_new_tokens: prompt.definition.max_new_tokens,
                ..GenerationOptions::default()
            };
            eprintln!(
                "running {:?} repetition {}/{}",
                prompt.definition.name,
                repetition + 1,
                args.repetitions
            );
            let before = ProcessSnapshot::capture()?;
            let result = runtime.generate(&prompt.rendered, &options)?;
            let after = ProcessSnapshot::capture()?;
            let prompt_count_matches =
                result.prompt_token_ids.len() == prompt.definition.expected_prompt_tokens;
            let prefix_status = correctness(
                prompt.definition.expected_token_prefix.as_deref(),
                &result.generated_token_ids,
            );
            let correctness_status = if !prompt_count_matches || prefix_status == "fail" {
                "fail"
            } else if prefix_status == "pass" {
                "pass"
            } else {
                "prompt_count_only"
            };
            let record = Record {
                schema_version: SCHEMA_VERSION,
                timestamp_unix_ms: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .context("system clock is before the Unix epoch")?
                    .as_millis(),
                source: &source,
                build: &build,
                host: &host,
                model: &identity,
                load: LoadInfo {
                    seconds: load_seconds,
                    process: &load_process,
                },
                warmup: WarmupInfo { ..warmup },
                run: RunInfo {
                    name: &prompt.definition.name,
                    category: &prompt.definition.category,
                    repetition,
                    residency,
                },
                workload: WorkloadInfo {
                    prompt: &prompt.definition.prompt,
                    rendered_prompt: &prompt.rendered,
                    chat: prompt.definition.chat,
                    system_prompt: prompt.definition.system_prompt.as_deref(),
                    expected_prompt_tokens: prompt.definition.expected_prompt_tokens,
                    max_new_tokens: prompt.definition.max_new_tokens,
                },
                tokens: TokenInfo {
                    prompt_ids: &result.prompt_token_ids,
                    evaluated_input_ids: &result.evaluated_input_token_ids,
                    generated_ids: &result.generated_token_ids,
                    generated_text: &result.text,
                },
                correctness: CorrectnessInfo {
                    status: correctness_status,
                    prompt_token_count_matches: prompt_count_matches,
                    expected_token_prefix: prompt.definition.expected_token_prefix.as_deref(),
                },
                performance: PerformanceInfo {
                    prompt_tokens: result.metrics.prompt_tokens,
                    evaluated_input_tokens: result.metrics.evaluated_input_tokens,
                    generated_tokens: result.metrics.generated_tokens,
                    decode_passes: result.metrics.generated_tokens.saturating_sub(1),
                    context_tokens: runtime.context_tokens(),
                    time_to_first_token_seconds: result.metrics.time_to_first_token.as_secs_f64(),
                    prefill_seconds: result.metrics.prefill_wall_time.as_secs_f64(),
                    prefill_tokens_per_second: result.metrics.prefill_tokens_per_second(),
                    decode_seconds: result.metrics.decode_wall_time.as_secs_f64(),
                    decode_tokens_per_second: result.metrics.decode_tokens_per_second(),
                },
                expert_cache: result.metrics.expert_cache,
                process: before.delta(&after),
                timings: TimingInfo {
                    prefill: result.metrics.prefill_profile.report(),
                    decode: result.metrics.decode_profile.report(),
                },
            };
            serde_json::to_writer(&mut output, &record)?;
            writeln!(output)?;
            output.flush()?;
            eprintln!(
                "completed {:?}: TTFT {:.3}s, decode {:.2} tok/s, correctness {}",
                prompt.definition.name,
                result.metrics.time_to_first_token.as_secs_f64(),
                result.metrics.decode_tokens_per_second(),
                correctness_status
            );
            if correctness_status == "fail" {
                bail!(
                    "correctness gate failed for workload {:?}",
                    prompt.definition.name
                );
            }
        }
    }
    Ok(())
}

fn correctness(expected: Option<&[u32]>, actual: &[u32]) -> &'static str {
    match expected {
        None => "not_checked",
        Some(expected) if actual.starts_with(expected) => "pass",
        Some(_) => "fail",
    }
}

#[cfg(test)]
mod tests {
    use super::correctness;

    #[test]
    fn correctness_checks_a_prefix() {
        assert_eq!(correctness(Some(&[1, 2]), &[1, 2, 3]), "pass");
        assert_eq!(correctness(Some(&[1, 2]), &[1, 4]), "fail");
        assert_eq!(correctness(None, &[1]), "not_checked");
    }
}
