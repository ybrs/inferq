use std::{
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Result, ensure};
use clap::Parser;
use qwen_engine::{
    GenerationOptions, GgufCheckpoint, GgufModelIdentity, QuantizedGenerationResult,
    QuantizedRuntime,
    profile::{ProcessDelta, ProcessSnapshot},
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
    /// Enable Qwen3.5/3.6 MTP speculation with at most N draft tokens per verification pass.
    #[arg(long, default_value_t = 0)]
    speculative_mtp: usize,
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
            "MTP speculation: {}/{} draft tokens accepted ({:.1}%); {} verification passes over {} tokens; {} rollback replays over {} tokens; draft {:.3}s, verify {:.3}s, resync {:.3}s",
            speculative.accepted_tokens,
            speculative.drafted_tokens,
            speculative.acceptance_rate() * 100.,
            speculative.verification_passes,
            speculative.verification_tokens,
            speculative.rollback_replays,
            speculative.replayed_tokens,
            speculative.draft_wall_time.as_secs_f64(),
            speculative.verification_wall_time.as_secs_f64(),
            speculative.resync_wall_time.as_secs_f64(),
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
    Ok(result)
}

fn user_turn_prompt(
    runtime: &QuantizedRuntime<'_>,
    user: &str,
    chat: bool,
    first_turn: bool,
    assistant_closed: bool,
    system_prompt: Option<&str>,
) -> Result<String> {
    if !chat {
        return Ok(user.to_owned());
    }
    if first_turn {
        runtime.tokenizer().initial_chat_prompt(user, system_prompt)
    } else {
        runtime
            .tokenizer()
            .chat_continuation(user, assistant_closed)
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
        )?;
        let result = generate_and_report(runtime, &prompt, options)?;
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
                )?;
                let result = generate_and_report(runtime, &prompt, options)?;
                assistant_closed = emitted_im_end(runtime, &result);
                first_turn = false;
            }
        }
    }
    Ok(())
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
    ensure!(
        args.speculative_mtp == 0
            || (args.routing_trace.is_none() && args.routing_census.is_none()),
        "--speculative-mtp does not yet support routing traces or censuses"
    );

    let checkpoint = GgufCheckpoint::open(&args.model)?;
    let expert_cache_bytes = args
        .expert_cache_mib
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("--expert-cache-mib is too large"))?;
    checkpoint.configure_expert_cache(expert_cache_bytes)?;
    let identity = checkpoint.identity()?;
    let load_started = Instant::now();
    let mut runtime = QuantizedRuntime::load(&checkpoint, &args.tokenizer_model)?;
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

    let options = GenerationOptions {
        max_new_tokens: args.max_new_tokens,
        speculative_mtp_draft_tokens: args.speculative_mtp,
        ..GenerationOptions::default()
    };
    if args.interactive {
        interactive(
            &mut runtime,
            args.prompt.as_deref(),
            &options,
            args.chat,
            args.system_prompt.as_deref(),
        )
    } else {
        let prompt = user_turn_prompt(
            &runtime,
            args.prompt.as_deref().expect("prompt validated above"),
            args.chat,
            true,
            false,
            args.system_prompt.as_deref(),
        )?;
        generate_and_report(&mut runtime, &prompt, &options)?;
        Ok(())
    }
}
