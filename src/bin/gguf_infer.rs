use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
    time::Instant,
};

use anyhow::{Result, ensure};
use clap::Parser;
use qwen_engine::{
    GenerationOptions, GgufCheckpoint, QuantizedGenerationResult, QuantizedRuntime,
    trace::{JsonRoutingCensus, JsonlRoutingTrace, RoutingTraceSet},
};

#[derive(Debug, Parser)]
#[command(about = "End-to-end quantized Qwen3-Coder-Next inference")]
struct Args {
    /// Qwen3-Coder-Next GGUF file.
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
    /// Write every layer-qualified routing decision as JSONL.
    #[arg(long)]
    routing_trace: Option<PathBuf>,
    /// Include all router logits in --routing-trace (large).
    #[arg(long)]
    trace_router_logits: bool,
    /// Write cumulative per-layer expert counts as a versioned JSON sidecar.
    #[arg(long)]
    routing_census: Option<PathBuf>,
    /// Retain recently used expert matrices in-process, bounded in MiB.
    #[arg(long, default_value_t = 0)]
    expert_cache_mib: usize,
}

fn report(result: &QuantizedGenerationResult, context_tokens: usize) -> Result<()> {
    println!("{}", result.text);
    io::stdout().flush()?;
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
    let cache = result.metrics.expert_cache;
    eprintln!(
        "expert cache: {}/{} hits ({:.1}%); loaded {:.1} MiB of GGUF ranges; resident {:.1}/{:.1} MiB in {} entries; {} evictions",
        cache.hits,
        cache.requests,
        cache.hit_rate() * 100.,
        cache.bytes_loaded as f64 / (1024. * 1024.),
        cache.resident_bytes as f64 / (1024. * 1024.),
        cache.capacity_bytes as f64 / (1024. * 1024.),
        cache.entries,
        cache.evictions,
    );
    Ok(())
}

fn generate_and_report(
    runtime: &mut QuantizedRuntime<'_>,
    prompt: &str,
    options: &GenerationOptions,
) -> Result<()> {
    let result = runtime.generate(prompt, options)?;
    report(&result, runtime.context_tokens())
}

fn interactive(
    runtime: &mut QuantizedRuntime<'_>,
    initial_prompt: Option<&str>,
    options: &GenerationOptions,
) -> Result<()> {
    eprintln!("interactive session ready; /reset clears sequence state, /quit exits");
    if let Some(prompt) = initial_prompt {
        generate_and_report(runtime, prompt, options)?;
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
                eprintln!("sequence state reset");
            }
            _ => generate_and_report(runtime, &line, options)?,
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

    let mut traces = RoutingTraceSet::default();
    if let Some(path) = args.routing_trace {
        traces.push(Box::new(JsonlRoutingTrace::create(
            path,
            args.trace_router_logits,
        )?));
    }
    if let Some(path) = args.routing_census {
        traces.push(Box::new(JsonRoutingCensus::create(path, identity)));
    }
    if !traces.is_empty() {
        runtime.set_trace(Some(Box::new(traces)));
    }

    let options = GenerationOptions {
        max_new_tokens: args.max_new_tokens,
        ..GenerationOptions::default()
    };
    if args.interactive {
        interactive(&mut runtime, args.prompt.as_deref(), &options)
    } else {
        generate_and_report(
            &mut runtime,
            args.prompt.as_deref().expect("prompt validated above"),
            &options,
        )
    }
}
