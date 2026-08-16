use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use qwen_engine::{GenerationOptions, Runtime, sampling::SamplingConfig, trace::JsonlRoutingTrace};

#[derive(Debug, Parser)]
#[command(about = "Run Qwen3-Coder-Next CPU inference")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.)]
    temperature: f32,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    min_p: Option<f32>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    stop_token: Vec<u32>,
    #[arg(long)]
    chat: bool,
    #[arg(long)]
    routing_trace: Option<PathBuf>,
    #[arg(long)]
    trace_router_logits: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let args = Args::parse();
    let mut runtime = Runtime::load(&args.model)?;
    eprintln!(
        "{}",
        serde_json::to_string_pretty(&runtime.model().checkpoint().summary())?
    );
    if let Some(path) = args.routing_trace {
        runtime.set_trace(Some(Box::new(JsonlRoutingTrace::create(
            path,
            args.trace_router_logits,
        )?)));
    }
    let prompt = if args.chat {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            args.prompt
        )
    } else {
        args.prompt
    };
    let result = runtime.generate(
        &prompt,
        &GenerationOptions {
            max_new_tokens: args.max_new_tokens,
            sampling: SamplingConfig {
                temperature: args.temperature,
                top_k: args.top_k,
                top_p: args.top_p,
                min_p: args.min_p,
                seed: args.seed,
            },
            stop_tokens: args.stop_token,
            add_special_tokens: false,
            speculative_mtp_draft_tokens: 0,
            speculative_mtp_min_margin: None,
            thinking_budget: None,
        },
    )?;
    print!("{}", result.text);
    eprintln!(
        "\nprompt token ids: {:?}\ngenerated token ids: {:?}",
        result.prompt_token_ids, result.generated_token_ids
    );
    eprintln!(
        "prompt: {} tokens in {:.3}s ({:.2} tok/s); decode: {} tokens in {:.3}s ({:.2} tok/s)",
        result.metrics.prompt_tokens,
        result.metrics.prefill_wall_time.as_secs_f64(),
        result.metrics.prefill_tokens_per_second(),
        result.metrics.generated_tokens,
        result.metrics.decode_wall_time.as_secs_f64(),
        result.metrics.decode_tokens_per_second()
    );
    Ok(())
}
