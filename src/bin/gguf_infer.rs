use std::{path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use candle_core::IndexOp;
use clap::Parser;
use qwen_engine::{
    GgufCheckpoint, Qwen3NextConfig, qwen::QuantizedModel, tokenizer::ModelTokenizer,
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
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 1)]
    max_new_tokens: usize,
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
    anyhow::ensure!(
        args.max_new_tokens > 0,
        "--max-new-tokens must be at least one"
    );
    let config = Qwen3NextConfig::from_path(args.tokenizer_model.join("config.json"))?;
    let tokenizer = ModelTokenizer::from_model_dir(&args.tokenizer_model)?;
    let checkpoint = GgufCheckpoint::open(&args.model)?;
    let load_started = Instant::now();
    let model = QuantizedModel::load(&checkpoint, config)?;
    let load_time = load_started.elapsed();
    let prompt_ids = tokenizer.encode(&args.prompt, false)?;
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt token sequence is empty");
    let mut state = model.new_state();
    let prefill_started = Instant::now();
    let (mut logits, prefill_timings) = model.forward(&prompt_ids, &mut state)?;
    let prefill_time = prefill_started.elapsed();
    let decode_started = Instant::now();
    let mut generated = Vec::with_capacity(args.max_new_tokens);
    for step in 0..args.max_new_tokens {
        let row = logits.i(logits.dim(0)? - 1)?.to_vec1::<f32>()?;
        let token = row
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .context("LM head produced no logits")?;
        generated.push(token);
        if step + 1 == args.max_new_tokens
            || model
                .config()
                .eos_token_id
                .as_ref()
                .is_some_and(|ids| ids.contains(token))
        {
            break;
        }
        (logits, _) = model.forward(&[token], &mut state)?;
    }
    let decode_time = decode_started.elapsed();
    let decode_passes = generated.len().saturating_sub(1);
    let text = tokenizer.decode(&generated, true)?;
    println!("{text}");
    eprintln!("prompt token ids: {prompt_ids:?}");
    eprintln!("generated token ids: {generated:?}");
    eprintln!(
        "model load: {:.3}s; prefill: {} tokens in {:.3}s ({:.2} tok/s); decode: {} passes in {:.3}s ({:.2} tok/s); model prefill: {:.3}s",
        load_time.as_secs_f64(),
        prompt_ids.len(),
        prefill_time.as_secs_f64(),
        prompt_ids.len() as f64 / prefill_time.as_secs_f64(),
        decode_passes,
        decode_time.as_secs_f64(),
        if decode_passes == 0 {
            0.
        } else {
            decode_passes as f64 / decode_time.as_secs_f64()
        },
        prefill_timings.wall.as_secs_f64(),
    );
    Ok(())
}
