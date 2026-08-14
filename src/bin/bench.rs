use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use qwen_engine::{GenerationOptions, Runtime};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(about = "Repeatable coding-workload benchmark")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "benchmarks/prompts.json")]
    prompts: PathBuf,
    #[arg(long, default_value_t = 1)]
    repetitions: usize,
}

#[derive(Debug, Deserialize)]
struct Prompt {
    name: String,
    category: String,
    prompt: String,
    max_new_tokens: usize,
}

#[derive(Debug, Serialize)]
struct Record<'a> {
    name: &'a str,
    category: &'a str,
    repetition: usize,
    prompt_tokens: usize,
    generated_tokens: usize,
    prefill_seconds: f64,
    prefill_tokens_per_second: f64,
    decode_seconds: f64,
    decode_tokens_per_second: f64,
    threads: usize,
    model: String,
    format: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let prompts: Vec<Prompt> = serde_json::from_slice(
        &fs::read(&args.prompts)
            .with_context(|| format!("failed to read {}", args.prompts.display()))?,
    )?;
    let mut runtime = Runtime::load(&args.model)?;
    let summary = runtime.model().checkpoint().summary();
    for prompt in &prompts {
        for repetition in 0..args.repetitions {
            let result = runtime.generate(
                &prompt.prompt,
                &GenerationOptions {
                    max_new_tokens: prompt.max_new_tokens,
                    ..Default::default()
                },
            )?;
            println!(
                "{}",
                serde_json::to_string(&Record {
                    name: &prompt.name,
                    category: &prompt.category,
                    repetition,
                    prompt_tokens: result.metrics.prompt_tokens,
                    generated_tokens: result.metrics.generated_tokens,
                    prefill_seconds: result.metrics.prefill_wall_time.as_secs_f64(),
                    prefill_tokens_per_second: result.metrics.prefill_tokens_per_second(),
                    decode_seconds: result.metrics.decode_wall_time.as_secs_f64(),
                    decode_tokens_per_second: result.metrics.decode_tokens_per_second(),
                    threads: std::thread::available_parallelism()
                        .map(usize::from)
                        .unwrap_or(1),
                    model: summary.architecture.clone(),
                    format: summary.format.clone(),
                })?
            );
        }
    }
    Ok(())
}
