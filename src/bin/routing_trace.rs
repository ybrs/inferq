use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use qwen_engine::{GenerationOptions, Runtime, trace::JsonlRoutingTrace};

#[derive(Debug, Parser)]
#[command(about = "Generate a deterministic Qwen3-Coder-Next routing trace")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,
    #[arg(long)]
    include_logits: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut runtime = Runtime::load(args.model)?;
    runtime.set_trace(Some(Box::new(JsonlRoutingTrace::create(
        args.output,
        args.include_logits,
    )?)));
    let result = runtime.generate(
        &args.prompt,
        &GenerationOptions {
            max_new_tokens: args.max_new_tokens,
            ..Default::default()
        },
    )?;
    println!("{}", result.text);
    Ok(())
}
