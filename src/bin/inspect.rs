use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use qwen_engine::{Checkpoint, GgufCheckpoint, inspect_gguf};

#[derive(Debug, Parser)]
#[command(about = "Validate and inspect a Qwen3-Coder-Next checkpoint")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tensors: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.model.is_file() {
        println!(
            "{}",
            serde_json::to_string_pretty(&inspect_gguf(&args.model)?)?
        );
        if args.tensors {
            for tensor in GgufCheckpoint::open(&args.model)?.tensor_infos() {
                println!("{}", serde_json::to_string(&tensor)?);
            }
        }
        return Ok(());
    }
    let checkpoint = Checkpoint::open(args.model)?;
    println!("{}", serde_json::to_string_pretty(&checkpoint.summary())?);
    if args.tensors {
        for tensor in checkpoint.tensor_infos() {
            println!("{}", serde_json::to_string(tensor)?);
        }
    }
    Ok(())
}
