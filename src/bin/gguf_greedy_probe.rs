//! Print the exact token ids a greedy turn produces, for equivalence checks.
use anyhow::Result;
use clap::Parser;
use qwen_engine::{GenerationOptions, GgufCheckpoint, QuantizedRuntime, SpeculativeMode};
use std::path::PathBuf;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer_model: PathBuf,
    #[arg(long, default_value_t = 32)]
    tokens: usize,
    /// Context depth to reach before decoding, so the scan is exercised.
    #[arg(long, default_value_t = 0)]
    depth: usize,
    #[arg(long, default_value_t = 0)]
    expert_cache_mib: usize,
    /// Run the speculative policy instead of plain decode. Verification
    /// evaluates several rows in one attention pass, so this exercises the
    /// multi-row path; it must produce the same tokens either way.
    #[arg(long)]
    speculative: bool,
}

fn main() -> Result<()> {
    qwen_engine::threading::init();
    let args = Args::parse();
    let checkpoint = GgufCheckpoint::open(&args.model)?;
    checkpoint.configure_expert_cache(args.expert_cache_mib.saturating_mul(1024 * 1024))?;
    let mut runtime = QuantizedRuntime::load(&checkpoint, &args.tokenizer_model)?;
    let filler =
        "The engine evaluates each token against every earlier token in the sequence. ".repeat(600);
    let tokens = runtime.tokenizer().encode(&filler, false)?;
    let options = if args.speculative {
        GenerationOptions {
            max_new_tokens: args.tokens,
            speculative_mode: SpeculativeMode::Auto,
            speculative_mtp_draft_tokens: 4,
            speculative_ngram_draft_tokens: 8,
            ..GenerationOptions::default()
        }
    } else {
        GenerationOptions {
            max_new_tokens: args.tokens,
            speculative_mode: SpeculativeMode::Off,
            speculative_mtp_draft_tokens: 0,
            speculative_ngram_draft_tokens: 0,
            ..GenerationOptions::default()
        }
    };
    runtime.reset();
    if args.depth > 0 {
        runtime.prefill_tokens(&tokens[..args.depth], false)?;
    }
    let result = runtime.generate_tokens_with_callback(
        &tokens[args.depth..args.depth + 1],
        &options,
        |_| Ok(()),
    )?;
    println!(
        "{}",
        result
            .generated_token_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}
