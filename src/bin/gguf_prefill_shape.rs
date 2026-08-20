//! Does the width of a prefill pass change what it costs?
//!
//! `QuantizedMatrix::forward` uses the fused multi-row kernel only for 2..=16
//! rows and otherwise falls through to the per-row path, so a prefill wider
//! than sixteen tokens may be paying to decode the same weight blocks once per
//! row. This prefills the same tokens in different pass widths and reports
//! what each width cost, which settles that without reading the kernel.
use anyhow::{Context, Result};
use clap::Parser;
use qwen_engine::{GenerationOptions, GgufCheckpoint, QuantizedRuntime, SpeculativeMode};
use std::{path::PathBuf, time::Instant};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer_model: PathBuf,
    /// Tokens prefilled in total, the same for every width.
    #[arg(long, default_value_t = 256)]
    tokens: usize,
    /// Pass widths to compare.
    #[arg(long, value_delimiter = ',', default_values_t = [16usize, 32, 64, 128, 256])]
    widths: Vec<usize>,
    #[arg(long, default_value_t = 0)]
    expert_cache_mib: usize,
    #[arg(long)]
    warmup_all_experts: bool,
}

fn main() -> Result<()> {
    qwen_engine::threading::init();
    let args = Args::parse();
    let checkpoint = GgufCheckpoint::open(&args.model)
        .with_context(|| format!("failed to open {}", args.model.display()))?;
    checkpoint.configure_expert_cache(args.expert_cache_mib.saturating_mul(1024 * 1024))?;
    let mut runtime = QuantizedRuntime::load(&checkpoint, &args.tokenizer_model)?;
    if args.warmup_all_experts {
        qwen_engine::warm_all_experts(&checkpoint, runtime.model().config(), |_| {})?;
    }
    let filler =
        "The engine evaluates each token against every earlier token in the sequence. ".repeat(600);
    let tokens = runtime.tokenizer().encode(&filler, false)?;
    anyhow::ensure!(
        tokens.len() >= args.tokens,
        "filler is shorter than --tokens"
    );
    let tokens = &tokens[..args.tokens];

    println!("width,passes,seconds,prefill_tok_s");
    for width in &args.widths {
        let width = (*width).min(args.tokens);
        runtime.reset();
        let started = Instant::now();
        let mut passes = 0;
        for chunk in tokens.chunks(width) {
            runtime.prefill_tokens(chunk, false)?;
            passes += 1;
        }
        let seconds = started.elapsed().as_secs_f64();
        println!(
            "{width},{passes},{seconds:.2},{:.2}",
            args.tokens as f64 / seconds
        );
    }

    // What one pass of each width costs PER TOKEN. A wider pass reads each
    // expert's weights once for more tokens, so if that reuse is being taken
    // the MoE column falls as the width rises; if it is not, it stays flat.
    println!();
    println!("one pass of N rows, milliseconds per token");
    println!("width  total    moe  linear   proj  recur    moe_load  moe_compute");
    for width in &args.widths {
        let width = (*width).min(args.tokens);
        let options = GenerationOptions {
            max_new_tokens: 1,
            speculative_mode: SpeculativeMode::Off,
            speculative_mtp_draft_tokens: 0,
            speculative_ngram_draft_tokens: 0,
            ..GenerationOptions::default()
        };
        runtime.reset();
        let result =
            runtime.generate_tokens_with_callback(&tokens[..width], &options, |_| Ok(()))?;
        let profile = &result.metrics.prefill_profile;
        let (mut moe, mut linear, mut proj, mut recur) = (0., 0., 0., 0.);
        let (mut load, mut compute) = (0., 0.);
        for layer in &profile.layer_details {
            moe += layer.moe.wall.as_secs_f64();
            linear += layer.delta.wall.as_secs_f64();
            proj += layer.delta.projections.as_secs_f64();
            recur += layer.delta.recurrence.as_secs_f64();
            load += layer.moe.expert_load.as_secs_f64();
            compute += layer.moe.expert_compute.as_secs_f64();
        }
        let per = |seconds: f64| seconds / width as f64 * 1000.;
        println!(
            "{width:>5} {:>6.2} {:>6.2} {:>7.2} {:>6.2} {:>6.2} {:>11.2} {:>12.2}",
            per(profile.wall.as_secs_f64()),
            per(moe),
            per(linear),
            per(proj),
            per(recur),
            per(load),
            per(compute),
        );
    }
    Ok(())
}
