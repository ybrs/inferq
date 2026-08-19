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

    // Where one wide pass actually spends itself. Prefill is the same layers
    // decode runs, so the interesting question is whether the ratios differ.
    runtime.reset();
    let options = GenerationOptions {
        max_new_tokens: 1,
        speculative_mode: SpeculativeMode::Off,
        speculative_mtp_draft_tokens: 0,
        speculative_ngram_draft_tokens: 0,
        ..GenerationOptions::default()
    };
    let result = runtime.generate_tokens_with_callback(tokens, &options, |_| Ok(()))?;
    let profile = &result.metrics.prefill_profile;
    let (mut scan, mut other, mut linear, mut moe) = (0., 0., 0., 0.);
    let mut delta = (0., 0., 0., 0., 0., 0.);
    for layer in &profile.layer_details {
        scan += layer.attention.attention.as_secs_f64();
        other += (layer.attention.wall - layer.attention.attention).as_secs_f64();
        linear += layer.delta.wall.as_secs_f64();
        moe += layer.moe.wall.as_secs_f64();
        delta.0 += layer.delta.projections.as_secs_f64();
        delta.1 += layer.delta.convolution.as_secs_f64();
        delta.2 += layer.delta.recurrence.as_secs_f64();
        delta.3 += layer.delta.gated_norm.as_secs_f64();
        delta.4 += layer.delta.output_projection.as_secs_f64();
        delta.5 += layer.delta.snapshot.as_secs_f64();
    }
    println!();
    println!(
        "one pass of {} tokens: {:.2} s",
        args.tokens,
        profile.wall.as_secs_f64()
    );
    for (name, seconds) in [
        ("moe", moe),
        ("linear", linear),
        ("attention scan", scan),
        ("attention other", other),
        ("lm_head", profile.lm_head.as_secs_f64()),
    ] {
        println!(
            "  {name:<16} {seconds:>7.2} s  {:>4.1}%",
            100. * seconds / profile.wall.as_secs_f64()
        );
    }
    println!("  linear splits into:");
    for (name, seconds) in [
        ("projections", delta.0),
        ("convolution", delta.1),
        ("recurrence", delta.2),
        ("gated_norm", delta.3),
        ("output_projection", delta.4),
        ("snapshot", delta.5),
    ] {
        println!(
            "    {name:<18} {seconds:>7.2} s  {:>4.1}%",
            100. * seconds / profile.wall.as_secs_f64()
        );
    }
    Ok(())
}
