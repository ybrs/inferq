//! Decode throughput as a function of how much context is already in the
//! sequence, with the cost attributed to a stage.
//!
//! Every other benchmark in this repository measures a short prompt — the
//! qualified sustained case reaches 151 context tokens — but an agent request
//! arrives with thousands, and decode is not the same operation at both
//! depths. Full-attention layers scan a KV cache that grows with the
//! conversation while the linear and MoE layers cost the same at any depth, so
//! a single decode figure hides which of them a long request is paying for.
//!
//! ```bash
//! CARGO_TARGET_DIR=target-native RUSTFLAGS='-C target-cpu=native' \
//!   cargo build --release --bin gguf_decode_depth
//!
//! INFERQ_NUM_THREADS=6 ./target-native/release/gguf_decode_depth \
//!   --model "${INFERQ_GGUF}" --tokenizer-model "${INFERQ_TOKENIZER_DIR}" \
//!   --expert-cache-mib 24000
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use qwen_engine::{GenerationOptions, GgufCheckpoint, QuantizedRuntime, SpeculativeMode};
use std::{path::PathBuf, time::Instant};

#[derive(Debug, Parser)]
#[command(about = "Decode throughput against context depth, attributed by stage")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer_model: PathBuf,
    /// Context depths to measure, in tokens.
    #[arg(long, value_delimiter = ',', default_values_t = [64usize, 512, 1024, 2048, 3072])]
    depths: Vec<usize>,
    /// Tokens decoded at each depth. Enough to average over, few enough that
    /// the depth barely moves while they are produced.
    #[arg(long, default_value_t = 16)]
    decode_tokens: usize,
    #[arg(long, default_value_t = 0)]
    expert_cache_mib: usize,
    /// Warm every expert first, so expert residency is not what is measured.
    #[arg(long)]
    warmup_all_experts: bool,
}

fn main() -> Result<()> {
    qwen_engine::threading::init();
    let args = Args::parse();
    anyhow::ensure!(args.decode_tokens > 0, "--decode-tokens must be at least 1");
    let checkpoint = GgufCheckpoint::open(&args.model)
        .with_context(|| format!("failed to open {}", args.model.display()))?;
    checkpoint.configure_expert_cache(args.expert_cache_mib.saturating_mul(1024 * 1024))?;
    let mut runtime = QuantizedRuntime::load(&checkpoint, &args.tokenizer_model)?;
    if args.warmup_all_experts {
        qwen_engine::warm_all_experts(&checkpoint, runtime.model().config(), |_| {})?;
    }

    // One long stream of ordinary prose to slice prefixes out of, so each
    // depth measures the same text and only the length differs.
    let filler =
        "The engine evaluates each token against every earlier token in the sequence. ".repeat(600);
    let tokens = runtime.tokenizer().encode(&filler, false)?;
    let deepest = args.depths.iter().copied().max().unwrap_or_default();
    anyhow::ensure!(
        tokens.len() > deepest,
        "the filler tokenises to {} tokens, which cannot reach a depth of {deepest}",
        tokens.len()
    );

    println!(
        "depth,prefill_tok_s,decode_tok_s,attn_scan_s,scores_s,softmax_s,weighted_s,\
         attn_other_s,linear_s,moe_s,lm_head_s"
    );
    println!("# linear split: proj,conv,recurrence,gated_norm,out_proj,snapshot");
    for depth in &args.depths {
        let depth = *depth;
        // Speculation off: this measures what the target itself costs, which
        // is what the arms are then trying to amortise.
        let options = GenerationOptions {
            max_new_tokens: args.decode_tokens,
            speculative_mode: SpeculativeMode::Off,
            speculative_mtp_draft_tokens: 0,
            speculative_ngram_draft_tokens: 0,
            ..GenerationOptions::default()
        };
        runtime.reset();
        let started = Instant::now();
        runtime.prefill_tokens(&tokens[..depth], false)?;
        let prefill = started.elapsed().as_secs_f64();
        let result =
            runtime
                .generate_tokens_with_callback(&tokens[depth..depth + 1], &options, |_| Ok(()))?;
        let metrics = &result.metrics;
        let profile = &metrics.decode_profile;
        let (mut scan, mut other, mut linear, mut moe) = (0., 0., 0., 0.);
        // The three parts are summed across threads, so they are reported as a
        // share of their own total rather than of the wall clock.
        let (mut scores, mut softmax, mut weighted) = (0., 0., 0.);
        let mut delta = (0., 0., 0., 0., 0., 0.);
        for layer in &profile.layer_details {
            scan += layer.attention.attention.as_secs_f64();
            other += (layer.attention.wall - layer.attention.attention).as_secs_f64();
            linear += layer.delta.wall.as_secs_f64();
            moe += layer.moe.wall.as_secs_f64();
            scores += layer.attention.scores.as_secs_f64();
            softmax += layer.attention.softmax.as_secs_f64();
            weighted += layer.attention.weighted_sum.as_secs_f64();
            delta.0 += layer.delta.projections.as_secs_f64();
            delta.1 += layer.delta.convolution.as_secs_f64();
            delta.2 += layer.delta.recurrence.as_secs_f64();
            delta.3 += layer.delta.gated_norm.as_secs_f64();
            delta.4 += layer.delta.output_projection.as_secs_f64();
            delta.5 += layer.delta.snapshot.as_secs_f64();
        }
        println!(
            "{depth},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            depth as f64 / prefill,
            metrics.generated_tokens as f64 / metrics.decode_wall_time.as_secs_f64(),
            scan,
            scores,
            softmax,
            weighted,
            other,
            linear,
            moe,
            profile.lm_head.as_secs_f64(),
        );
        println!(
            "# {depth}: {:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            delta.0, delta.1, delta.2, delta.3, delta.4, delta.5
        );
    }
    Ok(())
}
