use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, ensure};
use candle_core::IndexOp;
use clap::Parser;
use qwen_engine::{Checkpoint, qwen::Model};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Produce logits and compare them with a reference JSON vector")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokens: PathBuf,
    #[arg(long)]
    dump_logits: Option<PathBuf>,
    #[arg(long)]
    reference_logits: Option<PathBuf>,
    #[arg(long, default_value_t = 10)]
    top_n: usize,
}

#[derive(Debug, Serialize)]
struct Comparison {
    elements: usize,
    max_absolute_error: f32,
    mean_absolute_error: f64,
    cosine_similarity: f64,
    top_1_agreement: bool,
    top_n_overlap: usize,
}

fn top_n(values: &[f32], n: usize) -> Vec<usize> {
    let mut ids: Vec<_> = (0..values.len()).collect();
    ids.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]));
    ids.truncate(n.min(ids.len()));
    ids
}

fn compare(actual: &[f32], reference: &[f32], n: usize) -> Result<Comparison> {
    ensure!(
        actual.len() == reference.len(),
        "reference has {} logits, Rust produced {}",
        reference.len(),
        actual.len()
    );
    let mut max_absolute_error = 0f32;
    let mut sum_error = 0f64;
    let (mut dot, mut lhs_norm, mut rhs_norm) = (0f64, 0f64, 0f64);
    for (&a, &b) in actual.iter().zip(reference) {
        let error = (a - b).abs();
        max_absolute_error = max_absolute_error.max(error);
        sum_error += error as f64;
        dot += a as f64 * b as f64;
        lhs_norm += (a as f64).powi(2);
        rhs_norm += (b as f64).powi(2);
    }
    let actual_top = top_n(actual, n);
    let reference_top = top_n(reference, n);
    Ok(Comparison {
        elements: actual.len(),
        max_absolute_error,
        mean_absolute_error: sum_error / actual.len() as f64,
        cosine_similarity: dot / (lhs_norm.sqrt() * rhs_norm.sqrt()),
        top_1_agreement: actual_top.first() == reference_top.first(),
        top_n_overlap: actual_top
            .iter()
            .filter(|id| reference_top.contains(id))
            .count(),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let token_bytes = fs::read(&args.tokens)
        .with_context(|| format!("failed to read {}", args.tokens.display()))?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&token_bytes).context("tokens must be a JSON array of integers")?;
    ensure!(!tokens.is_empty(), "token file is empty");
    let model = Model::new(Checkpoint::open(&args.model)?);
    let mut state = model.new_state();
    let (logits, timings) = model.forward(&tokens, &mut state, None)?;
    let last = logits.i(logits.dim(0)? - 1)?.to_vec1::<f32>()?;
    if let Some(path) = args.dump_logits {
        let mut writer = BufWriter::new(fs::File::create(&path)?);
        for value in &last {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.flush()?;
    }
    if let Some(path) = args.reference_logits {
        let reference: Vec<f32> = serde_json::from_slice(&fs::read(path)?)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&compare(&last, &reference, args.top_n)?)?
        );
    } else {
        println!(
            "logits: {}; layer time: {:.3}s; lm head: {:.3}s",
            last.len(),
            timings
                .layers
                .iter()
                .map(|layer| layer.wall)
                .sum::<std::time::Duration>()
                .as_secs_f64(),
            timings.lm_head.as_secs_f64()
        );
    }
    Ok(())
}
