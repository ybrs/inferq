use std::{path::PathBuf, time::Instant};

use anyhow::Result;
use candle_core::{Device, Tensor};
use clap::Parser;
use qwen_engine::{GgufCheckpoint, profile::ProcessSnapshot};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Execute one GGUF matrix without whole-matrix dequantization")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tensor: String,
    /// Select one matrix from a fused [experts, rows, columns] tensor.
    #[arg(long)]
    expert: Option<usize>,
    #[arg(long, default_value_t = 1)]
    repetitions: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    tensor: String,
    expert: Option<usize>,
    dtype: String,
    shape: [usize; 2],
    compressed_storage_bytes: usize,
    load_seconds: f64,
    mean_forward_seconds: f64,
    output_l2_norm: f64,
    output_prefix: Vec<f32>,
    process: qwen_engine::profile::ProcessDelta,
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.repetitions > 0, "--repetitions must be at least one");
    let checkpoint = GgufCheckpoint::open(&args.model)?;
    let before = ProcessSnapshot::capture()?;
    let load_started = Instant::now();
    let matrix = match args.expert {
        Some(expert) => checkpoint.load_expert_matrix(&args.tensor, expert)?,
        None => checkpoint.load_matrix(&args.tensor)?,
    };
    let load_seconds = load_started.elapsed().as_secs_f64();
    let shape = matrix.shape();
    let values: Vec<f32> = (0..shape[1])
        .map(|index| ((index % 31) as f32 - 15.) / 16.)
        .collect();
    let input = Tensor::from_vec(values, (1, shape[1]), &Device::Cpu)?;
    let forward_started = Instant::now();
    let mut output = None;
    for _ in 0..args.repetitions {
        output = Some(matrix.forward(&input)?);
    }
    let mean_forward_seconds = forward_started.elapsed().as_secs_f64() / args.repetitions as f64;
    let values = output
        .expect("positive repetition count")
        .to_vec2::<f32>()?;
    let values = &values[0];
    let output_l2_norm = values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    let after = ProcessSnapshot::capture()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            tensor: args.tensor,
            expert: args.expert,
            dtype: matrix.dtype(),
            shape,
            compressed_storage_bytes: matrix.storage_bytes(),
            load_seconds,
            mean_forward_seconds,
            output_l2_norm,
            output_prefix: values.iter().copied().take(8).collect(),
            process: before.delta(&after),
        })?
    );
    Ok(())
}
