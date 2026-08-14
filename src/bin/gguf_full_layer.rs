use std::{path::PathBuf, time::Instant};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use qwen_engine::{
    Checkpoint, GgufCheckpoint,
    profile::ProcessSnapshot,
    qwen::{QuantizedFullLayer, reference_full_layer},
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Compare one complete quantized full-attention layer with BF16")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    reference_model: PathBuf,
    #[arg(long, default_value_t = 3)]
    layer: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    layer: usize,
    selected_experts: Vec<usize>,
    reference_experts: Vec<usize>,
    routing_match: bool,
    load_seconds: f64,
    wall_seconds: f64,
    normalization_seconds: f64,
    attention_seconds: f64,
    moe_seconds: f64,
    expert_load_seconds: f64,
    max_abs_error: f32,
    mean_abs_error: f64,
    root_mean_square_error: f64,
    reference_l2_norm: f64,
    quantized_l2_norm: f64,
    process: qwen_engine::profile::ProcessDelta,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let reference_checkpoint = Checkpoint::open(&args.reference_model)?;
    let config = reference_checkpoint.config();
    let checkpoint = GgufCheckpoint::open(&args.model)?;
    let before = ProcessSnapshot::capture()?;
    let load_started = Instant::now();
    let layer = QuantizedFullLayer::load(&checkpoint, config, args.layer)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let values: Vec<f32> = (0..config.hidden_size)
        .map(|index| ((index % 31) as f32 - 15.) / 16.)
        .collect();
    let input = Tensor::from_vec(values, (1, config.hidden_size), &Device::Cpu)?;
    let mut state = layer.new_state();
    let quantized = layer.forward(&input, 0, &mut state)?;
    let reference = reference_full_layer(
        &reference_checkpoint,
        config,
        args.layer,
        &input.to_dtype(DType::BF16)?,
    )?;
    let quantized_values = quantized.hidden.to_vec2::<f32>()?.remove(0);
    let reference_values = reference
        .hidden
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?
        .remove(0);
    let errors: Vec<f32> = reference_values
        .iter()
        .zip(&quantized_values)
        .map(|(reference, actual)| (reference - actual).abs())
        .collect();
    let squared_error = errors
        .iter()
        .map(|error| f64::from(*error) * f64::from(*error))
        .sum::<f64>();
    let l2 = |values: &[f32]| {
        values
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt()
    };
    let after = ProcessSnapshot::capture()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            layer: args.layer,
            selected_experts: quantized.routes[0].experts.clone(),
            reference_experts: reference.routes[0].experts.clone(),
            routing_match: quantized.routes[0].experts == reference.routes[0].experts,
            load_seconds,
            wall_seconds: quantized.timings.wall.as_secs_f64(),
            normalization_seconds: quantized.timings.normalization.as_secs_f64(),
            attention_seconds: quantized.timings.attention.wall.as_secs_f64(),
            moe_seconds: quantized.timings.moe.wall.as_secs_f64(),
            expert_load_seconds: quantized.timings.moe.expert_load.as_secs_f64(),
            max_abs_error: errors.iter().copied().fold(0., f32::max),
            mean_abs_error: errors.iter().map(|error| f64::from(*error)).sum::<f64>()
                / errors.len() as f64,
            root_mean_square_error: (squared_error / errors.len() as f64).sqrt(),
            reference_l2_norm: l2(&reference_values),
            quantized_l2_norm: l2(&quantized_values),
            process: before.delta(&after),
        })?
    );
    Ok(())
}
