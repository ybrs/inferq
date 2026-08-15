use std::{path::PathBuf, time::Instant};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use qwen_engine::{
    Checkpoint, GgufCheckpoint, LayerType,
    profile::ProcessSnapshot,
    qwen::{QuantizedAttentionLayer, ReferenceAttentionState, reference_attention_step},
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Compare one quantized GGUF full-attention mixer with BF16")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    reference_model: PathBuf,
    #[arg(long, default_value_t = 3)]
    layer: usize,
    #[arg(long, default_value_t = 1)]
    steps: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    layer: usize,
    steps: usize,
    load_seconds: f64,
    wall_seconds: f64,
    projections_seconds: f64,
    norm_rope_seconds: f64,
    attention_seconds: f64,
    output_projection_seconds: f64,
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
    anyhow::ensure!(
        config.layer_type(args.layer) == LayerType::FullAttention,
        "layer {} is not a full-attention layer",
        args.layer
    );
    let checkpoint = GgufCheckpoint::open(&args.model)?;
    let before = ProcessSnapshot::capture()?;
    let load_started = Instant::now();
    let layer = QuantizedAttentionLayer::load(&checkpoint, config, args.layer)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    anyhow::ensure!(args.steps > 0, "--steps must be at least one");
    let mut state = layer.new_state();
    let mut reference_state = ReferenceAttentionState::default();
    let mut final_result = None;
    for step in 0..args.steps {
        let values: Vec<f32> = (0..config.hidden_size)
            .map(|index| (((index + step * 7) % 31) as f32 - 15.) / 16.)
            .collect();
        let input = Tensor::from_vec(values, (1, config.hidden_size), &Device::Cpu)?;
        let (quantized, timings) = layer.forward(&input, step, &mut state)?;
        let reference = reference_attention_step(
            &reference_checkpoint,
            config,
            args.layer,
            &input.to_dtype(DType::BF16)?,
            step,
            &mut reference_state,
        )?;
        final_result = Some((quantized, reference, timings));
    }
    let (quantized, reference, timings) = final_result.expect("positive step count");
    let quantized = quantized.to_vec2::<f32>()?.remove(0);
    let reference = reference.to_dtype(DType::F32)?.to_vec2::<f32>()?.remove(0);
    let errors: Vec<f32> = reference
        .iter()
        .zip(&quantized)
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
            steps: args.steps,
            load_seconds,
            wall_seconds: timings.wall.as_secs_f64(),
            projections_seconds: timings.projections.as_secs_f64(),
            norm_rope_seconds: timings.norm_rope.as_secs_f64(),
            attention_seconds: timings.attention.as_secs_f64(),
            output_projection_seconds: timings.output_projection.as_secs_f64(),
            max_abs_error: errors.iter().copied().fold(0., f32::max),
            mean_abs_error: errors.iter().map(|error| f64::from(*error)).sum::<f64>()
                / errors.len() as f64,
            root_mean_square_error: (squared_error / errors.len() as f64).sqrt(),
            reference_l2_norm: l2(&reference),
            quantized_l2_norm: l2(&quantized),
            process: before.delta(&after),
        })?
    );
    Ok(())
}
