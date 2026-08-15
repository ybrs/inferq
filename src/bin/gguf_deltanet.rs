use std::{path::PathBuf, time::Instant};

use anyhow::Result;
use candle_core::{Device, Tensor};
use clap::Parser;
use qwen_engine::{
    Checkpoint, GgufCheckpoint,
    profile::ProcessSnapshot,
    qwen::{QuantizedDeltaLayer, reference_deltanet},
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Compare one quantized GGUF DeltaNet mixer with BF16")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    reference_model: PathBuf,
    #[arg(long, default_value_t = 0)]
    layer: usize,
}

#[derive(Debug, Serialize)]
struct TimingReport {
    layer_load_seconds: f64,
    wall_seconds: f64,
    projections_seconds: f64,
    convolution_seconds: f64,
    recurrence_seconds: f64,
    gated_norm_seconds: f64,
    output_projection_seconds: f64,
}

#[derive(Debug, Serialize)]
struct ErrorReport {
    max_abs_error: f32,
    mean_abs_error: f64,
    root_mean_square_error: f64,
    reference_l2_norm: f64,
    quantized_l2_norm: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    layer: usize,
    timings: TimingReport,
    errors: ErrorReport,
    quantized_output_prefix: Vec<f32>,
    reference_output_prefix: Vec<f32>,
    process: qwen_engine::profile::ProcessDelta,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let reference_checkpoint = Checkpoint::open(&args.reference_model)?;
    let config = reference_checkpoint.config();
    anyhow::ensure!(
        config.layer_type(args.layer) == qwen_engine::LayerType::LinearAttention,
        "layer {} is not a linear-attention layer",
        args.layer
    );
    let checkpoint = GgufCheckpoint::open(&args.model)?;
    let before = ProcessSnapshot::capture()?;
    let load_started = Instant::now();
    let layer = QuantizedDeltaLayer::load(&checkpoint, config, args.layer)?;
    let layer_load_seconds = load_started.elapsed().as_secs_f64();
    let values: Vec<f32> = (0..config.hidden_size)
        .map(|index| ((index % 31) as f32 - 15.) / 16.)
        .collect();
    let input = Tensor::from_vec(values, (1, config.hidden_size), &Device::Cpu)?;
    let mut state = layer.new_state();
    let (quantized, timings) = layer.forward(&input, &mut state)?;
    let reference = reference_deltanet(&reference_checkpoint, config, args.layer, &input)?;
    let quantized = quantized.to_vec2::<f32>()?.remove(0);
    let reference = reference
        .to_dtype(candle_core::DType::F32)?
        .to_vec2::<f32>()?
        .remove(0);
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
            timings: TimingReport {
                layer_load_seconds,
                wall_seconds: timings.wall.as_secs_f64(),
                projections_seconds: timings.projections.as_secs_f64(),
                convolution_seconds: timings.convolution.as_secs_f64(),
                recurrence_seconds: timings.recurrence.as_secs_f64(),
                gated_norm_seconds: timings.gated_norm.as_secs_f64(),
                output_projection_seconds: timings.output_projection.as_secs_f64(),
            },
            errors: ErrorReport {
                max_abs_error: errors.iter().copied().fold(0., f32::max),
                mean_abs_error: errors.iter().map(|error| f64::from(*error)).sum::<f64>()
                    / errors.len() as f64,
                root_mean_square_error: (squared_error / errors.len() as f64).sqrt(),
                reference_l2_norm: l2(&reference),
                quantized_l2_norm: l2(&quantized),
            },
            quantized_output_prefix: quantized.into_iter().take(8).collect(),
            reference_output_prefix: reference.into_iter().take(8).collect(),
            process: before.delta(&after),
        })?
    );
    Ok(())
}
