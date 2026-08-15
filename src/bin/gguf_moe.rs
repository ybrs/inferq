use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use clap::Parser;
use qwen_engine::{
    Checkpoint, GgufCheckpoint,
    profile::ProcessSnapshot,
    qwen::{
        QuantizedMoeLayer, QuantizedMoeOutput, QuantizedMoeTimings, reference_routes,
        reference_sparse_moe,
    },
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Execute one real routed GGUF MoE sublayer")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    layer: usize,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long, default_value_t = 1)]
    repetitions: usize,
    /// Optional SafeTensors model directory for routing and output comparison.
    #[arg(long)]
    reference_model: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct TimingReport {
    wall_seconds: f64,
    router_seconds: f64,
    top_k_seconds: f64,
    expert_load_seconds: f64,
    expert_compute_seconds: f64,
    shared_expert_seconds: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    layer: usize,
    top_k: usize,
    repetitions: usize,
    layer_load_seconds: f64,
    selected_experts: Vec<usize>,
    route_weights: Vec<f32>,
    mean_timings: TimingReport,
    output_l2_norm: f64,
    output_prefix: Vec<f32>,
    process: qwen_engine::profile::ProcessDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference: Option<ReferenceReport>,
}

#[derive(Debug, Serialize)]
struct ReferenceReport {
    selected_experts: Vec<usize>,
    routing_match: bool,
    output_max_abs_error: f32,
    output_mean_abs_error: f64,
    output_root_mean_square_error: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.repetitions > 0, "--repetitions must be at least one");
    let checkpoint = GgufCheckpoint::open(&args.model)?;
    let before = ProcessSnapshot::capture()?;
    let load_started = Instant::now();
    let layer = QuantizedMoeLayer::load(&checkpoint, args.layer, args.top_k, true)?;
    let layer_load_seconds = load_started.elapsed().as_secs_f64();
    let hidden_size = checkpoint
        .tensor_info(&format!("blk.{}.ffn_gate_inp.weight", args.layer))
        .context("validated router tensor disappeared")?
        .shape[1];
    let values: Vec<f32> = (0..hidden_size)
        .map(|index| ((index % 31) as f32 - 15.) / 16.)
        .collect();
    let input = Tensor::from_vec(values, (1, hidden_size), &Device::Cpu)?;
    let mut output = None;
    let mut total_timings = QuantizedMoeTimings::default();
    for _ in 0..args.repetitions {
        let sample = layer.forward(&input)?;
        total_timings.accumulate(&sample.timings);
        output = Some(sample);
    }
    let output = output.expect("positive repetition count");
    let scale = 1. / args.repetitions as f64;
    let reference = args
        .reference_model
        .as_deref()
        .map(|path| compare_reference(path, args.layer, &input, &output))
        .transpose()?;
    let values = output.hidden.to_vec2::<f32>()?;
    let values = &values[0];
    let after = ProcessSnapshot::capture()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            layer: args.layer,
            top_k: args.top_k,
            repetitions: args.repetitions,
            layer_load_seconds,
            selected_experts: output.routes[0].experts.clone(),
            route_weights: output.routes[0].weights.clone(),
            mean_timings: TimingReport {
                wall_seconds: total_timings.wall.as_secs_f64() * scale,
                router_seconds: total_timings.router.as_secs_f64() * scale,
                top_k_seconds: total_timings.top_k.as_secs_f64() * scale,
                expert_load_seconds: total_timings.expert_load.as_secs_f64() * scale,
                expert_compute_seconds: total_timings.expert_compute.as_secs_f64() * scale,
                shared_expert_seconds: total_timings.shared_expert.as_secs_f64() * scale,
            },
            output_l2_norm: values
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>()
                .sqrt(),
            output_prefix: values.iter().copied().take(8).collect(),
            process: before.delta(&after),
            reference,
        })?
    );
    Ok(())
}

fn compare_reference(
    model: &Path,
    layer: usize,
    input: &Tensor,
    quantized: &QuantizedMoeOutput,
) -> Result<ReferenceReport> {
    let checkpoint = Checkpoint::open(model)?;
    let config = checkpoint.config();
    let routes = reference_routes(&checkpoint, config, layer, input)?;
    let reference = reference_sparse_moe(&checkpoint, config, layer, input)?
        .to_vec2::<f32>()?
        .into_iter()
        .flatten();
    let quantized_values = quantized.hidden.to_vec2::<f32>()?.into_iter().flatten();
    let errors: Vec<f32> = reference
        .zip(quantized_values)
        .map(|(reference, actual)| (reference - actual).abs())
        .collect();
    let squared_error = errors
        .iter()
        .map(|error| f64::from(*error) * f64::from(*error))
        .sum::<f64>();
    Ok(ReferenceReport {
        selected_experts: routes[0].experts.clone(),
        routing_match: routes[0].experts == quantized.routes[0].experts,
        output_max_abs_error: errors.iter().copied().fold(0., f32::max),
        output_mean_abs_error: errors.iter().map(|error| f64::from(*error)).sum::<f64>()
            / errors.len() as f64,
        output_root_mean_square_error: (squared_error / errors.len() as f64).sqrt(),
    })
}
