//! Per-kernel timing for the multi-row quantized matmul.
//!
//! Times each available implementation on the same matrix and the same input
//! at a range of row counts, so the effect of a kernel change is measured
//! directly rather than inferred from end-to-end decode timings. Matrices come
//! from a real GGUF so block layouts, scale distributions and shapes are the
//! ones the model actually executes.

use std::{path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use clap::Parser;
use qwen_engine::{GgufCheckpoint, MultiRowPath, QuantizedMatrix};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Time the multi-row quantized matmul kernels")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    /// Tensors to time, as `name` or `name:expert` for a fused expert tensor.
    /// Repeatable.
    #[arg(long = "tensor", required = true)]
    tensors: Vec<String>,
    /// Input row counts to time.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,12,16")]
    rows: Vec<usize>,
    #[arg(long, default_value_t = 5)]
    repetitions: usize,
    /// Discard this many timed passes before measuring.
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long)]
    output: Option<PathBuf>,
    /// Restrict to these kernels. Running one kernel per process is the
    /// fairest comparison: each gets the same untouched cache and allocator
    /// state, and no kernel is charged for another's evictions.
    #[arg(long, value_delimiter = ',', default_value = "candle,smallm,fused")]
    paths: Vec<String>,
}

fn parse_path(name: &str) -> Result<MultiRowPath> {
    match name.to_ascii_lowercase().as_str() {
        "candle" => Ok(MultiRowPath::Candle),
        "smallm" | "small_m" => Ok(MultiRowPath::SmallM),
        "fused" => Ok(MultiRowPath::Fused),
        other => anyhow::bail!("unknown kernel {other:?}"),
    }
}

#[derive(Debug, Clone, Serialize)]
struct Measurement {
    tensor: String,
    dtype: String,
    rows: usize,
    columns: usize,
    storage_bytes: usize,
    path: String,
    input_rows: usize,
    /// Best of `repetitions`, which is the least noise-contaminated estimate.
    best_seconds: f64,
    median_seconds: f64,
    nanoseconds_per_input_row: f64,
    /// Compressed weight bytes divided by the time for one pass. Above the
    /// host's memory bandwidth this means the weights were served from cache.
    effective_gigabytes_per_second: f64,
    /// Largest absolute difference against the Candle path at the same rows.
    max_abs_diff_vs_candle: Option<f32>,
}

fn deterministic_input(rows: usize, columns: usize) -> Result<Tensor> {
    // Reproducible, mean-zero, and not degenerate across rows: a constant
    // input would let a kernel look good by accident on cache behavior.
    let values: Vec<f32> = (0..rows * columns)
        .map(|index| {
            let row = index / columns;
            let column = index % columns;
            (((column * 31 + row * 17) % 61) as f32 - 30.) / 32.
        })
        .collect();
    Ok(Tensor::from_vec(values, (rows, columns), &Device::Cpu)?)
}

/// Time every path with the repetitions interleaved.
///
/// Timing all repetitions of one path before starting the next lets clock
/// drift and thermal state bias whichever path ran later — enough, measured,
/// to move a ratio between 1.46x and 1.70x. Round-robin sampling puts every
/// path in the same conditions, and the per-path minimum then rejects the
/// remaining noise, which is one-sided.
fn time_paths(
    matrix: &QuantizedMatrix,
    input: &Tensor,
    paths: &[MultiRowPath],
    repetitions: usize,
    warmup: usize,
) -> Result<Vec<(f64, f64, Vec<f32>)>> {
    for _ in 0..warmup {
        for path in paths {
            matrix.forward_via(input, *path)?;
        }
    }
    let mut samples = vec![Vec::with_capacity(repetitions); paths.len()];
    let mut outputs: Vec<Option<Tensor>> = vec![None; paths.len()];
    for _ in 0..repetitions {
        for (index, path) in paths.iter().enumerate() {
            let started = Instant::now();
            let result = matrix.forward_via(input, *path)?;
            samples[index].push(started.elapsed().as_secs_f64());
            outputs[index] = Some(result);
        }
    }
    let mut measured = Vec::with_capacity(paths.len());
    for (index, mut samples) in samples.into_iter().enumerate() {
        samples.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
        let values = outputs[index]
            .take()
            .context("at least one repetition")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        measured.push((samples[0], samples[samples.len() / 2], values));
    }
    Ok(measured)
}

fn load(checkpoint: &GgufCheckpoint, spec: &str) -> Result<QuantizedMatrix> {
    match spec.split_once(':') {
        Some((name, expert)) => {
            let expert = expert
                .parse()
                .with_context(|| format!("expert index in {spec:?} is not a number"))?;
            checkpoint.load_expert_matrix(name, expert)
        }
        None => checkpoint.load_matrix(spec),
    }
}

fn main() -> Result<()> {
    qwen_engine::threading::init();
    let args = Args::parse();
    anyhow::ensure!(args.repetitions > 0, "--repetitions must be at least one");
    let checkpoint = GgufCheckpoint::open(&args.model)?;
    let selected = args
        .paths
        .iter()
        .map(|name| parse_path(name))
        .collect::<Result<Vec<_>>>()?;
    let mut measurements = Vec::new();

    for spec in &args.tensors {
        let matrix = load(&checkpoint, spec)?;
        let shape = matrix.shape();
        eprintln!(
            "{spec}: {} {:?}, {:.1} MiB compressed",
            matrix.dtype(),
            shape,
            matrix.storage_bytes() as f64 / (1024. * 1024.)
        );
        for &input_rows in &args.rows {
            let input = deterministic_input(input_rows, shape[1])?;
            let mut paths = selected.clone();
            if input_rows < 2 {
                paths.retain(|path| *path == MultiRowPath::Candle);
            }
            if paths.is_empty() {
                continue;
            }
            let measured = time_paths(&matrix, &input, &paths, args.repetitions, args.warmup)?;
            let reference = measured[0].2.clone();
            for (path, (best, median, values)) in paths.iter().zip(measured) {
                let max_abs_diff = reference
                    .iter()
                    .zip(&values)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                measurements.push(Measurement {
                    tensor: spec.clone(),
                    dtype: matrix.dtype(),
                    rows: shape[0],
                    columns: shape[1],
                    storage_bytes: matrix.storage_bytes(),
                    path: format!("{path:?}"),
                    input_rows,
                    best_seconds: best,
                    median_seconds: median,
                    nanoseconds_per_input_row: best * 1e9 / input_rows as f64,
                    effective_gigabytes_per_second: matrix.storage_bytes() as f64 / best / 1e9,
                    max_abs_diff_vs_candle: Some(max_abs_diff),
                });
                let last = measurements.last().expect("just pushed");
                eprintln!(
                    "  M={input_rows:<3} {:<7} {:8.3} ms  {:9.0} ns/row  {:6.2} GB/s  max|diff| {max_abs_diff:.2e}",
                    last.path,
                    best * 1e3,
                    last.nanoseconds_per_input_row,
                    last.effective_gigabytes_per_second,
                );
            }
        }
    }

    let json = serde_json::to_string_pretty(&measurements)?;
    match args.output {
        Some(path) => std::fs::write(path, json)?,
        None => println!("{json}"),
    }
    Ok(())
}
