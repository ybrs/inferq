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
use qwen_engine::{GgufCheckpoint, MultiRowPath, QuantizedMatrix, RowSpread};
use rayon::prelude::*;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Time the multi-row quantized matmul kernels")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    /// Tensors to time, as `name`, `name:expert` for one expert of a fused
    /// expert tensor, or `first+second:expert` for the row-concatenated pair
    /// the MoE actually multiplies by (gate and up share one call). Repeatable.
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
    /// Also time this many distinct experts of each `name:expert` tensor as
    /// one batch, under both schedules a MoE layer could use: one matrix at a
    /// time with its output rows across the pool, or every matrix at once with
    /// each one's rows on its own thread. A single matrix timed alone cannot
    /// tell those apart, and a wide MoE pass is always the batch.
    #[arg(long, default_value_t = 0)]
    expert_batch: usize,
}

fn parse_path(name: &str) -> Result<MultiRowPath> {
    match name.to_ascii_lowercase().as_str() {
        "candle" => Ok(MultiRowPath::Candle),
        "smallm" | "small_m" => Ok(MultiRowPath::SmallM),
        "fused" => Ok(MultiRowPath::Fused),
        "tiled" | "fusedtiled" | "fused_tiled" => Ok(MultiRowPath::FusedTiled),
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
    /// Candle's time divided by this path's, at the same rows and in the same
    /// process. Above one this path is faster. `None` when Candle was not
    /// among `--paths`, since the ratio is only meaningful interleaved.
    speedup_vs_candle: Option<f64>,
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
            matrix.forward_via(input, *path, RowSpread::Pool)?;
        }
    }
    let mut samples = vec![Vec::with_capacity(repetitions); paths.len()];
    let mut outputs: Vec<Option<Tensor>> = vec![None; paths.len()];
    for _ in 0..repetitions {
        for (index, path) in paths.iter().enumerate() {
            let started = Instant::now();
            let result = matrix.forward_via(input, *path, RowSpread::Pool)?;
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
            match name.split_once('+') {
                Some((first, second)) => checkpoint.expert_pair(first, second)?.load(expert),
                None => checkpoint.load_expert_matrix(name, expert),
            }
        }
        None => checkpoint.load_matrix(spec),
    }
}

/// Time one batch of expert matrices under both schedules.
///
/// `Sequential` is what the MoE runs today: the expert loop is serial and each
/// expert's matmul splits its own output rows across the pool, so the pass pays
/// one fork/join per expert per matmul. `Concurrent` inverts that — the experts
/// are the parallel unit and each matmul runs whole on one thread. Both do
/// exactly the same arithmetic on exactly the same rows.
fn time_expert_batch(
    matrices: &[QuantizedMatrix],
    input: &Tensor,
    path: MultiRowPath,
    repetitions: usize,
    warmup: usize,
) -> Result<(f64, f64)> {
    let sequential = |_: ()| -> Result<()> {
        for matrix in matrices {
            matrix.forward_via(input, path, RowSpread::Pool)?;
        }
        Ok(())
    };
    let concurrent = |_: ()| -> Result<()> {
        matrices
            .par_iter()
            .map(|matrix| {
                matrix
                    .forward_via(input, path, RowSpread::Caller)
                    .map(|_| ())
            })
            .collect::<Result<Vec<()>>>()?;
        Ok(())
    };
    for _ in 0..warmup {
        sequential(())?;
        concurrent(())?;
    }
    let (mut best_sequential, mut best_concurrent) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..repetitions {
        let started = Instant::now();
        sequential(())?;
        best_sequential = best_sequential.min(started.elapsed().as_secs_f64());
        let started = Instant::now();
        concurrent(())?;
        best_concurrent = best_concurrent.min(started.elapsed().as_secs_f64());
    }
    Ok((best_sequential, best_concurrent))
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
            let candle_seconds = paths
                .iter()
                .position(|path| *path == MultiRowPath::Candle)
                .map(|index| measured[index].0);
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
                    speedup_vs_candle: candle_seconds.map(|candle| candle / best),
                });
                let last = measurements.last().expect("just pushed");
                eprintln!(
                    "  M={input_rows:<3} {:<7} {:8.3} ms  {:9.0} ns/row  {:6.2} GB/s  {:>6}  max|diff| {max_abs_diff:.2e}",
                    last.path,
                    best * 1e3,
                    last.nanoseconds_per_input_row,
                    last.effective_gigabytes_per_second,
                    match last.speedup_vs_candle {
                        Some(ratio) => format!("{ratio:.2}x"),
                        None => "-".to_string(),
                    },
                );
            }
        }
    }

    for spec in &args.tensors {
        let Some((name, first)) = spec.rsplit_once(':') else {
            continue;
        };
        if args.expert_batch == 0 {
            continue;
        }
        let first: usize = first
            .parse()
            .with_context(|| format!("expert index in {spec:?} is not a number"))?;
        let matrices = (first..first + args.expert_batch)
            .map(|expert| load(&checkpoint, &format!("{name}:{expert}")))
            .collect::<Result<Vec<_>>>()?;
        let columns = matrices[0].shape()[1];
        eprintln!(
            "{name}: {} experts as one batch, {:.1} MiB compressed",
            matrices.len(),
            matrices
                .iter()
                .map(|matrix| matrix.storage_bytes())
                .sum::<usize>() as f64
                / (1024. * 1024.)
        );
        for &input_rows in &args.rows {
            if input_rows < 2 {
                continue;
            }
            let input = deterministic_input(input_rows, columns)?;
            for path in &selected {
                let (sequential, concurrent) = time_expert_batch(
                    &matrices,
                    &input,
                    *path,
                    args.repetitions,
                    args.warmup.max(1),
                )?;
                let per_row = |seconds: f64| seconds * 1e9 / (matrices.len() * input_rows) as f64;
                eprintln!(
                    "  M={input_rows:<3} {:<11} pool-per-matmul {:9.0} ns/row   thread-per-expert {:9.0} ns/row   {:.2}x",
                    format!("{path:?}"),
                    per_row(sequential),
                    per_row(concurrent),
                    sequential / concurrent,
                );
                measurements.push(Measurement {
                    tensor: format!("{name}:{first}+{}", args.expert_batch),
                    dtype: matrices[0].dtype(),
                    rows: matrices[0].shape()[0],
                    columns,
                    storage_bytes: matrices.iter().map(|m| m.storage_bytes()).sum(),
                    path: format!("{path:?}/thread-per-expert"),
                    input_rows,
                    best_seconds: concurrent,
                    median_seconds: concurrent,
                    nanoseconds_per_input_row: per_row(concurrent),
                    effective_gigabytes_per_second: matrices
                        .iter()
                        .map(|m| m.storage_bytes())
                        .sum::<usize>() as f64
                        / concurrent
                        / 1e9,
                    max_abs_diff_vs_candle: None,
                    speedup_vs_candle: None,
                });
                measurements.push(Measurement {
                    tensor: format!("{name}:{first}+{}", args.expert_batch),
                    dtype: matrices[0].dtype(),
                    rows: matrices[0].shape()[0],
                    columns,
                    storage_bytes: matrices.iter().map(|m| m.storage_bytes()).sum(),
                    path: format!("{path:?}/pool-per-matmul"),
                    input_rows,
                    best_seconds: sequential,
                    median_seconds: sequential,
                    nanoseconds_per_input_row: per_row(sequential),
                    effective_gigabytes_per_second: matrices
                        .iter()
                        .map(|m| m.storage_bytes())
                        .sum::<usize>() as f64
                        / sequential
                        / 1e9,
                    max_abs_diff_vs_candle: None,
                    speedup_vs_candle: None,
                });
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
