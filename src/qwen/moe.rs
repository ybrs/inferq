use std::{cmp::Ordering, time::Instant};

use anyhow::{Result, ensure};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::ops;

use crate::{
    Checkpoint, Qwen3NextConfig,
    trace::{RoutingRecord, RoutingTrace},
};

use super::{f32_tensor, linear_profiled, load_profiled, model::ForwardTimings};

#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    pub experts: Vec<usize>,
    pub weights: Vec<f32>,
    pub logits: Vec<f32>,
}

pub fn top_k_routes(router_logits: &Tensor, k: usize, normalize: bool) -> Result<Vec<Route>> {
    let logits = f32_tensor(router_logits)?.to_vec2::<f32>()?;
    let mut routes = Vec::with_capacity(logits.len());
    for row in logits {
        ensure!(
            k > 0 && k <= row.len(),
            "invalid router top-k {k} for {} experts",
            row.len()
        );
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = row.iter().map(|x| (*x - max).exp()).sum();
        let probabilities: Vec<f32> = row.iter().map(|x| (*x - max).exp() / denominator).collect();
        let mut ranked: Vec<usize> = (0..row.len()).collect();
        ranked.sort_unstable_by(|&a, &b| {
            probabilities[b]
                .partial_cmp(&probabilities[a])
                .unwrap_or(Ordering::Equal)
        });
        ranked.truncate(k);
        let mut weights: Vec<f32> = ranked.iter().map(|&i| probabilities[i]).collect();
        if normalize {
            let sum: f32 = weights.iter().sum();
            for weight in &mut weights {
                *weight /= sum;
            }
        }
        routes.push(Route {
            experts: ranked,
            weights,
            logits: row,
        });
    }
    Ok(routes)
}

fn mlp(
    xs: &Tensor,
    gate: &Tensor,
    up: &Tensor,
    down: &Tensor,
    timings: &mut ForwardTimings,
) -> Result<Tensor> {
    let gate = ops::silu(&linear_profiled(xs, gate, timings)?)?;
    linear_profiled(
        &gate.broadcast_mul(&linear_profiled(xs, up, timings)?)?,
        down,
        timings,
    )
    .map_err(Into::into)
}

pub(crate) fn dense_mlp(
    checkpoint: &Checkpoint,
    layer: usize,
    xs: &Tensor,
    timings: &mut ForwardTimings,
) -> Result<Tensor> {
    let dev = xs.device();
    let p = format!("model.layers.{layer}.mlp");
    let gate = load_profiled(checkpoint, &format!("{p}.gate_proj.weight"), dev, timings)?;
    let up = load_profiled(checkpoint, &format!("{p}.up_proj.weight"), dev, timings)?;
    let down = load_profiled(checkpoint, &format!("{p}.down_proj.weight"), dev, timings)?;
    mlp(xs, &gate, &up, &down, timings)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_moe(
    checkpoint: &Checkpoint,
    config: &Qwen3NextConfig,
    layer: usize,
    xs: &Tensor,
    token_ids: &[u32],
    token_offset: usize,
    mut trace: Option<&mut dyn RoutingTrace>,
    timings: &mut ForwardTimings,
) -> Result<Tensor> {
    let dev = xs.device();
    let p = format!("model.layers.{layer}.mlp");
    let flat = xs.reshape((xs.elem_count() / config.hidden_size, config.hidden_size))?;
    let router_started = Instant::now();
    let router_weight = load_profiled(checkpoint, &format!("{p}.gate.weight"), dev, timings)?;
    let router_logits = linear_profiled(&flat, &router_weight, timings)?;
    timings.router += router_started.elapsed();
    let top_k_started = Instant::now();
    let routes = top_k_routes(
        &router_logits,
        config.num_experts_per_tok,
        config.norm_topk_prob,
    )?;
    timings.top_k += top_k_started.elapsed();
    ensure!(
        routes.len() == token_ids.len(),
        "routing token count mismatch"
    );

    let routed_started = Instant::now();
    let mut outputs = Vec::with_capacity(routes.len());
    for (token_index, route) in routes.iter().enumerate() {
        let x = flat.i(token_index)?.unsqueeze(0)?;
        let mut combined = Tensor::zeros((1, config.hidden_size), DType::F32, dev)?;
        for (&expert, &route_weight) in route.experts.iter().zip(&route.weights) {
            let expert_prefix = format!("{p}.experts.{expert}");
            let gate = load_profiled(
                checkpoint,
                &format!("{expert_prefix}.gate_proj.weight"),
                dev,
                timings,
            )?;
            let up = load_profiled(
                checkpoint,
                &format!("{expert_prefix}.up_proj.weight"),
                dev,
                timings,
            )?;
            let down = load_profiled(
                checkpoint,
                &format!("{expert_prefix}.down_proj.weight"),
                dev,
                timings,
            )?;
            let expert_out = mlp(&x, &gate, &up, &down, timings)?.to_dtype(DType::F32)?;
            combined = (combined + (expert_out * route_weight as f64)?)?;
        }
        outputs.push(combined);
        if let Some(sink) = trace.as_deref_mut() {
            sink.record(&RoutingRecord {
                token_index: token_offset + token_index,
                token_id: token_ids[token_index],
                layer,
                selected_expert_ids: route.experts.clone(),
                router_weights: route.weights.clone(),
                router_logits: Some(route.logits.clone()),
            })?;
        }
    }
    timings.routed_experts += routed_started.elapsed();
    let routed = Tensor::cat(&outputs, 0)?;
    let shared_started = Instant::now();
    let shared_gate = load_profiled(
        checkpoint,
        &format!("{p}.shared_expert.gate_proj.weight"),
        dev,
        timings,
    )?;
    let shared_up = load_profiled(
        checkpoint,
        &format!("{p}.shared_expert.up_proj.weight"),
        dev,
        timings,
    )?;
    let shared_down = load_profiled(
        checkpoint,
        &format!("{p}.shared_expert.down_proj.weight"),
        dev,
        timings,
    )?;
    let shared =
        mlp(&flat, &shared_gate, &shared_up, &shared_down, timings)?.to_dtype(DType::F32)?;
    let shared_selector = load_profiled(
        checkpoint,
        &format!("{p}.shared_expert_gate.weight"),
        dev,
        timings,
    )?;
    let shared_selector =
        ops::sigmoid(&linear_profiled(&flat, &shared_selector, timings)?.to_dtype(DType::F32)?)?;
    let out = (routed + shared.broadcast_mul(&shared_selector)?)?.to_dtype(xs.dtype())?;
    timings.shared_expert += shared_started.elapsed();
    Ok(out.reshape(xs.shape())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn router_selects_and_normalizes_top_k() {
        let logits = Tensor::new(&[[1f32, 4., 2., 3.]], &Device::Cpu).unwrap();
        let routes = top_k_routes(&logits, 2, true).unwrap();
        assert_eq!(routes[0].experts, vec![1, 3]);
        assert!((routes[0].weights.iter().sum::<f32>() - 1.).abs() < 1e-6);
    }
}
