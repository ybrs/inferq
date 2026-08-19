use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::ops;
use rayon::prelude::*;
use serde::Serialize;

use crate::{GgufCheckpoint, GgufExpertPair, GgufExpertTensor, QuantizedMatrix, RowSpread};

use super::{Route, top_k_routes};

#[derive(Debug, Clone, Default, Serialize)]
pub struct QuantizedMoeRoutingStats {
    pub batches: usize,
    pub rows: usize,
    pub token_expert_assignments: usize,
    pub unique_experts_selected: usize,
    pub max_rows_per_expert: usize,
}

impl QuantizedMoeRoutingStats {
    pub fn from_routes(routes: &[Route], expert_count: usize) -> Self {
        let mut counts = vec![0usize; expert_count];
        for route in routes {
            for &expert in &route.experts {
                counts[expert] += 1;
            }
        }
        Self {
            batches: 1,
            rows: routes.len(),
            token_expert_assignments: counts.iter().sum(),
            unique_experts_selected: counts.iter().filter(|&&count| count > 0).count(),
            max_rows_per_expert: counts.into_iter().max().unwrap_or(0),
        }
    }

    pub fn accumulate(&mut self, other: &Self) {
        self.batches += other.batches;
        self.rows += other.rows;
        self.token_expert_assignments += other.token_expert_assignments;
        self.unique_experts_selected += other.unique_experts_selected;
        self.max_rows_per_expert = self.max_rows_per_expert.max(other.max_rows_per_expert);
    }

    pub fn duplicate_assignment_rate(&self) -> f64 {
        if self.token_expert_assignments == 0 {
            0.
        } else {
            1. - self.unique_experts_selected as f64 / self.token_expert_assignments as f64
        }
    }

    pub fn average_rows_per_selected_expert(&self) -> f64 {
        if self.unique_experts_selected == 0 {
            0.
        } else {
            self.token_expert_assignments as f64 / self.unique_experts_selected as f64
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct QuantizedMoeTimings {
    pub wall: Duration,
    pub router: Duration,
    pub top_k: Duration,
    pub expert_load: Duration,
    pub expert_compute: Duration,
    pub expert_gate_up: Duration,
    pub expert_activation: Duration,
    pub expert_down: Duration,
    pub expert_accumulation: Duration,
    pub shared_expert: Duration,
    pub routing: QuantizedMoeRoutingStats,
}

impl QuantizedMoeTimings {
    pub fn accumulate(&mut self, other: &Self) {
        self.wall += other.wall;
        self.router += other.router;
        self.top_k += other.top_k;
        self.expert_load += other.expert_load;
        self.expert_compute += other.expert_compute;
        self.expert_gate_up += other.expert_gate_up;
        self.expert_activation += other.expert_activation;
        self.expert_down += other.expert_down;
        self.expert_accumulation += other.expert_accumulation;
        self.shared_expert += other.shared_expert;
        self.routing.accumulate(&other.routing);
    }
}

#[derive(Debug)]
pub struct QuantizedMoeOutput {
    pub hidden: Tensor,
    pub routes: Vec<Route>,
    pub timings: QuantizedMoeTimings,
}

/// One GGUF MoE sublayer with resident decision/shared weights and directly
/// addressed routed experts.
pub struct QuantizedMoeLayer<'a> {
    layer: usize,
    hidden_size: usize,
    intermediate_size: usize,
    expert_count: usize,
    experts_per_token: usize,
    normalize_top_k: bool,
    router: QuantizedMatrix,
    shared_gate_up: QuantizedMatrix,
    shared_down: QuantizedMatrix,
    shared_selector: Tensor,
    gate_up_experts: GgufExpertPair<'a>,
    down_experts: GgufExpertTensor<'a>,
}

#[derive(Clone, Copy)]
enum RoutedExecution {
    TokenMajor,
    GroupedByExpert,
}

impl std::fmt::Debug for QuantizedMoeLayer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedMoeLayer")
            .field("layer", &self.layer)
            .field("hidden_size", &self.hidden_size)
            .field("intermediate_size", &self.intermediate_size)
            .field("expert_count", &self.expert_count)
            .field("experts_per_token", &self.experts_per_token)
            .finish_non_exhaustive()
    }
}

impl<'a> QuantizedMoeLayer<'a> {
    pub fn load(
        checkpoint: &'a GgufCheckpoint,
        layer: usize,
        experts_per_token: usize,
        normalize_top_k: bool,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer}");
        let router_name = format!("{prefix}.ffn_gate_inp.weight");
        let gate_experts_name = format!("{prefix}.ffn_gate_exps.weight");
        let down_experts_name = format!("{prefix}.ffn_down_exps.weight");
        let router_info = checkpoint
            .tensor_info(&router_name)
            .with_context(|| format!("GGUF is missing tensor {router_name:?}"))?;
        let gate_info = checkpoint
            .tensor_info(&gate_experts_name)
            .with_context(|| format!("GGUF is missing tensor {gate_experts_name:?}"))?;
        let down_info = checkpoint
            .tensor_info(&down_experts_name)
            .with_context(|| format!("GGUF is missing tensor {down_experts_name:?}"))?;
        ensure!(
            router_info.shape.len() == 2,
            "router {router_name:?} is not a matrix"
        );
        ensure!(
            gate_info.shape.len() == 3 && down_info.shape.len() == 3,
            "layer {layer} fused expert tensors are not rank three"
        );
        let expert_count = router_info.shape[0];
        let hidden_size = router_info.shape[1];
        let intermediate_size = gate_info.shape[1];
        ensure!(
            experts_per_token > 0 && experts_per_token <= expert_count,
            "invalid top-k {experts_per_token} for {expert_count} experts"
        );
        ensure!(
            gate_info.shape == [expert_count, intermediate_size, hidden_size],
            "invalid gate expert shape {:?}",
            gate_info.shape
        );
        ensure!(
            down_info.shape == [expert_count, hidden_size, intermediate_size],
            "invalid down expert shape {:?}",
            down_info.shape
        );
        let up_info = checkpoint
            .tensor_info(&format!("{prefix}.ffn_up_exps.weight"))
            .context("GGUF is missing fused up expert weights")?;
        ensure!(
            up_info.shape == gate_info.shape,
            "up expert shape {:?} does not match gate shape {:?}",
            up_info.shape,
            gate_info.shape
        );
        let router = checkpoint.load_matrix(&router_name)?;
        let shared_gate = checkpoint.load_matrix(&format!("{prefix}.ffn_gate_shexp.weight"))?;
        let shared_up = checkpoint.load_matrix(&format!("{prefix}.ffn_up_shexp.weight"))?;
        let shared_down = checkpoint.load_matrix(&format!("{prefix}.ffn_down_shexp.weight"))?;
        let shared_selector = checkpoint
            .load_f32_vector(&format!("{prefix}.ffn_gate_inp_shexp.weight"))?
            .reshape((1, hidden_size))?;
        let down_experts = checkpoint.expert_tensor(&down_experts_name)?;
        ensure!(
            shared_gate.shape()[1] == hidden_size
                && shared_up.shape() == shared_gate.shape()
                && shared_down.shape() == [hidden_size, shared_gate.shape()[0]],
            "invalid shared expert matrix shapes"
        );
        let shared_gate_up = shared_gate.concatenate_rows(&shared_up)?;
        let gate_up_experts =
            checkpoint.expert_pair(&gate_experts_name, &format!("{prefix}.ffn_up_exps.weight"))?;
        Ok(Self {
            layer,
            hidden_size,
            intermediate_size,
            expert_count,
            experts_per_token,
            normalize_top_k,
            router,
            shared_gate_up,
            shared_down,
            shared_selector,
            gate_up_experts,
            down_experts,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<QuantizedMoeOutput> {
        let rows = xs.elem_count() / self.hidden_size;
        let execution = if rows > 1 {
            RoutedExecution::GroupedByExpert
        } else {
            RoutedExecution::TokenMajor
        };
        self.forward_with_execution(xs, execution)
    }

    fn forward_with_execution(
        &self,
        xs: &Tensor,
        execution: RoutedExecution,
    ) -> Result<QuantizedMoeOutput> {
        let wall_started = Instant::now();
        ensure!(
            xs.elem_count().is_multiple_of(self.hidden_size),
            "MoE input element count is not divisible by hidden size {}",
            self.hidden_size
        );
        let flat = xs
            .to_dtype(DType::F32)?
            .reshape((xs.elem_count() / self.hidden_size, self.hidden_size))?;
        let mut timings = QuantizedMoeTimings::default();
        let router_started = Instant::now();
        let router_logits = self.router.forward(&flat)?;
        timings.router = router_started.elapsed();
        let top_k_started = Instant::now();
        let routes = top_k_routes(&router_logits, self.experts_per_token, self.normalize_top_k)?;
        timings.top_k = top_k_started.elapsed();
        timings.routing = QuantizedMoeRoutingStats::from_routes(&routes, self.expert_count);

        let routed = match execution {
            RoutedExecution::TokenMajor => self.routed_token_major(&flat, &routes, &mut timings)?,
            RoutedExecution::GroupedByExpert => {
                self.routed_grouped_by_expert(&flat, &routes, &mut timings)?
            }
        };
        let shared_started = Instant::now();
        let shared_gate_up = self.shared_gate_up.forward(&flat)?;
        let shared_gate = shared_gate_up.narrow(1, 0, self.intermediate_size)?;
        let shared_up = shared_gate_up.narrow(1, self.intermediate_size, self.intermediate_size)?;
        let shared = ops::silu(&shared_gate)?.broadcast_mul(&shared_up)?;
        let shared = self.shared_down.forward(&shared)?;
        let selector = ops::sigmoid(&flat.matmul(&self.shared_selector.t()?)?)?;
        let hidden = (routed + shared.broadcast_mul(&selector)?)?.reshape(xs.shape())?;
        timings.shared_expert = shared_started.elapsed();
        timings.wall = wall_started.elapsed();
        Ok(QuantizedMoeOutput {
            hidden,
            routes,
            timings,
        })
    }

    fn routed_token_major(
        &self,
        flat: &Tensor,
        routes: &[Route],
        timings: &mut QuantizedMoeTimings,
    ) -> Result<Tensor> {
        let mut outputs = Vec::with_capacity(routes.len());
        for (token_index, route) in routes.iter().enumerate() {
            let x = flat.i(token_index)?.unsqueeze(0)?;
            let mut combined = Tensor::zeros((1, self.hidden_size), DType::F32, flat.device())?;
            for (&expert, &route_weight) in route.experts.iter().zip(&route.weights) {
                let load_started = Instant::now();
                let gate_up = self.gate_up_experts.load(expert)?;
                let down = self.down_experts.load(expert)?;
                timings.expert_load += load_started.elapsed();
                let compute_started = Instant::now();
                let gate_up_started = Instant::now();
                let gate_up = gate_up.forward(&x)?;
                timings.expert_gate_up += gate_up_started.elapsed();
                let activation_started = Instant::now();
                let gate = gate_up.narrow(1, 0, self.intermediate_size)?;
                let up = gate_up.narrow(1, self.intermediate_size, self.intermediate_size)?;
                let activated = ops::silu(&gate)?.broadcast_mul(&up)?;
                timings.expert_activation += activation_started.elapsed();
                let down_started = Instant::now();
                let value = down.forward(&activated)?;
                timings.expert_down += down_started.elapsed();
                let accumulation_started = Instant::now();
                combined = (combined + (value * f64::from(route_weight))?)?;
                timings.expert_accumulation += accumulation_started.elapsed();
                timings.expert_compute += compute_started.elapsed();
            }
            outputs.push(combined);
        }
        Ok(Tensor::cat(&outputs, 0)?)
    }

    /// Run every routed expert, the experts in parallel.
    ///
    /// The expert loop used to be serial while each expert's matmul split its
    /// own few hundred output rows across the thread pool: one fork/join per
    /// expert per matmul, 20,480 of them in a forty-layer pass, each dividing
    /// work smaller than one core's share. The experts are the natural
    /// parallel unit — 256 independent matrices per layer, none of which reads
    /// another's output — so they take the pool and each matmul runs whole on
    /// one thread ([`RowSpread::Caller`]).
    ///
    /// Nothing about the arithmetic changes. Each expert sees the same rows in
    /// the same order and multiplies them by the same weights; the weighted
    /// additions that combine an expert's outputs are still a serial pass in
    /// token-then-route order below, unchanged and unparallelised, because
    /// that order is what makes a grouped pass agree with the token-major
    /// reference bit for bit.
    fn routed_grouped_by_expert(
        &self,
        flat: &Tensor,
        routes: &[Route],
        timings: &mut QuantizedMoeTimings,
    ) -> Result<Tensor> {
        let mut grouped = BTreeMap::<usize, Vec<(usize, usize)>>::new();
        for (row, route) in routes.iter().enumerate() {
            for (route_index, &expert) in route.experts.iter().enumerate() {
                grouped.entry(expert).or_default().push((row, route_index));
            }
        }
        let assignments: Vec<(usize, Vec<(usize, usize)>)> = grouped.into_iter().collect();

        // Where each (token, route) slot's result will be found once the
        // experts have run: which expert group, and which row within it.
        let mut slots = routes
            .iter()
            .map(|route| vec![None; route.experts.len()])
            .collect::<Vec<Vec<Option<(usize, usize)>>>>();
        for (group, (_, expert_assignments)) in assignments.iter().enumerate() {
            for (expert_row, &(row, route_index)) in expert_assignments.iter().enumerate() {
                slots[row][route_index] = Some((group, expert_row));
            }
        }

        let flat_values = flat.flatten_all()?.to_vec1::<f32>()?;
        let compute_started = Instant::now();
        let executed = assignments
            .par_iter()
            .map(|(expert, expert_assignments)| {
                self.run_one_expert(&flat_values, *expert, expert_assignments)
            })
            .collect::<Result<Vec<ExpertRun>>>()?;
        // Wall time of the whole grouped region, which is what the pass spent.
        // The gate_up/activation/down/load splits below are summed across the
        // workers and are therefore thread time, not wall; see docs/profiling.md.
        timings.expert_compute = compute_started.elapsed();
        for run in &executed {
            timings.expert_load += run.load;
            timings.expert_gate_up += run.gate_up;
            timings.expert_activation += run.activation;
            timings.expert_down += run.down;
        }

        // Preserve the target-only path's token and route accumulation order.
        // Only expert matrix execution is grouped; weighted additions remain
        // byte-for-byte ordered like the readable token-major reference.
        let accumulation_started = Instant::now();
        let mut combined = vec![0f32; routes.len() * self.hidden_size];
        for (row, route) in routes.iter().enumerate() {
            let output = &mut combined[row * self.hidden_size..(row + 1) * self.hidden_size];
            for (route_index, &route_weight) in route.weights.iter().enumerate() {
                let (group, expert_row) = slots[row][route_index]
                    .context("grouped expert result is missing a routed assignment")?;
                let value = &executed[group].values
                    [expert_row * self.hidden_size..(expert_row + 1) * self.hidden_size];
                for (output, value) in output.iter_mut().zip(value) {
                    // `combined + (value * weight)` as the reference wrote it:
                    // two roundings, multiply then add, never contracted.
                    *output += *value * route_weight;
                }
            }
        }
        timings.expert_accumulation = accumulation_started.elapsed();
        Ok(Tensor::from_vec(
            combined,
            (routes.len(), self.hidden_size),
            flat.device(),
        )?)
    }

    /// Gather one expert's rows, run its two matmuls, and return its outputs.
    ///
    /// Called from inside a parallel iterator over experts, so everything here
    /// stays on the calling thread: the matmuls take [`RowSpread::Caller`], and
    /// the activation is written out by hand rather than through candle, whose
    /// element-wise ops would each allocate a tensor per expert per layer.
    fn run_one_expert(
        &self,
        flat_values: &[f32],
        expert: usize,
        expert_assignments: &[(usize, usize)],
    ) -> Result<ExpertRun> {
        let hidden = self.hidden_size;
        let rows = expert_assignments.len();
        let mut inputs = vec![0f32; rows * hidden];
        for (expert_row, &(row, _)) in expert_assignments.iter().enumerate() {
            inputs[expert_row * hidden..(expert_row + 1) * hidden]
                .copy_from_slice(&flat_values[row * hidden..(row + 1) * hidden]);
        }

        let load_started = Instant::now();
        let gate_up_weights = self.gate_up_experts.load(expert)?;
        let down_weights = self.down_experts.load(expert)?;
        let load = load_started.elapsed();

        let gate_up_started = Instant::now();
        let gate_up = gate_up_weights.forward_rows(&inputs, rows, RowSpread::Caller)?;
        let gate_up_elapsed = gate_up_started.elapsed();

        let activation_started = Instant::now();
        let intermediate = self.intermediate_size;
        let mut activated = vec![0f32; rows * intermediate];
        for row in 0..rows {
            let (gate, up) = gate_up[row * 2 * intermediate..(row + 1) * 2 * intermediate]
                .split_at(intermediate);
            let output = &mut activated[row * intermediate..(row + 1) * intermediate];
            for ((output, gate), up) in output.iter_mut().zip(gate).zip(up) {
                // candle's `Silu`, elementwise: `v / (1 + exp(-v))` in f32.
                *output = (gate / (1. + (-gate).exp())) * up;
            }
        }
        let activation = activation_started.elapsed();

        let down_started = Instant::now();
        let values = down_weights.forward_rows(&activated, rows, RowSpread::Caller)?;
        Ok(ExpertRun {
            values,
            load,
            gate_up: gate_up_elapsed,
            activation,
            down: down_started.elapsed(),
        })
    }
}

/// One expert's outputs, row-major, with what its own thread spent producing
/// them. The durations are per-worker and are summed rather than added to wall
/// time by the caller.
struct ExpertRun {
    values: Vec<f32>,
    load: Duration,
    gate_up: Duration,
    activation: Duration,
    down: Duration,
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use candle_core::{
        Device,
        quantized::{GgmlDType, QTensor, gguf_file},
    };

    use super::*;

    fn tensor(values: Vec<f32>, shape: impl Into<candle_core::Shape>, dtype: GgmlDType) -> QTensor {
        let tensor = Tensor::from_vec(values, shape, &Device::Cpu).unwrap();
        QTensor::quantize(&tensor, dtype).unwrap()
    }

    #[test]
    fn executes_real_routing_and_selected_expert_ranges() {
        let hidden = 256;
        let intermediate = 256;
        let router = tensor(
            (0..2 * hidden)
                .map(|index| if index < hidden { 0. } else { 0.01 })
                .collect(),
            (2, hidden),
            GgmlDType::F32,
        );
        let fused_values: Vec<f32> = (0..2 * intermediate * hidden)
            .map(|index| (index % 17) as f32 / 256.)
            .collect();
        let gate = tensor(
            fused_values.clone(),
            (2, intermediate, hidden),
            GgmlDType::Q4K,
        );
        let up = tensor(fused_values, (2, intermediate, hidden), GgmlDType::Q4K);
        let down = tensor(
            vec![0.001; 2 * hidden * intermediate],
            (2, hidden, intermediate),
            GgmlDType::Q5K,
        );
        let shared_gate = tensor(
            vec![0.001; intermediate * hidden],
            (intermediate, hidden),
            GgmlDType::Q8_0,
        );
        let shared_up = tensor(
            vec![0.001; intermediate * hidden],
            (intermediate, hidden),
            GgmlDType::Q8_0,
        );
        let shared_down = tensor(
            vec![0.001; hidden * intermediate],
            (hidden, intermediate),
            GgmlDType::Q8_0,
        );
        let selector = tensor(vec![0.; hidden], hidden, GgmlDType::F32);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("moe.gguf");
        let mut file = File::create(&path).unwrap();
        gguf_file::write(
            &mut file,
            &[],
            &[
                ("blk.0.ffn_gate_inp.weight", &router),
                ("blk.0.ffn_gate_exps.weight", &gate),
                ("blk.0.ffn_up_exps.weight", &up),
                ("blk.0.ffn_down_exps.weight", &down),
                ("blk.0.ffn_gate_shexp.weight", &shared_gate),
                ("blk.0.ffn_up_shexp.weight", &shared_up),
                ("blk.0.ffn_down_shexp.weight", &shared_down),
                ("blk.0.ffn_gate_inp_shexp.weight", &selector),
            ],
        )
        .unwrap();
        drop(file);

        let checkpoint = GgufCheckpoint::open(path).unwrap();
        let layer = QuantizedMoeLayer::load(&checkpoint, 0, 1, true).unwrap();
        let input = Tensor::ones((1, hidden), DType::F32, &Device::Cpu).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.routes[0].experts, vec![1]);
        assert_eq!(output.hidden.dims(), &[1, hidden]);
        assert!(
            output.hidden.to_vec2::<f32>().unwrap()[0]
                .iter()
                .all(|value| value.is_finite())
        );

        let batch = Tensor::ones((3, hidden), DType::F32, &Device::Cpu).unwrap();
        let reference = layer
            .forward_with_execution(&batch, RoutedExecution::TokenMajor)
            .unwrap();
        let grouped = layer
            .forward_with_execution(&batch, RoutedExecution::GroupedByExpert)
            .unwrap();
        assert_eq!(grouped.routes.len(), reference.routes.len());
        for (actual, expected) in grouped.routes.iter().zip(&reference.routes) {
            assert_eq!(actual.experts, expected.experts);
            assert_eq!(actual.weights, expected.weights);
        }
        let actual = grouped
            .hidden
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = reference
            .hidden
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Identical, not merely close. Grouping changes which rows share a
        // matmul and which thread runs it, never what is summed or in what
        // order, and speculative decoding is defined by the target path and
        // the grouped path agreeing on the token.
        assert_eq!(actual, expected, "grouped MoE diverged from token-major");
    }
}
