use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::ops;

use crate::{GgufCheckpoint, GgufExpertPair, GgufExpertTensor, QuantizedMatrix};

use super::{Route, top_k_routes};

#[derive(Debug, Clone, Default)]
pub struct QuantizedMoeTimings {
    pub wall: Duration,
    pub router: Duration,
    pub top_k: Duration,
    pub expert_load: Duration,
    pub expert_compute: Duration,
    pub shared_expert: Duration,
}

impl QuantizedMoeTimings {
    pub fn accumulate(&mut self, other: &Self) {
        self.wall += other.wall;
        self.router += other.router;
        self.top_k += other.top_k;
        self.expert_load += other.expert_load;
        self.expert_compute += other.expert_compute;
        self.shared_expert += other.shared_expert;
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

        let mut outputs = Vec::with_capacity(routes.len());
        for (token_index, route) in routes.iter().enumerate() {
            let x = flat.i(token_index)?.unsqueeze(0)?;
            let mut combined = Tensor::zeros((1, self.hidden_size), DType::F32, xs.device())?;
            for (&expert, &route_weight) in route.experts.iter().zip(&route.weights) {
                let load_started = Instant::now();
                let gate_up = self.gate_up_experts.load(expert)?;
                let down = self.down_experts.load(expert)?;
                timings.expert_load += load_started.elapsed();
                let compute_started = Instant::now();
                let gate_up = gate_up.forward(&x)?;
                let gate = gate_up.narrow(1, 0, self.intermediate_size)?;
                let up = gate_up.narrow(1, self.intermediate_size, self.intermediate_size)?;
                let activated = ops::silu(&gate)?.broadcast_mul(&up)?;
                let value = down.forward(&activated)?;
                combined = (combined + (value * f64::from(route_weight))?)?;
                timings.expert_compute += compute_started.elapsed();
            }
            outputs.push(combined);
        }
        let routed = Tensor::cat(&outputs, 0)?;
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
    }
}
