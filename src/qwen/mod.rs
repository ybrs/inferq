//! Readable scalar/tensor reference implementation of Qwen3-Next.

mod attention;
mod deltanet;
mod model;
mod moe;
mod norm;
mod quantized_attention;
mod quantized_deltanet;
mod quantized_layer;
mod quantized_model;
mod quantized_moe;
mod quantized_mtp;

pub use attention::{ReferenceAttentionState, reference_attention, reference_attention_step};
pub use deltanet::reference_deltanet;
pub use model::{
    ForwardTimingReport, ForwardTimings, Model, ModelState, ReferenceLayerOutput,
    reference_full_layer, reference_linear_layer,
};
pub use moe::{Route, reference_routes, reference_sparse_moe, top_k_routes};
pub use norm::{l2_normalize, rms_norm, rms_norm_gated};
pub use quantized_attention::{
    QuantizedAttentionLayer, QuantizedAttentionState, QuantizedAttentionTimings,
};
pub use quantized_deltanet::{
    QuantizedDeltaCheckpoint, QuantizedDeltaLayer, QuantizedDeltaSnapshots, QuantizedDeltaState,
    QuantizedDeltaTimings,
};
pub use quantized_layer::{
    QuantizedFullLayer, QuantizedLayerOutput, QuantizedLayerTimings, QuantizedLinearLayer,
};
pub use quantized_model::{
    QuantizedForwardOutput, QuantizedForwardTimingReport, QuantizedForwardTimings,
    QuantizedLayerTimingReport, QuantizedModel, QuantizedModelCheckpoint, QuantizedModelState,
    QuantizedOperationTimingReport, QuantizedStageTimingReport, QuantizedStateSnapshots,
};
pub use quantized_moe::{
    QuantizedMoeLayer, QuantizedMoeOutput, QuantizedMoeRoutingStats, QuantizedMoeTimings,
};
pub use quantized_mtp::{
    QuantizedMtpHead, QuantizedMtpOutput, QuantizedMtpState, QuantizedMtpTimings,
};

use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};

use crate::Checkpoint;

#[cfg(test)]
pub(crate) fn linear(xs: &Tensor, weight: &Tensor) -> Result<Tensor> {
    linear_inner(xs, weight, None)
}

pub(crate) fn linear_profiled(
    xs: &Tensor,
    weight: &Tensor,
    timings: &mut ForwardTimings,
) -> Result<Tensor> {
    linear_inner(xs, weight, Some(timings))
}

fn linear_inner(
    xs: &Tensor,
    weight: &Tensor,
    mut timings: Option<&mut ForwardTimings>,
) -> Result<Tensor> {
    // Candle's CPU backend does not implement BF16/F16 matmul. Compute the
    // reference projection in F32, then restore the checkpoint dtype so layer
    // boundaries match the published BF16 model.
    let output_dtype = weight.dtype();
    let conversion_started = Instant::now();
    let xs = f32_tensor(xs)?;
    let weight = f32_tensor(weight)?;
    if let Some(profile) = timings.as_deref_mut() {
        profile.dtype_conversion += conversion_started.elapsed();
    }
    let dims = xs.dims();
    let hidden = *dims.last().unwrap_or(&0);
    let rows = xs.elem_count() / hidden;
    let output_size = weight.dim(0)?;
    let xs = xs.reshape((rows, hidden))?;
    let weight = weight.t()?;
    let matmul_started = Instant::now();
    let out = xs.matmul(&weight)?;
    if let Some(profile) = timings.as_deref_mut() {
        profile.matmul += matmul_started.elapsed();
    }
    let mut shape = dims.to_vec();
    *shape.last_mut().unwrap() = output_size;
    let out = out.reshape(shape)?;
    let conversion_started = Instant::now();
    let out = out.to_dtype(output_dtype)?;
    if let Some(profile) = timings {
        profile.dtype_conversion += conversion_started.elapsed();
    }
    Ok(out)
}

pub(crate) fn load_profiled(
    checkpoint: &Checkpoint,
    name: &str,
    device: &Device,
    timings: &mut ForwardTimings,
) -> anyhow::Result<Tensor> {
    let started = Instant::now();
    let tensor = checkpoint.load(name, device)?;
    timings.weight_load += started.elapsed();
    Ok(tensor)
}

pub(crate) fn load_f32_profiled(
    checkpoint: &Checkpoint,
    name: &str,
    device: &Device,
    timings: &mut ForwardTimings,
) -> anyhow::Result<Tensor> {
    let tensor = load_profiled(checkpoint, name, device, timings)?;
    if tensor.dtype() == DType::F32 {
        Ok(tensor)
    } else {
        let started = Instant::now();
        let tensor = tensor.to_dtype(DType::F32)?;
        timings.dtype_conversion += started.elapsed();
        Ok(tensor)
    }
}

pub(crate) fn f32_tensor(xs: &Tensor) -> Result<Tensor> {
    if xs.dtype() == DType::F32 {
        Ok(xs.clone())
    } else {
        xs.to_dtype(DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::linear;

    #[test]
    fn linear_promotes_bf16_for_cpu_matmul() {
        let x = Tensor::new(&[[1f32, 2.]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let weight = Tensor::new(&[[3f32, 4.]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let result = linear(&x, &weight).unwrap();
        assert_eq!(result.dtype(), DType::BF16);
        assert_eq!(
            result
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![11.]]
        );
    }
}
