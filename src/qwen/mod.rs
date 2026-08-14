//! Readable scalar/tensor reference implementation of Qwen3-Next.

mod attention;
mod deltanet;
mod model;
mod moe;
mod norm;

pub use model::{Model, ModelState};
pub use moe::{Route, top_k_routes};
pub use norm::{l2_normalize, rms_norm, rms_norm_gated};

use candle_core::{DType, Result, Tensor};

pub(crate) fn linear(xs: &Tensor, weight: &Tensor) -> Result<Tensor> {
    // Candle's CPU backend does not implement BF16/F16 matmul. Compute the
    // reference projection in F32, then restore the checkpoint dtype so layer
    // boundaries match the published BF16 model.
    let output_dtype = weight.dtype();
    let xs = f32_tensor(xs)?;
    let weight = f32_tensor(weight)?;
    let dims = xs.dims();
    let hidden = *dims.last().unwrap_or(&0);
    let rows = xs.elem_count() / hidden;
    let out = xs.reshape((rows, hidden))?.matmul(&weight.t()?)?;
    let mut shape = dims.to_vec();
    *shape.last_mut().unwrap() = weight.dim(0)?;
    out.reshape(shape)?.to_dtype(output_dtype)
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
