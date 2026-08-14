use candle_core::{D, Result, Tensor};
use candle_nn::ops;

use super::f32_tensor;

/// Qwen3-Next RMSNorm uses `(1 + weight)`, unlike standard Llama RMSNorm.
pub fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs = f32_tensor(xs)?;
    let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
    let normalized = xs.broadcast_div(&(variance + eps)?.sqrt()?)?;
    let weight = (f32_tensor(weight)? + 1.)?;
    normalized.broadcast_mul(&weight)?.to_dtype(dtype)
}

pub fn rms_norm_gated(xs: &Tensor, weight: &Tensor, gate: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs = f32_tensor(xs)?;
    let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
    let normalized = xs.broadcast_div(&(variance + eps)?.sqrt()?)?;
    let normalized = normalized.broadcast_mul(&f32_tensor(weight)?)?;
    normalized
        .broadcast_mul(&ops::silu(&f32_tensor(gate)?)?)?
        .to_dtype(dtype)
}

pub fn l2_normalize(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs = f32_tensor(xs)?;
    let norm = (xs.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?;
    xs.broadcast_div(&norm)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn qwen_rms_norm_is_one_centered() {
        let x = Tensor::new(&[[3f32, 4.]], &Device::Cpu).unwrap();
        let w = Tensor::zeros(2, candle_core::DType::F32, &Device::Cpu).unwrap();
        let got = rms_norm(&x, &w, 0.).unwrap().to_vec2::<f32>().unwrap();
        let scale = (12.5f32).sqrt();
        assert!((got[0][0] - 3. / scale).abs() < 1e-6);
        assert!((got[0][1] - 4. / scale).abs() < 1e-6);
    }
}
