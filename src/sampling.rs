use std::cmp::Ordering;

use anyhow::{Result, bail, ensure};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.,
            top_k: None,
            top_p: None,
            min_p: None,
            seed: 0,
        }
    }
}

pub struct Sampler {
    config: SamplingConfig,
    rng: StdRng,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Result<Self> {
        ensure!(
            config.temperature >= 0. && config.temperature.is_finite(),
            "temperature must be finite and non-negative"
        );
        if let Some(p) = config.top_p {
            ensure!(p > 0. && p <= 1., "top_p must be in (0, 1]");
        }
        if let Some(p) = config.min_p {
            ensure!((0. ..=1.).contains(&p), "min_p must be in [0, 1]");
        }
        if let Some(k) = config.top_k {
            ensure!(k > 0, "top_k must be greater than zero");
        }
        let rng = StdRng::seed_from_u64(config.seed);
        Ok(Self { config, rng })
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        ensure!(!logits.is_empty(), "cannot sample empty logits");
        if self.config.temperature == 0. {
            return argmax(logits).map(|v| v as u32);
        }
        let mut candidates: Vec<(usize, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, x)| x.is_finite())
            .collect();
        ensure!(!candidates.is_empty(), "all logits are non-finite");
        candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        if let Some(k) = self.config.top_k {
            candidates.truncate(k.min(candidates.len()));
        }
        let max = candidates[0].1 / self.config.temperature;
        let mut total = 0.;
        for (_, p) in &mut candidates {
            *p = (*p / self.config.temperature - max).exp();
            total += *p;
        }
        for (_, p) in &mut candidates {
            *p /= total;
        }
        if let Some(min_p) = self.config.min_p {
            let cutoff = candidates[0].1 * min_p;
            candidates.retain(|(_, p)| *p >= cutoff);
        }
        if let Some(top_p) = self.config.top_p {
            let mut cumulative = 0.;
            let mut keep = 0;
            for (_, p) in &candidates {
                cumulative += *p;
                keep += 1;
                if cumulative >= top_p {
                    break;
                }
            }
            candidates.truncate(keep);
        }
        let total: f32 = candidates.iter().map(|(_, p)| p).sum();
        let mut draw = self.rng.random::<f32>() * total;
        for (id, p) in candidates {
            if draw <= p {
                return Ok(id as u32);
            }
            draw -= p;
        }
        bail!("sampling failed due to invalid probability mass")
    }
}

pub fn argmax(values: &[f32]) -> Result<usize> {
    values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
        .ok_or_else(|| anyhow::anyhow!("all logits are non-finite"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_deterministic() {
        let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
        assert_eq!(sampler.sample(&[-1., 3., 2.]).unwrap(), 1);
    }

    #[test]
    fn seeded_sampling_is_repeatable() {
        let cfg = SamplingConfig {
            temperature: 1.,
            top_k: Some(2),
            top_p: None,
            min_p: None,
            seed: 42,
        };
        let mut a = Sampler::new(cfg.clone()).unwrap();
        let mut b = Sampler::new(cfg).unwrap();
        let lhs: Vec<_> = (0..20).map(|_| a.sample(&[1., 2., 3.]).unwrap()).collect();
        let rhs: Vec<_> = (0..20).map(|_| b.sample(&[1., 2., 3.]).unwrap()).collect();
        assert_eq!(lhs, rhs);
    }
}
