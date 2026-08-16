//! Reusable expert residency preparation for quantized execution.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::{ExpertCacheStats, GgufCheckpoint, Qwen3NextConfig};

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FullExpertWarmupMode {
    OsPageCache,
    PinnedExpertCache,
}

#[derive(Debug, Clone, Copy)]
pub struct FullExpertWarmupProgress {
    pub tensors_completed: usize,
    pub tensors_total: usize,
    pub bytes_loaded: usize,
    pub bytes_total: usize,
}

#[derive(Debug, Clone)]
pub struct FullExpertWarmupReport {
    pub mode: FullExpertWarmupMode,
    pub tensor_count: usize,
    pub bytes_loaded: usize,
    pub elapsed: Duration,
    pub cache: ExpertCacheStats,
}

/// Sequentially visit every fused expert tensor in GGUF file order.
///
/// A configured expert cache pins every expert matrix, then combines compatible
/// gate/up rows without changing resident bytes. With zero cache capacity the
/// same traversal only warms the OS page cache.
pub fn warm_all_experts(
    checkpoint: &GgufCheckpoint,
    config: &Qwen3NextConfig,
    mut on_progress: impl FnMut(FullExpertWarmupProgress),
) -> Result<FullExpertWarmupReport> {
    let resident_layers = config.num_hidden_layers + config.mtp_num_hidden_layers;
    let mut tensors = Vec::with_capacity(resident_layers * 3);
    for layer in 0..resident_layers {
        for suffix in [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ] {
            let name = format!("blk.{layer}.{suffix}");
            let info = checkpoint
                .tensor_info(&name)
                .with_context(|| format!("GGUF is missing expert tensor {name:?}"))?;
            tensors.push((info.offset, name, info.storage_bytes));
        }
    }
    tensors.sort_unstable_by_key(|(offset, _, _)| *offset);
    let total_bytes = tensors.iter().try_fold(0usize, |total, (_, _, bytes)| {
        total
            .checked_add(*bytes)
            .context("full expert warmup byte count overflowed")
    })?;
    let cache_before = checkpoint.expert_cache_stats()?;
    let pin_in_process = cache_before.capacity_bytes > 0;
    if pin_in_process {
        ensure!(
            cache_before.capacity_bytes >= total_bytes,
            "full expert warmup needs at least {:.1} MiB of expert-cache capacity to pin all experts; configured {:.1} MiB",
            total_bytes as f64 / (1024. * 1024.),
            cache_before.capacity_bytes as f64 / (1024. * 1024.)
        );
    }

    let started = Instant::now();
    let mut loaded = 0usize;
    for (index, (_, name, expected_bytes)) in tensors.iter().enumerate() {
        let bytes = if pin_in_process {
            let tensor = checkpoint.expert_tensor(name)?;
            (0..config.num_experts).try_fold(0usize, |total, expert| {
                total
                    .checked_add(tensor.warm(expert)?)
                    .context("full expert warmup byte count overflowed")
            })?
        } else {
            checkpoint.warm_tensor_pages(name)?
        };
        ensure!(
            bytes == *expected_bytes,
            "warmup byte count changed for {name:?}: expected {expected_bytes}, got {bytes}"
        );
        loaded = loaded
            .checked_add(bytes)
            .context("full expert warmup byte count overflowed")?;
        on_progress(FullExpertWarmupProgress {
            tensors_completed: index + 1,
            tensors_total: tensors.len(),
            bytes_loaded: loaded,
            bytes_total: total_bytes,
        });
    }
    if pin_in_process {
        // File-order warmup keeps HDD access sequential. Once all matrices are
        // resident, replace each gate/up pair with one row-concatenated cache
        // entry without changing the byte footprint or rereading the GGUF.
        for layer in 0..resident_layers {
            let prefix = format!("blk.{layer}");
            checkpoint.fuse_cached_expert_pair(
                &format!("{prefix}.ffn_gate_exps.weight"),
                &format!("{prefix}.ffn_up_exps.weight"),
            )?;
        }
        checkpoint.mark_expert_cache_fully_resident()?;
    }
    let elapsed = started.elapsed();
    let cache = checkpoint
        .expert_cache_stats()?
        .activity_since(cache_before);
    Ok(FullExpertWarmupReport {
        mode: if pin_in_process {
            FullExpertWarmupMode::PinnedExpertCache
        } else {
            FullExpertWarmupMode::OsPageCache
        },
        tensor_count: tensors.len(),
        bytes_loaded: loaded,
        elapsed,
        cache,
    })
}
