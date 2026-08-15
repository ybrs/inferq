use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::GgufModelIdentity;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutingRecord {
    pub token_index: usize,
    pub token_id: u32,
    pub layer: usize,
    pub selected_expert_ids: Vec<usize>,
    pub router_weights: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_logits: Option<Vec<f32>>,
}

pub trait RoutingTrace: Send {
    fn record(&mut self, record: &RoutingRecord) -> Result<()>;
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Default)]
pub struct RoutingTraceSet {
    sinks: Vec<Box<dyn RoutingTrace>>,
}

impl RoutingTraceSet {
    pub fn push(&mut self, sink: Box<dyn RoutingTrace>) {
        self.sinks.push(sink);
    }

    pub fn is_empty(&self) -> bool {
        self.sinks.is_empty()
    }
}

impl RoutingTrace for RoutingTraceSet {
    fn record(&mut self, record: &RoutingRecord) -> Result<()> {
        for sink in &mut self.sinks {
            sink.record(record)?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        for sink in &mut self.sinks {
            sink.flush()?;
        }
        Ok(())
    }
}

pub struct JsonlRoutingTrace {
    writer: BufWriter<File>,
    include_logits: bool,
}

impl JsonlRoutingTrace {
    pub fn create(path: impl AsRef<Path>, include_logits: bool) -> Result<Self> {
        let path = path.as_ref();
        let file = File::create(path)
            .with_context(|| format!("failed to create routing trace {}", path.display()))?;
        Ok(Self {
            writer: BufWriter::new(file),
            include_logits,
        })
    }
}

impl RoutingTrace for JsonlRoutingTrace {
    fn record(&mut self, record: &RoutingRecord) -> Result<()> {
        if self.include_logits {
            serde_json::to_writer(&mut self.writer, record)?;
        } else {
            let mut compact = record.clone();
            compact.router_logits = None;
            serde_json::to_writer(&mut self.writer, &compact)?;
        }
        self.writer.write_all(b"\n")?;
        Ok(())
    }
    fn flush(&mut self) -> Result<()> {
        Ok(self.writer.flush()?)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayerRoutingCensus {
    pub routed_tokens: u64,
    pub expert_selections: u64,
    pub expert_counts: BTreeMap<usize, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingCensusArtifact {
    pub schema_version: u32,
    pub model: GgufModelIdentity,
    pub routing_records: u64,
    pub layers: BTreeMap<usize, LayerRoutingCensus>,
}

impl RoutingCensusArtifact {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read routing census {}", path.display()))?;
        let artifact: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid routing census {}", path.display()))?;
        ensure!(
            artifact.schema_version == 1,
            "unsupported routing census schema version {}",
            artifact.schema_version
        );
        Ok(artifact)
    }

    pub fn validate_for(
        &self,
        model: &GgufModelIdentity,
        layer_count: usize,
        expert_count: usize,
    ) -> Result<()> {
        ensure!(
            self.model.layout_fingerprint == model.layout_fingerprint
                && self.model.size_bytes == model.size_bytes
                && self.model.quantization == model.quantization,
            "routing census model identity does not match the loaded GGUF"
        );
        for (layer, census) in &self.layers {
            ensure!(
                *layer < layer_count,
                "routing census contains invalid layer {layer}; model has {layer_count} layers"
            );
            for expert in census.expert_counts.keys() {
                ensure!(
                    *expert < expert_count,
                    "routing census layer {layer} contains invalid expert {expert}; model has {expert_count} experts"
                );
            }
        }
        Ok(())
    }

    pub fn hottest_experts(&self, per_layer: usize) -> Vec<(usize, Vec<usize>)> {
        self.layers
            .iter()
            .map(|(layer, census)| {
                let mut experts: Vec<_> = census
                    .expert_counts
                    .iter()
                    .map(|(expert, count)| (*expert, *count))
                    .collect();
                experts.sort_unstable_by(
                    |(left_expert, left_count), (right_expert, right_count)| {
                        right_count
                            .cmp(left_count)
                            .then_with(|| left_expert.cmp(right_expert))
                    },
                );
                experts.truncate(per_layer);
                (
                    *layer,
                    experts.into_iter().map(|(expert, _)| expert).collect(),
                )
            })
            .collect()
    }

    fn matches_model(&self, model: &GgufModelIdentity) -> bool {
        self.model.layout_fingerprint == model.layout_fingerprint
            && self.model.size_bytes == model.size_bytes
            && self.model.quantization == model.quantization
    }
}

/// Accumulates real model routes and rewrites a compact JSON sidecar on flush.
pub struct JsonRoutingCensus {
    path: std::path::PathBuf,
    artifact: RoutingCensusArtifact,
}

impl JsonRoutingCensus {
    pub fn create(path: impl AsRef<Path>, model: GgufModelIdentity) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            artifact: RoutingCensusArtifact {
                schema_version: 1,
                model,
                routing_records: 0,
                layers: BTreeMap::new(),
            },
        }
    }

    pub fn artifact(&self) -> &RoutingCensusArtifact {
        &self.artifact
    }

    pub fn resume(path: impl AsRef<Path>, model: GgufModelIdentity) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::create(path, model));
        }
        let artifact = RoutingCensusArtifact::from_path(path)?;
        ensure!(
            artifact.matches_model(&model),
            "existing routing census model identity does not match the loaded GGUF"
        );
        Ok(Self {
            path: path.to_path_buf(),
            artifact,
        })
    }
}

impl RoutingTrace for JsonRoutingCensus {
    fn record(&mut self, record: &RoutingRecord) -> Result<()> {
        let layer = self.artifact.layers.entry(record.layer).or_default();
        layer.routed_tokens += 1;
        layer.expert_selections += record.selected_expert_ids.len() as u64;
        for expert in &record.selected_expert_ids {
            *layer.expert_counts.entry(*expert).or_default() += 1;
        }
        self.artifact.routing_records += 1;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        let temporary = self.path.with_extension("census.tmp");
        let file = File::create(&temporary).with_context(|| {
            format!(
                "failed to create temporary routing census {}",
                temporary.display()
            )
        })?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &self.artifact)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        drop(writer);
        std::fs::rename(&temporary, &self.path).with_context(|| {
            format!(
                "failed to replace routing census {} from {}",
                self.path.display(),
                temporary.display()
            )
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> GgufModelIdentity {
        GgufModelIdentity {
            path: "model.gguf".into(),
            size_bytes: 42,
            modified_unix_nanos: Some(7),
            layout_fingerprint: "fnv1a64:test".into(),
            quantization: vec!["Q4K".into()],
        }
    }

    #[test]
    fn census_counts_layer_qualified_experts() {
        let directory = tempfile::tempdir().unwrap();
        let mut census =
            JsonRoutingCensus::create(directory.path().join("census.json"), identity());
        for (layer, experts) in [(0, vec![1, 3]), (0, vec![1, 4]), (1, vec![1, 4])] {
            census
                .record(&RoutingRecord {
                    token_index: 0,
                    token_id: 10,
                    layer,
                    selected_expert_ids: experts,
                    router_weights: vec![0.5, 0.5],
                    router_logits: None,
                })
                .unwrap();
        }
        assert_eq!(census.artifact().routing_records, 3);
        assert_eq!(census.artifact().layers[&0].expert_counts[&1], 2);
        assert_eq!(census.artifact().layers[&1].expert_counts[&1], 1);
        census.flush().unwrap();
        let artifact =
            RoutingCensusArtifact::from_path(directory.path().join("census.json")).unwrap();
        assert_eq!(
            artifact.hottest_experts(2),
            [(0, vec![1, 3]), (1, vec![1, 4])]
        );
        artifact.validate_for(&identity(), 2, 5).unwrap();
        let mut resumed =
            JsonRoutingCensus::resume(directory.path().join("census.json"), identity()).unwrap();
        resumed
            .record(&RoutingRecord {
                token_index: 1,
                token_id: 11,
                layer: 0,
                selected_expert_ids: vec![1],
                router_weights: vec![1.],
                router_logits: None,
            })
            .unwrap();
        assert_eq!(resumed.artifact().routing_records, 4);
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.path().join("census.json")).unwrap())
                .unwrap();
        assert_eq!(value["schema_version"], 1);
    }
}
