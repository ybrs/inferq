use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use anyhow::{Context, Result};
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

#[derive(Debug, Clone, Serialize)]
pub struct RoutingCensusArtifact {
    pub schema_version: u32,
    pub model: GgufModelIdentity,
    pub routing_records: u64,
    pub layers: BTreeMap<usize, LayerRoutingCensus>,
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
        let file = File::create(&self.path)
            .with_context(|| format!("failed to create routing census {}", self.path.display()))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &self.artifact)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
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
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.path().join("census.json")).unwrap())
                .unwrap();
        assert_eq!(value["schema_version"], 1);
    }
}
