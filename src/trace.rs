use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

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
