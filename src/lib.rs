//! Qwen3-Coder-Next CPU inference primitives and runtime.

pub mod config;
pub mod gguf;
pub mod loader;
pub mod ngram;
pub mod profile;
pub mod qgemm;
pub mod qwen;
pub mod residency;
pub mod runtime;
pub mod sampling;
pub mod threading;
pub mod tokenizer;
pub mod trace;

pub use config::{LayerType, Qwen3NextConfig};
pub use gguf::{
    ExpertCacheStats, GgufCheckpoint, GgufExpertPair, GgufExpertTensor, GgufModelIdentity,
    GgufSummary, GgufTensorInfo, MultiRowPath, QuantizedEmbedding, QuantizedMatrix, inspect_gguf,
};
pub use loader::{Checkpoint, ModelSummary, TensorInfo};
pub use residency::{
    FullExpertWarmupMode, FullExpertWarmupProgress, FullExpertWarmupReport, warm_all_experts,
};
pub use runtime::{
    GenerationMetrics, GenerationOptions, GenerationResult, NgramMatchLengthStats,
    QuantizedDraftObservation, QuantizedGenerationMetrics, QuantizedGenerationResult,
    QuantizedNgramMetrics, QuantizedRuntime, QuantizedSpeculativeMetrics, Runtime, ThinkingMetrics,
};
