//! An OpenAI-compatible HTTP surface over the quantized runtime.
//!
//! The engine decodes one sequence at a time (see [`engine`]), so the server
//! is a queue in front of a single inference slot rather than a batching
//! scheduler. Requests are stateless: each one renders its whole conversation
//! and prefills it.

pub mod api;
pub mod engine;
pub mod http;
pub mod request;
mod stop;

pub use engine::{EngineConfig, EngineHandle, ModelInfo, Warmup};
pub use http::{ServerState, serve};
