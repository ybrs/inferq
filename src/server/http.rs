//! Routes, authentication and the OpenAI response shapes.

use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Path, Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, KeepAlive, Sse},
    },
    routing::{get, post},
};
use rand::Rng;
use tokio::{net::TcpListener, sync::mpsc::unbounded_channel};
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::{
    api::{
        ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice, ChunkChoice,
        Delta, ErrorBody, ErrorEnvelope, Model, ModelList, ResponseFunctionCall, ResponseMessage,
        ResponseToolCall, Usage,
    },
    engine::{EngineHandle, Event, JobRequest, SubmitError},
    request::{opens_thinking, render_history_prefix, render_prompt, resolve_options},
};
use crate::tool_calls::ParsedToolCall;

/// Largest request body accepted, chosen to fit long conversations without
/// letting one request buffer unbounded memory.
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

/// How often a comment is written on an idle stream. Prefill on CPU can run
/// for minutes before the first token, and intermediaries drop silent
/// connections well before that.
const KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(10);

pub struct ServerState {
    pub engine: EngineHandle,
    /// When set, every `/v1` request must present this key.
    pub api_key: Option<String>,
    /// Whether an assistant turn opens a `<think>` block unless the request
    /// says otherwise.
    pub default_enable_thinking: bool,
}

/// An error in the OpenAI envelope, so clients surface it rather than a bare
/// status code.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }

    pub fn overloaded(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "server_error",
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "server_error",
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        // A client that cannot use this server usually reports nothing more
        // than "request failed", so every refusal is logged with its reason.
        tracing::warn!(
            status = self.status.as_u16(),
            kind = self.kind,
            message = %self.message,
            "rejected a request"
        );
        let body = ErrorEnvelope {
            error: ErrorBody {
                message: self.message,
                kind: self.kind,
                param: None,
                code: None,
            },
        };
        let mut response = (self.status, axum::Json(body)).into_response();
        if self.status == StatusCode::SERVICE_UNAVAILABLE {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, header::HeaderValue::from_static("1"));
        }
        response
    }
}

pub fn router(state: Arc<ServerState>) -> Router {
    let protected = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{model}", get(retrieve_model))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            require_api_key,
        ));
    Router::new()
        .route("/health", get(health))
        .merge(protected)
        .fallback(unknown_route)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

/// Bind and serve until the process is asked to stop.
pub async fn serve(address: SocketAddr, state: Arc<ServerState>) -> Result<()> {
    let listener = TcpListener::bind(address)
        .await
        .with_context(|| format!("failed to bind {address}"))?;
    let local = listener.local_addr().unwrap_or(address);
    tracing::info!(address = %local, model = %state.engine.info().id, "listening");
    eprintln!(
        "inferq: OpenAI-compatible API on http://{local}/v1 serving `{}`{}",
        state.engine.info().id,
        if state.api_key.is_some() {
            " (API key required)"
        } else {
            ""
        }
    );
    let engine = state.engine.clone();
    let result = axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("the HTTP server failed");
    // The last request's cache entry may still be with the writer thread.
    engine.flush_prompt_cache(crate::prompt_cache::DRAIN_TIMEOUT);
    result
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::warn!(error = %error, "failed to listen for shutdown signals");
        std::future::pending::<()>().await;
    }
    eprintln!("\ninferq: shutting down; in-flight requests are allowed to finish");
}

async fn require_api_key(
    State(state): State<Arc<ServerState>>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let Some(expected) = state.api_key.as_deref() else {
        return Ok(next.run(request).await);
    };
    let headers = request.headers();
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
        });
    match presented {
        Some(key) if constant_time_eq(key.as_bytes(), expected.as_bytes()) => {
            Ok(next.run(request).await)
        }
        Some(_) => Err(ApiError::unauthorized("invalid API key")),
        None => Err(ApiError::unauthorized(
            "missing API key: send it as `Authorization: Bearer <key>`",
        )),
    }
}

/// Compare without an early exit on the first differing byte. The length is
/// not secret; the key's contents are.
fn constant_time_eq(presented: &[u8], expected: &[u8]) -> bool {
    presented.len() == expected.len()
        && presented
            .iter()
            .zip(expected)
            .fold(0u8, |differences, (a, b)| differences | (a ^ b))
            == 0
}

async fn health(State(state): State<Arc<ServerState>>) -> Response {
    let info = state.engine.info();
    axum::Json(serde_json::json!({
        "status": "ok",
        "model": info.id,
        "quantization": info.quantization,
        "layout_fingerprint": info.layout_fingerprint,
        "max_position_embeddings": info.max_position_embeddings,
        "pending_requests": state.engine.pending(),
        "max_queue": state.engine.max_queue(),
        "prompt_cache": state.engine.prompt_cache_stats().map(|stats| serde_json::json!({
            "entries": stats.entries,
            "bytes": stats.bytes,
            "budget_bytes": stats.budget_bytes,
            "hits": stats.hits,
            "misses": stats.misses,
            "reused_tokens": stats.reused_tokens,
            "writes": stats.writes,
            "write_skips": stats.write_skips,
            "evictions": stats.evictions,
            "failures": stats.failures,
        })),
    }))
    .into_response()
}

async fn unknown_route(request: Request) -> ApiError {
    ApiError::not_found(format!(
        "unknown route {} {}",
        request.method(),
        request.uri().path()
    ))
}

async fn list_models(State(state): State<Arc<ServerState>>) -> Response {
    axum::Json(ModelList {
        object: "list",
        data: vec![describe_model(&state)],
    })
    .into_response()
}

async fn retrieve_model(
    State(state): State<Arc<ServerState>>,
    Path(model): Path<String>,
) -> Result<Response, ApiError> {
    if model != state.engine.info().id {
        return Err(ApiError::not_found(format!("unknown model `{model}`")));
    }
    Ok(axum::Json(describe_model(&state)).into_response())
}

fn describe_model(state: &ServerState) -> Model {
    Model {
        id: state.engine.info().id.clone(),
        object: "model",
        created: unix_seconds(),
        owned_by: "local",
    }
}

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let request: ChatCompletionRequest = serde_json::from_slice(&body)
        .map_err(|error| ApiError::bad_request(format!("invalid request body: {error}")))?;
    request
        .validate()
        .map_err(|error| ApiError::bad_request(format!("{error:#}")))?;
    let prompt = render_prompt(
        state.engine.tokenizer(),
        &request,
        state.default_enable_thinking,
    )
    .map_err(|error| ApiError::bad_request(format!("{error:#}")))?;
    // Everything before the final message is what the next request in an agent
    // session repeats; the engine caches at a boundary inside it.
    let stable_prefix = render_history_prefix(state.engine.tokenizer(), &request)
        .map_err(|error| ApiError::bad_request(format!("{error:#}")))?;
    let options = resolve_options(state.engine.defaults(), &request)
        .map_err(|error| ApiError::bad_request(format!("{error:#}")))?;
    // Enough to see what a client is actually asking for when it reports
    // nothing more useful than "request failed".
    tracing::info!(
        model = request.model.as_deref().unwrap_or("<unset>"),
        messages = request.messages.len(),
        tools = request.tool_definitions().len(),
        stream = request.stream,
        max_tokens = options.max_new_tokens,
        temperature = options.sampling.temperature,
        "accepted a chat completion"
    );
    let events = state
        .engine
        .submit(JobRequest {
            prompt,
            stable_prefix,
            options,
            stop_strings: request.stop_strings(),
            tools_enabled: !request.tool_definitions().is_empty(),
            thinking_open: opens_thinking(
                state.engine.tokenizer(),
                &request,
                state.default_enable_thinking,
            ),
        })
        .map_err(|error| match error {
            SubmitError::Busy => ApiError::overloaded(
                "the inference queue is full; this server decodes one request at a time",
            ),
            SubmitError::Stopped => ApiError::internal("the inference worker is not running"),
        })?;

    let id = completion_id();
    let model = state.engine.info().id.clone();
    let created = unix_seconds();
    if request.stream {
        let include_usage = request
            .stream_options
            .as_ref()
            .is_some_and(|options| options.include_usage);
        Ok(stream_response(
            events,
            id,
            model,
            created,
            include_usage,
            request,
        ))
    } else {
        collect_response(events, id, model, created, &request).await
    }
}

/// Type a turn's calls with the schemas the request declared, and give each
/// one the identifier the client will send back with its result.
fn response_tool_calls(
    calls: &[ParsedToolCall],
    request: &ChatCompletionRequest,
    indexed: bool,
) -> Vec<ResponseToolCall> {
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| ResponseToolCall {
            index: indexed.then_some(index),
            id: tool_call_id(),
            kind: "function",
            function: ResponseFunctionCall {
                name: call.name.clone(),
                arguments: call.arguments(request.tool_schema(&call.name)).to_string(),
            },
        })
        .collect()
}

/// Drain the job, returning one complete chat completion.
async fn collect_response(
    mut events: tokio::sync::mpsc::UnboundedReceiver<Event>,
    id: String,
    model: String,
    created: u64,
    request: &ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let mut content = String::new();
    while let Some(event) = events.recv().await {
        match event {
            Event::Delta(text) => content.push_str(&text),
            Event::Failed(message) => return Err(ApiError::internal(message)),
            Event::Done(completion) => {
                let tool_calls = response_tool_calls(&completion.tool_calls, request, false);
                return Ok(axum::Json(ChatCompletionResponse {
                    id,
                    object: "chat.completion",
                    created,
                    model,
                    choices: vec![Choice {
                        index: 0,
                        message: ResponseMessage {
                            role: "assistant",
                            // OpenAI reports null content for a turn that was
                            // nothing but calls.
                            content: if !tool_calls.is_empty() && content.trim().is_empty() {
                                None
                            } else {
                                Some(content)
                            },
                            tool_calls,
                        },
                        finish_reason: completion.finish_reason,
                    }],
                    usage: Usage::new(completion.prompt_tokens, completion.completion_tokens),
                })
                .into_response());
            }
        }
    }
    Err(ApiError::internal(
        "the inference worker ended the request without a result",
    ))
}

/// Relay the job as `text/event-stream` chunks.
///
/// A forwarding task owns the job's receiver, so when the client goes away the
/// SSE body is dropped, the forward fails, and dropping the receiver tells the
/// worker to abandon the request.
fn stream_response(
    mut events: tokio::sync::mpsc::UnboundedReceiver<Event>,
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
    request: ChatCompletionRequest,
) -> Response {
    let (chunks, receiver) = unbounded_channel::<Result<SseEvent, Infallible>>();
    tokio::spawn(async move {
        let chunk = |choices, usage| ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices,
            usage,
        };
        let send = |value: &ChatCompletionChunk| match serde_json::to_string(value) {
            Ok(json) => chunks.send(Ok(SseEvent::default().data(json))).is_ok(),
            Err(error) => {
                tracing::error!(error = %error, "failed to serialise a chunk");
                false
            }
        };
        // The first chunk carries the role and no content, as OpenAI's does.
        if !send(&chunk(
            vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant"),
                    content: None,
                    tool_calls: Vec::new(),
                },
                finish_reason: None,
            }],
            None,
        )) {
            return;
        }
        while let Some(event) = events.recv().await {
            match event {
                Event::Delta(text) => {
                    if !send(&chunk(
                        vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(text),
                                tool_calls: Vec::new(),
                            },
                            finish_reason: None,
                        }],
                        None,
                    )) {
                        return;
                    }
                }
                Event::Done(completion) => {
                    // A turn's calls arrive complete rather than assembled
                    // across chunks: this engine has the whole turn before it
                    // knows a call was made at all.
                    let tool_calls = response_tool_calls(&completion.tool_calls, &request, true);
                    if !tool_calls.is_empty()
                        && !send(&chunk(
                            vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                    tool_calls,
                                },
                                finish_reason: None,
                            }],
                            None,
                        ))
                    {
                        return;
                    }
                    let usage = Usage::new(completion.prompt_tokens, completion.completion_tokens);
                    let finished = send(&chunk(
                        vec![ChunkChoice {
                            index: 0,
                            delta: Delta::default(),
                            finish_reason: Some(completion.finish_reason),
                        }],
                        None,
                    ));
                    // A usage-only chunk carries no choices, which is how
                    // OpenAI reports usage on a stream.
                    if finished && (!include_usage || send(&chunk(vec![], Some(usage)))) {
                        let _ = chunks.send(Ok(SseEvent::default().data("[DONE]")));
                    }
                    return;
                }
                Event::Failed(message) => {
                    // The response has already begun, so the status line is
                    // spent; the error goes out as a final SSE payload in the
                    // same envelope a non-streamed failure would use.
                    let body = ErrorEnvelope {
                        error: ErrorBody {
                            message,
                            kind: "server_error",
                            param: None,
                            code: None,
                        },
                    };
                    if let Ok(json) = serde_json::to_string(&body) {
                        let _ = chunks.send(Ok(SseEvent::default().data(json)));
                    }
                    return;
                }
            }
        }
    });
    Sse::new(UnboundedReceiverStream::new(receiver))
        .keep_alive(KeepAlive::new().interval(KEEP_ALIVE_INTERVAL))
        .into_response()
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or_default()
}

fn completion_id() -> String {
    random_id("chatcmpl-", 24)
}

/// Identifier a client sends back on the `tool` message carrying the result.
fn tool_call_id() -> String {
    random_id("call_", 16)
}

fn random_id(prefix: &str, digits: usize) -> String {
    let mut id = String::from(prefix);
    let mut rng = rand::rng();
    for _ in 0..digits {
        id.push(char::from_digit(rng.random_range(0..16), 16).unwrap_or('0'));
    }
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_ids_look_like_openais_and_differ() {
        let id = completion_id();
        assert!(id.starts_with("chatcmpl-"));
        assert_eq!(id.len(), "chatcmpl-".len() + 24);
        assert_ne!(id, completion_id());
    }

    #[test]
    fn key_comparison_accepts_only_the_exact_key() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secrez"));
        assert!(!constant_time_eq(b"secret", b"secre"));
        assert!(!constant_time_eq(b"", b"secret"));
    }

    #[test]
    fn errors_carry_the_openai_envelope() {
        let response = ApiError::bad_request("nope").into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let busy = ApiError::overloaded("full").into_response();
        assert_eq!(busy.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(busy.headers().contains_key(header::RETRY_AFTER));
    }
}
