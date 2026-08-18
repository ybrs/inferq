//! End-to-end checks for the OpenAI-compatible server.
//!
//! These need the real GGUF and its tokenizer directory, so they are opt-in:
//! set `INFERQ_TEST_GGUF` and `INFERQ_TEST_MODEL_DIR` to run them. Without
//! those they skip with a message rather than failing.
//!
//! The client here is deliberately hand-rolled: one request per connection,
//! `Connection: close`, read to end of stream. That is enough to exercise the
//! routes without pulling an HTTP client into the dependency tree.

use std::{net::SocketAddr, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use qwen_engine::{
    GenerationOptions, PromptCacheConfig, SpeculativeMode,
    server::{EngineConfig, ServerState, Warmup, engine, http::router},
};
use serde_json::Value;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

const API_KEY: &str = "test-key-4f2c";
const MAX_NEW_TOKENS: usize = 8;

struct Server {
    address: SocketAddr,
    runtime: tokio::runtime::Runtime,
}

/// Boot the engine and bind the router on an ephemeral port.
fn server() -> Option<Server> {
    server_with(None)
}

/// As above, with a prompt cache. The block size is deliberately tiny so a
/// short test prompt still crosses a boundary.
fn server_with(prompt_cache: Option<PromptCacheConfig>) -> Option<Server> {
    let (Ok(gguf), Ok(model_dir)) = (
        std::env::var("INFERQ_TEST_GGUF"),
        std::env::var("INFERQ_TEST_MODEL_DIR"),
    ) else {
        eprintln!(
            "skipping: set INFERQ_TEST_GGUF and INFERQ_TEST_MODEL_DIR to run \
             the OpenAI server integration tests"
        );
        return None;
    };
    qwen_engine::threading::init();
    let handle = engine::start(EngineConfig {
        model: gguf.into(),
        tokenizer_model: model_dir.into(),
        served_model_name: Some("test-model".to_owned()),
        expert_cache_bytes: 8 * 1024 * 1024 * 1024,
        warmup: Warmup::None,
        snapshot_nontemporal: true,
        defaults: GenerationOptions {
            max_new_tokens: MAX_NEW_TOKENS,
            speculative_mode: SpeculativeMode::Auto,
            ..GenerationOptions::default()
        },
        max_queue: 4,
        prompt_cache,
        prefix_reuse: true,
    })
    .expect("start the engine");
    let state = Arc::new(ServerState {
        engine: handle,
        api_key: Some(API_KEY.to_owned()),
        default_enable_thinking: false,
    });
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build a runtime");
    let listener = runtime
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind an ephemeral port");
    let address = listener.local_addr().expect("read the bound address");
    runtime.spawn(async move {
        let _ = axum::serve(listener, router(state)).await;
    });
    Some(Server { address, runtime })
}

struct Response {
    status: u16,
    body: String,
}

impl Response {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body)
            .unwrap_or_else(|error| panic!("expected JSON, got {error}: {}", self.body))
    }
}

async fn send(
    address: SocketAddr,
    method: &str,
    path: &str,
    key: Option<&str>,
    body: Option<&str>,
) -> Result<Response> {
    let mut stream = TcpStream::connect(address).await.context("connect")?;
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nAccept: */*\r\n"
    );
    if let Some(key) = key {
        request.push_str(&format!("Authorization: Bearer {key}\r\n"));
    }
    if let Some(body) = body {
        request.push_str("Content-Type: application/json\r\n");
        request.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    request.push_str("\r\n");
    if let Some(body) = body {
        request.push_str(body);
    }
    stream
        .write_all(request.as_bytes())
        .await
        .context("write")?;
    stream.flush().await.context("flush")?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.context("read")?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text
        .split_once("\r\n\r\n")
        .context("response has no header terminator")?;
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .context("response has no status code")?;
    Ok(Response {
        status,
        body: body.to_owned(),
    })
}

fn chat_body(prompt: &str, stream: bool) -> String {
    serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW_TOKENS,
        "stream": stream,
        "stream_options": {"include_usage": true},
    })
    .to_string()
}

/// Reassemble an SSE body into its deltas and its terminal metadata.
fn parse_stream(body: &str) -> (String, Option<String>, Option<Value>) {
    let mut content = String::new();
    let mut finish_reason = None;
    let mut usage = None;
    let mut saw_done = false;
    for line in body.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            saw_done = true;
            continue;
        }
        let chunk: Value = serde_json::from_str(payload).expect("chunk is JSON");
        assert_eq!(chunk["object"], "chat.completion.chunk");
        if let Some(choice) = chunk["choices"].get(0) {
            if let Some(delta) = choice["delta"]["content"].as_str() {
                content.push_str(delta);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_owned());
            }
        }
        if chunk["usage"].is_object() {
            usage = Some(chunk["usage"].clone());
        }
    }
    assert!(saw_done, "stream did not end with [DONE]: {body}");
    (content, finish_reason, usage)
}

#[test]
fn health_is_public_but_the_api_requires_the_key() -> Result<()> {
    let Some(server) = server() else {
        return Ok(());
    };
    server.runtime.block_on(async {
        let health = send(server.address, "GET", "/health", None, None).await?;
        assert_eq!(health.status, 200);
        assert_eq!(health.json()["model"], "test-model");

        let anonymous = send(server.address, "GET", "/v1/models", None, None).await?;
        assert_eq!(anonymous.status, 401);
        assert_eq!(anonymous.json()["error"]["type"], "invalid_request_error");

        let wrong = send(server.address, "GET", "/v1/models", Some("nope"), None).await?;
        assert_eq!(wrong.status, 401);

        let models = send(server.address, "GET", "/v1/models", Some(API_KEY), None).await?;
        assert_eq!(models.status, 200);
        let body = models.json();
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["id"], "test-model");

        let missing = send(server.address, "GET", "/v1/nonsense", Some(API_KEY), None).await?;
        assert_eq!(missing.status, 404);
        anyhow::Ok(())
    })
}

#[test]
fn rejects_bodies_it_cannot_honour() -> Result<()> {
    let Some(server) = server() else {
        return Ok(());
    };
    server.runtime.block_on(async {
        for body in [
            "not json",
            r#"{"messages":[]}"#,
            r#"{"messages":[{"role":"user","content":"hi"}],"n":2}"#,
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#,
            r#"{"messages":[{"role":"user","content":"hi"}],"temperature":-1}"#,
        ] {
            let response = send(
                server.address,
                "POST",
                "/v1/chat/completions",
                Some(API_KEY),
                Some(body),
            )
            .await?;
            assert_eq!(response.status, 400, "body {body} should be rejected");
            assert!(
                response.json()["error"]["message"].is_string(),
                "error envelope missing for {body}"
            );
        }
        anyhow::Ok(())
    })
}

#[test]
fn streaming_and_buffered_completions_agree() -> Result<()> {
    let Some(server) = server() else {
        return Ok(());
    };
    server.runtime.block_on(async {
        let prompt = "Reply with the single word: ready.";
        let buffered = send(
            server.address,
            "POST",
            "/v1/chat/completions",
            Some(API_KEY),
            Some(&chat_body(prompt, false)),
        )
        .await?;
        assert_eq!(buffered.status, 200);
        let body = buffered.json();
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .expect("content is a string")
            .to_owned();
        assert!(!content.is_empty(), "no content in {body}");
        let usage = body["usage"].clone();
        assert!(usage["prompt_tokens"].as_u64().unwrap_or(0) > 0);
        let completion_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
        assert!(completion_tokens > 0 && completion_tokens <= MAX_NEW_TOKENS as u64);
        assert_eq!(
            usage["total_tokens"],
            usage["prompt_tokens"].as_u64().unwrap_or(0) + completion_tokens
        );

        // Both requests decode greedily from the same prompt, so the streamed
        // deltas have to reassemble into exactly the buffered content.
        let streamed = send(
            server.address,
            "POST",
            "/v1/chat/completions",
            Some(API_KEY),
            Some(&chat_body(prompt, true)),
        )
        .await?;
        assert_eq!(streamed.status, 200);
        let (text, finish_reason, streamed_usage) = parse_stream(&streamed.body);
        assert_eq!(text, content);
        assert_eq!(
            finish_reason.as_deref(),
            body["choices"][0]["finish_reason"].as_str()
        );
        assert_eq!(streamed_usage.expect("usage chunk"), usage);
        anyhow::Ok(())
    })
}

#[test]
fn stop_strings_truncate_the_response() -> Result<()> {
    let Some(server) = server() else {
        return Ok(());
    };
    server.runtime.block_on(async {
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "Count: one two three four five"}],
            "max_tokens": 32,
            "stop": ["three"],
        })
        .to_string();
        let response = tokio::time::timeout(
            Duration::from_secs(600),
            send(
                server.address,
                "POST",
                "/v1/chat/completions",
                Some(API_KEY),
                Some(&body),
            ),
        )
        .await
        .context("request timed out")??;
        assert_eq!(response.status, 200);
        let value = response.json();
        let content = value["choices"][0]["message"]["content"]
            .as_str()
            .expect("content is a string");
        assert!(
            !content.contains("three"),
            "stop sequence leaked into {content:?}"
        );
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
        anyhow::Ok(())
    })
}

#[test]
fn a_repeated_prompt_is_restored_from_the_cache() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let Some(server) = server_with(Some(PromptCacheConfig {
        dir: directory.path().to_path_buf(),
        budget_bytes: 64 * 1024 * 1024 * 1024,
        block_tokens: 8,
        min_tokens: 8,
    })) else {
        return Ok(());
    };
    server.runtime.block_on(async {
        let prompt = "Describe, in one sentence, what a prompt cache is for.";
        let first = send(
            server.address,
            "POST",
            "/v1/chat/completions",
            Some(API_KEY),
            Some(&chat_body(prompt, false)),
        )
        .await?;
        assert_eq!(first.status, 200);
        let answer = first.json()["choices"][0]["message"]["content"].clone();

        // The write is asynchronous; wait for it before asking again.
        for _ in 0..600 {
            let health = send(server.address, "GET", "/health", None, None).await?;
            if health.json()["prompt_cache"]["writes"]
                .as_u64()
                .unwrap_or(0)
                > 0
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let second = send(
            server.address,
            "POST",
            "/v1/chat/completions",
            Some(API_KEY),
            Some(&chat_body(prompt, false)),
        )
        .await?;
        assert_eq!(second.status, 200);
        // A restored prefix must not change the answer.
        assert_eq!(second.json()["choices"][0]["message"]["content"], answer);
        assert_eq!(
            second.json()["usage"]["prompt_tokens"],
            first.json()["usage"]["prompt_tokens"]
        );

        let stats = send(server.address, "GET", "/health", None, None)
            .await?
            .json();
        let cache = &stats["prompt_cache"];
        assert!(cache["writes"].as_u64().unwrap_or(0) >= 1, "{cache}");
        assert!(cache["hits"].as_u64().unwrap_or(0) >= 1, "{cache}");
        assert!(cache["reused_tokens"].as_u64().unwrap_or(0) >= 8, "{cache}");
        assert_eq!(cache["failures"], 0);
        assert_eq!(cache["entries"].as_u64().unwrap_or(0), 1, "{cache}");
        anyhow::Ok(())
    })
}
