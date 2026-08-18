//! OpenAI-compatible request and response bodies.
//!
//! Only the subset this engine can honour is modelled. Unknown fields are
//! ignored so that clients sending the full OpenAI schema still work, but a
//! field that would silently change results if ignored is rejected instead;
//! see [`ChatCompletionRequest::validate`].

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::tokenizer::{ChatMessage, ChatRole, ChatToolCall};

/// `POST /v1/chat/completions`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<RequestMessage>,
    /// Superseded by `max_completion_tokens` in the OpenAI schema; both are
    /// accepted and the newer one wins.
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// Not in the OpenAI schema; accepted because this engine samples with it.
    pub top_k: Option<usize>,
    /// Not in the OpenAI schema; accepted because this engine samples with it.
    pub min_p: Option<f32>,
    pub seed: Option<u64>,
    pub stop: Option<StopSequences>,
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub n: Option<usize>,
    /// OpenAI's reasoning knob. It is categorical rather than a token count,
    /// so the server maps each level to a budget; see [`ReasoningEffort`].
    pub reasoning_effort: Option<String>,
    /// Qwen's convention for selecting the thinking generation prefix.
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// Flat spelling of `chat_template_kwargs.enable_thinking`.
    pub enable_thinking: Option<bool>,
    /// Not in the OpenAI schema: force-close `<think>` after exactly N
    /// committed tokens, which is what Anthropic and Google expose and OpenAI
    /// does not. Takes precedence over `reasoning_effort`.
    pub thinking_budget: Option<usize>,
    /// Function definitions the model may call, in OpenAI's shape:
    /// `{"type": "function", "function": {"name", "description", "parameters"}}`.
    pub tools: Option<Vec<serde_json::Value>>,
    /// `auto` and `none` are honoured. Forcing a call is not: this engine does
    /// not constrain decoding, so it cannot promise one.
    pub tool_choice: Option<serde_json::Value>,
    // Fields below are rejected when set, because ignoring them would return a
    // response that quietly does not match what was asked for.
    pub functions: Option<Vec<serde_json::Value>>,
    pub logprobs: Option<bool>,
    pub response_format: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ChatTemplateKwargs {
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(stop) => vec![stop],
            Self::Many(stops) => stops,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RequestMessage {
    pub role: String,
    pub content: Option<MessageContent>,
    /// Calls this assistant turn made, echoed back by the client so the model
    /// sees its own tool use.
    pub tool_calls: Option<Vec<RequestToolCall>>,
    /// Sent with a `tool` message; carried only so a client that requires it
    /// round-trips cleanly.
    pub tool_call_id: Option<String>,
    /// Some clients keep an assistant turn's thinking out of `content`.
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RequestToolCall {
    pub id: Option<String>,
    pub function: Option<RequestFunctionCall>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RequestFunctionCall {
    pub name: String,
    /// A JSON object encoded as a string, as OpenAI specifies.
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
}

impl MessageContent {
    /// Flatten to plain text. Multi-part content is concatenated the way the
    /// OpenAI clients that emit it expect; a non-text part is an error rather
    /// than a silently dropped input.
    pub fn into_text(self) -> Result<String> {
        match self {
            Self::Text(text) => Ok(text),
            Self::Parts(parts) => {
                let mut text = String::new();
                for part in parts {
                    ensure!(
                        part.kind == "text",
                        "unsupported content part type `{}`: this engine accepts text only",
                        part.kind
                    );
                    text.push_str(part.text.as_deref().unwrap_or_default());
                }
                Ok(text)
            }
        }
    }
}

impl ChatCompletionRequest {
    /// Reject what cannot be honoured, before any work is queued.
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.messages.is_empty(), "`messages` must not be empty");
        ensure!(
            self.n.is_none_or(|n| n == 1),
            "`n` must be 1: this engine decodes one sequence at a time"
        );
        ensure!(
            self.functions.as_ref().is_none_or(Vec::is_empty),
            "the deprecated `functions` field is not supported; use `tools`"
        );
        if let Some(choice) = &self.tool_choice {
            let choice = choice.as_str().unwrap_or("named");
            ensure!(
                matches!(choice, "auto" | "none"),
                "`tool_choice` must be `auto` or `none`: this engine does not \
                 constrain decoding and cannot guarantee a forced call"
            );
        }
        ensure!(self.logprobs != Some(true), "`logprobs` is not supported");
        ensure!(
            self.response_format.is_none()
                || self
                    .response_format
                    .as_ref()
                    .and_then(|format| format.get("type"))
                    .and_then(serde_json::Value::as_str)
                    == Some("text"),
            "`response_format` other than `text` is not supported"
        );
        Ok(())
    }

    /// Requested output length, if the client bounded it.
    pub fn max_new_tokens(&self) -> Option<usize> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    pub fn stop_strings(&self) -> Vec<String> {
        self.stop
            .clone()
            .map(StopSequences::into_vec)
            .unwrap_or_default()
            .into_iter()
            .filter(|stop| !stop.is_empty())
            .collect()
    }

    /// Whether the assistant turn should open an unclosed `<think>` block.
    ///
    /// `reasoning_effort: "none"` says the same thing in OpenAI's vocabulary.
    pub fn enable_thinking(&self) -> Option<bool> {
        if self.reasoning_effort() == Some(ReasoningEffort::None) {
            return Some(false);
        }
        self.enable_thinking.or(self
            .chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.enable_thinking))
    }

    /// The requested effort level, ignoring anything unrecognised: a client
    /// that sends a level this build does not know still gets an answer, at
    /// the server's default effort.
    pub fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.reasoning_effort
            .as_deref()
            .and_then(ReasoningEffort::parse)
    }

    /// The tools the model may call, or nothing when the request disabled them.
    pub fn tool_definitions(&self) -> &[serde_json::Value] {
        if self
            .tool_choice
            .as_ref()
            .and_then(serde_json::Value::as_str)
            == Some("none")
        {
            return &[];
        }
        self.tools.as_deref().unwrap_or_default()
    }

    /// The schema of one named tool, used to type a parsed call's arguments.
    pub fn tool_schema(&self, name: &str) -> Option<&serde_json::Value> {
        self.tool_definitions().iter().find_map(|tool| {
            let function = tool.get("function")?;
            (function.get("name")?.as_str()? == name).then_some(function)
        })
    }

    /// Convert to the renderer's message form, validating roles.
    pub fn chat_messages(&self) -> Result<Vec<ChatMessage>> {
        self.messages
            .iter()
            .map(|message| {
                let role = match message.role.as_str() {
                    "system" | "developer" => ChatRole::System,
                    "user" => ChatRole::User,
                    "assistant" => ChatRole::Assistant,
                    "tool" | "function" => ChatRole::Tool,
                    other => bail!("unsupported message role `{other}`"),
                };
                let content = message
                    .content
                    .clone()
                    .map(MessageContent::into_text)
                    .transpose()?
                    .unwrap_or_default();
                let tool_calls = message
                    .tool_calls
                    .iter()
                    .flatten()
                    .map(|call| {
                        let function = call
                            .function
                            .as_ref()
                            .context("a tool call needs a `function`")?;
                        ensure!(!function.name.is_empty(), "a tool call needs a name");
                        // OpenAI encodes arguments as a JSON string. A client
                        // that sends something else is told so rather than
                        // having its call quietly dropped.
                        let arguments = match function.arguments.as_deref() {
                            None | Some("") => serde_json::Value::Object(Default::default()),
                            Some(text) => serde_json::from_str(text).with_context(|| {
                                format!("tool call `{}` has unparseable arguments", function.name)
                            })?,
                        };
                        Ok(ChatToolCall {
                            name: function.name.clone(),
                            arguments,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(ChatMessage {
                    role,
                    content,
                    tool_calls,
                    reasoning: message.reasoning_content.clone(),
                })
            })
            .collect()
    }
}

/// OpenAI's `reasoning_effort` levels.
///
/// OpenAI has no token budget in its API — effort is categorical, and
/// `max_completion_tokens` bounds reasoning and answer together. This server
/// therefore maps each level to a budget the operator chooses, because what
/// "high" can afford depends entirely on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    /// Not an OpenAI level on every model, but clients send it.
    XHigh,
}

impl ReasoningEffort {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Some(Self::None),
            "minimal" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" | "max" => Some(Self::XHigh),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }

    /// Every level a budget can be configured for. `None` is not one: it turns
    /// the thinking section off rather than bounding it.
    pub const BUDGETED: [Self; 5] = [
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::XHigh,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// A stop token, a stop string, or the end of the assistant turn.
    Stop,
    /// The token budget ran out first.
    Length,
    /// The turn ended by calling tools.
    ToolCalls,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Present when the turn had a thinking section, as OpenAI reports it for
    /// its reasoning models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: usize,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens_details: None,
        }
    }

    /// Report how much of the output was thinking. `None` means the turn had
    /// no thinking section at all, which is different from having spent none.
    pub fn with_reasoning(mut self, reasoning_tokens: Option<usize>) -> Self {
        self.completion_tokens_details =
            reasoning_tokens.map(|reasoning_tokens| CompletionTokensDetails { reasoning_tokens });
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseMessage {
    pub role: &'static str,
    /// Null when the turn was nothing but tool calls, as OpenAI reports it.
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ResponseToolCall>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseToolCall {
    /// Position in this turn's list, which is how a streamed call is matched
    /// to its pieces.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ResponseFunctionCall,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseFunctionCall {
    pub name: String,
    /// A JSON object encoded as a string, as OpenAI specifies.
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ResponseMessage,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ResponseToolCall>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// Sent only on the final chunk, and only when the client asked for it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Model {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<Model>,
}

/// The OpenAI error envelope. Clients surface `error.message` verbatim.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub param: Option<String>,
    pub code: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorEnvelope {
    pub error: ErrorBody,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> ChatCompletionRequest {
        serde_json::from_str(body).expect("request parses")
    }

    #[test]
    fn parses_a_minimal_request() {
        let request = parse(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        request.validate().expect("valid");
        assert!(!request.stream);
        assert_eq!(request.max_new_tokens(), None);
        let messages = request.chat_messages().expect("roles map");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, ChatRole::User);
        assert_eq!(messages[0].content, "hi");
    }

    #[test]
    fn ignores_unknown_fields_and_prefers_max_completion_tokens() {
        let request = parse(
            r#"{"messages":[{"role":"user","content":"hi"}],"user":"someone",
                "presence_penalty":0.5,"max_tokens":10,"max_completion_tokens":20}"#,
        );
        assert_eq!(request.max_new_tokens(), Some(20));
    }

    #[test]
    fn parses_multi_part_content_and_rejects_non_text_parts() {
        let request = parse(
            r#"{"messages":[{"role":"user","content":[
                {"type":"text","text":"a"},{"type":"text","text":"b"}]}]}"#,
        );
        assert_eq!(
            request.chat_messages().expect("text parts")[0].content,
            "ab"
        );
        let image = parse(
            r#"{"messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"http://x"}}]}]}"#,
        );
        assert!(image.chat_messages().is_err());
    }

    #[test]
    fn parses_both_stop_shapes() {
        assert_eq!(
            parse(r#"{"messages":[],"stop":"END"}"#).stop_strings(),
            vec!["END".to_owned()]
        );
        assert_eq!(
            parse(r#"{"messages":[],"stop":["a","","b"]}"#).stop_strings(),
            vec!["a".to_owned(), "b".to_owned()]
        );
        assert!(
            parse(r#"{"messages":[],"stop":null}"#)
                .stop_strings()
                .is_empty()
        );
    }

    #[test]
    fn reads_thinking_from_either_spelling() {
        assert_eq!(
            parse(r#"{"messages":[],"chat_template_kwargs":{"enable_thinking":false}}"#)
                .enable_thinking(),
            Some(false)
        );
        assert_eq!(
            parse(r#"{"messages":[],"enable_thinking":true}"#).enable_thinking(),
            Some(true)
        );
        assert_eq!(parse(r#"{"messages":[]}"#).enable_thinking(), None);
    }

    #[test]
    fn rejects_requests_it_cannot_honour() {
        assert!(parse(r#"{"messages":[]}"#).validate().is_err());
        let one = r#"{"role":"user","content":"hi"}"#;
        assert!(
            parse(&format!(r#"{{"messages":[{one}],"n":2}}"#))
                .validate()
                .is_err()
        );
        assert!(
            parse(&format!(r#"{{"messages":[{one}],"logprobs":true}}"#))
                .validate()
                .is_err()
        );
        assert!(
            parse(&format!(
                r#"{{"messages":[{one}],"tool_choice":"required"}}"#
            ))
            .validate()
            .is_err()
        );
        assert!(
            parse(&format!(
                r#"{{"messages":[{one}],"functions":[{{"name":"f"}}]}}"#
            ))
            .validate()
            .is_err()
        );
        assert!(
            parse(&format!(
                r#"{{"messages":[{one}],"response_format":{{"type":"json_object"}}}}"#
            ))
            .validate()
            .is_err()
        );
        // Tools, `tool_choice: auto` and a text response format are honoured.
        assert!(
            parse(&format!(
                r#"{{"messages":[{one}],"tools":[{{"type":"function","function":{{"name":"f"}}}}],
                    "tool_choice":"auto","response_format":{{"type":"text"}}}}"#
            ))
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn maps_developer_role_to_system() {
        let request = parse(r#"{"messages":[{"role":"developer","content":"rules"}]}"#);
        assert_eq!(
            request.chat_messages().expect("roles map")[0].role,
            ChatRole::System
        );
        let tool = parse(r#"{"messages":[{"role":"tool","content":"x","tool_call_id":"c1"}]}"#);
        assert_eq!(
            tool.chat_messages().expect("roles map")[0].role,
            ChatRole::Tool
        );
        let bad = parse(r#"{"messages":[{"role":"nonsense","content":"x"}]}"#);
        assert!(bad.chat_messages().is_err());
    }
}
