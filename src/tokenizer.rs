use std::{fs, path::Path};

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct ModelTokenizer {
    tokenizer: Tokenizer,
    chat_template: Option<String>,
    thinking_generation_prompt: bool,
    /// The token the tokenizer's own config calls the end of a turn.
    ///
    /// `config.json` is where a generation loop normally reads this, but not
    /// every published checkpoint fills in `eos_token_id` — Qwen3.6-35B-A3B
    /// does not — and a turn that never ends is not a turn. The tokenizer's
    /// config names it in either release, so it is read here too.
    eos_token: Option<u32>,
}

impl ModelTokenizer {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .with_context(|| format!("failed to load tokenizer {}", path.display()))?;
        let config_path = model_dir.as_ref().join("tokenizer_config.json");
        let config: Option<Value> =
            if config_path.exists() {
                let bytes = fs::read(&config_path).with_context(|| {
                    format!("failed to read tokenizer config {}", config_path.display())
                })?;
                Some(serde_json::from_slice(&bytes).with_context(|| {
                    format!("invalid tokenizer config {}", config_path.display())
                })?)
            } else {
                None
            };
        let chat_template = config
            .as_ref()
            .and_then(|config| config.get("chat_template"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        // `eos_token` is a string in every Qwen release; a checkpoint that
        // spells it as an object carries the text under `content`.
        let eos_token = config
            .as_ref()
            .and_then(|config| config.get("eos_token"))
            .and_then(|eos| {
                eos.as_str()
                    .or_else(|| eos.get("content").and_then(Value::as_str))
            })
            .and_then(|eos| tokenizer.token_to_id(eos));
        let thinking_generation_prompt = chat_template.as_deref().is_some_and(|template| {
            template.contains("enable_thinking") && template.contains("<think>")
        });
        Ok(Self {
            tokenizer,
            chat_template,
            thinking_generation_prompt,
            eos_token,
        })
    }

    /// The end-of-turn token named by `tokenizer_config.json`, if it names one
    /// this tokenizer knows.
    pub fn eos_token(&self) -> Option<u32> {
        self.eos_token
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    pub fn decode_token(&self, id: u32) -> Result<String> {
        self.decode(&[id], false)
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    pub fn supports_thinking_generation(&self) -> bool {
        self.thinking_generation_prompt
    }

    pub fn decode_stream(
        &self,
        skip_special_tokens: bool,
    ) -> impl FnMut(u32) -> Result<Option<String>> + '_ {
        let mut stream = self.tokenizer.decode_stream(skip_special_tokens);
        move |token| {
            stream
                .step(token)
                .map_err(|error| anyhow::anyhow!(error.to_string()))
        }
    }

    fn ensure_qwen_chat_template(&self) -> Result<()> {
        let template = self
            .chat_template
            .as_deref()
            .context("tokenizer_config.json does not define a chat_template")?;
        ensure!(
            template.contains("<|im_start|>") && template.contains("<|im_end|>"),
            "unsupported chat template: expected Qwen im_start/im_end markers"
        );
        Ok(())
    }

    /// Render the official template's plain-message, no-tools subset.
    pub fn initial_chat_prompt(&self, user: &str, system: Option<&str>) -> Result<String> {
        self.initial_chat_prompt_with_thinking(user, system, true)
    }

    pub fn initial_chat_prompt_with_thinking(
        &self,
        user: &str,
        system: Option<&str>,
        enable_thinking: bool,
    ) -> Result<String> {
        self.ensure_qwen_chat_template()?;
        Ok(qwen_initial_chat_prompt(
            user,
            system,
            self.thinking_generation_prompt,
            enable_thinking,
        ))
    }

    /// Render a whole conversation, ending with the assistant generation
    /// prompt. Unlike [`Self::chat_continuation`], nothing here depends on
    /// retained sequence state: the caller supplies every turn, which is what
    /// a stateless request needs.
    pub fn render_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: &[Value],
        enable_thinking: bool,
    ) -> Result<String> {
        self.ensure_qwen_chat_template()?;
        ensure!(!messages.is_empty(), "a conversation needs one message");
        ensure!(
            messages
                .last()
                .is_some_and(|m| m.role != ChatRole::Assistant),
            "the last message must not be an assistant message"
        );
        ensure!(
            messages
                .iter()
                .skip(1)
                .all(|message| message.role != ChatRole::System),
            "a system message must come first"
        );
        ensure!(
            messages
                .iter()
                .any(|message| message.role == ChatRole::User),
            "a conversation needs a user message"
        );
        Ok(qwen_chat_prompt(
            messages,
            tools,
            self.thinking_generation_prompt,
            enable_thinking,
        ))
    }

    /// Render everything before the final message: the part of a conversation
    /// that an agent's next request will almost always repeat verbatim.
    ///
    /// This is a string prefix of [`Self::render_chat_prompt`] for the same
    /// messages, and the caller can rely on it being a token prefix too — it
    /// ends immediately after an `<|im_end|>` marker, which no merge crosses.
    /// Returns `None` when there is nothing before the final message.
    pub fn render_chat_history_prefix(
        &self,
        messages: &[ChatMessage],
        tools: &[Value],
    ) -> Result<Option<String>> {
        self.ensure_qwen_chat_template()?;
        if messages.len() < 2 {
            return Ok(None);
        }
        let (prompt, offset) = qwen_chat_messages_with_last_offset(messages, tools);
        Ok(offset.filter(|offset| *offset > 0).map(|offset| {
            let mut prefix = prompt;
            prefix.truncate(offset);
            prefix
        }))
    }

    /// Append a user turn to an existing assistant generation. If the model
    /// already emitted `<|im_end|>`, only its required trailing newline is
    /// inserted; otherwise the assistant message is closed first.
    pub fn chat_continuation(&self, user: &str, assistant_closed: bool) -> Result<String> {
        self.chat_continuation_with_thinking(user, assistant_closed, true)
    }

    pub fn chat_continuation_with_thinking(
        &self,
        user: &str,
        assistant_closed: bool,
        enable_thinking: bool,
    ) -> Result<String> {
        self.ensure_qwen_chat_template()?;
        Ok(qwen_chat_continuation(
            user,
            assistant_closed,
            self.thinking_generation_prompt,
            enable_thinking,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    /// A tool result. The template folds these into a user turn wrapped in
    /// `<tool_response>` markers rather than giving them a role of their own.
    Tool,
}

impl ChatRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// One call an assistant turn made, as it will be replayed into the prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatToolCall {
    pub name: String,
    /// The call's arguments as a JSON object.
    pub arguments: Value,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// Set on assistant turns that called tools.
    pub tool_calls: Vec<ChatToolCall>,
    /// An explicit reasoning section, when the client kept it separate from
    /// `content` instead of leaving the `<think>` block inline.
    pub reasoning: Option<String>,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            reasoning: None,
        }
    }
}

/// The instruction block the template emits with a tool list, verbatim.
const TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the \
following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n\
<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n\
This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n\
</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the \
specified format: an inner <function=...></function> block must be nested within \
<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide \
optional reasoning for your function call in natural language BEFORE the function call, but NOT \
after\n- If there is no function call available, answer the question like normal with your \
current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Index of the last real user turn: the point after which an assistant turn
/// keeps its thinking section. A user message that is only a tool response
/// does not count, which is what keeps a multi-step tool exchange inside one
/// query.
fn last_query_index(messages: &[ChatMessage]) -> Option<usize> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| {
            let content = message.content.trim();
            let is_tool_response =
                content.starts_with("<tool_response>") && content.ends_with("</tool_response>");
            (message.role == ChatRole::User && !is_tool_response).then_some(index)
        })
}

/// Render the leading system block, which is also where a tool list lives.
fn qwen_system_block(messages: &[ChatMessage], tools: &[Value]) -> String {
    let system = messages
        .first()
        .filter(|message| message.role == ChatRole::System)
        .map(|message| message.content.trim())
        .filter(|content| !content.is_empty());
    let mut prompt = String::new();
    if tools.is_empty() {
        if let Some(system) = system {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(system);
            prompt.push_str("<|im_end|>\n");
        }
        return prompt;
    }
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for tool in tools {
        prompt.push('\n');
        prompt.push_str(&python_json(tool));
    }
    prompt.push_str("\n</tools>");
    prompt.push_str(TOOL_INSTRUCTIONS);
    if let Some(system) = system {
        prompt.push_str("\n\n");
        prompt.push_str(system);
    }
    prompt.push_str("<|im_end|>\n");
    prompt
}

/// JSON with Python's default separators and the key order the request used,
/// which is what the reference template produces through Jinja's `tojson`
/// (transformers overrides it with `sort_keys=False, ensure_ascii=False`).
///
/// `serde_json` is built with `preserve_order` so a tool definition renders in
/// the order the client declared it, byte for byte like the reference.
fn python_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let fields: Vec<String> = map
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}: {}",
                        python_json(&Value::String(key.clone())),
                        python_json(value)
                    )
                })
                .collect();
            format!("{{{}}}", fields.join(", "))
        }
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(python_json).collect();
            format!("[{}]", items.join(", "))
        }
        other => other.to_string(),
    }
}

fn qwen_chat_messages(messages: &[ChatMessage], tools: &[Value]) -> String {
    qwen_chat_messages_with_last_offset(messages, tools).0
}

/// Render the conversation, and report where the final message's own text
/// begins.
///
/// The offset is how a caller gets the conversation without its last message
/// without re-rendering it: re-rendering a shorter list moves the last real
/// user turn, which changes whether earlier assistant turns keep their
/// thinking section, so the shorter rendering is not always a prefix of the
/// longer one.
fn qwen_chat_messages_with_last_offset(
    messages: &[ChatMessage],
    tools: &[Value],
) -> (String, Option<usize>) {
    let mut prompt = qwen_system_block(messages, tools);
    let last_query = last_query_index(messages);
    let mut last_offset = None;
    for (index, message) in messages.iter().enumerate() {
        if index + 1 == messages.len() {
            last_offset = Some(prompt.len());
        }
        let content = message.content.trim();
        match message.role {
            // Handled by the system block, and only valid in first position.
            ChatRole::System => {}
            ChatRole::User => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(content);
                prompt.push_str("<|im_end|>\n");
            }
            ChatRole::Assistant => {
                prompt.push_str("<|im_start|>assistant\n");
                // The template keeps a turn's reasoning only while it is still
                // inside the current query — a multi-step tool exchange — and
                // drops it from older turns.
                let (reasoning, visible) = split_reasoning(message);
                if last_query.is_some_and(|last| index > last) {
                    prompt.push_str("<think>\n");
                    prompt.push_str(reasoning.trim());
                    prompt.push_str("\n</think>\n\n");
                }
                prompt.push_str(visible);
                for (position, call) in message.tool_calls.iter().enumerate() {
                    if position == 0 {
                        if !visible.is_empty() {
                            prompt.push_str("\n\n");
                        }
                    } else {
                        prompt.push('\n');
                    }
                    crate::tool_calls::render(&call.name, &call.arguments, &mut prompt);
                }
                prompt.push_str("<|im_end|>\n");
            }
            ChatRole::Tool => {
                let follows_tool = index
                    .checked_sub(1)
                    .and_then(|previous| messages.get(previous))
                    .is_some_and(|previous| previous.role == ChatRole::Tool);
                if !follows_tool {
                    prompt.push_str("<|im_start|>user");
                }
                prompt.push_str("\n<tool_response>\n");
                prompt.push_str(content);
                prompt.push_str("\n</tool_response>");
                let last_of_run = messages
                    .get(index + 1)
                    .is_none_or(|next| next.role != ChatRole::Tool);
                if last_of_run {
                    prompt.push_str("<|im_end|>\n");
                }
            }
        }
    }
    (prompt, last_offset)
}

/// An assistant turn's reasoning and its visible answer.
///
/// The reference template takes the reasoning from before the *first*
/// `</think>` and the answer from after the *last* one, so text between two
/// closing markers belongs to neither.
fn split_reasoning(message: &ChatMessage) -> (&str, &str) {
    let content = message.content.trim();
    if let Some(reasoning) = &message.reasoning {
        return (reasoning.as_str(), content);
    }
    let Some(first) = content.find("</think>") else {
        return ("", content);
    };
    let head = content[..first].trim_end_matches('\n');
    let reasoning = head
        .rsplit_once("<think>")
        .map_or(head, |(_, after)| after)
        .trim_start_matches('\n');
    let last = content
        .rfind("</think>")
        .map_or(first, |last| last + "</think>".len());
    (reasoning, content[last..].trim_start_matches('\n'))
}

fn qwen_chat_prompt(
    messages: &[ChatMessage],
    tools: &[Value],
    thinking_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let mut prompt = qwen_chat_messages(messages, tools);
    prompt.push_str("<|im_start|>assistant\n");
    if thinking_generation_prompt {
        prompt.push_str(if enable_thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        });
    }
    prompt
}

fn qwen_initial_chat_prompt(
    user: &str,
    system: Option<&str>,
    thinking_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let mut prompt = String::new();
    if let Some(system) = system {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(user);
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
    if thinking_generation_prompt {
        prompt.push_str(if enable_thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        });
    }
    prompt
}

fn qwen_chat_continuation(
    user: &str,
    assistant_closed: bool,
    thinking_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let mut prompt = if assistant_closed {
        "\n".to_owned()
    } else {
        "<|im_end|>\n".to_owned()
    };
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(user);
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
    if thinking_generation_prompt {
        prompt.push_str(if enable_thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        });
    }
    prompt
}

#[cfg(test)]
mod tests {
    use super::{
        ChatMessage, ChatRole, ChatToolCall, Value, qwen_chat_continuation,
        qwen_chat_messages_with_last_offset, qwen_chat_prompt, qwen_initial_chat_prompt,
        split_reasoning,
    };

    fn message(role: ChatRole, content: &str) -> ChatMessage {
        ChatMessage::new(role, content)
    }

    /// A tool list as a client would send it.
    fn tools() -> Vec<Value> {
        vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "limit": {"type": "integer"}
                    },
                    "required": ["path"]
                }
            }
        })]
    }

    #[test]
    fn renders_a_multi_turn_conversation() {
        let messages = [
            message(ChatRole::System, "be concise"),
            message(ChatRole::User, "hello"),
            message(ChatRole::Assistant, "hi"),
            message(ChatRole::User, "again"),
        ];
        assert_eq!(
            qwen_chat_prompt(&messages, &[], false, true),
            "<|im_start|>system\nbe concise<|im_end|>\n\
             <|im_start|>user\nhello<|im_end|>\n\
             <|im_start|>assistant\nhi<|im_end|>\n\
             <|im_start|>user\nagain<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn conversation_rendering_matches_the_first_turn_helper() {
        let messages = [
            message(ChatRole::System, "be concise"),
            message(ChatRole::User, "hello"),
        ];
        for thinking_prompt in [false, true] {
            for enable_thinking in [false, true] {
                assert_eq!(
                    qwen_chat_prompt(&messages, &[], thinking_prompt, enable_thinking),
                    qwen_initial_chat_prompt(
                        "hello",
                        Some("be concise"),
                        thinking_prompt,
                        enable_thinking
                    ),
                );
            }
        }
    }

    #[test]
    fn the_history_prefix_is_a_prefix_of_the_whole_prompt() {
        let messages = [
            message(ChatRole::System, "be concise"),
            message(ChatRole::User, "hello"),
            message(ChatRole::Assistant, "<think>\nhm\n</think>\n\nhi"),
            message(ChatRole::User, "again"),
        ];
        let (whole, offset) = qwen_chat_messages_with_last_offset(&messages, &[]);
        let history = &whole[..offset.expect("a last message")];
        assert!(whole.starts_with(history));
        assert!(history.ends_with("<|im_end|>\n"));
        // Only the final turn is excluded, and the assistant turn is rendered
        // the same way in both — which re-rendering the shorter list would not
        // guarantee, since dropping the final user turn moves the last query.
        assert!(!history.contains("again"));
        assert!(history.contains("hi"));
        assert!(!history.contains("<think>"), "{history:?}");
        for thinking_prompt in [false, true] {
            let prompt = qwen_chat_prompt(&messages, &[], thinking_prompt, true);
            assert!(prompt.starts_with(history), "{history:?} vs {prompt:?}");
        }
    }

    /// These expectations were produced by rendering the same fixtures through
    /// Qwen3.6-35B-A3B's own `chat_template` with Jinja2, using the filter
    /// overrides transformers applies (`tojson` with `sort_keys=False,
    /// ensure_ascii=False`). They are the contract this renderer implements.
    #[test]
    fn matches_the_reference_template_with_tools() {
        let tools = tools();
        let messages = [
            message(ChatRole::System, "Be terse."),
            message(ChatRole::User, "Read src/lib.rs"),
        ];
        assert_eq!(
            qwen_chat_prompt(&messages, &tools, true, false),
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"read_file\", \"description\": \"Read a file\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}, \"limit\": {\"type\": \"integer\"}}, \"required\": [\"path\"]}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\nBe terse.<|im_end|>\n<|im_start|>user\nRead src/lib.rs<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn matches_the_reference_template_for_a_tool_round_trip() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
            }
        })];
        let mut called = ChatMessage::new(
            ChatRole::Assistant,
            "<think>\nneed the file\n</think>\n\nReading it.",
        );
        called.tool_calls = vec![ChatToolCall {
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "src/lib.rs"}),
        }];
        let messages = [
            message(ChatRole::User, "Read src/lib.rs"),
            called,
            message(ChatRole::Tool, "pub mod config;"),
            message(ChatRole::User, "Summarise it"),
        ];
        let rendered = qwen_chat_prompt(&messages, &tools, true, false);
        let (_, body) = rendered
            .split_once("</IMPORTANT>")
            .expect("the tool block is rendered");
        assert_eq!(
            body,
            "<|im_end|>\n<|im_start|>user\nRead src/lib.rs<|im_end|>\n<|im_start|>assistant\nReading it.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/lib.rs\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\npub mod config;\n</tool_response><|im_end|>\n<|im_start|>user\nSummarise it<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn matches_the_reference_template_without_tools() {
        let messages = [
            message(ChatRole::System, "Be concise."),
            message(ChatRole::User, "hello"),
            message(ChatRole::Assistant, "<think>\nhm\n</think>\n\nhi"),
            message(ChatRole::User, "again"),
        ];
        assert_eq!(
            qwen_chat_prompt(&messages, &[], true, true),
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nhi<|im_end|>\n<|im_start|>user\nagain<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn a_turn_inside_the_current_query_keeps_its_thinking() {
        // Nothing after the last user turn: the assistant's own reasoning is
        // still part of the query being answered, so the template keeps it.
        let mut called = ChatMessage::new(ChatRole::Assistant, "<think>\nlook\n</think>\n\nok");
        called.tool_calls = vec![ChatToolCall {
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "a"}),
        }];
        let messages = [
            message(ChatRole::User, "read a"),
            called,
            message(ChatRole::Tool, "contents"),
        ];
        let rendered = qwen_chat_prompt(&messages, &tools(), true, false);
        assert!(
            rendered
                .contains("<|im_start|>assistant\n<think>\nlook\n</think>\n\nok\n\n<tool_call>"),
            "{rendered}"
        );
    }

    #[test]
    fn consecutive_tool_results_share_one_user_turn() {
        let messages = [
            message(ChatRole::User, "go"),
            message(ChatRole::Assistant, "ok"),
            message(ChatRole::Tool, "first"),
            message(ChatRole::Tool, "second"),
            message(ChatRole::User, "thanks"),
        ];
        let rendered = qwen_chat_prompt(&messages, &[], false, true);
        assert!(
            rendered.contains(
                "<|im_start|>user\n<tool_response>\nfirst\n</tool_response>\n<tool_response>\nsecond\n</tool_response><|im_end|>\n"
            ),
            "{rendered}"
        );
    }

    #[test]
    fn drops_reasoning_from_assistant_history() {
        let split = |content: &str| {
            let message = ChatMessage::new(ChatRole::Assistant, content);
            let (reasoning, visible) = split_reasoning(&message);
            (reasoning.to_owned(), visible.to_owned())
        };
        assert_eq!(
            split("<think>\nwhy\n</think>\n\nanswer"),
            ("why".to_owned(), "answer".to_owned())
        );
        assert_eq!(split("plain"), (String::new(), "plain".to_owned()));
        assert_eq!(split("a</think>b</think>c").1, "c");
        // A client that keeps reasoning in its own field is taken at its word.
        let mut separated = ChatMessage::new(ChatRole::Assistant, "answer");
        separated.reasoning = Some("why".to_owned());
        assert_eq!(split_reasoning(&separated), ("why", "answer"));
        let messages = [
            message(ChatRole::User, "hello"),
            message(ChatRole::Assistant, "<think>\nwhy\n</think>\n\nhi"),
            message(ChatRole::User, "again"),
        ];
        assert_eq!(
            qwen_chat_prompt(&messages, &[], true, true),
            "<|im_start|>user\nhello<|im_end|>\n\
             <|im_start|>assistant\nhi<|im_end|>\n\
             <|im_start|>user\nagain<|im_end|>\n\
             <|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn renders_plain_qwen_chat_turns() {
        assert_eq!(
            qwen_initial_chat_prompt("hello", Some("be concise"), false, true),
            "<|im_start|>system\nbe concise<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            qwen_chat_continuation("next", false, false, true),
            "<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            qwen_chat_continuation("next", true, false, true),
            "\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn renders_thinking_generation_prefix() {
        assert_eq!(
            qwen_initial_chat_prompt("hello", None, true, true),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            qwen_chat_continuation("next", false, true, true),
            "<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn renders_non_thinking_generation_prefix_with_closed_block() {
        assert_eq!(
            qwen_initial_chat_prompt("hello", None, true, false),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            qwen_chat_continuation("next", true, true, false),
            "\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
