//! Qwen3.6 tool-call syntax: rendering it into a prompt and reading it back.
//!
//! This checkpoint's chat template does not use the JSON-in-`<tool_call>` form
//! some Qwen releases do. It asks for nested XML-ish tags:
//!
//! ```text
//! <tool_call>
//! <function=read_file>
//! <parameter=path>
//! src/lib.rs
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Parameter values are raw text, so the declared JSON type is what decides
//! whether `42` means the number or the string. The tool schema the request
//! supplied is therefore part of parsing, not just of prompting.

use serde_json::{Map, Value};

const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

/// One call as the model wrote it: a name and its parameters as raw text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    pub parameters: Vec<(String, String)>,
}

impl ParsedToolCall {
    /// Convert the raw parameter text into a JSON object, using the tool's
    /// declared parameter types where they exist.
    ///
    /// A value that should be JSON but is not parseable is kept as a string
    /// rather than dropped: handing the caller what the model actually wrote
    /// beats inventing a value or failing the whole call.
    pub fn arguments(&self, schema: Option<&Value>) -> Value {
        let properties = schema
            .and_then(|schema| schema.get("parameters"))
            .and_then(|parameters| parameters.get("properties"));
        let mut arguments = Map::new();
        for (name, raw) in &self.parameters {
            let declared = properties
                .and_then(|properties| properties.get(name))
                .and_then(|property| property.get("type"))
                .and_then(Value::as_str);
            arguments.insert(name.clone(), coerce(raw, declared));
        }
        Value::Object(arguments)
    }
}

fn coerce(raw: &str, declared: Option<&str>) -> Value {
    match declared {
        Some("string") => Value::String(raw.to_owned()),
        // An undeclared parameter is treated like a typed one: the model wrote
        // JSON for anything structured, and a bare word simply fails to parse.
        _ => serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_owned())),
    }
}

/// Split generated text into the part meant for the user and the calls it
/// requested.
///
/// The template tells the model that reasoning may precede a call but nothing
/// may follow one, so text after the last call is dropped rather than shown.
pub fn parse(text: &str) -> (String, Vec<ParsedToolCall>) {
    let Some(first) = text.find(CALL_OPEN) else {
        return (text.to_owned(), Vec::new());
    };
    let mut calls = Vec::new();
    let mut rest = &text[first..];
    while let Some(start) = rest.find(CALL_OPEN) {
        let body = &rest[start + CALL_OPEN.len()..];
        // A block ends at its own close tag. Parsing past it would let a block
        // that carries no function of its own adopt the next one's, and then
        // the scan would read that same function again from its real block.
        // A truncated final block has no close tag and simply runs to the end.
        let (block, after) = match body.find(CALL_CLOSE) {
            Some(end) => (&body[..end], Some(&body[end + CALL_CLOSE.len()..])),
            None => (body, None),
        };
        // A block that parses to nothing is skipped rather than ending the
        // scan: the blocks after it are still the model's own calls.
        if let Some(call) = parse_call(block) {
            calls.push(call);
        }
        let Some(remainder) = after else {
            break;
        };
        rest = remainder;
    }
    if calls.is_empty() {
        return (text.to_owned(), calls);
    }
    (text[..first].trim_end().to_owned(), calls)
}

/// Parse one `<function=…>…</function>` block from the text after a
/// `<tool_call>` marker.
fn parse_call(body: &str) -> Option<ParsedToolCall> {
    let open = body.find(FUNCTION_OPEN)?;
    let after_open = &body[open + FUNCTION_OPEN.len()..];
    let name_end = after_open.find('>')?;
    let name = after_open[..name_end].trim().to_owned();
    if name.is_empty() {
        return None;
    }
    let mut inner = &after_open[name_end + 1..];
    let close = inner.find(FUNCTION_CLOSE)?;
    inner = &inner[..close];
    let mut parameters = Vec::new();
    let mut cursor = inner;
    while let Some(start) = cursor.find(PARAMETER_OPEN) {
        let after = &cursor[start + PARAMETER_OPEN.len()..];
        let Some(key_end) = after.find('>') else {
            break;
        };
        let key = after[..key_end].trim().to_owned();
        let value_text = &after[key_end + 1..];
        let Some(value_end) = value_text.find(PARAMETER_CLOSE) else {
            break;
        };
        // The template frames a value with a newline on each side; anything
        // else inside the tags is the value itself.
        let value = value_text[..value_end]
            .strip_prefix('\n')
            .unwrap_or(&value_text[..value_end]);
        let value = value.strip_suffix('\n').unwrap_or(value);
        parameters.push((key, value.to_owned()));
        cursor = &value_text[value_end + PARAMETER_CLOSE.len()..];
    }
    Some(ParsedToolCall { name, parameters })
}

/// Render an assistant turn's calls back into the prompt, as the template does
/// when a tool-calling conversation is replayed.
pub fn render(name: &str, arguments: &Value, out: &mut String) {
    out.push_str(CALL_OPEN);
    out.push('\n');
    out.push_str(FUNCTION_OPEN);
    out.push_str(name);
    out.push_str(">\n");
    if let Some(arguments) = arguments.as_object() {
        for (key, value) in arguments {
            out.push_str(PARAMETER_OPEN);
            out.push_str(key);
            out.push_str(">\n");
            match value {
                // A string parameter is written raw; everything else is JSON.
                Value::String(text) => out.push_str(text),
                other => out.push_str(&other.to_string()),
            }
            out.push('\n');
            out.push_str(PARAMETER_CLOSE);
            out.push('\n');
        }
    }
    out.push_str(FUNCTION_CLOSE);
    out.push('\n');
    out.push_str(CALL_CLOSE);
}

/// The marker that tells a streaming caller to stop showing text to the user.
pub const CALL_MARKER: &str = CALL_OPEN;

/// The marker that ends a turn.
///
/// The template instructs the model to reply with a call and *no suffix*, and
/// a turn that keeps going past one is not following it: left unbounded, this
/// checkpoint will emit call after call until it runs out of budget. Treating
/// the closing tag as the end of the turn is what makes a tool-calling turn
/// terminate, at the cost of one call per turn.
pub const CALL_END_MARKER: &str = CALL_CLOSE;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn plain_text_has_no_calls() {
        let (content, calls) = parse("just an answer");
        assert_eq!(content, "just an answer");
        assert!(calls.is_empty());
    }

    #[test]
    fn parses_one_call_with_its_parameters() {
        let text = "I will look.\n\n<tool_call>\n<function=read_file>\n\
                    <parameter=path>\nsrc/lib.rs\n</parameter>\n\
                    <parameter=limit>\n40\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = parse(text);
        assert_eq!(content, "I will look.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(
            calls[0].parameters,
            vec![
                ("path".to_owned(), "src/lib.rs".to_owned()),
                ("limit".to_owned(), "40".to_owned()),
            ]
        );
    }

    #[test]
    fn parses_several_calls_in_one_turn() {
        let text = "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n\
                    </function>\n</tool_call>\n\
                    <tool_call>\n<function=b>\n</function>\n</tool_call>";
        let (content, calls) = parse(text);
        assert!(content.is_empty());
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert!(calls[1].parameters.is_empty());
    }

    #[test]
    fn a_truncated_call_still_yields_what_completed() {
        // The turn hit its token budget after the function block closed.
        let text = "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>";
        let (_, calls) = parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].parameters, vec![("x".to_owned(), "1".to_owned())]);

        // Nothing usable yet: no closing function tag.
        let (content, calls) = parse("<tool_call>\n<function=a>\n<parameter=x>\n1");
        assert!(calls.is_empty());
        assert!(
            content.contains("<tool_call>"),
            "the text is returned as-is"
        );
    }

    #[test]
    fn a_block_never_adopts_the_next_blocks_function() {
        // Regression: the scan used to look for `<function=` past the first
        // block's own `</tool_call>`, so the empty block took the second
        // block's call and the second block then reported it again.
        let text = "<tool_call>\n</tool_call>\n\
                    <tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n\
                    </function>\n</tool_call>";
        let (_, calls) = parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[0].parameters, vec![("x".to_owned(), "1".to_owned())]);

        // The same block with text in it, and with a well-formed call ahead of
        // the malformed one rather than behind it.
        let (_, calls) = parse(
            "<tool_call>\nnot a call\n</tool_call>\n\
             <tool_call>\n<function=b>\n</function>\n</tool_call>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "b");
        let (_, calls) = parse(
            "<tool_call>\n<function=a>\n</function>\n</tool_call>\n\
             <tool_call>\nnothing here\n</tool_call>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "a");
    }

    #[test]
    fn keeps_multi_line_values_intact() {
        let text = "<tool_call>\n<function=write>\n<parameter=body>\nline one\nline two\n\
                    </parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse(text);
        assert_eq!(calls[0].parameters[0].1, "line one\nline two");
    }

    #[test]
    fn typing_follows_the_declared_schema() {
        let schema = json!({
            "parameters": {"properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
                "deep": {"type": "boolean"},
                "opts": {"type": "object"}
            }}
        });
        let call = ParsedToolCall {
            name: "read".into(),
            parameters: vec![
                ("path".into(), "42".into()),
                ("limit".into(), "40".into()),
                ("deep".into(), "true".into()),
                ("opts".into(), "{\"a\": 1}".into()),
                ("extra".into(), "hello".into()),
            ],
        };
        let arguments = call.arguments(Some(&schema));
        // A declared string stays a string even when it looks like a number.
        assert_eq!(arguments["path"], json!("42"));
        assert_eq!(arguments["limit"], json!(40));
        assert_eq!(arguments["deep"], json!(true));
        assert_eq!(arguments["opts"], json!({"a": 1}));
        // An undeclared parameter that is not JSON stays text.
        assert_eq!(arguments["extra"], json!("hello"));
    }

    #[test]
    fn unparseable_json_is_kept_as_written() {
        let schema = json!({"parameters": {"properties": {"n": {"type": "integer"}}}});
        let call = ParsedToolCall {
            name: "f".into(),
            parameters: vec![("n".into(), "about forty".into())],
        };
        assert_eq!(call.arguments(Some(&schema))["n"], json!("about forty"));
    }

    #[test]
    fn rendering_round_trips_through_the_parser() {
        let mut rendered = String::new();
        render(
            "read_file",
            &json!({"path": "src/lib.rs", "limit": 40, "deep": true}),
            &mut rendered,
        );
        let (content, calls) = parse(&rendered);
        assert!(content.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        let schema = serde_json::json!({
            "parameters": {"properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
                "deep": {"type": "boolean"}
            }}
        });
        assert_eq!(
            calls[0].arguments(Some(&schema)),
            json!({"path": "src/lib.rs", "limit": 40, "deep": true})
        );
    }
}
