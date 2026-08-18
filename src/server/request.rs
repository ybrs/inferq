//! Turning a validated API request into what the runtime needs.

use anyhow::{Result, ensure};

use crate::{GenerationOptions, tokenizer::ModelTokenizer};

use super::thinking::ThinkingPlan;

use super::{api::ChatCompletionRequest, engine::disable_speculation_for_sampling};

/// Upper bound on a single request's output, whatever it asks for. Without one
/// a client can pin the single inference slot for as long as it likes.
pub const MAX_OUTPUT_TOKENS: usize = 32_768;

/// Merge a request over the server's defaults.
///
/// Anything the request does not set keeps the value the server was started
/// with, so `--temperature`, the speculative flags and `--max-new-tokens`
/// remain the operator's policy for clients that do not care.
pub fn resolve_options(
    defaults: &GenerationOptions,
    request: &ChatCompletionRequest,
    thinking: ThinkingPlan,
) -> Result<GenerationOptions> {
    let mut options = defaults.clone();
    if let Some(max_new_tokens) = request.max_new_tokens() {
        ensure!(max_new_tokens > 0, "`max_tokens` must be at least 1");
        options.max_new_tokens = max_new_tokens;
    }
    options.max_new_tokens = options.max_new_tokens.min(MAX_OUTPUT_TOKENS);
    if let Some(temperature) = request.temperature {
        ensure!(
            temperature.is_finite() && temperature >= 0.,
            "`temperature` must be finite and non-negative"
        );
        options.sampling.temperature = temperature;
    }
    if let Some(top_p) = request.top_p {
        ensure!(
            top_p > 0. && top_p <= 1.,
            "`top_p` must be in the range (0, 1]"
        );
        // OpenAI's clients send top_p=1 to mean "unrestricted"; keeping it as a
        // filter would only cost a sort.
        options.sampling.top_p = (top_p < 1.).then_some(top_p);
    }
    if let Some(top_k) = request.top_k {
        ensure!(top_k > 0, "`top_k` must be at least 1");
        options.sampling.top_k = Some(top_k);
    }
    if let Some(min_p) = request.min_p {
        ensure!(
            (0. ..=1.).contains(&min_p),
            "`min_p` must be in the range [0, 1]"
        );
        options.sampling.min_p = (min_p > 0.).then_some(min_p);
    }
    if let Some(seed) = request.seed {
        options.sampling.seed = seed;
    }
    // The runtime's budget assumes the turn begins inside an open block, so a
    // closed one must carry no budget at all: it would force a second closure.
    options.thinking_budget = thinking.budget.filter(|_| thinking.open);
    disable_speculation_for_sampling(&mut options);
    Ok(options)
}

/// Render the conversation into the model's chat format.
pub fn render_prompt(
    tokenizer: &ModelTokenizer,
    request: &ChatCompletionRequest,
    thinking: ThinkingPlan,
) -> Result<String> {
    let messages = request.chat_messages()?;
    tokenizer.render_chat_prompt(&messages, request.tool_definitions(), thinking.open)
}

/// Render the conversation minus its final message, which is the part a later
/// request in the same session is expected to send again unchanged.
pub fn render_history_prefix(
    tokenizer: &ModelTokenizer,
    request: &ChatCompletionRequest,
) -> Result<Option<String>> {
    let messages = request.chat_messages()?;
    tokenizer.render_chat_history_prefix(&messages, request.tool_definitions())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SpeculativeMode;

    fn request(body: &str) -> ChatCompletionRequest {
        serde_json::from_str(body).expect("request parses")
    }

    fn thinking() -> ThinkingPlan {
        ThinkingPlan {
            open: true,
            budget: None,
        }
    }

    fn defaults() -> GenerationOptions {
        GenerationOptions {
            max_new_tokens: 256,
            speculative_mode: SpeculativeMode::Auto,
            speculative_mtp_draft_tokens: 4,
            speculative_ngram_draft_tokens: 8,
            ..GenerationOptions::default()
        }
    }

    #[test]
    fn keeps_server_defaults_when_the_request_is_silent() {
        let options = resolve_options(&defaults(), &request(r#"{"messages":[]}"#), thinking())
            .expect("valid");
        assert_eq!(options.max_new_tokens, 256);
        assert_eq!(options.sampling.temperature, 0.);
        assert_eq!(options.speculative_mode, SpeculativeMode::Auto);
    }

    #[test]
    fn applies_request_overrides() {
        let options = resolve_options(
            &defaults(),
            &request(r#"{"messages":[],"max_tokens":16,"top_p":0.9,"top_k":40,"seed":7}"#),
            thinking(),
        )
        .expect("valid");
        assert_eq!(options.max_new_tokens, 16);
        assert_eq!(options.sampling.top_p, Some(0.9));
        assert_eq!(options.sampling.top_k, Some(40));
        assert_eq!(options.sampling.seed, 7);
    }

    #[test]
    fn caps_the_output_length() {
        let options = resolve_options(
            &defaults(),
            &request(r#"{"messages":[],"max_tokens":1000000}"#),
            thinking(),
        )
        .expect("valid");
        assert_eq!(options.max_new_tokens, MAX_OUTPUT_TOKENS);
    }

    #[test]
    fn treats_unrestricted_filters_as_absent() {
        let options = resolve_options(
            &defaults(),
            &request(r#"{"messages":[],"top_p":1.0,"min_p":0.0}"#),
            thinking(),
        )
        .expect("valid");
        assert_eq!(options.sampling.top_p, None);
        assert_eq!(options.sampling.min_p, None);
    }

    #[test]
    fn sampling_turns_speculation_off() {
        let options = resolve_options(
            &defaults(),
            &request(r#"{"messages":[],"temperature":0.7}"#),
            thinking(),
        )
        .expect("valid");
        assert_eq!(options.sampling.temperature, 0.7);
        assert_eq!(options.speculative_mode, SpeculativeMode::Off);
        assert_eq!(options.speculative_mtp_draft_tokens, 0);
        assert_eq!(options.speculative_ngram_draft_tokens, 0);

        // Greedy requests keep it.
        let options = resolve_options(
            &defaults(),
            &request(r#"{"messages":[],"temperature":0}"#),
            thinking(),
        )
        .expect("valid");
        assert_eq!(options.speculative_mode, SpeculativeMode::Auto);
    }

    #[test]
    fn rejects_out_of_range_sampling_parameters() {
        for body in [
            r#"{"messages":[],"max_tokens":0}"#,
            r#"{"messages":[],"temperature":-1}"#,
            r#"{"messages":[],"top_p":0}"#,
            r#"{"messages":[],"top_p":1.5}"#,
            r#"{"messages":[],"top_k":0}"#,
            r#"{"messages":[],"min_p":2}"#,
        ] {
            assert!(
                resolve_options(&defaults(), &request(body), thinking()).is_err(),
                "{body} should be rejected"
            );
        }
    }
}
