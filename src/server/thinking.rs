//! How much a turn may think, and who decides.
//!
//! OpenAI's API has no thinking budget. It has `reasoning_effort` — an
//! categorical knob (`none`, `minimal`, `low`, `medium`, `high`, and `xhigh`
//! on newer models) — with `max_completion_tokens` bounding reasoning and
//! answer together, and `usage.completion_tokens_details.reasoning_tokens`
//! reporting what the reasoning cost. Anthropic and Google are the ones that
//! take an explicit token count (`thinking.budget_tokens`,
//! `thinkingConfig.thinkingBudget`).
//!
//! This model's runtime takes a token count, so the two are bridged here: the
//! operator says what each effort level can afford on their host, because a
//! level's cost is a property of the machine rather than of the request.

use std::collections::BTreeMap;

use anyhow::{Result, bail, ensure};

use super::api::{ChatCompletionRequest, ReasoningEffort};

/// Default tokens per effort level. Chosen for a CPU host where thinking is
/// paid for a token at a time rather than in a burst.
pub const DEFAULT_EFFORT_BUDGETS: [(ReasoningEffort, usize); 5] = [
    (ReasoningEffort::Minimal, 64),
    (ReasoningEffort::Low, 256),
    (ReasoningEffort::Medium, 1024),
    (ReasoningEffort::High, 4096),
    (ReasoningEffort::XHigh, 16384),
];

/// The operator's side of the bargain: a default, a ceiling, and what each
/// effort level is worth.
#[derive(Debug, Clone)]
pub struct ThinkingPolicy {
    /// Budget for a request that asks for neither a level nor a count.
    /// `None` leaves thinking unbounded, which is the model's own behaviour.
    pub default_budget: Option<usize>,
    /// Ceiling on anything a request asks for. `None` means the client may
    /// ask for as much as it likes.
    pub max_budget: Option<usize>,
    pub effort_budgets: BTreeMap<ReasoningEffort, usize>,
    /// Whether an assistant turn opens a thinking block when the request says
    /// nothing either way.
    pub enabled_by_default: bool,
}

impl Default for ThinkingPolicy {
    fn default() -> Self {
        Self {
            default_budget: None,
            max_budget: None,
            effort_budgets: DEFAULT_EFFORT_BUDGETS.into_iter().collect(),
            enabled_by_default: true,
        }
    }
}

impl ThinkingPolicy {
    pub fn validate(&self) -> Result<()> {
        for (effort, budget) in &self.effort_budgets {
            ensure!(
                *budget > 0,
                "the `{}` reasoning budget must be at least one token; \
                 use `reasoning_effort: none` to turn thinking off",
                effort.as_str()
            );
        }
        ensure!(
            self.default_budget.is_none_or(|budget| budget > 0),
            "the default thinking budget must be at least one token"
        );
        ensure!(
            self.max_budget.is_none_or(|budget| budget > 0),
            "the maximum thinking budget must be at least one token"
        );
        Ok(())
    }

    /// Parse a `level=tokens` pair, as the command line takes them.
    pub fn parse_effort_budget(value: &str) -> Result<(ReasoningEffort, usize)> {
        let (level, tokens) = value
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("expected `level=tokens`, got `{value}`"))?;
        let Some(effort) = ReasoningEffort::parse(level) else {
            bail!(
                "unknown reasoning effort `{level}`; expected one of {}",
                ReasoningEffort::BUDGETED
                    .iter()
                    .map(|effort| effort.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        };
        ensure!(
            effort != ReasoningEffort::None,
            "`none` turns thinking off and takes no budget"
        );
        let tokens = tokens
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("`{tokens}` is not a token count"))?;
        Ok((effort, tokens))
    }
}

/// What one request's thinking section will be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingPlan {
    /// Whether the prompt leaves a `<think>` block open.
    pub open: bool,
    /// Tokens the block may spend before it is closed for the model.
    pub budget: Option<usize>,
}

/// Decide a request's thinking section.
///
/// Precedence, most specific first: an explicit `thinking_budget`, then
/// `reasoning_effort`, then the server's default. A request that turns
/// thinking off gets no budget at all — the block is already closed in the
/// prompt, and a budget would only force a second closure.
/// `supports_thinking` is whether the model's template can open a block at
/// all; a model without one never thinks, whatever the request asks for.
pub fn plan(
    request: &ChatCompletionRequest,
    policy: &ThinkingPolicy,
    supports_thinking: bool,
) -> ThinkingPlan {
    let open = supports_thinking
        && request
            .enable_thinking()
            .unwrap_or(policy.enabled_by_default);
    if !open {
        return ThinkingPlan {
            open: false,
            budget: None,
        };
    }
    let budget = request
        .thinking_budget
        .or_else(|| {
            request
                .reasoning_effort()
                .and_then(|effort| policy.effort_budgets.get(&effort).copied())
        })
        .or(policy.default_budget);
    let budget = match (budget, policy.max_budget) {
        (Some(budget), Some(max)) => Some(budget.min(max)),
        (Some(budget), None) => Some(budget),
        (None, max) => max,
    };
    ThinkingPlan {
        open: true,
        // Zero would force a closure before the first token; the request
        // asked for thinking, so give it at least one token to think with.
        budget: budget.map(|budget| budget.max(1)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: &str) -> ChatCompletionRequest {
        serde_json::from_str(body).expect("request parses")
    }

    fn policy() -> ThinkingPolicy {
        ThinkingPolicy {
            default_budget: Some(512),
            max_budget: Some(2048),
            ..ThinkingPolicy::default()
        }
    }

    fn plan_with(supports_thinking: bool, body: &str, policy: &ThinkingPolicy) -> ThinkingPlan {
        plan(&request(body), policy, supports_thinking)
    }

    #[test]
    fn effort_levels_map_to_budgets() {
        let policy = policy();
        for (level, expected) in [
            ("minimal", 64),
            ("low", 256),
            ("medium", 1024),
            ("high", 2048),  // clamped by the ceiling
            ("xhigh", 2048), // clamped by the ceiling
        ] {
            let body = format!(r#"{{"messages":[],"reasoning_effort":"{level}"}}"#);
            assert_eq!(
                plan_with(true, &body, &policy).budget,
                Some(expected),
                "{level}"
            );
        }
    }

    #[test]
    fn an_explicit_budget_beats_an_effort_level() {
        let plan = plan_with(
            true,
            r#"{"messages":[],"reasoning_effort":"high","thinking_budget":32}"#,
            &policy(),
        );
        assert_eq!(plan.budget, Some(32));
    }

    #[test]
    fn none_turns_thinking_off_and_takes_no_budget() {
        let plan = plan_with(
            true,
            r#"{"messages":[],"reasoning_effort":"none","thinking_budget":900}"#,
            &policy(),
        );
        assert_eq!(
            plan,
            ThinkingPlan {
                open: false,
                budget: None
            }
        );
        // As does the Qwen spelling.
        assert!(
            !plan_with(
                true,
                r#"{"messages":[],"chat_template_kwargs":{"enable_thinking":false}}"#,
                &policy()
            )
            .open
        );
    }

    #[test]
    fn a_silent_request_gets_the_server_default() {
        assert_eq!(
            plan_with(true, r#"{"messages":[]}"#, &policy()).budget,
            Some(512)
        );
        let unbounded = ThinkingPolicy {
            default_budget: None,
            max_budget: None,
            ..ThinkingPolicy::default()
        };
        assert_eq!(
            plan_with(true, r#"{"messages":[]}"#, &unbounded).budget,
            None
        );
    }

    #[test]
    fn the_ceiling_applies_even_without_a_default() {
        let capped = ThinkingPolicy {
            default_budget: None,
            max_budget: Some(128),
            ..ThinkingPolicy::default()
        };
        // Nothing asked for, so the ceiling is the budget.
        assert_eq!(
            plan_with(true, r#"{"messages":[]}"#, &capped).budget,
            Some(128)
        );
        assert_eq!(
            plan_with(true, r#"{"messages":[],"thinking_budget":9999}"#, &capped).budget,
            Some(128)
        );
    }

    #[test]
    fn an_unknown_level_falls_back_rather_than_failing() {
        assert_eq!(
            plan_with(
                true,
                r#"{"messages":[],"reasoning_effort":"turbo"}"#,
                &policy()
            )
            .budget,
            Some(512)
        );
    }

    #[test]
    fn a_model_without_a_thinking_template_never_opens_one() {
        let plan = plan_with(
            false,
            r#"{"messages":[],"reasoning_effort":"high"}"#,
            &policy(),
        );
        assert!(!plan.open);
        assert_eq!(plan.budget, None);
    }

    #[test]
    fn effort_budget_pairs_parse() {
        assert_eq!(
            ThinkingPolicy::parse_effort_budget("high=8192").expect("parses"),
            (ReasoningEffort::High, 8192)
        );
        assert!(ThinkingPolicy::parse_effort_budget("high").is_err());
        assert!(ThinkingPolicy::parse_effort_budget("turbo=10").is_err());
        assert!(ThinkingPolicy::parse_effort_budget("none=10").is_err());
        assert!(ThinkingPolicy::parse_effort_budget("high=lots").is_err());
    }

    #[test]
    fn a_zero_budget_is_rejected_rather_than_silently_disabling_thinking() {
        let policy = ThinkingPolicy {
            default_budget: Some(0),
            ..ThinkingPolicy::default()
        };
        assert!(policy.validate().is_err());
    }
}
