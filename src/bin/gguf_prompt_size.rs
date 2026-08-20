//! Token cost of a captured OpenAI request, rendered as the server renders it.
//!
//! Answers "how much does this client spend before it has said anything", which
//! is a prefill bill and, with a prompt cache, the part that is paid once.
use anyhow::{Context, Result};
use clap::Parser;
use qwen_engine::{
    server::api::ChatCompletionRequest, server::request::render_prompt,
    server::thinking::ThinkingPlan, tokenizer::ModelTokenizer,
};
use std::path::PathBuf;

#[derive(Debug, Parser)]
struct Args {
    /// A captured `POST /v1/chat/completions` body.
    #[arg(long)]
    request: PathBuf,
    #[arg(long)]
    tokenizer_model: PathBuf,
}

fn count(tokenizer: &ModelTokenizer, request: &ChatCompletionRequest, open: bool) -> Result<usize> {
    let plan = ThinkingPlan { open, budget: None };
    let prompt = render_prompt(tokenizer, request, plan)?;
    Ok(tokenizer.encode(&prompt, false)?.len())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tokenizer = ModelTokenizer::from_model_dir(&args.tokenizer_model)?;
    let body = std::fs::read(&args.request)
        .with_context(|| format!("failed to read {}", args.request.display()))?;
    let request: ChatCompletionRequest = serde_json::from_slice(&body)?;

    let full = count(&tokenizer, &request, true)?;
    // Each variant removes one part, so the difference is what that part costs
    // once the template has wrapped it.
    let mut without_tools = request.clone();
    without_tools.tools = Some(Vec::new());
    let mut without_system = request.clone();
    without_system.messages.retain(|m| m.role != "system");
    let mut bare = without_tools.clone();
    bare.messages.retain(|m| m.role != "system");

    let no_tools = count(&tokenizer, &without_tools, true)?;
    let no_system = count(&tokenizer, &without_system, true)?;
    let neither = count(&tokenizer, &bare, true)?;
    let closed = count(&tokenizer, &request, false)?;

    println!("total prompt tokens        {full:>7}");
    println!("  tool definitions         {:>7}", full - no_tools);
    println!("  system message           {:>7}", full - no_system);
    println!("  user turn plus template  {:>7}", neither);
    println!();
    println!("thinking block open        {full:>7}");
    println!("thinking block closed      {closed:>7}");
    for tool in request.tool_definitions() {
        let name = tool
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("?");
        let mut one = request.clone();
        one.tools = Some(vec![tool.clone()]);
        println!(
            "    tool {name:<8} {:>7}",
            count(&tokenizer, &one, true)? - no_tools
        );
    }
    Ok(())
}
