//! End-to-end checks for snapshot rollback and n-gram speculation.
//!
//! These need the real Qwen3.6 GGUF and its tokenizer directory, so they are
//! opt-in: set `INFERQ_TEST_GGUF` and `INFERQ_TEST_MODEL_DIR` to run them.
//! Without those they skip with a message rather than failing.

use anyhow::Result;
use candle_core::IndexOp;
use qwen_engine::{
    GenerationOptions, GgufCheckpoint, QuantizedRuntime, Qwen3NextConfig,
    qwen::{QuantizedModel, QuantizedStateSnapshots},
    sampling::argmax,
};

/// Deterministic prompt shared with `gguf_verify_bench`.
const PROMPT_TOKENS: [u32; 8] = [8160, 579, 264, 7047, 1817, 25, 271, 16];
const EXPERT_CACHE_BYTES: usize = 8 * 1024 * 1024 * 1024;

struct TestCheckpoint {
    checkpoint: GgufCheckpoint,
    model_dir: std::path::PathBuf,
}

fn checkpoint() -> Option<TestCheckpoint> {
    let (Ok(gguf), Ok(model_dir)) = (
        std::env::var("INFERQ_TEST_GGUF"),
        std::env::var("INFERQ_TEST_MODEL_DIR"),
    ) else {
        eprintln!(
            "skipping: set INFERQ_TEST_GGUF and INFERQ_TEST_MODEL_DIR to run \
             the speculative decoding integration tests"
        );
        return None;
    };
    qwen_engine::threading::init();
    let checkpoint = GgufCheckpoint::open(&gguf).expect("open the test GGUF");
    checkpoint
        .configure_expert_cache(EXPERT_CACHE_BYTES)
        .expect("configure the expert cache");
    Some(TestCheckpoint {
        checkpoint,
        model_dir: model_dir.into(),
    })
}

/// Greedy-decode `count` tokens one row at a time, the reference behavior
/// every speculative path has to reproduce exactly.
fn decode_sequentially(
    model: &QuantizedModel<'_>,
    state: &mut qwen_engine::qwen::QuantizedModelState,
    mut pending: u32,
    count: usize,
) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(count);
    for _ in 0..count {
        let output = model.forward_detailed(&[pending], state)?;
        let logits = output.logits.i(0)?.to_vec1::<f32>()?;
        pending = argmax(&logits)? as u32;
        tokens.push(pending);
    }
    Ok(tokens)
}

#[test]
fn partial_acceptance_rollback_matches_sequential_decoding() -> Result<()> {
    let Some(test) = checkpoint() else {
        return Ok(());
    };
    let config = Qwen3NextConfig::from_path(test.model_dir.join("config.json"))?;
    let model = QuantizedModel::load(&test.checkpoint, config)?;

    // Reference: prefill, then decode one token per pass.
    let mut reference_state = model.new_state();
    let prefill = model.forward_detailed(&PROMPT_TOKENS, &mut reference_state)?;
    let rows = prefill.logits.dim(0)?;
    let first = argmax(&prefill.logits.i(rows - 1)?.to_vec1::<f32>()?)? as u32;
    let mut reference = vec![first];
    reference.extend(decode_sequentially(
        &model,
        &mut reference_state,
        first,
        40,
    )?);

    // Speculative: verify the pending token plus a draft whose first two
    // proposals are right and whose last two are deliberately wrong, so the
    // pass has to roll back to an interior row boundary.
    let mut state = model.new_state();
    model.forward_detailed(&PROMPT_TOKENS, &mut state)?;
    let wrong = [100_u32, 101];
    let drafts = [reference[1], reference[2], wrong[0], wrong[1]];
    let mut verification_tokens = vec![reference[0]];
    verification_tokens.extend_from_slice(&drafts);
    let mut snapshots = QuantizedStateSnapshots::default();
    snapshots.set_nontemporal(true);
    let verified =
        model.forward_detailed_with_snapshots(&verification_tokens, &mut state, &mut snapshots)?;
    let verifier_logits = verified.logits.to_vec2::<f32>()?;
    let mut accepted = 0;
    while accepted < drafts.len() && argmax(&verifier_logits[accepted])? as u32 == drafts[accepted]
    {
        accepted += 1;
    }
    assert_eq!(
        accepted, 2,
        "expected exactly the two correct proposals to verify"
    );
    let authoritative = argmax(&verifier_logits[accepted])? as u32;
    assert_eq!(
        authoritative, reference[3],
        "the token after the last accepted row must be the target's own choice"
    );

    let committed_rows = 1 + accepted;
    state.rollback(&snapshots, committed_rows)?;
    assert_eq!(state.position, PROMPT_TOKENS.len() + committed_rows);

    // Continuing from the rolled-back state must reproduce the reference
    // token for token; any surviving draft state would diverge here.
    let continued = decode_sequentially(&model, &mut state, authoritative, 32)?;
    assert_eq!(
        continued,
        reference[4..36],
        "decoding after a partial-acceptance rollback diverged from sequential decoding"
    );
    Ok(())
}

#[test]
fn ngram_speculation_reproduces_target_only_tokens_without_replays() -> Result<()> {
    let Some(test) = checkpoint() else {
        return Ok(());
    };
    // A prompt whose answer copies long spans out of the prompt, so the
    // drafter fires often and both full and partial acceptances occur.
    let prompt = "Repeat the following list exactly twice, in order:\n\
                  alpha bravo charlie delta echo foxtrot golf hotel india juliet\n";
    let max_new_tokens = 48;

    let mut runtime = QuantizedRuntime::load(&test.checkpoint, &test.model_dir)?;
    let baseline = runtime.generate(
        prompt,
        &GenerationOptions {
            max_new_tokens,
            ..GenerationOptions::default()
        },
    )?;

    runtime.reset();
    let speculative = runtime.generate(
        prompt,
        &GenerationOptions {
            max_new_tokens,
            speculative_ngram_draft_tokens: 7,
            ..GenerationOptions::default()
        },
    )?;

    assert_eq!(
        speculative.generated_token_ids, baseline.generated_token_ids,
        "n-gram speculation changed the greedy token sequence"
    );
    let ngram = &speculative.metrics.ngram;
    assert!(
        ngram.drafts_issued > 0,
        "the drafter never fired; the test prompt no longer exercises speculation"
    );
    assert_eq!(
        ngram.rollback_replays, 0,
        "the snapshot rollback must never replay a forward pass"
    );
    assert_eq!(ngram.replayed_tokens, 0);
    assert_eq!(
        ngram.steps,
        ngram.steps_with_match + ngram.steps_without_match
    );
    assert!(
        ngram.draft_tokens_accepted <= ngram.draft_tokens_proposed,
        "acceptance cannot exceed what was proposed"
    );
    Ok(())
}

#[test]
fn ngram_speculation_respects_the_thinking_budget() -> Result<()> {
    let Some(test) = checkpoint() else {
        return Ok(());
    };
    let mut runtime = QuantizedRuntime::load(&test.checkpoint, &test.model_dir)?;
    let prompt = runtime.tokenizer().initial_chat_prompt_with_thinking(
        "List the numbers one to twenty.",
        None,
        true,
    )?;
    let budget = 8;
    let result = runtime.generate(
        &prompt,
        &GenerationOptions {
            max_new_tokens: 64,
            speculative_ngram_draft_tokens: 7,
            thinking_budget: Some(budget),
            ..GenerationOptions::default()
        },
    )?;
    assert_eq!(result.metrics.thinking.budget, Some(budget));
    assert!(
        result.metrics.thinking.committed_thinking_tokens <= budget,
        "accepted drafts pushed {} thinking tokens past the {budget}-token budget",
        result.metrics.thinking.committed_thinking_tokens
    );
    Ok(())
}
