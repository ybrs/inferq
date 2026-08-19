//! End-to-end checks for the unified speculative policy.
//!
//! These need the real Qwen3.6 GGUF and its tokenizer directory, so they are
//! opt-in: set `INFERQ_TEST_GGUF` and `INFERQ_TEST_MODEL_DIR` to run them.
//! Without those they skip with a message rather than failing.
//!
//! What they exist to prove is that speculation changes speed and nothing
//! else. Every mode has to emit the token sequence target-only decoding would
//! emit, and the lazy MTP catch-up has to leave the predictor in exactly the
//! state an eager resynchronisation would have left it in.

use anyhow::Result;
use qwen_engine::{
    GenerationOptions, GgufCheckpoint, QuantizedRuntime, SpeculativeMode,
    runtime::PolicyTuning,
    speculative::{DEFAULT_MTP_DEPTH_CAP, DEFAULT_NGRAM_DRAFT_CAP},
};

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
             the speculative policy integration tests"
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

fn options(mode: SpeculativeMode, max_new_tokens: usize) -> GenerationOptions {
    GenerationOptions {
        max_new_tokens,
        speculative_mode: mode,
        ..GenerationOptions::default()
    }
}

/// A prompt whose answer copies long literal spans out of itself, so the
/// n-gram arm fires often, and whose second half asks for structural
/// repetition the index cannot see, so the MTP arm gets its turn.
const MIXED_PROMPT: &str = "Repeat the following list exactly twice, in order:\n\
                            alpha bravo charlie delta echo foxtrot golf hotel india juliet\n\
                            Then count from one to twenty, one number per line.\n";

#[test]
fn every_mode_reproduces_the_target_only_token_sequence() -> Result<()> {
    let Some(test) = checkpoint() else {
        return Ok(());
    };
    let max_new_tokens = 64;
    let mut runtime = QuantizedRuntime::load(&test.checkpoint, &test.model_dir)?;
    let baseline =
        runtime.generate(MIXED_PROMPT, &options(SpeculativeMode::Off, max_new_tokens))?;

    for mode in [
        SpeculativeMode::Auto,
        SpeculativeMode::Ngram,
        SpeculativeMode::Mtp,
    ] {
        runtime.reset();
        let speculative = runtime.generate(MIXED_PROMPT, &options(mode, max_new_tokens))?;
        assert_eq!(
            speculative.generated_token_ids,
            baseline.generated_token_ids,
            "mode {} changed the greedy token sequence",
            mode.as_str()
        );
        let policy = &speculative.metrics.policy;
        assert_eq!(policy.mode, mode);
        assert_eq!(
            policy.steps,
            policy.ngram_steps + policy.mtp_steps + policy.plain_steps,
            "every step belongs to exactly one arm"
        );
        assert!(
            speculative.metrics.ngram.rollbacks <= policy.rollbacks,
            "the n-gram arm cannot have rolled back more often than the loop did"
        );
        if mode == SpeculativeMode::Ngram {
            assert_eq!(
                policy.mtp_steps, 0,
                "the MTP arm must not fire in ngram mode"
            );
        }
        if mode == SpeculativeMode::Mtp {
            assert_eq!(
                policy.ngram_steps, 0,
                "the n-gram arm must not fire in mtp mode"
            );
            assert!(
                policy.steps_with_ngram_match > 0,
                "literal evidence is recorded even when the n-gram arm is off"
            );
        }
    }
    Ok(())
}

#[test]
fn auto_uses_both_arms_and_never_runs_them_in_the_same_step() -> Result<()> {
    let Some(test) = checkpoint() else {
        return Ok(());
    };
    let mut runtime = QuantizedRuntime::load(&test.checkpoint, &test.model_dir)?;
    let result = runtime.generate(MIXED_PROMPT, &options(SpeculativeMode::Auto, 96))?;
    let policy = &result.metrics.policy;
    assert!(
        policy.ngram_steps > 0,
        "the copying half of the prompt should have fired the n-gram arm"
    );
    assert!(
        policy.mtp_steps > 0,
        "the counting half of the prompt should have fired the MTP arm"
    );
    // The tie rule: a step the n-gram arm won never also drafted with MTP, so
    // no MTP proposal can carry an n-gram match alongside it.
    assert_eq!(
        policy.mtp_proposed_on_ngram_match, 0,
        "an n-gram match must take the step outright"
    );
    // The record carries each controller's state *after* the step's outcome
    // was folded in, so a step that shrank its arm reports a length below what
    // it proposed. The invariant that holds either way is the cap.
    for record in &policy.records {
        assert!(
            record.proposed <= DEFAULT_NGRAM_DRAFT_CAP.max(DEFAULT_MTP_DEPTH_CAP),
            "step {} proposed {} tokens, past every cap",
            record.step,
            record.proposed
        );
        assert!(
            record.accepted <= record.proposed,
            "step {} accepted more than it proposed",
            record.step
        );
    }
    Ok(())
}

#[test]
fn lazy_mtp_catch_up_decodes_exactly_like_eager_resynchronisation() -> Result<()> {
    let Some(test) = checkpoint() else {
        return Ok(());
    };
    // The point of the comparison is a long stretch in which the MTP block is
    // not the one committing tokens: the copying half of this prompt is
    // n-gram-arm work, and the MTP arm then has to draft from a position it
    // last synchronised many tokens ago.
    let max_new_tokens = 96;
    let mut runtime = QuantizedRuntime::load(&test.checkpoint, &test.model_dir)?;
    let lazy = runtime.generate(
        MIXED_PROMPT,
        &options(SpeculativeMode::Auto, max_new_tokens),
    )?;

    runtime.reset();
    let eager = runtime.generate(
        MIXED_PROMPT,
        &GenerationOptions {
            policy: PolicyTuning {
                eager_mtp_resync: true,
                ..PolicyTuning::default()
            },
            ..options(SpeculativeMode::Auto, max_new_tokens)
        },
    )?;

    assert_eq!(
        lazy.generated_token_ids, eager.generated_token_ids,
        "lazy MTP catch-up changed the decoded tokens"
    );
    // Same drafts, not merely the same output: a stale predictor would still
    // be corrected by verification, so token equality alone would not prove
    // the catch-up worked. Equal per-arm acceptance does.
    assert_eq!(
        lazy.metrics.policy.mtp_arm.proposed_tokens, eager.metrics.policy.mtp_arm.proposed_tokens,
        "the two schemes proposed different numbers of MTP tokens"
    );
    assert_eq!(
        lazy.metrics.policy.mtp_arm.accepted_tokens, eager.metrics.policy.mtp_arm.accepted_tokens,
        "the MTP arm drafted differently under lazy catch-up"
    );
    assert!(
        lazy.metrics.policy.resync_passes <= eager.metrics.policy.resync_passes,
        "the lazy scheme cannot resynchronise more often than the eager one"
    );
    assert!(
        lazy.metrics.policy.resync_tokens <= eager.metrics.policy.resync_tokens,
        "the lazy scheme never resynchronises rows the eager one did not: a gap \
         left behind when the arm suspends is never closed at all"
    );
    assert!(
        lazy.metrics.policy.max_resync_tokens >= eager.metrics.policy.max_resync_tokens,
        "the lazy scheme is the one that closes long gaps in a single pass"
    );
    Ok(())
}

#[test]
fn the_thinking_budget_bounds_continuation_and_mtp_drafts_alike() -> Result<()> {
    let Some(test) = checkpoint() else {
        return Ok(());
    };
    let mut runtime = QuantizedRuntime::load(&test.checkpoint, &test.model_dir)?;
    let prompt = runtime.tokenizer().initial_chat_prompt_with_thinking(
        "List the numbers one to twenty, then list them again.",
        None,
        true,
    )?;
    // Budgets that fall inside a draft rather than on a pass boundary are the
    // interesting ones, so sweep a few.
    for budget in [3, 8, 11] {
        for mode in [
            SpeculativeMode::Auto,
            SpeculativeMode::Ngram,
            SpeculativeMode::Mtp,
        ] {
            runtime.reset();
            let result = runtime.generate(
                &prompt,
                &GenerationOptions {
                    thinking_budget: Some(budget),
                    ..options(mode, 64)
                },
            )?;
            assert_eq!(result.metrics.thinking.budget, Some(budget));
            assert!(
                result.metrics.thinking.committed_thinking_tokens <= budget,
                "mode {} with budget {budget} committed {} thinking tokens",
                mode.as_str(),
                result.metrics.thinking.committed_thinking_tokens
            );
            assert_eq!(
                result.metrics.generated_tokens,
                result.generated_token_ids.len()
            );
        }
    }
    Ok(())
}

#[test]
fn a_turn_limit_that_lands_inside_a_draft_still_emits_exactly_that_many_tokens() -> Result<()> {
    let Some(test) = checkpoint() else {
        return Ok(());
    };
    let mut runtime = QuantizedRuntime::load(&test.checkpoint, &test.model_dir)?;
    // Lengths chosen to land mid-draft for a controller sitting at 7 and at 4.
    for max_new_tokens in [5, 9, 13, 30] {
        runtime.reset();
        let baseline =
            runtime.generate(MIXED_PROMPT, &options(SpeculativeMode::Off, max_new_tokens))?;
        runtime.reset();
        let auto = runtime.generate(
            MIXED_PROMPT,
            &options(SpeculativeMode::Auto, max_new_tokens),
        )?;
        assert_eq!(auto.generated_token_ids.len(), max_new_tokens);
        assert_eq!(
            auto.generated_token_ids, baseline.generated_token_ids,
            "the turn limit changed what {max_new_tokens} tokens were emitted"
        );
    }
    Ok(())
}

#[test]
fn disabling_the_controllers_pins_both_arms_at_their_caps() -> Result<()> {
    let Some(test) = checkpoint() else {
        return Ok(());
    };
    let mut runtime = QuantizedRuntime::load(&test.checkpoint, &test.model_dir)?;
    let pinned = PolicyTuning {
        span_continuation: false,
        adaptive_length: false,
        ewma_backoff: false,
        ..PolicyTuning::default()
    };
    let result = runtime.generate(
        MIXED_PROMPT,
        &GenerationOptions {
            policy: PolicyTuning {
                mtp_depth_start: 7,
                ..pinned
            },
            ..options(SpeculativeMode::Auto, 64)
        },
    )?;
    let policy = &result.metrics.policy;
    assert_eq!(policy.ngram_span_steps, 0, "span continuation was disabled");
    assert_eq!(policy.ngram_arm.suspensions, 0, "backoff was disabled");
    assert_eq!(policy.mtp_arm.suspensions, 0, "backoff was disabled");
    for record in &policy.records {
        assert_eq!(record.ngram_len, 7, "the n-gram length must not move");
        assert_eq!(record.mtp_depth, 7, "the MTP depth must not move");
    }
    Ok(())
}

/// Two identical target-only generations in one process must agree.
///
/// This is not about speculation at all: it is the control for every other
/// test in this file. Each of those compares a speculative run against a
/// target-only baseline decoded by the *same* long-lived runtime, so if
/// repeated decoding in one process is not reproducible, those comparisons
/// cannot attribute a difference to speculation.
#[test]
fn repeated_target_only_decoding_is_reproducible_in_one_process() -> Result<()> {
    let Some(test) = checkpoint() else {
        return Ok(());
    };
    let mut runtime = QuantizedRuntime::load(&test.checkpoint, &test.model_dir)?;
    let first = runtime.generate(MIXED_PROMPT, &options(SpeculativeMode::Off, 64))?;
    for round in 0..3 {
        runtime.reset();
        let again = runtime.generate(MIXED_PROMPT, &options(SpeculativeMode::Off, 64))?;
        assert_eq!(
            again.generated_token_ids, first.generated_token_ids,
            "target-only decoding differed between repetitions in round {round}"
        );
    }
    Ok(())
}

/// Row `r` of a multi-row pass must equal the one-row pass for that position.
///
/// This is the assumption every speculative mode rests on: a verification pass
/// evaluates row `r` from exactly the prefix a sequential decode would have fed
/// it, so accepting a draft commits the token the target would have produced on
/// its own. If widening a pass perturbs the arithmetic, that equality holds only
/// What a multi-row pass does and does not guarantee against single-row decoding.
///
/// This is the assumption every speculative mode rests on. A verification pass
/// evaluates row `r` from exactly the prefix a sequential decode would have fed
/// it, so accepting a draft commits the token the target would have produced on
/// its own. If widening a pass perturbed the arithmetic, that equality would
/// hold only up to floating-point noise and a near-tie in the argmax could
/// flip, making greedy equivalence probabilistic rather than guaranteed.
///
/// It *is* probabilistic today, and this test records the size of the gap
/// rather than pretending otherwise. Dispatch sends M=1 to Candle's kernel and
/// M>=2 to the fused one. On the LM head those agree to 4.77e-7 — about four
/// ULP, a summation-order difference — but forty layers of feedback amplify it:
/// every one of the 248,320 logits differs, by up to 1.28 against a tightest
/// observed top1/top2 margin of 0.25.
///
/// Unifying the dispatch does make it exact, and was verified to; it costs
/// ~20% of unspeculated decode because the fused kernel's advantage is reusing
/// a decoded weight block across M input rows and at M=1 there is nothing to
/// reuse. See the task tracking a bit-exact fused kernel, which would give
/// exactness at no cost. Until then this asserts the property that does hold:
/// the argmax is stable across widths on this sample.
#[test]
fn a_multi_row_pass_agrees_with_single_row_decoding_on_the_greedy_choice() -> Result<()> {
    let Some(test) = checkpoint() else {
        return Ok(());
    };
    use candle_core::IndexOp;
    let config = qwen_engine::Qwen3NextConfig::from_path(test.model_dir.join("config.json"))?;
    let model = qwen_engine::qwen::QuantizedModel::load(&test.checkpoint, config)?;
    let prompt: [u32; 8] = [8160, 579, 264, 7047, 1817, 25, 271, 16];

    // Decode a reference sequence one row at a time.
    let mut sequential = model.new_state();
    model.forward_detailed(&prompt, &mut sequential)?;
    let mut pending = 16u32;
    let mut follow = vec![pending];
    let mut singles = Vec::new();
    for _ in 0..8 {
        let out = model.forward_detailed(&[pending], &mut sequential)?;
        let row = out.logits.i(0)?.to_vec1::<f32>()?;
        pending = qwen_engine::sampling::argmax(&row)? as u32;
        singles.push(row);
        follow.push(pending);
    }

    let mut worst_delta = 0f32;
    let mut tightest_margin = f32::MAX;
    let mut mismatched = 0usize;
    // Re-evaluate the same positions as rows of progressively wider passes.
    for width in [2usize, 4, 8] {
        let mut state = model.new_state();
        model.forward_detailed(&prompt, &mut state)?;
        let wide = model.forward_detailed(&follow[..width], &mut state)?;
        for (row, single) in singles.iter().enumerate().take(width) {
            let batched = wide.logits.i(row)?.to_vec1::<f32>()?;
            let mismatches = single
                .iter()
                .zip(&batched)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let worst = single
                .iter()
                .zip(&batched)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            worst_delta = worst_delta.max(worst);
            mismatched += mismatches;
            let mut sorted = batched.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            tightest_margin = tightest_margin.min(sorted[0] - sorted[1]);
            assert_eq!(
                qwen_engine::sampling::argmax(single.as_slice())?,
                qwen_engine::sampling::argmax(&batched)?,
                "row {row} of a {width}-row pass chose a different token than the \
                 one-row pass; |delta| up to {worst:e}"
            );
        }
    }
    eprintln!(
        "batching vs single-row: {mismatched} logits differ, worst |delta| \
         {worst_delta:e}, tightest top1/top2 margin {tightest_margin:e}"
    );
    // Bit-exact, not merely agreeing on the argmax. The argmax assertion above
    // is what this test could claim while the two kernels reached different
    // summation orders; opening a checkpoint now refuses on a build where they
    // do, so the stronger statement is available and is the one worth pinning.
    // If this fails and the argmax assertions did not, the kernels have drifted
    // apart again and speculative decoding is back to being probabilistic
    // rather than exact -- see docs/speculative-decoding.md.
    assert_eq!(
        mismatched, 0,
        "a multi-row pass and one-row decoding must produce identical logits, \
         not merely the same argmax; {mismatched} differ by up to {worst_delta:e} \
         against a tightest margin of {tightest_margin:e}"
    );
    Ok(())
}

/// The draft-only LM head must never touch the target's decisions.
///
/// A vocabulary prefix is admissible for drafting because a wrong draft is
/// rejected. It would be a correctness bug anywhere else, so this pins both
/// halves: the slice really is the leading rows of the full head, and a run
/// using it still emits exactly what target-only decoding emits.
#[test]
fn a_shortlisted_draft_head_changes_speed_but_not_output() -> Result<()> {
    let Some(test) = checkpoint() else {
        return Ok(());
    };
    let config = qwen_engine::Qwen3NextConfig::from_path(test.model_dir.join("config.json"))?;
    let model = qwen_engine::qwen::QuantizedModel::load(&test.checkpoint, config)?;
    let mtp = model.mtp().expect("this checkpoint carries an MTP block");

    // The slice is the leading rows of the shared head, byte for byte.
    let full = mtp.vocab_size();
    let sliced = mtp.draft_head(1024)?;
    assert_eq!(sliced.shape(), [1024, model.config().hidden_size]);
    assert!(
        sliced.storage_bytes() * full / 1024 == mtp.draft_head(full)?.storage_bytes(),
        "a leading row slice must be a contiguous byte prefix"
    );

    let mut runtime = QuantizedRuntime::load(&test.checkpoint, &test.model_dir)?;
    let baseline = runtime.generate(MIXED_PROMPT, &options(SpeculativeMode::Off, 64))?;
    for vocab in [0usize, 8192, 32768] {
        runtime.reset();
        let shortlisted = runtime.generate(
            MIXED_PROMPT,
            &GenerationOptions {
                mtp_draft_vocab: vocab,
                ..options(SpeculativeMode::Mtp, 64)
            },
        )?;
        assert_eq!(
            shortlisted.generated_token_ids, baseline.generated_token_ids,
            "draft vocabulary {vocab} changed the emitted tokens"
        );
        let policy = &shortlisted.metrics.policy;
        if vocab > 0 {
            assert_eq!(policy.draft_vocab, vocab.min(policy.full_vocab));
        }
    }
    Ok(())
}
