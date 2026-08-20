//! End-to-end checks that a restored prefix decodes like one that was never
//! interrupted.
//!
//! These need the real GGUF and its tokenizer directory, so they are opt-in:
//! set `INFERQ_TEST_GGUF` and `INFERQ_TEST_MODEL_DIR` to run them. Without
//! those they skip with a message rather than failing.

use anyhow::Result;
use qwen_engine::{
    GenerationOptions, GgufCheckpoint, PromptCache, PromptCacheConfig, QuantizedRuntime,
    SpeculativeMode, prompt_cache::LayerKind,
};

const EXPERT_CACHE_BYTES: usize = 8 * 1024 * 1024 * 1024;
/// Long enough to cross a boundary at the small block size used here.
const PROMPT: &str = "Write a short Rust function that sums a slice of integers, \
                      then explain what it does in one sentence.";
const BOUNDARY: usize = 8;
const NEW_TOKENS: usize = 24;

struct Fixture {
    checkpoint: GgufCheckpoint,
    model_dir: std::path::PathBuf,
}

fn fixture() -> Option<Fixture> {
    let (Ok(gguf), Ok(model_dir)) = (
        std::env::var("INFERQ_TEST_GGUF"),
        std::env::var("INFERQ_TEST_MODEL_DIR"),
    ) else {
        eprintln!(
            "skipping: set INFERQ_TEST_GGUF and INFERQ_TEST_MODEL_DIR to run \
             the prompt cache integration tests"
        );
        return None;
    };
    qwen_engine::threading::init();
    let checkpoint = GgufCheckpoint::open(&gguf).expect("open the test GGUF");
    checkpoint
        .configure_expert_cache(EXPERT_CACHE_BYTES)
        .expect("configure the expert cache");
    Some(Fixture {
        checkpoint,
        model_dir: model_dir.into(),
    })
}

fn options(mode: SpeculativeMode) -> GenerationOptions {
    GenerationOptions {
        max_new_tokens: NEW_TOKENS,
        speculative_mode: mode,
        speculative_mtp_draft_tokens: if mode.allows_mtp() { 4 } else { 0 },
        speculative_ngram_draft_tokens: if mode.allows_ngram() { 8 } else { 0 },
        ..GenerationOptions::default()
    }
}

/// Decode `tokens` after prefilling `tokens[..BOUNDARY]` as its own pass, the
/// chunking a cached prefix reproduces.
fn decode_in_two_passes(
    runtime: &mut QuantizedRuntime<'_>,
    tokens: &[u32],
    options: &GenerationOptions,
) -> Result<Vec<u32>> {
    runtime.reset();
    runtime.prefill_tokens(&tokens[..BOUNDARY], options.speculative_mode.allows_mtp())?;
    Ok(runtime
        .generate_tokens_with_callback(&tokens[BOUNDARY..], options, |_| Ok(()))?
        .generated_token_ids)
}

/// The other half of the chain the two tests below rely on.
///
/// They compare a restored session against a *chunked* prefill, so both sides
/// share the boundary and neither would notice if chunking itself moved the
/// output. Splitting a prefill is not bit-for-bit the same computation as
/// running it in one pass — the batched reductions differ in their last bits,
/// as they do for any chunked prefill — so this asserts the property that
/// actually matters to a client: that the difference stays below the token
/// the model commits to.
///
/// A failure here is not automatically a defect. It means the documented
/// last-bits caveat reached an argmax on this host, prompt and thread count;
/// `docs/prompt-cache.md` explains why that is possible. It is worth
/// investigating before it is dismissed.
#[test]
fn a_chunked_prefill_decodes_like_a_single_pass() -> Result<()> {
    let Some(fixture) = fixture() else {
        return Ok(());
    };
    let mut runtime = QuantizedRuntime::load(&fixture.checkpoint, &fixture.model_dir)?;
    let tokens = runtime.tokenizer().encode(PROMPT, false)?;
    assert!(tokens.len() > BOUNDARY, "the test prompt is too short");

    for mode in [SpeculativeMode::Off, SpeculativeMode::Auto] {
        let options = options(mode);
        runtime.reset();
        let single = runtime
            .generate_tokens_with_callback(&tokens, &options, |_| Ok(()))?
            .generated_token_ids;
        let chunked = decode_in_two_passes(&mut runtime, &tokens, &options)?;
        assert_eq!(
            chunked, single,
            "mode {mode:?}: splitting the prefill at {BOUNDARY} changed the output"
        );
        assert_eq!(single.len(), NEW_TOKENS);
    }
    Ok(())
}

#[test]
fn a_restored_prefix_decodes_exactly_like_an_uninterrupted_one() -> Result<()> {
    let Some(fixture) = fixture() else {
        return Ok(());
    };
    let mut runtime = QuantizedRuntime::load(&fixture.checkpoint, &fixture.model_dir)?;
    let tokens = runtime.tokenizer().encode(PROMPT, false)?;
    assert!(tokens.len() > BOUNDARY, "the test prompt is too short");

    for mode in [SpeculativeMode::Off, SpeculativeMode::Auto] {
        let options = options(mode);
        let reference = decode_in_two_passes(&mut runtime, &tokens, &options)?;

        // Capture the boundary, then rebuild the session from that image alone.
        runtime.reset();
        runtime.prefill_tokens(&tokens[..BOUNDARY], mode.allows_mtp())?;
        let image = runtime.session_image(tokens[..BOUNDARY].to_vec())?;
        assert_eq!(image.position(), BOUNDARY);
        if mode.allows_mtp() {
            // Without the predictor's own cache a restored session decodes
            // with the MTP arm sitting out — same tokens, quietly slower.
            assert!(image.mtp.is_some(), "the image carries the MTP cache");
            assert!(image.last_target_hidden.is_some());
        }
        runtime.reset();
        runtime.restore_session(&image)?;
        assert_eq!(runtime.context_tokens(), BOUNDARY);
        assert_eq!(
            runtime.mtp_arm_ready(),
            mode.allows_mtp(),
            "a restored session must speculate exactly as a prefilled one does"
        );
        let restored = runtime
            .generate_tokens_with_callback(&tokens[BOUNDARY..], &options, |_| Ok(()))?
            .generated_token_ids;
        assert_eq!(
            restored, reference,
            "mode {mode:?} diverged after a restore"
        );
        assert_eq!(restored.len(), NEW_TOKENS);
    }
    Ok(())
}

#[test]
fn an_entry_written_to_disk_restores_in_a_new_runtime() -> Result<()> {
    let Some(fixture) = fixture() else {
        return Ok(());
    };
    let directory = tempfile::tempdir()?;
    let identity = fixture.checkpoint.identity()?;
    let options = options(SpeculativeMode::Auto);

    let reference = {
        let mut runtime = QuantizedRuntime::load(&fixture.checkpoint, &fixture.model_dir)?;
        let tokens = runtime.tokenizer().encode(PROMPT, false)?;
        let reference = decode_in_two_passes(&mut runtime, &tokens, &options)?;
        // Write the boundary the way the server does.
        runtime.reset();
        runtime.prefill_tokens(&tokens[..BOUNDARY], true)?;
        let image = runtime.session_image(tokens[..BOUNDARY].to_vec())?;
        let config = runtime.model().config();
        let cache = PromptCache::open(
            PromptCacheConfig {
                dir: directory.path().to_path_buf(),
                budget_bytes: 64 * 1024 * 1024 * 1024,
                block_tokens: BOUNDARY,
                min_tokens: BOUNDARY,
            },
            &identity.layout_fingerprint,
            &identity.quantization.join("+"),
            LayerKind::for_config(config),
        )?;
        assert!(cache.store(image));
        for _ in 0..600 {
            if cache.stats().writes > 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert_eq!(cache.stats().writes, 1, "{:?}", cache.stats());
        reference
    };

    // A second process would see exactly this: a fresh runtime, a cache
    // directory, and no state of its own.
    let mut runtime = QuantizedRuntime::load(&fixture.checkpoint, &fixture.model_dir)?;
    let tokens = runtime.tokenizer().encode(PROMPT, false)?;
    let config = runtime.model().config();
    let cache = PromptCache::open(
        PromptCacheConfig {
            dir: directory.path().to_path_buf(),
            budget_bytes: 64 * 1024 * 1024 * 1024,
            block_tokens: BOUNDARY,
            min_tokens: BOUNDARY,
        },
        &identity.layout_fingerprint,
        &identity.quantization.join("+"),
        LayerKind::for_config(config),
    )?;
    assert_eq!(cache.stats().entries, 1);
    let image = cache.lookup(&tokens, 0).expect("the entry is a prefix hit");
    assert_eq!(image.position(), BOUNDARY);
    assert!(image.mtp.is_some(), "the predictor cache survived the disk");
    runtime.restore_session(&image)?;
    assert!(runtime.mtp_arm_ready());
    let restored = runtime
        .generate_tokens_with_callback(&tokens[BOUNDARY..], &options, |_| Ok(()))?
        .generated_token_ids;
    assert_eq!(restored, reference, "a disk round trip changed decoding");
    Ok(())
}
