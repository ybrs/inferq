//! Single startup point that unifies the CPU thread pools candle and inferq
//! each spin up on their own.
//!
//! candle-core 0.11 (see its `src/utils.rs`) maintains two private, lazily
//! initialized pools of its own: a native-thread `BarrierPool` sized by
//! `CANDLE_NUM_THREADS` that runs the quantized k_quants matmul (the K=1
//! decode matvec hot path, `k_quants.rs`), and a separate `rayon::ThreadPool`
//! sized by `RAYON_NUM_THREADS` that wraps general CPU op dispatch via
//! `Device::Cpu`'s `with_threadpool`. Neither exposes a public setter; both
//! read their env var exactly once, on first use, via `OnceLock`. Inferq's
//! own multi-row dense path (`gguf.rs`, `QuantizedMatrix::small_m_forward`)
//! does not go through either candle pool — it calls `par_chunks_mut`
//! directly, which dispatches on the third pool in play: the process-global
//! rayon pool.
//!
//! With no coordinating logic, `CANDLE_NUM_THREADS` and `RAYON_NUM_THREADS`
//! could disagree, or agree but leave the global rayon pool sized by
//! whatever rayon's own default happens to be, silently serializing or
//! oversubscribing one of the three pools relative to the others. `init`
//! resolves one thread count and aligns all three: it sets
//! `CANDLE_NUM_THREADS` and `RAYON_NUM_THREADS` (so candle's own lazy env
//! reads, whichever fires first, see the same value) and explicitly builds
//! the global rayon pool to that count, rather than relying on rayon reading
//! `RAYON_NUM_THREADS` itself, since build order between candle's first CPU
//! op and inferq's first `par_chunks_mut` call is not otherwise guaranteed.

use std::sync::Once;

static INIT: Once = Once::new();

/// Resolve a single thread count and align every CPU thread pool inferq or
/// candle can spin up to it. Idempotent and safe to call from multiple
/// binaries' `main()` and from library entry points such as
/// `GgufCheckpoint::open`; only the first call in a process has any effect.
pub fn init() {
    INIT.call_once(|| {
        let (threads, source) = resolve_thread_count();
        align_env_vars(threads);
        let pool_mode = match rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
        {
            Ok(()) => "built".to_string(),
            Err(err) => format!(
                "global pool already initialized by something else ({err}); \
                 CANDLE_NUM_THREADS/RAYON_NUM_THREADS still aligned to {threads}, \
                 but the existing global rayon pool's size was not changed"
            ),
        };
        eprintln!(
            "inferq: threading: resolved {threads} threads (source: {source}); \
             global rayon pool: {pool_mode}"
        );
    });
}

fn resolve_thread_count() -> (usize, &'static str) {
    if let Some(n) = parse_env("INFERQ_NUM_THREADS") {
        return (n, "INFERQ_NUM_THREADS");
    }
    let candle = parse_env("CANDLE_NUM_THREADS");
    let rayon = parse_env("RAYON_NUM_THREADS");
    match (candle, rayon) {
        (Some(c), Some(r)) if c == r => (c, "CANDLE_NUM_THREADS/RAYON_NUM_THREADS (agree)"),
        (Some(c), Some(r)) => {
            eprintln!(
                "inferq: threading: CANDLE_NUM_THREADS={c} and RAYON_NUM_THREADS={r} disagree; \
                 using {c} for both. CANDLE_NUM_THREADS governs candle's quantized-matvec \
                 BarrierPool, the K=1 decode hot path, so it takes priority."
            );
            (c, "CANDLE_NUM_THREADS (conflict override)")
        }
        (Some(c), None) => (c, "CANDLE_NUM_THREADS"),
        (None, Some(r)) => (r, "RAYON_NUM_THREADS"),
        (None, None) => (physical_core_count(), "detected physical core count"),
    }
}

fn parse_env(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
}

fn physical_core_count() -> usize {
    // Matches candle's own fallback (`default_num_threads` in its
    // `src/utils.rs`) so an unconfigured host gets the same number candle
    // would have picked for itself.
    num_cpus::get_physical().max(1)
}

fn align_env_vars(threads: usize) {
    let value = threads.to_string();
    // SAFETY: `init` runs at most once (guarded by `Once`) from the first
    // line of every inference binary's `main()`, or from `GgufCheckpoint::
    // open` before any inference work starts — in both cases before any
    // other thread in the process has been spawned, so nothing can be
    // concurrently reading or writing the environment.
    unsafe {
        std::env::set_var("CANDLE_NUM_THREADS", &value);
        std::env::set_var("RAYON_NUM_THREADS", &value);
        std::env::set_var("INFERQ_NUM_THREADS", &value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_env_rejects_zero_and_garbage() {
        // SAFETY: single-threaded test, no other test in this process reads
        // these specific env var names concurrently.
        unsafe {
            std::env::set_var("INFERQ_TEST_THREADS_ZERO", "0");
            std::env::set_var("INFERQ_TEST_THREADS_GARBAGE", "nope");
            std::env::remove_var("INFERQ_TEST_THREADS_UNSET");
        }
        assert_eq!(parse_env("INFERQ_TEST_THREADS_ZERO"), None);
        assert_eq!(parse_env("INFERQ_TEST_THREADS_GARBAGE"), None);
        assert_eq!(parse_env("INFERQ_TEST_THREADS_UNSET"), None);
        // SAFETY: same as above.
        unsafe {
            std::env::remove_var("INFERQ_TEST_THREADS_ZERO");
            std::env::remove_var("INFERQ_TEST_THREADS_GARBAGE");
        }
    }

    #[test]
    fn physical_core_count_is_at_least_one() {
        assert!(physical_core_count() >= 1);
    }
}
