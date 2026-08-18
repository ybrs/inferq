//! The inference slot behind the HTTP surface.
//!
//! [`QuantizedRuntime`] owns one sequence: its recurrent state, KV cache and
//! MTP synchronisation point all describe a single conversation, and
//! generation takes `&mut self`. So the engine is a single worker thread that
//! owns the checkpoint and the runtime for the process's lifetime and takes
//! one job at a time off a queue.
//!
//! Requests are stateless in what they mean — every job is decoded against
//! exactly the conversation it sent, and nothing carries over implicitly — but
//! not in what they compute. Prefill is the expensive half of a request, so a
//! job starts from the longest state it can prove describes a prefix of its
//! own tokens: the previous request's session when this one continues it, or a
//! [`PromptCache`] entry when one is on disk. See `src/prompt_cache`.

use std::{
    fmt,
    path::PathBuf,
    sync::{
        Arc, atomic,
        atomic::AtomicUsize,
        mpsc::{self, Receiver, SyncSender},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use crate::{
    GenerationOptions, GgufCheckpoint, GgufModelIdentity, PromptCache, PromptCacheConfig,
    PromptCacheStats, QuantizedRuntime, Qwen3NextConfig, SpeculativeMode, config::EosTokenId,
    prompt_cache::LayerKind, residency::warm_all_experts, runtime::effective_speculative_mode,
    tokenizer::ModelTokenizer, tool_calls, tool_calls::ParsedToolCall,
    trace::RoutingCensusArtifact,
};

use super::{api::FinishReason, stop::StopBuffer};

/// Marker the model writes to end its thinking section.
const THINK_CLOSE: &str = "</think>";

/// Counts what a turn spent inside its thinking block.
///
/// The block is opened by the prompt, so counting starts immediately and ends
/// at the first `</think>` the model writes — including the one a forced
/// closure inserts, which is part of the reasoning section either way.
#[derive(Debug, Default)]
struct ThinkingCounter {
    open: bool,
    tokens: usize,
    /// Where to resume searching for the marker. Kept on a character
    /// boundary: the model writes plenty of text that is not ASCII.
    scanned: usize,
}

impl ThinkingCounter {
    fn new(open: bool) -> Self {
        Self {
            open,
            tokens: 0,
            scanned: 0,
        }
    }

    /// Record one emitted token, given everything decoded so far.
    fn observe(&mut self, raw: &str) {
        if !self.open {
            return;
        }
        self.tokens += 1;
        if raw[self.scanned..].contains(THINK_CLOSE) {
            self.open = false;
            return;
        }
        // Only a tail shorter than the marker can still complete it.
        let mut resume = raw.len().saturating_sub(THINK_CLOSE.len() - 1);
        while resume > 0 && !raw.is_char_boundary(resume) {
            resume -= 1;
        }
        self.scanned = resume;
    }
}

/// Which experts to pull into memory before the first request.
#[derive(Debug, Clone, Default)]
pub enum Warmup {
    #[default]
    None,
    /// Every fused expert tensor, in GGUF file order.
    AllExperts,
    /// The hottest experts per layer from a prior routing census.
    Census {
        path: PathBuf,
        experts_per_layer: usize,
    },
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model: PathBuf,
    pub tokenizer_model: PathBuf,
    /// Name the server reports and accepts as `model`. Defaults to the GGUF
    /// file stem.
    pub served_model_name: Option<String>,
    pub expert_cache_bytes: usize,
    pub warmup: Warmup,
    pub snapshot_nontemporal: bool,
    /// Generation settings a request inherits when it does not override them.
    pub defaults: GenerationOptions,
    /// Jobs allowed to be queued or running before new ones are refused.
    pub max_queue: usize,
    /// Persist prefix state under this directory. `None` keeps everything in
    /// memory, so only a request that continues the live session reuses work.
    pub prompt_cache: Option<PromptCacheConfig>,
    /// Whether a request may start from state it did not compute itself.
    /// Turning this off makes every request prefill its whole prompt, which is
    /// the behaviour reuse is measured against.
    pub prefix_reuse: bool,
}

/// What the server can report about the loaded checkpoint.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub layout_fingerprint: String,
    pub quantization: Vec<String>,
    pub max_position_embeddings: usize,
    pub load_wall_time: Duration,
}

/// One unit of streamed output. Exactly one terminal event (`Done` or
/// `Failed`) is sent per job unless the receiver was dropped first.
#[derive(Debug, Clone)]
pub enum Event {
    /// Text delta, already stripped of special tokens and of any text held
    /// back as a possible stop-string prefix.
    Delta(String),
    Done(Completion),
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub finish_reason: FinishReason,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Prompt tokens this request did not have to prefill.
    pub reused_tokens: usize,
    pub reuse: Reuse,
    /// Calls the turn made, with their parameters still as the model wrote
    /// them. Typing them needs the request's tool schemas.
    pub tool_calls: Vec<ParsedToolCall>,
    /// Tokens spent inside the thinking block, or `None` when the turn had no
    /// thinking section at all.
    pub reasoning_tokens: Option<usize>,
}

/// Where a request's starting state came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reuse {
    /// Nothing: the prompt was prefilled from an empty session.
    None,
    /// The previous request's session, which this prompt continues.
    Live,
    /// An entry the prompt cache had on disk.
    Cache,
}

impl Reuse {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Live => "live",
            Self::Cache => "cache",
        }
    }
}

/// Holds one place in the queue. Dropping it frees the place, so a job that is
/// never queued, is abandoned while queued, or runs to completion all account
/// for themselves without the HTTP layer having to remember to.
struct QueueSlot {
    pending: Arc<AtomicUsize>,
}

impl Drop for QueueSlot {
    fn drop(&mut self) {
        self.pending.fetch_sub(1, atomic::Ordering::AcqRel);
    }
}

/// One request as the engine needs it.
#[derive(Debug, Clone)]
pub struct JobRequest {
    pub prompt: String,
    /// The part of `prompt` a later request is expected to repeat: a
    /// conversation without its final message. Only ever a hint, and only used
    /// to decide where a cache entry is worth storing.
    pub stable_prefix: Option<String>,
    pub options: GenerationOptions,
    pub stop_strings: Vec<String>,
    /// Whether the prompt offered tools. Tool-call syntax is then held back
    /// from the text stream and reported as calls instead.
    pub tools_enabled: bool,
    /// Whether the prompt left a `<think>` block open. The tag is then part of
    /// the response, so what the client receives is a complete block rather
    /// than text that ends with an unmatched `</think>`.
    pub thinking_open: bool,
}

struct Job {
    request: JobRequest,
    events: UnboundedSender<Event>,
    _slot: QueueSlot,
}

/// Why generation stopped early, carried out of the token callback.
///
/// The callback's error is returned verbatim by every generation path, so a
/// sentinel is how the callback stops the loop without being mistaken for a
/// real failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Halt {
    StopString,
    Disconnected,
}

impl fmt::Display for Halt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StopString => f.write_str("a stop sequence was generated"),
            Self::Disconnected => f.write_str("the client disconnected"),
        }
    }
}

impl std::error::Error for Halt {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    /// The queue is at `max_queue`.
    Busy,
    /// The worker thread is gone; the process cannot serve anything further.
    Stopped,
}

impl fmt::Display for SubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy => f.write_str("the inference queue is full"),
            Self::Stopped => f.write_str("the inference worker is not running"),
        }
    }
}

impl std::error::Error for SubmitError {}

/// Handle to the worker thread. Cloneable and safe to share across tasks.
#[derive(Clone)]
pub struct EngineHandle {
    jobs: mpsc::Sender<Job>,
    /// Queued plus running jobs.
    pending: Arc<AtomicUsize>,
    max_queue: usize,
    info: ModelInfo,
    /// A clone of the runtime's tokenizer, so prompts render (and fail) on the
    /// request path rather than after a job has been queued.
    tokenizer: ModelTokenizer,
    defaults: GenerationOptions,
    /// Kept here only so the HTTP layer can report cache counters; the worker
    /// thread is the only one that reads or writes entries.
    prompt_cache: Option<Arc<PromptCache>>,
}

impl EngineHandle {
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    pub fn prompt_cache_stats(&self) -> Option<PromptCacheStats> {
        self.prompt_cache.as_ref().map(|cache| cache.stats())
    }

    /// Let a queued cache write finish before the process exits.
    pub fn flush_prompt_cache(&self, timeout: Duration) {
        if let Some(cache) = &self.prompt_cache
            && !cache.wait_for_writes(timeout)
        {
            tracing::warn!("a prompt cache write was still in flight at shutdown");
        }
    }

    pub fn tokenizer(&self) -> &ModelTokenizer {
        &self.tokenizer
    }

    pub fn defaults(&self) -> &GenerationOptions {
        &self.defaults
    }

    /// Queued plus running jobs.
    pub fn pending(&self) -> usize {
        self.pending.load(atomic::Ordering::Relaxed)
    }

    pub fn max_queue(&self) -> usize {
        self.max_queue
    }

    /// Queue a job, returning the stream of its output events.
    ///
    /// Dropping the receiver cancels the job: if it has not started the worker
    /// skips it, and if it has the worker abandons it at the next token.
    pub fn submit(&self, request: JobRequest) -> Result<UnboundedReceiver<Event>, SubmitError> {
        let slot = self.reserve_slot()?;
        let (events, receiver) = unbounded_channel();
        self.jobs
            .send(Job {
                request,
                events,
                _slot: slot,
            })
            .map_err(|_| SubmitError::Stopped)?;
        Ok(receiver)
    }

    fn reserve_slot(&self) -> Result<QueueSlot, SubmitError> {
        let mut pending = self.pending.load(atomic::Ordering::Acquire);
        loop {
            if pending >= self.max_queue {
                return Err(SubmitError::Busy);
            }
            match self.pending.compare_exchange_weak(
                pending,
                pending + 1,
                atomic::Ordering::AcqRel,
                atomic::Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(QueueSlot {
                        pending: Arc::clone(&self.pending),
                    });
                }
                Err(observed) => pending = observed,
            }
        }
    }
}

/// Load the model on a dedicated thread and return once it is ready to serve.
///
/// The checkpoint and the runtime that borrows it both live on that thread for
/// the process's lifetime, which is what keeps the borrow local and the
/// runtime off every other thread.
pub fn start(config: EngineConfig) -> Result<EngineHandle> {
    let (jobs, job_receiver) = mpsc::channel::<Job>();
    let (ready, ready_receiver) = mpsc::sync_channel::<Result<Ready>>(1);
    let worker_config = config.clone();
    thread::Builder::new()
        .name("inferq-engine".to_owned())
        .spawn(move || worker(worker_config, job_receiver, ready))
        .context("failed to spawn the inference worker thread")?;
    let ready = ready_receiver
        .recv()
        .context("the inference worker exited before reporting readiness")??;
    Ok(EngineHandle {
        jobs,
        pending: Arc::new(AtomicUsize::new(0)),
        max_queue: config.max_queue.max(1),
        info: ready.info,
        tokenizer: ready.tokenizer,
        defaults: config.defaults,
        prompt_cache: ready.prompt_cache,
    })
}

/// What the worker hands back once the model is loaded and the cache is open.
struct Ready {
    info: ModelInfo,
    tokenizer: ModelTokenizer,
    prompt_cache: Option<Arc<PromptCache>>,
}

fn worker(config: EngineConfig, jobs: Receiver<Job>, ready: SyncSender<Result<Ready>>) {
    let checkpoint = match GgufCheckpoint::open(&config.model)
        .with_context(|| format!("failed to open {}", config.model.display()))
    {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let (mut runtime, prompt_cache) = match prepare(&checkpoint, &config) {
        Ok((runtime, info, prompt_cache)) => {
            let tokenizer = runtime.tokenizer().clone();
            let sent = ready.send(Ok(Ready {
                info,
                tokenizer,
                prompt_cache: prompt_cache.clone(),
            }));
            if sent.is_err() {
                return;
            }
            (runtime, prompt_cache)
        }
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    // The tokens the live session's state represents, so the next request can
    // tell whether it continues this conversation or starts another one.
    let mut live: Option<Vec<u32>> = None;
    while let Ok(job) = jobs.recv() {
        run_job(
            &mut runtime,
            job,
            &mut live,
            prompt_cache.as_deref().filter(|_| config.prefix_reuse),
            config.prefix_reuse,
        );
    }
}

/// Open the runtime, apply warmup, and describe what was loaded.
#[allow(clippy::type_complexity)]
fn prepare<'a>(
    checkpoint: &'a GgufCheckpoint,
    config: &EngineConfig,
) -> Result<(QuantizedRuntime<'a>, ModelInfo, Option<Arc<PromptCache>>)> {
    checkpoint.configure_expert_cache(config.expert_cache_bytes)?;
    let identity = checkpoint.identity()?;
    let started = Instant::now();
    let mut runtime = QuantizedRuntime::load(checkpoint, &config.tokenizer_model)?;
    runtime.set_snapshot_nontemporal(config.snapshot_nontemporal);
    let load_wall_time = started.elapsed();
    let id = config.served_model_name.clone().unwrap_or_else(|| {
        config
            .model
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "qwen".to_owned())
    });
    let info = ModelInfo {
        id,
        layout_fingerprint: identity.layout_fingerprint.clone(),
        quantization: identity.quantization.clone(),
        max_position_embeddings: runtime.model().config().max_position_embeddings,
        load_wall_time,
    };
    tracing::info!(
        model = %info.id,
        layout = %info.layout_fingerprint,
        quantization = %info.quantization.join("+"),
        seconds = load_wall_time.as_secs_f64(),
        "model loaded"
    );
    match &config.warmup {
        Warmup::None => {}
        Warmup::AllExperts => {
            let report = warm_all_experts(checkpoint, runtime.model().config(), |_| {})?;
            tracing::info!(
                gib = report.bytes_loaded as f64 / (1024. * 1024. * 1024.),
                seconds = report.elapsed.as_secs_f64(),
                "warmed every expert"
            );
        }
        Warmup::Census {
            path,
            experts_per_layer,
        } => {
            let census = RoutingCensusArtifact::from_path(path)?;
            census.validate_for(
                &identity,
                runtime.model().config().num_hidden_layers,
                runtime.model().config().num_experts,
            )?;
            let started = Instant::now();
            let mut experts = 0usize;
            for (layer, layer_experts) in census.hottest_experts(*experts_per_layer) {
                for expert in layer_experts {
                    checkpoint.warm_expert(layer, expert)?;
                    experts += 1;
                }
            }
            tracing::info!(
                experts,
                seconds = started.elapsed().as_secs_f64(),
                "warmed the hottest experts from a census"
            );
        }
    }
    let prompt_cache = config
        .prompt_cache
        .clone()
        .map(|cache_config| {
            let model_config = runtime.model().config();
            PromptCache::open(
                cache_config,
                &state_fingerprint(&identity, model_config)?,
                &identity.quantization.join("+"),
                LayerKind::for_config(model_config),
            )
            .map(Arc::new)
        })
        .transpose()?;
    Ok((runtime, info, prompt_cache))
}

/// What a cached state must have been produced by.
///
/// The GGUF's own fingerprint covers its tensors, but the numerics that shape
/// a state also come from `config.json` — norm epsilon, RoPE base, the layer
/// pattern, every head dimension — and two revisions of it can name the same
/// weights. Both are folded in, so an entry written under one configuration is
/// never read under another.
fn state_fingerprint(identity: &GgufModelIdentity, config: &Qwen3NextConfig) -> Result<String> {
    let serialized =
        serde_json::to_vec(config).context("failed to describe the model configuration")?;
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in serialized {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(format!("{}+cfg{hash:016x}", identity.layout_fingerprint))
}

fn run_job(
    runtime: &mut QuantizedRuntime<'_>,
    job: Job,
    live: &mut Option<Vec<u32>>,
    cache: Option<&PromptCache>,
    prefix_reuse: bool,
) {
    let Job {
        request,
        events,
        _slot,
    } = job;
    // A client that gave up while queued costs nothing beyond this check.
    if events.is_closed() {
        return;
    }
    let started = Instant::now();
    match generate(runtime, &request, &events, live, cache, prefix_reuse) {
        Ok(Some(completion)) => {
            let seconds = started.elapsed().as_secs_f64();
            tracing::info!(
                prompt_tokens = completion.prompt_tokens,
                completion_tokens = completion.completion_tokens,
                reused_tokens = completion.reused_tokens,
                reuse = completion.reuse.as_str(),
                tool_calls = completion.tool_calls.len(),
                reasoning_tokens = completion.reasoning_tokens.unwrap_or_default(),
                finish = ?completion.finish_reason,
                tokens_per_second = completion.completion_tokens as f64 / seconds.max(f64::EPSILON),
                seconds,
                "request complete"
            );
            let _ = events.send(Event::Done(completion));
        }
        Ok(None) => tracing::debug!("request cancelled by the client"),
        Err(error) => {
            tracing::warn!(error = %format!("{error:#}"), "request failed");
            let _ = events.send(Event::Failed(format!("{error:#}")));
        }
    }
}

/// The state a request starts from, and how many of its tokens that covers.
struct Start {
    reuse: Reuse,
    tokens: usize,
}

/// Put the session into the longest state that is a prefix of `tokens`.
///
/// The live session is free to continue and is checked first; the cache is
/// consulted only for a boundary beyond what the live session already covers,
/// so a hit is never read from disk just to be discarded.
fn prepare_session(
    runtime: &mut QuantizedRuntime<'_>,
    tokens: &[u32],
    live: &mut Option<Vec<u32>>,
    cache: Option<&PromptCache>,
    prefix_reuse: bool,
    options: &GenerationOptions,
) -> Result<Start> {
    if !prefix_reuse {
        runtime.reset();
        *live = None;
        return Ok(Start {
            reuse: Reuse::None,
            tokens: 0,
        });
    }
    // A live session is reusable only when this prompt continues it exactly.
    // Anything else — an edited history, a reply the client did not send back
    // verbatim — has to start from a boundary, because recurrent state cannot
    // be rewound to the point where the two diverge.
    let live_tokens = live
        .as_ref()
        .filter(|history| !history.is_empty() && tokens.len() > history.len())
        .filter(|history| tokens.starts_with(history))
        .map_or(0, Vec::len);
    if let Some(image) = cache.and_then(|cache| cache.lookup(tokens, live_tokens)) {
        let restored = image.position();
        runtime
            .restore_session(&image)
            .context("failed to restore a cached prefix")?;
        if serves_requested_arms(runtime, options) {
            *live = Some(image.tokens);
            return Ok(Start {
                reuse: Reuse::Cache,
                tokens: restored,
            });
        }
        tracing::debug!("a cached prefix cannot serve this request's speculative mode");
    } else if live_tokens > 0 && serves_requested_arms(runtime, options) {
        return Ok(Start {
            reuse: Reuse::Live,
            tokens: live_tokens,
        });
    }
    runtime.reset();
    *live = None;
    Ok(Start {
        reuse: Reuse::None,
        tokens: 0,
    })
}

/// Reuse is only worth having when the session can still decode the way this
/// request asked. The single-arm MTP mode is the one case where it cannot:
/// generation rejects a session whose predictor state has fallen behind, so
/// such a session is dropped here rather than turned into a failed request.
fn serves_requested_arms(runtime: &QuantizedRuntime<'_>, options: &GenerationOptions) -> bool {
    effective_speculative_mode(options) != SpeculativeMode::Mtp || runtime.mtp_arm_ready()
}

/// Evaluate up to the next cache boundary and leave an entry there.
///
/// Returns the new reuse point. Storing costs one extra prefill pass boundary
/// and one state copy; it buys every later request that shares this prefix the
/// whole prefill below the boundary.
fn store_boundary_entry(
    runtime: &mut QuantizedRuntime<'_>,
    tokens: &[u32],
    from: usize,
    stable: Option<usize>,
    cache: &PromptCache,
    options: &GenerationOptions,
) -> Result<usize> {
    let Some(boundary) = cache.store_boundary(tokens, from, stable) else {
        return Ok(from);
    };
    let wants_mtp =
        effective_speculative_mode(options).allows_mtp() && runtime.model().mtp().is_some();
    runtime.prefill_tokens(&tokens[from..boundary], wants_mtp)?;
    let image = runtime.session_image(tokens[..boundary].to_vec())?;
    // An entry is written once and then reused forever, so it must not
    // capture a session that had already lost its predictor state — a halted
    // turn earlier in this session, for instance. Skipping leaves the key free
    // for a later request that can fill it properly.
    if wants_mtp && image.mtp.is_none() {
        tracing::debug!(
            boundary,
            "not storing a prompt cache entry: this session cannot supply the MTP predictor's state"
        );
        return Ok(boundary);
    }
    let bytes = image.bytes();
    if cache.store(image) {
        tracing::debug!(
            boundary,
            mib = bytes as f64 / (1024. * 1024.),
            "queued a prompt cache write"
        );
    }
    Ok(boundary)
}

/// Run one request to completion. `Ok(None)` means the client disconnected.
fn generate(
    runtime: &mut QuantizedRuntime<'_>,
    request: &JobRequest,
    events: &UnboundedSender<Event>,
    live: &mut Option<Vec<u32>>,
    cache: Option<&PromptCache>,
    prefix_reuse: bool,
) -> Result<Option<Completion>> {
    let JobRequest {
        prompt,
        stable_prefix,
        options,
        stop_strings,
        tools_enabled,
        thinking_open,
    } = request;
    let tokenizer = runtime.tokenizer().clone();
    let tokens = tokenizer
        .encode(prompt, options.add_special_tokens)
        .context("failed to encode the prompt")?;
    anyhow::ensure!(!tokens.is_empty(), "the prompt encoded to no tokens");
    let prompt_tokens = tokens.len();

    // From here on the session is being mutated, so any failure has to leave
    // the live history unset rather than describing a state that never came
    // to be.
    // How much of this prompt a later request is expected to repeat. Storing
    // an entry above that point would key it on the tokens of one specific
    // turn, which nothing else ever sends.
    let stable = stable_prefix
        .as_deref()
        .and_then(|prefix| stable_token_count(&tokenizer, prefix, &tokens));

    let start = prepare_session(runtime, &tokens, live, cache, prefix_reuse, options)
        .inspect_err(|_| *live = None)?;
    let reused = start.tokens;
    let mut prefilled_to = reused;
    if let Some(cache) = cache {
        match store_boundary_entry(runtime, &tokens, reused, stable, cache, options) {
            Ok(boundary) => {
                if boundary > reused {
                    *live = Some(tokens[..boundary].to_vec());
                    prefilled_to = boundary;
                }
            }
            Err(error) => {
                *live = None;
                return Err(error.context("failed to prefill up to a cache boundary"));
            }
        }
    }

    // The prompt opened the block, so the tag is not in the model's output;
    // sending it first is what makes the response a complete `<think>` block.
    // It is not model output, so nothing downstream matches against it.
    if *thinking_open && events.send(Event::Delta("<think>\n".to_owned())).is_err() {
        return Ok(None);
    }
    let eos = runtime.model().config().eos_token_id.clone();
    let mut decode = tokenizer.decode_stream(true);
    let mut stops = StopBuffer::new(stop_strings);
    // Everything the model wrote, including tool-call markup that never
    // reaches the client as text.
    let mut raw = String::new();
    // When tools are offered, `<tool_call>` ends the visible answer: the rest
    // of the turn is markup, reported as calls instead of shown.
    let mut tools = tools_enabled.then(|| StopBuffer::new(&[tool_calls::CALL_MARKER.to_owned()]));
    let mut calling = false;
    let mut completion_tokens = 0usize;
    let mut generated = Vec::new();
    let mut halted = None;
    let mut reasoning = ThinkingCounter::new(*thinking_open);
    let result = runtime.generate_tokens_with_callback(&tokens[prefilled_to..], options, |token| {
        completion_tokens += 1;
        generated.push(token);
        let Some(chunk) = decode(token)? else {
            return Ok(());
        };
        raw.push_str(&chunk);
        reasoning.observe(&raw);
        let visible = match (&mut tools, calling) {
            (Some(_), true) => String::new(),
            (Some(guard), false) => {
                let step = guard.push(&chunk);
                calling = step.matched;
                step.emit
            }
            (None, _) => chunk,
        };
        // A closed call ends the turn. Without this the model runs to its
        // token budget emitting call after call, which is both wrong and, on
        // CPU, expensive.
        if calling
            && raw
                .rfind(tool_calls::CALL_MARKER)
                .is_some_and(|start| raw[start..].contains(tool_calls::CALL_END_MARKER))
        {
            halted = Some(Halt::StopString);
            bail!(Halt::StopString);
        }
        if visible.is_empty() {
            return Ok(());
        }
        let step = stops.push(&visible);
        if !step.emit.is_empty() && events.send(Event::Delta(step.emit)).is_err() {
            halted = Some(Halt::Disconnected);
            bail!(Halt::Disconnected);
        }
        if step.matched {
            halted = Some(Halt::StopString);
            bail!(Halt::StopString);
        }
        Ok(())
    });
    // Every committed token reaches the callback exactly once, and a halt
    // leaves the session holding the last of them as its pending token, so the
    // prompt plus what was emitted is what the state now represents.
    let mut session = tokens;
    session.extend_from_slice(&generated);
    *live = Some(session);
    let parsed = tools_enabled
        .then(|| tool_calls::parse(&raw).1)
        .unwrap_or_default();
    let completion = |finish_reason| Completion {
        finish_reason,
        prompt_tokens,
        completion_tokens,
        reused_tokens: reused,
        reuse: start.reuse,
        tool_calls: parsed.clone(),
        reasoning_tokens: thinking_open.then_some(reasoning.tokens),
    };
    match result {
        Ok(result) => {
            let stopped_on_token = result
                .generated_token_ids
                .last()
                .is_some_and(|token| is_stop_token(eos.as_ref(), options, *token));
            // Text held back as a possible marker prefix is just text once the
            // turn is over without one.
            let held = tools
                .as_mut()
                .filter(|_| !calling)
                .map_or(String::new(), StopBuffer::flush);
            if !held.is_empty() {
                let step = stops.push(&held);
                if !step.emit.is_empty() && events.send(Event::Delta(step.emit)).is_err() {
                    return Ok(None);
                }
            }
            let tail = stops.flush();
            if !tail.is_empty() && events.send(Event::Delta(tail)).is_err() {
                return Ok(None);
            }
            let finish_reason = if !parsed.is_empty() {
                FinishReason::ToolCalls
            } else if !stopped_on_token && completion_tokens >= options.max_new_tokens {
                FinishReason::Length
            } else {
                FinishReason::Stop
            };
            Ok(Some(completion(finish_reason)))
        }
        Err(error) => match halted {
            Some(Halt::Disconnected) => Ok(None),
            // A closed tool call halts the turn the same way a stop string
            // does, but the client needs to be told which of the two it was.
            Some(Halt::StopString) => Ok(Some(completion(if parsed.is_empty() {
                FinishReason::Stop
            } else {
                FinishReason::ToolCalls
            }))),
            None => {
                // A failure inside a pass can leave the session anywhere; the
                // next request rebuilds from a boundary rather than trusting it.
                *live = None;
                Err(error)
            }
        },
    }
}

/// How many of `tokens` the caller's stable prefix covers.
///
/// The prefix is re-encoded rather than trusted: it is only a hint, and one
/// that does not actually tokenise to a prefix of this prompt would move the
/// stored boundary somewhere it does not belong.
fn stable_token_count(
    tokenizer: &ModelTokenizer,
    stable_prefix: &str,
    tokens: &[u32],
) -> Option<usize> {
    let prefix = tokenizer.encode(stable_prefix, false).ok()?;
    tokens.starts_with(&prefix).then_some(prefix.len())
}

fn is_stop_token(eos: Option<&EosTokenId>, options: &GenerationOptions, token: u32) -> bool {
    eos.is_some_and(|ids| ids.contains(token)) || options.stop_tokens.contains(&token)
}

/// Speculative decoding verifies against the target's argmax, so it is defined
/// only for greedy decoding. A request that asks to sample gets plain decode
/// rather than an error, because most OpenAI clients send a non-zero
/// temperature by default and would otherwise never reach the model.
pub fn disable_speculation_for_sampling(options: &mut GenerationOptions) {
    if options.sampling.temperature > 0. && effective_speculative_mode(options).is_speculative() {
        options.speculative_mode = SpeculativeMode::Off;
        options.speculative_mtp_draft_tokens = 0;
        options.speculative_ngram_draft_tokens = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed decoded chunks the way the token callback does.
    fn count(open: bool, chunks: &[&str]) -> usize {
        let mut counter = ThinkingCounter::new(open);
        let mut raw = String::new();
        for chunk in chunks {
            raw.push_str(chunk);
            counter.observe(&raw);
        }
        counter.tokens
    }

    #[test]
    fn counts_tokens_until_the_block_closes() {
        assert_eq!(count(true, &["think", "ing", "</think>", "answer", "!"]), 3);
        // Never opened: nothing is reasoning.
        assert_eq!(count(false, &["answer", "!"]), 0);
        // Never closed: the whole turn was thinking.
        assert_eq!(count(true, &["still", " going"]), 2);
    }

    #[test]
    fn finds_a_marker_split_across_chunks() {
        assert_eq!(count(true, &["a", "</thi", "nk>", "b"]), 3);
    }

    #[test]
    fn does_not_split_a_multi_byte_character() {
        // The scan window lands inside these characters unless it is moved to
        // a boundary, which would panic rather than miscount.
        assert_eq!(
            count(true, &["日本語です", "がんばって", "</think>", "答え"]),
            3
        );
        assert_eq!(count(true, &["🙂🙂🙂", "🙂", "</think>"]), 3);
    }
}
