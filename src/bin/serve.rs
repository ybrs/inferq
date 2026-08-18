//! OpenAI-compatible HTTP server over the quantized runtime.
//!
//! One model, one inference slot: requests queue and are served in order. See
//! `docs/openai-server.md` for the supported request surface.

use std::{fs, net::IpAddr, path::PathBuf, sync::Arc};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use qwen_engine::{
    GenerationOptions, PromptCacheConfig, SpeculativeMode,
    prompt_cache::{DEFAULT_BLOCK_TOKENS, DEFAULT_BUDGET_MIB, DEFAULT_MIN_TOKENS},
    runtime::{DEFAULT_NGRAM_MIN_MATCH, PolicyTuning},
    sampling::SamplingConfig,
    server::{EngineConfig, ServerState, Warmup, engine, serve},
    speculative::{
        DEFAULT_BACKOFF_CAP, DEFAULT_BACKOFF_TOKENS, DEFAULT_EWMA_ALPHA, DEFAULT_MTP_DEPTH_CAP,
        DEFAULT_MTP_DEPTH_FLOOR, DEFAULT_MTP_DEPTH_START, DEFAULT_MTP_SUSPEND_BELOW,
        DEFAULT_NGRAM_DRAFT_CAP, DEFAULT_NGRAM_DRAFT_FLOOR, DEFAULT_NGRAM_SUSPEND_BELOW,
    },
};

/// HTTP worker threads. The async side only moves bytes; all model work
/// happens on the engine thread, and taking more would oversubscribe the CPU
/// the model is using.
const HTTP_WORKER_THREADS: usize = 2;

#[derive(Debug, Parser)]
#[command(about = "OpenAI-compatible HTTP server for Qwen3-Next-family GGUF models")]
struct Args {
    /// Supported Qwen3-Next or Qwen3.5/3.6 MoE GGUF file.
    #[arg(long)]
    model: PathBuf,
    /// Hugging Face model directory supplying config.json and tokenizer.json.
    #[arg(long)]
    tokenizer_model: PathBuf,
    /// Address to bind. The default accepts local clients only.
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Require this key on every /v1 request, as `Authorization: Bearer <key>`.
    #[arg(long, value_name = "KEY", conflicts_with = "api_key_file")]
    api_key: Option<String>,
    /// Read the required API key from a file, so it stays out of the process
    /// list. Leading and trailing whitespace is trimmed.
    #[arg(long, value_name = "PATH")]
    api_key_file: Option<PathBuf>,
    /// Name this server reports and accepts. Defaults to the GGUF file stem.
    #[arg(long, value_name = "NAME")]
    served_model_name: Option<String>,
    /// Requests allowed to be queued or running before new ones get 503.
    #[arg(long, value_name = "N", default_value_t = 8)]
    max_queue: usize,
    /// Persist prefix state here, so a prompt an earlier run already prefilled
    /// starts from a file read instead of a full prefill. Entries contain the
    /// token ids of cached prompts; the directory is created owner-only.
    #[arg(long, value_name = "PATH")]
    prompt_cache_dir: Option<PathBuf>,
    /// Disk the prompt cache may use before evicting least-recently-used
    /// entries.
    #[arg(long, value_name = "MIB", default_value_t = DEFAULT_BUDGET_MIB)]
    prompt_cache_mib: u64,
    /// Token boundary entries are stored at. A prompt that diverges from a
    /// cached one re-prefills at most this many extra tokens.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_BLOCK_TOKENS)]
    prompt_cache_block: usize,
    /// Shortest prefix worth an entry.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MIN_TOKENS)]
    prompt_cache_min_tokens: usize,
    /// Prefill every prompt in full, reusing neither the live session nor the
    /// prompt cache. This is what reuse is measured against.
    #[arg(long)]
    no_prefix_reuse: bool,
    /// Output length for requests that do not set `max_tokens`.
    #[arg(long, value_name = "N", default_value_t = 512)]
    max_new_tokens: usize,
    /// Sampling temperature for requests that do not set one. Zero is greedy
    /// decoding, which is also what speculative decoding requires.
    #[arg(long, value_name = "F", default_value_t = 0.)]
    temperature: f32,
    #[arg(long, value_name = "F")]
    top_p: Option<f32>,
    #[arg(long, value_name = "N")]
    top_k: Option<usize>,
    #[arg(long, value_name = "F")]
    min_p: Option<f32>,
    #[arg(long, value_name = "N", default_value_t = 0)]
    seed: u64,
    /// Render assistant turns with the thinking block already closed, unless a
    /// request asks otherwise through `chat_template_kwargs.enable_thinking`.
    #[arg(long)]
    no_thinking: bool,
    /// Force-close Qwen's thinking block after N committed generated tokens.
    #[arg(long, value_name = "N")]
    thinking_budget: Option<usize>,
    /// Retain recently used expert matrices in-process, bounded in MiB.
    #[arg(long, default_value_t = 0)]
    expert_cache_mib: usize,
    /// Warm the hottest layer-qualified experts from a prior census.
    #[arg(long)]
    warmup_census: Option<PathBuf>,
    /// Number of hottest experts to warm in each observed layer.
    #[arg(long, default_value_t = 10)]
    warmup_experts_per_layer: usize,
    /// Warm every fused expert tensor, pinning it when an expert cache is
    /// configured.
    #[arg(long, conflicts_with = "warmup_census")]
    warmup_all_experts: bool,
    /// Which draft sources decoding may use. `auto` is the unified policy.
    /// Ignored for a request that samples, since verification is defined
    /// against the target's argmax.
    #[arg(long, value_enum, default_value_t = Speculative::Auto)]
    speculative: Speculative,
    /// Skip MTP proposals whose raw top-1/top-2 logit margin is below this value.
    #[arg(long, value_name = "MARGIN")]
    speculative_mtp_min_margin: Option<f32>,
    /// Shortest token suffix the n-gram drafter will match on.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_NGRAM_MIN_MATCH)]
    ngram_min_match: usize,
    /// Longest draft the n-gram arm may propose. Its controller starts here.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_NGRAM_DRAFT_CAP)]
    ngram_draft_cap: usize,
    /// Shortest draft the n-gram controller will shrink to.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_NGRAM_DRAFT_FLOOR)]
    ngram_draft_floor: usize,
    /// Deepest chained MTP draft.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MTP_DEPTH_CAP)]
    mtp_depth_cap: usize,
    /// Shallowest depth the MTP controller will shrink to.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MTP_DEPTH_FLOOR)]
    mtp_depth_floor: usize,
    /// Depth the MTP controller starts each request at.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MTP_DEPTH_START)]
    mtp_depth_start: usize,
    /// Acceptance EWMA below which the n-gram arm suspends itself.
    #[arg(long, value_name = "F", default_value_t = DEFAULT_NGRAM_SUSPEND_BELOW)]
    ngram_suspend_below: f64,
    /// Acceptance EWMA below which the MTP arm suspends itself.
    #[arg(long, value_name = "F", default_value_t = DEFAULT_MTP_SUSPEND_BELOW)]
    mtp_suspend_below: f64,
    /// Weight of the newest proposal in each arm's acceptance EWMA.
    #[arg(long, value_name = "F", default_value_t = DEFAULT_EWMA_ALPHA)]
    ewma_alpha: f64,
    /// Committed tokens a first suspension lasts.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_BACKOFF_TOKENS)]
    backoff_tokens: usize,
    /// Longest suspension repeated failed probes can reach.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_BACKOFF_CAP)]
    backoff_cap: usize,
    /// Disable continuing an accepted n-gram draft's source span.
    #[arg(long)]
    no_span_continuation: bool,
    /// Disable adaptive draft length and depth.
    #[arg(long)]
    no_adaptive_length: bool,
    /// Disable EWMA backoff, so neither arm ever suspends.
    #[arg(long)]
    no_ewma_backoff: bool,
    /// Resynchronise the MTP block after every committing pass.
    #[arg(long)]
    eager_mtp_resync: bool,
    /// How verification snapshots copy recurrent state.
    #[arg(long, value_enum, default_value_t = SnapshotCopy::Streaming)]
    snapshot_copy: SnapshotCopy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Speculative {
    /// Ordinary autoregressive decoding.
    Off,
    /// The unified policy: both arms, each under its own controller.
    Auto,
    /// The n-gram arm only.
    Ngram,
    /// The MTP arm only.
    Mtp,
}

impl From<Speculative> for SpeculativeMode {
    fn from(value: Speculative) -> Self {
        match value {
            Speculative::Off => Self::Off,
            Speculative::Auto => Self::Auto,
            Speculative::Ngram => Self::Ngram,
            Speculative::Mtp => Self::Mtp,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum SnapshotCopy {
    /// Non-temporal stores, skipping read-for-ownership and L3 pollution.
    Streaming,
    /// Ordinary `copy_from_slice`.
    Plain,
}

impl Args {
    fn api_key(&self) -> Result<Option<String>> {
        if let Some(path) = &self.api_key_file {
            let key = fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.display()))?
                .trim()
                .to_owned();
            ensure!(!key.is_empty(), "{} is empty", path.display());
            return Ok(Some(key));
        }
        Ok(self.api_key.clone())
    }

    fn prompt_cache(&self) -> Option<PromptCacheConfig> {
        self.prompt_cache_dir.as_ref().map(|dir| PromptCacheConfig {
            dir: dir.clone(),
            budget_bytes: self.prompt_cache_mib.saturating_mul(1024 * 1024),
            block_tokens: self.prompt_cache_block,
            min_tokens: self.prompt_cache_min_tokens,
        })
    }

    fn warmup(&self) -> Warmup {
        if self.warmup_all_experts {
            Warmup::AllExperts
        } else if let Some(path) = &self.warmup_census {
            Warmup::Census {
                path: path.clone(),
                experts_per_layer: self.warmup_experts_per_layer,
            }
        } else {
            Warmup::None
        }
    }

    /// Generation settings a request inherits when it does not override them.
    fn defaults(&self) -> GenerationOptions {
        let mode = SpeculativeMode::from(self.speculative);
        let (ngram_cap, mtp_cap) = if mode.is_speculative() {
            (self.ngram_draft_cap, self.mtp_depth_cap)
        } else {
            (0, 0)
        };
        GenerationOptions {
            max_new_tokens: self.max_new_tokens,
            sampling: SamplingConfig {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                min_p: self.min_p,
                seed: self.seed,
            },
            speculative_mode: mode,
            policy: PolicyTuning {
                ngram_draft_floor: self.ngram_draft_floor.min(ngram_cap.max(1)),
                mtp_depth_floor: self.mtp_depth_floor.min(mtp_cap.max(1)),
                mtp_depth_start: self.mtp_depth_start,
                ngram_suspend_below: self.ngram_suspend_below,
                mtp_suspend_below: self.mtp_suspend_below,
                ewma_alpha: self.ewma_alpha,
                backoff_tokens: self.backoff_tokens,
                backoff_cap: self.backoff_cap,
                span_continuation: !self.no_span_continuation,
                adaptive_length: !self.no_adaptive_length,
                ewma_backoff: !self.no_ewma_backoff,
                eager_mtp_resync: self.eager_mtp_resync,
            },
            speculative_mtp_draft_tokens: mtp_cap,
            speculative_mtp_min_margin: self.speculative_mtp_min_margin,
            speculative_ngram_draft_tokens: ngram_cap,
            ngram_min_match: self.ngram_min_match,
            thinking_budget: self.thinking_budget,
            ..GenerationOptions::default()
        }
    }
}

fn main() -> Result<()> {
    qwen_engine::threading::init();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                // The model's own spans are per layer and per pass, so only
                // the server's request log is on by default. `RUST_LOG=info`
                // turns the rest back on.
                .unwrap_or_else(|_| {
                    tracing_subscriber::EnvFilter::new(
                        "warn,qwen_engine::server=info,qwen_engine::prompt_cache=info",
                    )
                }),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    ensure!(
        args.max_new_tokens > 0,
        "--max-new-tokens must be at least one"
    );
    ensure!(args.max_queue > 0, "--max-queue must be at least one");
    ensure!(
        args.prompt_cache_dir.is_none() || !args.no_prefix_reuse,
        "--prompt-cache-dir and --no-prefix-reuse contradict each other"
    );
    ensure!(
        args.ngram_draft_cap > 0 && args.mtp_depth_cap > 0,
        "draft caps must be at least one"
    );
    let api_key = args.api_key()?;
    if api_key.is_none() && !args.host.is_loopback() {
        eprintln!(
            "inferq: warning: binding {} without --api-key; anyone who can reach this port can use the model",
            args.host
        );
    }
    let expert_cache_bytes = args
        .expert_cache_mib
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("--expert-cache-mib is too large"))?;

    let config = EngineConfig {
        model: args.model.clone(),
        tokenizer_model: args.tokenizer_model.clone(),
        served_model_name: args.served_model_name.clone(),
        expert_cache_bytes,
        warmup: args.warmup(),
        snapshot_nontemporal: args.snapshot_copy == SnapshotCopy::Streaming,
        defaults: args.defaults(),
        max_queue: args.max_queue,
        prompt_cache: args.prompt_cache(),
        prefix_reuse: !args.no_prefix_reuse,
    };
    // Loading happens on the engine thread and blocks until the model is
    // ready, so a failed checkpoint is reported before anything is bound.
    let handle = engine::start(config)?;
    let state = Arc::new(ServerState {
        engine: handle,
        api_key,
        default_enable_thinking: !args.no_thinking,
    });
    let address = (args.host, args.port).into();
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(HTTP_WORKER_THREADS)
        .enable_all()
        .build()
        .context("failed to start the HTTP runtime")?
        .block_on(serve(address, state))
}
