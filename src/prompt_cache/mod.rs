//! A persistent cache of sequence state, keyed by token prefix.
//!
//! An agent opens every task with the same long preamble — system prompt, tool
//! definitions, project context — and prefill is the expensive half of this
//! engine: attention scores each new token against every earlier one, so the
//! cost of a preamble grows with its square. Caching the state that preamble
//! produces turns the second and every later run of it into a file read.
//!
//! What makes that possible is that a sequence's state is fully described by
//! plain floats: KV rows for the full-attention layers, conv and recurrent
//! state for the linear ones. What makes it necessary to key on a *prefix* is
//! that DeltaNet recurrence is destructive — state can be extended but never
//! rewound — so reuse has to come from an image captured at a boundary rather
//! than from trimming a longer state back.
//!
//! Entries are written at token boundaries that are multiples of
//! [`PromptCacheConfig::block_tokens`], which is what lets two requests that
//! share a preamble but diverge later land on the same key. The cost of that
//! quantisation is at most one block of re-prefilled tokens.
//!
//! Entries hold the token ids of the prefix they describe, so a cache
//! directory contains recoverable prompt text. It is created with owner-only
//! permissions and is never enabled unless a directory is configured.

pub mod format;

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, SyncSender, TrySendError},
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};

use crate::{SessionImage, ngram::fnv1a_tokens};

pub use format::LayerKind;

/// Default token boundary entries are stored at. A miss costs at most this
/// many re-prefilled tokens, and a smaller block means more distinct entries
/// for the same conversation.
pub const DEFAULT_BLOCK_TOKENS: usize = 256;

/// Default shortest prefix worth an entry. Below this the fixed recurrent
/// state dominates the file and the prefill it saves is not worth the write.
pub const DEFAULT_MIN_TOKENS: usize = 512;

/// Default disk budget before least-recently-used entries are removed.
pub const DEFAULT_BUDGET_MIB: u64 = 20 * 1024;

#[derive(Debug, Clone)]
pub struct PromptCacheConfig {
    pub dir: PathBuf,
    pub budget_bytes: u64,
    pub block_tokens: usize,
    pub min_tokens: usize,
}

impl PromptCacheConfig {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            budget_bytes: DEFAULT_BUDGET_MIB * 1024 * 1024,
            block_tokens: DEFAULT_BLOCK_TOKENS,
            min_tokens: DEFAULT_MIN_TOKENS,
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.block_tokens > 0,
            "the prompt cache block size must be at least one token"
        );
        ensure!(
            self.min_tokens >= self.block_tokens,
            "the shortest cached prefix ({}) must be at least one block ({})",
            self.min_tokens,
            self.block_tokens
        );
        Ok(())
    }
}

/// Boundaries an entry may be stored at or looked up from, longest first.
///
/// A boundary is strictly below `len` because a restored session still has to
/// evaluate at least one token to produce logits.
fn boundaries(len: usize, block: usize, min: usize) -> impl Iterator<Item = usize> {
    let highest = len
        .saturating_sub(1)
        .checked_div(block)
        .map_or(0, |blocks| blocks * block);
    (0..=highest)
        .rev()
        .step_by(block.max(1))
        .filter(move |boundary| *boundary >= min && *boundary > 0)
}

/// The boundary this request should leave behind, if any.
///
/// `stable` is how much of the prompt the caller expects a later request to
/// repeat — for a conversation, everything before its final message. Storing
/// above that point produces an entry keyed on tokens no other request will
/// send, so the boundary is capped there when the caller knows it.
///
/// Nothing is stored when the reused prefix already reaches the highest
/// usable boundary: the entry that would be written is the one just read.
fn store_boundary(
    len: usize,
    reused: usize,
    stable: Option<usize>,
    block: usize,
    min: usize,
) -> Option<usize> {
    let ceiling = stable.map_or(len, |stable| stable.min(len));
    boundaries(len, block, min)
        .filter(|boundary| *boundary <= ceiling)
        .find(|boundary| *boundary > reused)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EntryKey {
    tokens: usize,
    hash: u64,
}

impl EntryKey {
    fn of(tokens: &[u32]) -> Self {
        Self {
            tokens: tokens.len(),
            hash: fnv1a_tokens(tokens),
        }
    }

    fn file_name(&self, fingerprint: &str) -> String {
        format!(
            "{}-{:08}-{:016x}.{}",
            short_fingerprint(fingerprint),
            self.tokens,
            self.hash,
            format::EXTENSION
        )
    }
}

/// The identifying half of a layout fingerprint, for use in a file name.
fn short_fingerprint(fingerprint: &str) -> &str {
    fingerprint
        .rsplit_once(':')
        .map_or(fingerprint, |(_, hash)| hash)
}

#[derive(Debug, Clone)]
struct EntryMeta {
    path: PathBuf,
    bytes: u64,
    used: SystemTime,
}

#[derive(Debug, Default)]
struct Index {
    /// Entries belonging to the loaded checkpoint.
    entries: HashMap<EntryKey, EntryMeta>,
    /// Entries left by another checkpoint. Unusable, but they occupy the same
    /// budget and are the first thing eviction reclaims.
    foreign: Vec<EntryMeta>,
    bytes: u64,
}

impl Index {
    fn insert(&mut self, key: EntryKey, meta: EntryMeta) {
        if let Some(previous) = self.entries.insert(key, meta.clone()) {
            self.bytes = self.bytes.saturating_sub(previous.bytes);
        }
        self.bytes = self.bytes.saturating_add(meta.bytes);
    }

    fn remove(&mut self, key: &EntryKey) -> Option<EntryMeta> {
        let meta = self.entries.remove(key)?;
        self.bytes = self.bytes.saturating_sub(meta.bytes);
        Some(meta)
    }
}

#[derive(Debug, Default)]
struct Stats {
    hits: AtomicU64,
    misses: AtomicU64,
    reused_tokens: AtomicU64,
    writes: AtomicU64,
    write_skips: AtomicU64,
    evictions: AtomicU64,
    failures: AtomicU64,
}

/// What the cache reports about itself, for `/health` and the logs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptCacheStats {
    pub entries: usize,
    pub bytes: u64,
    pub budget_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub reused_tokens: u64,
    pub writes: u64,
    pub write_skips: u64,
    pub evictions: u64,
    pub failures: u64,
}

struct Shared {
    /// Writes queued or in flight, so a shutdown can wait for them.
    outstanding: AtomicUsize,
    config: PromptCacheConfig,
    fingerprint: String,
    quantization: String,
    layer_kinds: Vec<LayerKind>,
    index: Mutex<Index>,
    stats: Stats,
}

impl Shared {
    fn path_for(&self, key: &EntryKey) -> PathBuf {
        self.config.dir.join(key.file_name(&self.fingerprint))
    }

    /// Forget an entry whose file turned out to be unusable.
    fn discard(&self, key: &EntryKey, path: &Path, reason: &str) {
        tracing::warn!(path = %path.display(), reason, "discarding a prompt cache entry");
        self.stats.failures.fetch_add(1, Ordering::Relaxed);
        let _ = fs::remove_file(path);
        if let Ok(mut index) = self.index.lock() {
            index.remove(key);
        }
    }

    fn write(&self, image: &SessionImage) -> Result<()> {
        let key = EntryKey::of(&image.tokens);
        let path = self.path_for(&key);
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs());
        let started = std::time::Instant::now();
        let bytes = format::write(&path, image, &self.fingerprint, &self.quantization, created)?;
        {
            let mut index = self.index.lock().map_err(|_| {
                anyhow::anyhow!("the prompt cache index is poisoned; a writer panicked")
            })?;
            index.insert(
                key,
                EntryMeta {
                    path: path.clone(),
                    bytes,
                    used: SystemTime::now(),
                },
            );
        }
        self.stats.writes.fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            tokens = image.tokens.len(),
            mib = bytes as f64 / (1024. * 1024.),
            seconds = started.elapsed().as_secs_f64(),
            "stored a prompt cache entry"
        );
        self.evict(Some(key))?;
        Ok(())
    }

    /// Remove least-recently-used entries until the directory fits the budget.
    ///
    /// `keep` is the entry that was just written: evicting it would discard
    /// the prefill this request paid for and leave the cache no warmer.
    fn evict(&self, keep: Option<EntryKey>) -> Result<()> {
        let mut index = self
            .index
            .lock()
            .map_err(|_| anyhow::anyhow!("the prompt cache index is poisoned"))?;
        if index.bytes <= self.config.budget_bytes {
            return Ok(());
        }
        // Foreign entries first: they cost the same and can never be read.
        let mut candidates: Vec<(SystemTime, Option<EntryKey>, PathBuf, u64)> = index
            .foreign
            .iter()
            .map(|meta| (UNIX_EPOCH, None, meta.path.clone(), meta.bytes))
            .chain(
                index
                    .entries
                    .iter()
                    .filter(|(key, _)| Some(**key) != keep)
                    .map(|(key, meta)| (meta.used, Some(*key), meta.path.clone(), meta.bytes)),
            )
            .collect();
        candidates.sort_by_key(|(used, _, path, _)| (*used, path.clone()));
        for (_, key, path, bytes) in candidates {
            if index.bytes <= self.config.budget_bytes {
                break;
            }
            if let Err(error) = fs::remove_file(&path) {
                tracing::warn!(path = %path.display(), error = %error, "failed to evict an entry");
                continue;
            }
            match key {
                Some(key) => {
                    index.remove(&key);
                }
                None => {
                    index.foreign.retain(|meta| meta.path != path);
                    index.bytes = index.bytes.saturating_sub(bytes);
                }
            }
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            tracing::info!(path = %path.display(), "evicted a prompt cache entry");
        }
        Ok(())
    }
}

pub struct PromptCache {
    shared: Arc<Shared>,
    writes: SyncSender<SessionImage>,
}

impl PromptCache {
    /// Open (or create) a cache directory for one checkpoint.
    ///
    /// Entries written by another checkpoint or another format version stay on
    /// disk and count against the budget, but are never read.
    pub fn open(
        config: PromptCacheConfig,
        fingerprint: &str,
        quantization: &str,
        layer_kinds: Vec<LayerKind>,
    ) -> Result<Self> {
        config.validate()?;
        create_private_dir(&config.dir)?;
        let index = scan(&config.dir, fingerprint)?;
        let shared = Arc::new(Shared {
            outstanding: AtomicUsize::new(0),
            config,
            fingerprint: fingerprint.to_owned(),
            quantization: quantization.to_owned(),
            layer_kinds,
            index: Mutex::new(index),
            stats: Stats::default(),
        });
        shared.evict(None)?;
        // One slot: a write already in flight means the next one is skipped
        // rather than queued, because a queued image is a whole state copy
        // held in memory and the request it came from has already finished.
        let (writes, jobs) = mpsc::sync_channel::<SessionImage>(1);
        let writer = Arc::clone(&shared);
        thread::Builder::new()
            .name("inferq-prompt-cache".to_owned())
            .spawn(move || {
                while let Ok(image) = jobs.recv() {
                    if let Err(error) = writer.write(&image) {
                        tracing::warn!(
                            error = %format!("{error:#}"),
                            "failed to store a prompt cache entry"
                        );
                        writer.stats.failures.fetch_add(1, Ordering::Relaxed);
                    }
                    writer.outstanding.fetch_sub(1, Ordering::AcqRel);
                }
            })
            .context("failed to spawn the prompt cache writer thread")?;
        Ok(Self { shared, writes })
    }

    pub fn block_tokens(&self) -> usize {
        self.shared.config.block_tokens
    }

    pub fn min_tokens(&self) -> usize {
        self.shared.config.min_tokens
    }

    /// The longest cached prefix of `tokens` above `above`, if one is on disk.
    ///
    /// `above` is what the caller can already continue from without reading
    /// anything — a live session's own history — so a shorter entry is never
    /// loaded just to throw the longer prefix away.
    ///
    /// The stored token ids are compared exactly before any state is read, so
    /// a hash collision costs one header read rather than a wrong answer.
    pub fn lookup(&self, tokens: &[u32], above: usize) -> Option<SessionImage> {
        let config = &self.shared.config;
        for boundary in boundaries(tokens.len(), config.block_tokens, config.min_tokens) {
            if boundary <= above {
                break;
            }
            let prefix = &tokens[..boundary];
            let key = EntryKey::of(prefix);
            let Some(path) = self.entry_path(&key) else {
                continue;
            };
            match format::read_header(&path, &self.shared.fingerprint) {
                Ok(header) if header.tokens == prefix => {}
                Ok(_) => {
                    self.shared
                        .discard(&key, &path, "stored tokens differ from the request");
                    continue;
                }
                Err(error) => {
                    self.shared.discard(&key, &path, &format!("{error:#}"));
                    continue;
                }
            }
            match format::read(&path, &self.shared.fingerprint, &self.shared.layer_kinds) {
                Ok(image) => {
                    self.touch(&key, &path);
                    self.shared.stats.hits.fetch_add(1, Ordering::Relaxed);
                    self.shared
                        .stats
                        .reused_tokens
                        .fetch_add(boundary as u64, Ordering::Relaxed);
                    return Some(image);
                }
                Err(error) => self.shared.discard(&key, &path, &format!("{error:#}")),
            }
        }
        self.shared.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Where this request should leave an entry, given what it already reused
    /// and how much of its prompt is expected to recur (`stable`).
    ///
    /// `None` means there is nothing new worth storing: the prompt is too
    /// short, the reused prefix already covers the highest usable boundary, or
    /// that boundary is already on disk.
    pub fn store_boundary(
        &self,
        tokens: &[u32],
        reused: usize,
        stable: Option<usize>,
    ) -> Option<usize> {
        let config = &self.shared.config;
        let boundary = store_boundary(
            tokens.len(),
            reused,
            stable,
            config.block_tokens,
            config.min_tokens,
        )?;
        let key = EntryKey::of(&tokens[..boundary]);
        self.entry_path(&key).is_none().then_some(boundary)
    }

    /// Hand an image to the writer thread. Returns false when a write is
    /// already in flight and this one was dropped.
    pub fn store(&self, image: SessionImage) -> bool {
        self.shared.outstanding.fetch_add(1, Ordering::AcqRel);
        match self.writes.try_send(image) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => {
                self.shared.outstanding.fetch_sub(1, Ordering::AcqRel);
                self.shared
                    .stats
                    .write_skips
                    .fetch_add(1, Ordering::Relaxed);
                tracing::info!("skipped a prompt cache write: the writer is still busy");
                false
            }
            Err(TrySendError::Disconnected(_)) => {
                self.shared
                    .stats
                    .write_skips
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!("the prompt cache writer thread is gone");
                false
            }
        }
    }

    /// Wait for queued writes to land. Returns false if the timeout expired
    /// with work still outstanding.
    ///
    /// A request that has just paid for a long prefill has its entry in the
    /// writer's hands, not on disk; without this, stopping the server throws
    /// that work away.
    pub fn wait_for_writes(&self, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        while self.shared.outstanding.load(Ordering::Acquire) > 0 {
            if std::time::Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(20));
        }
        true
    }

    pub fn stats(&self) -> PromptCacheStats {
        let stats = &self.shared.stats;
        let (entries, bytes) = self
            .shared
            .index
            .lock()
            .map(|index| (index.entries.len(), index.bytes))
            .unwrap_or_default();
        PromptCacheStats {
            entries,
            bytes,
            budget_bytes: self.shared.config.budget_bytes,
            hits: stats.hits.load(Ordering::Relaxed),
            misses: stats.misses.load(Ordering::Relaxed),
            reused_tokens: stats.reused_tokens.load(Ordering::Relaxed),
            writes: stats.writes.load(Ordering::Relaxed),
            write_skips: stats.write_skips.load(Ordering::Relaxed),
            evictions: stats.evictions.load(Ordering::Relaxed),
            failures: stats.failures.load(Ordering::Relaxed),
        }
    }

    fn entry_path(&self, key: &EntryKey) -> Option<PathBuf> {
        self.shared
            .index
            .lock()
            .ok()?
            .entries
            .get(key)
            .map(|meta| meta.path.clone())
    }

    /// Record a use, in memory and in the file's modification time, so that
    /// eviction order survives a restart.
    fn touch(&self, key: &EntryKey, path: &Path) {
        let now = SystemTime::now();
        if let Ok(mut index) = self.shared.index.lock()
            && let Some(meta) = index.entries.get_mut(key)
        {
            meta.used = now;
        }
        if let Ok(file) = fs::File::options().write(true).open(path) {
            let _ = file.set_times(fs::FileTimes::new().set_modified(now));
        }
    }
}

#[cfg(unix)]
fn create_private_dir(dir: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    // Entries hold the token ids of every cached prompt.
    fs::set_permissions(dir, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("failed to restrict {}", dir.display()))
}

#[cfg(not(unix))]
fn create_private_dir(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))
}

/// Build the index from the directory's contents.
///
/// A file whose name does not parse belongs to another version or another
/// tool; it is counted against the budget so the directory stays bounded, and
/// otherwise left alone. Half-written files from a killed process are removed.
fn scan(dir: &Path, fingerprint: &str) -> Result<Index> {
    let mut index = Index::default();
    let entries = fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("failed to list {}", dir.display()))?;
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "writing")
        {
            let _ = fs::remove_file(&path);
            continue;
        }
        if !format::is_entry(&path) {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(metadata) if metadata.is_file() => metadata,
            _ => continue,
        };
        let meta = EntryMeta {
            path: path.clone(),
            bytes: metadata.len(),
            used: metadata.modified().unwrap_or(UNIX_EPOCH),
        };
        index.bytes = index.bytes.saturating_add(meta.bytes);
        match parse_file_name(&path, fingerprint) {
            Some(key) => {
                index.entries.insert(key, meta);
            }
            None => index.foreign.push(meta),
        }
    }
    tracing::info!(
        dir = %dir.display(),
        entries = index.entries.len(),
        foreign = index.foreign.len(),
        mib = index.bytes as f64 / (1024. * 1024.),
        "opened the prompt cache"
    );
    Ok(index)
}

/// `<fingerprint>-<tokens>-<hash>.inferq-prompt`, and only for this model.
fn parse_file_name(path: &Path, fingerprint: &str) -> Option<EntryKey> {
    let stem = path.file_stem()?.to_str()?;
    let mut parts = stem.split('-');
    let file_fingerprint = parts.next()?;
    let tokens = parts.next()?.parse().ok()?;
    let hash = u64::from_str_radix(parts.next()?, 16).ok()?;
    if parts.next().is_some() || file_fingerprint != short_fingerprint(fingerprint) {
        return None;
    }
    Some(EntryKey { tokens, hash })
}

/// How long a shutdown waits for a cache write that is still in flight.
pub const DRAIN_TIMEOUT: Duration = Duration::from_secs(120);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen::{
        LayerStateImage, QuantizedAttentionImage, QuantizedDeltaCheckpoint, QuantizedStateImage,
    };

    fn collected(len: usize, block: usize, min: usize) -> Vec<usize> {
        boundaries(len, block, min).collect()
    }

    #[test]
    fn boundaries_are_multiples_below_the_prompt() {
        assert_eq!(collected(1000, 256, 256), vec![768, 512, 256]);
        // A prompt that ends exactly on a boundary cannot use that boundary:
        // a restored session still needs a token to produce logits from.
        assert_eq!(collected(768, 256, 256), vec![512, 256]);
        assert_eq!(collected(769, 256, 256), vec![768, 512, 256]);
        assert_eq!(collected(300, 256, 256), vec![256]);
        assert!(collected(256, 256, 256).is_empty());
        assert!(collected(0, 256, 256).is_empty());
    }

    #[test]
    fn boundaries_respect_the_minimum() {
        assert_eq!(
            collected(3000, 256, 1024),
            vec![2816, 2560, 2304, 2048, 1792, 1536, 1280, 1024]
        );
        assert!(collected(1000, 256, 1024).is_empty());
    }

    #[test]
    fn store_boundary_advances_past_what_was_reused() {
        assert_eq!(store_boundary(1000, 0, None, 256, 256), Some(768));
        assert_eq!(store_boundary(1000, 768, None, 256, 256), None);
        assert_eq!(store_boundary(1000, 700, None, 256, 256), Some(768));
        assert_eq!(store_boundary(1000, 500, None, 256, 256), Some(768));
        assert_eq!(store_boundary(400, 0, None, 256, 512), None);
    }

    #[test]
    fn store_boundary_stays_inside_the_stable_prefix() {
        // The last turn of this prompt starts at token 700, so an entry at 768
        // would be keyed on tokens no later request repeats.
        assert_eq!(store_boundary(1000, 0, Some(700), 256, 256), Some(512));
        assert_eq!(store_boundary(1000, 512, Some(700), 256, 256), None);
        // A stable prefix beyond the prompt is simply no constraint.
        assert_eq!(store_boundary(1000, 0, Some(5000), 256, 256), Some(768));
        // Too little stable text to reach the minimum: nothing is stored.
        assert_eq!(store_boundary(1000, 0, Some(100), 256, 256), None);
    }

    #[test]
    fn file_names_round_trip() {
        let key = EntryKey {
            tokens: 4096,
            hash: 0x1234_5678_9abc_def0,
        };
        let name = key.file_name("fnv1a64:7ae605bb0922e4ef");
        assert_eq!(
            name,
            "7ae605bb0922e4ef-00004096-123456789abcdef0.inferq-prompt"
        );
        let parsed = parse_file_name(Path::new(&name), "fnv1a64:7ae605bb0922e4ef");
        assert_eq!(parsed, Some(key));
        assert_eq!(parse_file_name(Path::new(&name), "fnv1a64:other"), None);
        assert_eq!(
            parse_file_name(Path::new("junk.inferq-prompt"), "fnv1a64:x"),
            None
        );
    }

    fn image(tokens: &[u32]) -> SessionImage {
        let position = tokens.len();
        SessionImage {
            tokens: tokens.to_vec(),
            model: QuantizedStateImage {
                layers: vec![
                    LayerStateImage::Linear(QuantizedDeltaCheckpoint::from_parts(
                        vec![0.25; 4],
                        vec![0.5; 8],
                    )),
                    LayerStateImage::Full(QuantizedAttentionImage {
                        keys: vec![1.; position * 2],
                        values: vec![2.; position * 2],
                        positions: position,
                    }),
                ],
                position,
            },
            mtp: None,
            last_target_hidden: None,
        }
    }

    fn open_cache(dir: &Path, budget_bytes: u64) -> PromptCache {
        PromptCache::open(
            PromptCacheConfig {
                dir: dir.to_path_buf(),
                budget_bytes,
                block_tokens: 4,
                min_tokens: 4,
            },
            "fnv1a64:abc",
            "Q4K",
            vec![LayerKind::Linear, LayerKind::Full { stride: 2 }],
        )
        .expect("open the cache")
    }

    /// The writer is asynchronous; wait for the entry count to settle.
    fn wait_for_writes(cache: &PromptCache, expected: usize) {
        assert!(
            cache.wait_for_writes(Duration::from_secs(10)),
            "cache writes did not drain: {:?}",
            cache.stats()
        );
        assert!(
            cache.stats().entries >= expected,
            "cache writes did not complete: {:?}",
            cache.stats()
        );
    }

    #[test]
    fn stores_and_reuses_the_longest_matching_prefix() {
        let directory = tempfile::tempdir().expect("temp dir");
        let cache = open_cache(directory.path(), 1 << 30);
        let tokens: Vec<u32> = (0..20).collect();
        assert_eq!(cache.store_boundary(&tokens, 0, None), Some(16));
        cache.store(image(&tokens[..16]));
        wait_for_writes(&cache, 1);

        // A later prompt that shares the first sixteen tokens restores them.
        let mut next = tokens[..16].to_vec();
        next.extend([99, 98, 97]);
        let restored = cache.lookup(&next, 0).expect("hit");
        assert_eq!(restored.tokens, tokens[..16]);
        assert_eq!(restored.position(), 16);
        assert_eq!(cache.stats().hits, 1);
        // That boundary is now on disk, so nothing new is stored for it.
        assert_eq!(cache.store_boundary(&next, 16, None), None);
    }

    #[test]
    fn a_divergent_prefix_misses() {
        let directory = tempfile::tempdir().expect("temp dir");
        let cache = open_cache(directory.path(), 1 << 30);
        let tokens: Vec<u32> = (0..20).collect();
        cache.store(image(&tokens[..16]));
        wait_for_writes(&cache, 1);
        let other: Vec<u32> = (100..120).collect();
        assert!(cache.lookup(&other, 0).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn survives_a_restart() {
        let directory = tempfile::tempdir().expect("temp dir");
        let tokens: Vec<u32> = (0..20).collect();
        {
            let cache = open_cache(directory.path(), 1 << 30);
            cache.store(image(&tokens[..16]));
            wait_for_writes(&cache, 1);
        }
        let reopened = open_cache(directory.path(), 1 << 30);
        assert_eq!(reopened.stats().entries, 1);
        assert!(reopened.lookup(&tokens, 0).is_some());
    }

    #[test]
    fn another_checkpoints_entries_are_never_read() {
        let directory = tempfile::tempdir().expect("temp dir");
        let tokens: Vec<u32> = (0..20).collect();
        {
            let cache = open_cache(directory.path(), 1 << 30);
            cache.store(image(&tokens[..16]));
            wait_for_writes(&cache, 1);
        }
        let other = PromptCache::open(
            PromptCacheConfig {
                dir: directory.path().to_path_buf(),
                budget_bytes: 1 << 30,
                block_tokens: 4,
                min_tokens: 4,
            },
            "fnv1a64:different",
            "Q4K",
            vec![LayerKind::Linear, LayerKind::Full { stride: 2 }],
        )
        .expect("open the cache");
        assert_eq!(other.stats().entries, 0);
        assert!(other.lookup(&tokens, 0).is_none());
        assert!(other.stats().bytes > 0, "foreign entries still cost budget");
    }

    #[test]
    fn eviction_keeps_the_directory_under_budget() {
        let directory = tempfile::tempdir().expect("temp dir");
        let cache = open_cache(directory.path(), 1 << 30);
        let first: Vec<u32> = (0..8).collect();
        cache.store(image(&first));
        wait_for_writes(&cache, 1);
        let entry_bytes = cache.stats().bytes;

        // Reopen with room for a single entry and add a second one.
        drop(cache);
        let cache = open_cache(directory.path(), entry_bytes + 1);
        let second: Vec<u32> = (100..108).collect();
        cache.store(image(&second));
        wait_for_writes(&cache, 1);
        for _ in 0..200 {
            if cache.stats().evictions > 0 {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let stats = cache.stats();
        assert_eq!(stats.evictions, 1, "{stats:?}");
        assert!(stats.bytes <= entry_bytes + 1, "{stats:?}");
        // A prompt that extends the second entry still restores it, and the
        // first entry's prefix no longer resolves.
        let mut extends_second = second.clone();
        extends_second.extend([200, 201]);
        assert!(cache.lookup(&extends_second, 0).is_some(), "{stats:?}");
        let mut extends_first = first.clone();
        extends_first.extend([200, 201]);
        assert!(cache.lookup(&extends_first, 0).is_none(), "{stats:?}");
    }

    #[test]
    fn a_corrupt_entry_is_discarded_rather_than_returned() {
        let directory = tempfile::tempdir().expect("temp dir");
        let cache = open_cache(directory.path(), 1 << 30);
        let tokens: Vec<u32> = (0..20).collect();
        cache.store(image(&tokens[..16]));
        wait_for_writes(&cache, 1);
        let path = directory
            .path()
            .join(EntryKey::of(&tokens[..16]).file_name("fnv1a64:abc"));
        fs::write(&path, b"not a safetensors file").expect("corrupt the entry");
        assert!(cache.lookup(&tokens, 0).is_none());
        assert!(!path.exists(), "the unusable entry was removed");
        assert_eq!(cache.stats().failures, 1);
    }

    #[test]
    fn rejects_a_minimum_below_one_block() {
        let directory = tempfile::tempdir().expect("temp dir");
        let config = PromptCacheConfig {
            dir: directory.path().to_path_buf(),
            budget_bytes: 1 << 20,
            block_tokens: 256,
            min_tokens: 128,
        };
        assert!(
            PromptCache::open(
                config,
                "fnv1a64:abc",
                "Q4K",
                vec![LayerKind::Full { stride: 2 }]
            )
            .is_err()
        );
    }
}
