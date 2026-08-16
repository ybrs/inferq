use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap, VecDeque},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, ensure};
use candle_core::{
    DType, Device, Tensor,
    quantized::{
        GgmlDType, QStorage, QTensor,
        gguf_file::{Content, Value},
        k_quants::{BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8_0, GgmlType},
    },
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const SMALL_M_MIN_STORAGE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Serialize)]
pub struct GgufSummary {
    pub architecture: String,
    pub layers: usize,
    pub auxiliary_predictor_layers: usize,
    pub hidden_size: usize,
    pub experts_per_layer: usize,
    pub experts_selected: usize,
    pub vocab_size: usize,
    pub full_attention_layers: usize,
    pub linear_attention_layers: usize,
    pub tensor_count: usize,
    pub dtypes: Vec<String>,
    pub format: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub storage_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufModelIdentity {
    pub path: String,
    pub size_bytes: u64,
    pub modified_unix_nanos: Option<u128>,
    /// Stable FNV-1a digest of file metadata and the GGUF tensor layout. This
    /// identifies a local checkpoint without reading all weight bytes.
    pub layout_fingerprint: String,
    pub quantization: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ExpertCacheStats {
    pub capacity_bytes: usize,
    pub resident_bytes: usize,
    pub entries: usize,
    /// Full warmup proved that all model experts are pinned, so recency
    /// bookkeeping is disabled until an unexpected miss occurs.
    pub fully_resident: bool,
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Compressed expert bytes copied through the GGUF reader. This is not a
    /// physical-storage counter because reads may be served by the page cache.
    pub bytes_loaded: u64,
}

impl ExpertCacheStats {
    pub fn activity_since(self, earlier: Self) -> Self {
        Self {
            capacity_bytes: self.capacity_bytes,
            resident_bytes: self.resident_bytes,
            entries: self.entries,
            fully_resident: self.fully_resident,
            requests: self.requests.saturating_sub(earlier.requests),
            hits: self.hits.saturating_sub(earlier.hits),
            misses: self.misses.saturating_sub(earlier.misses),
            evictions: self.evictions.saturating_sub(earlier.evictions),
            bytes_loaded: self.bytes_loaded.saturating_sub(earlier.bytes_loaded),
        }
    }

    pub fn hit_rate(self) -> f64 {
        if self.requests == 0 {
            0.
        } else {
            self.hits as f64 / self.requests as f64
        }
    }
}

/// An executable GGUF matrix that keeps weights in their on-disk numeric
/// representation during multiplication.
///
/// The API intentionally exposes no whole-matrix dequantization method. F32
/// activations are multiplied directly by Candle's ggml block kernels.
#[derive(Clone)]
pub struct QuantizedMatrix {
    tensor: Arc<QTensor>,
    dtype: GgmlDType,
    rows: usize,
    columns: usize,
    storage_bytes: usize,
}

#[derive(Clone)]
pub struct QuantizedEmbedding {
    tensor: Arc<QTensor>,
    rows: usize,
    columns: usize,
    storage_bytes: usize,
}

/// A validated, checkpoint-bound view of one fused expert tensor.
///
/// Resolving the tensor once avoids string allocation, GGUF metadata lookup,
/// and shape arithmetic on every routed-expert cache access.
pub struct GgufExpertTensor<'a> {
    checkpoint: &'a GgufCheckpoint,
    name: String,
    dtype: GgmlDType,
    expert_count: usize,
    rows: usize,
    columns: usize,
    expert_bytes: usize,
    tensor_start: u64,
}

/// Two compatible fused expert tensors viewed as one row-concatenated matrix
/// per expert. This preserves each compressed row byte-for-byte.
pub struct GgufExpertPair<'a> {
    checkpoint: &'a GgufCheckpoint,
    first: GgufExpertTensor<'a>,
    second: GgufExpertTensor<'a>,
}

impl std::fmt::Debug for GgufExpertTensor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufExpertTensor")
            .field("name", &self.name)
            .field("dtype", &self.dtype)
            .field("expert_count", &self.expert_count)
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}

impl GgufExpertTensor<'_> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn expert_count(&self) -> usize {
        self.expert_count
    }

    pub fn load(&self, expert: usize) -> Result<QuantizedMatrix> {
        self.checkpoint.load_resolved_expert_matrix(self, expert)
    }

    pub fn warm(&self, expert: usize) -> Result<usize> {
        Ok(self.load(expert)?.storage_bytes())
    }
}

impl std::fmt::Debug for GgufExpertPair<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufExpertPair")
            .field("first", &self.first.name)
            .field("second", &self.second.name)
            .field("expert_count", &self.first.expert_count)
            .field("rows", &(self.first.rows + self.second.rows))
            .field("columns", &self.first.columns)
            .finish_non_exhaustive()
    }
}

impl GgufExpertPair<'_> {
    pub fn expert_count(&self) -> usize {
        self.first.expert_count
    }

    pub fn load(&self, expert: usize) -> Result<QuantizedMatrix> {
        self.checkpoint.load_resolved_expert_pair(self, expert)
    }

    pub fn warm(&self, expert: usize) -> Result<usize> {
        Ok(self.load(expert)?.storage_bytes())
    }
}

impl QuantizedEmbedding {
    fn new(tensor: QTensor) -> Result<Self> {
        ensure!(
            matches!(
                tensor.dtype(),
                GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K | GgmlDType::Q8_0 | GgmlDType::F32
            ),
            "unsupported GGUF embedding dtype {:?}",
            tensor.dtype()
        );
        let (rows, columns) = tensor.shape().dims2()?;
        let storage_bytes = tensor.storage_size_in_bytes();
        Ok(Self {
            tensor: Arc::new(tensor),
            rows,
            columns,
            storage_bytes,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.columns]
    }

    pub fn storage_bytes(&self) -> usize {
        self.storage_bytes
    }

    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        Ok(self.tensor.embedding(token_ids)?)
    }
}

impl std::fmt::Debug for QuantizedMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedMatrix")
            .field("dtype", &self.dtype)
            .field("shape", &[self.rows, self.columns])
            .field("storage_bytes", &self.storage_bytes)
            .finish()
    }
}

impl QuantizedMatrix {
    fn new(tensor: QTensor) -> Result<Self> {
        ensure!(
            matches!(
                tensor.dtype(),
                GgmlDType::Q4K
                    | GgmlDType::Q5K
                    | GgmlDType::Q6K
                    | GgmlDType::Q8_0
                    | GgmlDType::F32
                    | GgmlDType::BF16
            ),
            "unsupported executable GGUF matrix dtype {:?}; expected Q4_K, Q5_K, Q6_K, Q8_0, F32, or BF16",
            tensor.dtype()
        );
        let (rows, columns) = tensor.shape().dims2()?;
        let storage_bytes = tensor.storage_size_in_bytes();
        Ok(Self {
            dtype: tensor.dtype(),
            tensor: Arc::new(tensor),
            rows,
            columns,
            storage_bytes,
        })
    }

    pub fn dtype(&self) -> String {
        format!("{:?}", self.dtype)
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.columns]
    }

    pub fn storage_bytes(&self) -> usize {
        self.storage_bytes
    }

    /// Concatenate two identically quantized matrices along their output rows.
    /// Quantized GGUF blocks are row-major, so this preserves every row's bytes
    /// while allowing callers to share one kernel launch for related outputs.
    pub fn concatenate_rows(&self, other: &Self) -> Result<Self> {
        ensure!(
            self.dtype == other.dtype,
            "cannot concatenate quantized matrices with dtypes {:?} and {:?}",
            self.dtype,
            other.dtype
        );
        ensure!(
            self.columns == other.columns,
            "cannot concatenate quantized matrices with {} and {} columns",
            self.columns,
            other.columns
        );
        ensure!(
            self.tensor.device().is_cpu() && other.tensor.device().is_cpu(),
            "row concatenation currently supports CPU quantized matrices only"
        );
        let first = self.tensor.data()?;
        let second = other.tensor.data()?;
        let mut data = Vec::with_capacity(first.len() + second.len());
        data.extend_from_slice(&first);
        data.extend_from_slice(&second);
        let storage = QStorage::from_data(Cow::Owned(data), &Device::Cpu, self.dtype)?;
        let tensor = QTensor::new(storage, (self.rows + other.rows, self.columns))?;
        Self::new(tensor)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        ensure!(
            xs.rank() >= 2,
            "quantized matrix input must have rank at least two"
        );
        ensure!(
            xs.dim(xs.rank() - 1)? == self.columns,
            "quantized matrix input has final dimension {}, expected {}",
            xs.dim(xs.rank() - 1)?,
            self.columns
        );
        let xs = if xs.dtype() == DType::F32 {
            xs.clone()
        } else {
            xs.to_dtype(DType::F32)?
        };
        let input_rows = xs.elem_count() / self.columns;
        if (2..=8).contains(&input_rows)
            && self.tensor.device().is_cpu()
            && (self.storage_bytes >= SMALL_M_MIN_STORAGE_BYTES || cfg!(test))
        {
            let values = xs.flatten_all()?.to_vec1::<f32>()?;
            let output = match self.dtype {
                GgmlDType::Q4K => self.small_m_forward::<BlockQ4K>(&values, input_rows)?,
                GgmlDType::Q5K => self.small_m_forward::<BlockQ5K>(&values, input_rows)?,
                GgmlDType::Q6K => self.small_m_forward::<BlockQ6K>(&values, input_rows)?,
                GgmlDType::Q8_0 => self.small_m_forward::<BlockQ8_0>(&values, input_rows)?,
                _ => return Ok(xs.apply_op1_no_bwd(self.tensor.as_ref())?),
            };
            let mut shape = xs.dims().to_vec();
            *shape.last_mut().expect("rank validated above") = self.rows;
            return Ok(Tensor::from_vec(output, shape, xs.device())?);
        }
        Ok(xs.apply_op1_no_bwd(self.tensor.as_ref())?)
    }

    fn small_m_forward<T: GgmlType>(&self, xs: &[f32], input_rows: usize) -> Result<Vec<f32>> {
        ensure!(
            self.dtype == T::DTYPE,
            "small-M kernel dtype mismatch: matrix {:?}, block {:?}",
            self.dtype,
            T::DTYPE
        );
        let blocks_per_row = self.columns.div_ceil(T::BLCK_SIZE);
        let bytes = self.tensor.data()?;
        // SAFETY: Candle's CPU QStorage owns a correctly aligned Vec<T> and
        // exposes its exact initialized bytes here. The four concrete block
        // types dispatched above are repr(C) numeric PODs for which every bit
        // pattern is valid. Prefix/suffix checks below reject misalignment or
        // a partial block before the typed slice is used.
        let (prefix, weights, suffix) = unsafe { bytes.align_to::<T>() };
        ensure!(
            prefix.is_empty() && suffix.is_empty(),
            "quantized matrix bytes are not aligned to complete {:?} blocks",
            T::DTYPE
        );
        ensure!(
            weights.len() == self.rows * blocks_per_row,
            "quantized matrix has {} {:?} blocks, expected {}",
            weights.len(),
            T::DTYPE,
            self.rows * blocks_per_row
        );

        let mut quantized_inputs = vec![T::VecDotType::zeros(); input_rows * blocks_per_row];
        for row in 0..input_rows {
            T::VecDotType::from_float(
                &xs[row * self.columns..(row + 1) * self.columns],
                &mut quantized_inputs[row * blocks_per_row..(row + 1) * blocks_per_row],
            );
        }

        // Store output-column-major while parallel workers stream disjoint
        // compressed rows. Applying all M inputs before advancing retains a
        // weight row in cache instead of traversing the complete matrix M
        // times as Candle's generic CPU loop does.
        let mut transposed = vec![0f32; self.rows * input_rows];
        transposed
            .par_chunks_mut(input_rows)
            .enumerate()
            .for_each(|(output_row, output)| {
                let weight =
                    &weights[output_row * blocks_per_row..(output_row + 1) * blocks_per_row];
                for input_row in 0..input_rows {
                    let input = &quantized_inputs
                        [input_row * blocks_per_row..(input_row + 1) * blocks_per_row];
                    output[input_row] = T::vec_dot(self.columns, weight, input);
                }
            });
        let mut output = vec![0f32; input_rows * self.rows];
        for input_row in 0..input_rows {
            for output_row in 0..self.rows {
                output[input_row * self.rows + output_row] =
                    transposed[output_row * input_rows + input_row];
            }
        }
        Ok(output)
    }
}

/// Parsed GGUF metadata plus an owned file handle used to load executable
/// compressed tensors on demand. Loaded matrices own only their compressed
/// bytes; they do not retain a dequantized copy.
pub struct GgufCheckpoint {
    path: PathBuf,
    file: Mutex<File>,
    content: Content,
    expert_cache: Mutex<ExpertMatrixCache>,
}

struct CachedExpertMatrix {
    matrix: QuantizedMatrix,
    storage_bytes: usize,
    last_access: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ExpertMatrixKey {
    tensor_start: u64,
    second_tensor_start: Option<u64>,
    expert: usize,
}

#[derive(Default)]
struct ExpertMatrixCache {
    capacity_bytes: usize,
    resident_bytes: usize,
    access_clock: u64,
    requests: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    bytes_loaded: u64,
    entries: HashMap<ExpertMatrixKey, CachedExpertMatrix>,
    lru: VecDeque<(ExpertMatrixKey, u64)>,
    fully_resident: bool,
}

impl ExpertMatrixCache {
    fn compact_lru_if_needed(&mut self) {
        let limit = self.entries.len().saturating_mul(4).saturating_add(1024);
        if self.lru.len() <= limit {
            return;
        }
        let mut current: Vec<_> = self
            .entries
            .iter()
            .map(|(key, entry)| (*key, entry.last_access))
            .collect();
        current.sort_unstable_by_key(|(_, access)| *access);
        self.lru = current.into();
    }

    fn stats(&self) -> ExpertCacheStats {
        ExpertCacheStats {
            capacity_bytes: self.capacity_bytes,
            resident_bytes: self.resident_bytes,
            entries: self.entries.len(),
            fully_resident: self.fully_resident,
            requests: self.requests,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            bytes_loaded: self.bytes_loaded,
        }
    }

    fn get(&mut self, key: ExpertMatrixKey) -> Option<QuantizedMatrix> {
        self.requests += 1;
        if self.capacity_bytes == 0 {
            self.misses += 1;
            return None;
        }
        if self.fully_resident {
            if let Some(entry) = self.entries.get(&key) {
                self.hits += 1;
                return Some(entry.matrix.clone());
            }
            self.misses += 1;
            return None;
        }
        self.access_clock += 1;
        if let Some(entry) = self.entries.get_mut(&key) {
            self.hits += 1;
            entry.last_access = self.access_clock;
            self.lru.push_back((key, self.access_clock));
            let matrix = entry.matrix.clone();
            self.compact_lru_if_needed();
            Some(matrix)
        } else {
            self.misses += 1;
            None
        }
    }

    fn record_read_and_insert(&mut self, key: ExpertMatrixKey, matrix: &QuantizedMatrix) {
        if self.fully_resident {
            self.fully_resident = false;
            let mut current: Vec<_> = self
                .entries
                .iter()
                .map(|(key, entry)| (*key, entry.last_access))
                .collect();
            current.sort_unstable_by_key(|(_, access)| *access);
            self.lru = current.into();
        }
        let storage_bytes = matrix.storage_bytes();
        self.bytes_loaded = self.bytes_loaded.saturating_add(storage_bytes as u64);
        if self.capacity_bytes == 0 || storage_bytes > self.capacity_bytes {
            return;
        }
        while self.resident_bytes + storage_bytes > self.capacity_bytes {
            let Some((oldest, access)) = self.lru.pop_front() else {
                break;
            };
            let is_current = self
                .entries
                .get(&oldest)
                .is_some_and(|entry| entry.last_access == access);
            if is_current && let Some(evicted) = self.entries.remove(&oldest) {
                self.resident_bytes = self.resident_bytes.saturating_sub(evicted.storage_bytes);
                self.evictions += 1;
            }
        }
        self.access_clock += 1;
        let previous = self.entries.insert(
            key,
            CachedExpertMatrix {
                matrix: matrix.clone(),
                storage_bytes,
                last_access: self.access_clock,
            },
        );
        if let Some(previous) = previous {
            self.resident_bytes = self.resident_bytes.saturating_sub(previous.storage_bytes);
        }
        self.resident_bytes += storage_bytes;
        self.lru.push_back((key, self.access_clock));
        self.compact_lru_if_needed();
    }

    fn fuse_pair(&mut self, pair: &GgufExpertPair<'_>) -> Result<usize> {
        ensure!(
            self.capacity_bytes > 0,
            "cannot fuse expert matrices in a disabled cache"
        );
        let mut fused_bytes = 0usize;
        for expert in 0..pair.first.expert_count {
            let first_key = ExpertMatrixKey {
                tensor_start: pair.first.tensor_start,
                second_tensor_start: None,
                expert,
            };
            let second_key = ExpertMatrixKey {
                tensor_start: pair.second.tensor_start,
                second_tensor_start: None,
                expert,
            };
            let pair_key = ExpertMatrixKey {
                tensor_start: pair.first.tensor_start,
                second_tensor_start: Some(pair.second.tensor_start),
                expert,
            };
            if let Some(existing) = self.entries.get(&pair_key) {
                fused_bytes = fused_bytes
                    .checked_add(existing.storage_bytes)
                    .context("fused expert byte count overflowed")?;
                continue;
            }
            let first = self
                .entries
                .get(&first_key)
                .with_context(|| {
                    format!(
                        "expert {expert} from tensor {:?} is not resident for fusion",
                        pair.first.name
                    )
                })?
                .matrix
                .clone();
            let second = self
                .entries
                .get(&second_key)
                .with_context(|| {
                    format!(
                        "expert {expert} from tensor {:?} is not resident for fusion",
                        pair.second.name
                    )
                })?
                .matrix
                .clone();
            let combined = first.concatenate_rows(&second)?;
            let combined_bytes = combined.storage_bytes();
            ensure!(
                combined_bytes == first.storage_bytes() + second.storage_bytes(),
                "fused expert storage changed from {} to {} bytes",
                first.storage_bytes() + second.storage_bytes(),
                combined_bytes
            );
            let removed_first = self
                .entries
                .remove(&first_key)
                .context("validated first expert cache entry disappeared")?;
            let removed_second = self
                .entries
                .remove(&second_key)
                .context("validated second expert cache entry disappeared")?;
            self.resident_bytes = self
                .resident_bytes
                .saturating_sub(removed_first.storage_bytes)
                .saturating_sub(removed_second.storage_bytes);
            self.access_clock += 1;
            self.entries.insert(
                pair_key,
                CachedExpertMatrix {
                    matrix: combined,
                    storage_bytes: combined_bytes,
                    last_access: self.access_clock,
                },
            );
            self.resident_bytes += combined_bytes;
            fused_bytes = fused_bytes
                .checked_add(combined_bytes)
                .context("fused expert byte count overflowed")?;
        }
        Ok(fused_bytes)
    }

    fn mark_fully_resident(&mut self) {
        self.fully_resident = true;
        self.lru.clear();
    }
}

impl std::fmt::Debug for GgufCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufCheckpoint")
            .field("path", &self.path)
            .field("tensor_count", &self.content.tensor_infos.len())
            .finish_non_exhaustive()
    }
}

impl GgufCheckpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file =
            File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
        let content = Content::read(&mut file)
            .with_context(|| format!("failed to parse GGUF header {}", path.display()))?;
        Ok(Self {
            path,
            file: Mutex::new(file),
            content,
            expert_cache: Mutex::new(ExpertMatrixCache::default()),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn identity(&self) -> Result<GgufModelIdentity> {
        let metadata = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("GGUF file lock was poisoned"))?
            .metadata()?;
        let modified_unix_nanos = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos());
        let infos = self.tensor_infos();
        let mut hash = 0xcbf29ce484222325_u64;
        fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
            for byte in bytes {
                *hash ^= u64::from(*byte);
                *hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash_bytes(&mut hash, &metadata.len().to_le_bytes());
        if let Some(modified) = modified_unix_nanos {
            hash_bytes(&mut hash, &modified.to_le_bytes());
        }
        let mut quantization = BTreeSet::new();
        for info in &infos {
            hash_bytes(&mut hash, info.name.as_bytes());
            hash_bytes(&mut hash, info.dtype.as_bytes());
            for dimension in &info.shape {
                hash_bytes(&mut hash, &dimension.to_le_bytes());
            }
            hash_bytes(&mut hash, &info.offset.to_le_bytes());
            hash_bytes(&mut hash, &info.storage_bytes.to_le_bytes());
            quantization.insert(info.dtype.clone());
        }
        Ok(GgufModelIdentity {
            path: self.path.display().to_string(),
            size_bytes: metadata.len(),
            modified_unix_nanos,
            layout_fingerprint: format!("fnv1a64:{hash:016x}"),
            quantization: quantization.into_iter().collect(),
        })
    }

    pub fn configure_expert_cache(&self, capacity_bytes: usize) -> Result<()> {
        let mut cache = self
            .expert_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("expert cache lock was poisoned"))?;
        *cache = ExpertMatrixCache {
            capacity_bytes,
            ..ExpertMatrixCache::default()
        };
        Ok(())
    }

    pub fn expert_cache_stats(&self) -> Result<ExpertCacheStats> {
        let cache = self
            .expert_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("expert cache lock was poisoned"))?;
        Ok(cache.stats())
    }

    /// Mark a completed full-model warmup so cache hits no longer maintain an
    /// eviction queue. A later miss automatically restores LRU tracking.
    pub fn mark_expert_cache_fully_resident(&self) -> Result<()> {
        let mut cache = self
            .expert_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("expert cache lock was poisoned"))?;
        ensure!(
            cache.capacity_bytes > 0 && !cache.entries.is_empty(),
            "cannot mark an empty or disabled expert cache as fully resident"
        );
        cache.mark_fully_resident();
        Ok(())
    }

    /// Read the three compressed matrices for one layer-qualified expert.
    /// With a zero-capacity explicit cache, dropping the returned matrix views
    /// leaves residency decisions to the OS page cache.
    pub fn warm_expert(&self, layer: usize, expert: usize) -> Result<usize> {
        let prefix = format!("blk.{layer}");
        let gate_up = self.expert_pair(
            &format!("{prefix}.ffn_gate_exps.weight"),
            &format!("{prefix}.ffn_up_exps.weight"),
        )?;
        let down = self.expert_tensor(&format!("{prefix}.ffn_down_exps.weight"))?;
        gate_up
            .warm(expert)?
            .checked_add(down.warm(expert)?)
            .context("expert warmup byte count overflowed")
    }

    /// Load one expert matrix through the configured cache and return its
    /// compressed storage size.
    pub fn warm_expert_matrix(&self, name: &str, expert: usize) -> Result<usize> {
        self.expert_tensor(name)?.warm(expert)
    }

    /// Stream one tensor through a bounded buffer to populate the OS page
    /// cache without retaining a second copy in the process heap.
    pub fn warm_tensor_pages(&self, name: &str) -> Result<usize> {
        const BUFFER_BYTES: usize = 8 * 1024 * 1024;
        let info = self
            .content
            .tensor_infos
            .get(name)
            .with_context(|| format!("GGUF is missing tensor {name:?}"))?;
        let storage_bytes =
            info.shape.elem_count() / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
        let start = self.content.tensor_data_offset.saturating_add(info.offset);
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("GGUF file lock was poisoned"))?;
        file.seek(SeekFrom::Start(start))?;
        let mut buffer = vec![0u8; BUFFER_BYTES.min(storage_bytes)];
        let mut remaining = storage_bytes;
        while remaining > 0 {
            let chunk = remaining.min(buffer.len());
            file.read_exact(&mut buffer[..chunk])
                .with_context(|| format!("failed to warm GGUF tensor {name:?}"))?;
            remaining -= chunk;
        }
        Ok(storage_bytes)
    }

    pub fn tensor_infos(&self) -> Vec<GgufTensorInfo> {
        let mut infos: Vec<_> = self
            .content
            .tensor_infos
            .iter()
            .map(|(name, info)| GgufTensorInfo {
                name: name.clone(),
                dtype: format!("{:?}", info.ggml_dtype),
                shape: info.shape.dims().to_vec(),
                offset: self.content.tensor_data_offset + info.offset,
                storage_bytes: info.shape.elem_count() / info.ggml_dtype.block_size()
                    * info.ggml_dtype.type_size(),
            })
            .collect();
        infos.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        infos
    }

    pub fn tensor_info(&self, name: &str) -> Option<GgufTensorInfo> {
        let info = self.content.tensor_infos.get(name)?;
        Some(GgufTensorInfo {
            name: name.to_owned(),
            dtype: format!("{:?}", info.ggml_dtype),
            shape: info.shape.dims().to_vec(),
            offset: self.content.tensor_data_offset + info.offset,
            storage_bytes: info.shape.elem_count() / info.ggml_dtype.block_size()
                * info.ggml_dtype.type_size(),
        })
    }

    pub fn load_matrix(&self, name: &str) -> Result<QuantizedMatrix> {
        let info = self
            .content
            .tensor_infos
            .get(name)
            .with_context(|| format!("GGUF is missing tensor {name:?}"))?;
        ensure!(
            info.shape.rank() == 2,
            "GGUF tensor {name:?} has shape {:?}, expected a matrix",
            info.shape
        );
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("GGUF file lock was poisoned"))?;
        let tensor = info
            .read(&mut *file, self.content.tensor_data_offset, &Device::Cpu)
            .with_context(|| format!("failed to load GGUF matrix {name:?}"))?;
        QuantizedMatrix::new(tensor)
    }

    pub fn load_embedding(&self, name: &str) -> Result<QuantizedEmbedding> {
        let info = self
            .content
            .tensor_infos
            .get(name)
            .with_context(|| format!("GGUF is missing tensor {name:?}"))?;
        ensure!(
            info.shape.rank() == 2,
            "GGUF tensor {name:?} has shape {:?}, expected an embedding matrix",
            info.shape
        );
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("GGUF file lock was poisoned"))?;
        let tensor = info
            .read(&mut *file, self.content.tensor_data_offset, &Device::Cpu)
            .with_context(|| format!("failed to load GGUF embedding {name:?}"))?;
        QuantizedEmbedding::new(tensor)
    }

    /// Load one matrix from a GGUF tensor shaped `[experts, rows, columns]`.
    /// Only the selected expert's compressed byte range is read.
    pub fn load_expert_matrix(&self, name: &str, expert: usize) -> Result<QuantizedMatrix> {
        self.expert_tensor(name)?.load(expert)
    }

    /// Resolve and validate a fused expert tensor for repeated routed access.
    pub fn expert_tensor(&self, name: &str) -> Result<GgufExpertTensor<'_>> {
        let info = self
            .content
            .tensor_infos
            .get(name)
            .with_context(|| format!("GGUF is missing tensor {name:?}"))?;
        let dims = info.shape.dims();
        ensure!(
            dims.len() == 3,
            "GGUF tensor {name:?} has shape {:?}, expected [experts, rows, columns]",
            info.shape
        );
        let rows = dims[1];
        let columns = dims[2];
        ensure!(
            columns.is_multiple_of(info.ggml_dtype.block_size()),
            "GGUF expert matrix {name:?} width {columns} is not divisible by {:?} block size {}",
            info.ggml_dtype,
            info.ggml_dtype.block_size()
        );
        let blocks = rows
            .checked_mul(columns / info.ggml_dtype.block_size())
            .context("GGUF expert matrix block count overflowed")?;
        let expert_bytes = blocks
            .checked_mul(info.ggml_dtype.type_size())
            .context("GGUF expert matrix byte size overflowed")?;
        Ok(GgufExpertTensor {
            checkpoint: self,
            name: name.to_owned(),
            dtype: info.ggml_dtype,
            expert_count: dims[0],
            rows,
            columns,
            expert_bytes,
            tensor_start: self.content.tensor_data_offset.saturating_add(info.offset),
        })
    }

    /// Resolve two compatible `[experts, rows, columns]` tensors whose rows
    /// can be executed by one quantized matrix multiplication.
    pub fn expert_pair(&self, first: &str, second: &str) -> Result<GgufExpertPair<'_>> {
        let first = self.expert_tensor(first)?;
        let second = self.expert_tensor(second)?;
        ensure!(
            first.dtype == second.dtype,
            "cannot pair expert tensors {:?} and {:?} with dtypes {:?} and {:?}",
            first.name,
            second.name,
            first.dtype,
            second.dtype
        );
        ensure!(
            first.expert_count == second.expert_count
                && first.rows == second.rows
                && first.columns == second.columns,
            "cannot pair expert tensors {:?} [{}, {}, {}] and {:?} [{}, {}, {}]",
            first.name,
            first.expert_count,
            first.rows,
            first.columns,
            second.name,
            second.expert_count,
            second.rows,
            second.columns
        );
        Ok(GgufExpertPair {
            checkpoint: self,
            first,
            second,
        })
    }

    fn load_resolved_expert_matrix(
        &self,
        tensor: &GgufExpertTensor<'_>,
        expert: usize,
    ) -> Result<QuantizedMatrix> {
        ensure!(
            expert < tensor.expert_count,
            "expert index {expert} is outside tensor {:?} with {} experts",
            tensor.name,
            tensor.expert_count
        );
        let key = ExpertMatrixKey {
            tensor_start: tensor.tensor_start,
            second_tensor_start: None,
            expert,
        };
        if let Some(matrix) = self
            .expert_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("expert cache lock was poisoned"))?
            .get(key)
        {
            return Ok(matrix);
        }
        let expert_offset = expert
            .checked_mul(tensor.expert_bytes)
            .context("GGUF expert matrix offset overflowed")?;
        let start = tensor.tensor_start.saturating_add(expert_offset as u64);
        let mut raw = vec![0; tensor.expert_bytes];
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("GGUF file lock was poisoned"))?;
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut raw).with_context(|| {
            format!(
                "failed to read expert {expert} from GGUF tensor {:?}",
                tensor.name
            )
        })?;
        let storage = QStorage::from_data(Cow::Owned(raw), &Device::Cpu, tensor.dtype)?;
        let matrix = QuantizedMatrix::new(QTensor::new(storage, (tensor.rows, tensor.columns))?)?;
        self.expert_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("expert cache lock was poisoned"))?
            .record_read_and_insert(key, &matrix);
        Ok(matrix)
    }

    fn load_resolved_expert_pair(
        &self,
        pair: &GgufExpertPair<'_>,
        expert: usize,
    ) -> Result<QuantizedMatrix> {
        ensure!(
            expert < pair.first.expert_count,
            "expert index {expert} is outside paired tensors with {} experts",
            pair.first.expert_count
        );
        let key = ExpertMatrixKey {
            tensor_start: pair.first.tensor_start,
            second_tensor_start: Some(pair.second.tensor_start),
            expert,
        };
        if let Some(matrix) = self
            .expert_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("expert cache lock was poisoned"))?
            .get(key)
        {
            return Ok(matrix);
        }
        let mut raw = Vec::with_capacity(pair.first.expert_bytes + pair.second.expert_bytes);
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("GGUF file lock was poisoned"))?;
        for tensor in [&pair.first, &pair.second] {
            let expert_offset = expert
                .checked_mul(tensor.expert_bytes)
                .context("GGUF expert matrix offset overflowed")?;
            let start = tensor.tensor_start.saturating_add(expert_offset as u64);
            file.seek(SeekFrom::Start(start))?;
            let previous_len = raw.len();
            raw.resize(previous_len + tensor.expert_bytes, 0);
            file.read_exact(&mut raw[previous_len..]).with_context(|| {
                format!(
                    "failed to read expert {expert} from GGUF tensor {:?}",
                    tensor.name
                )
            })?;
        }
        drop(file);
        let storage = QStorage::from_data(Cow::Owned(raw), &Device::Cpu, pair.first.dtype)?;
        let matrix = QuantizedMatrix::new(QTensor::new(
            storage,
            (pair.first.rows + pair.second.rows, pair.first.columns),
        )?)?;
        self.expert_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("expert cache lock was poisoned"))?
            .record_read_and_insert(key, &matrix);
        Ok(matrix)
    }

    /// Replace separately cached pair members with byte-identical combined
    /// matrices. Full warmup uses this after sequential file-order loading.
    pub fn fuse_cached_expert_pair(&self, first: &str, second: &str) -> Result<usize> {
        let pair = self.expert_pair(first, second)?;
        self.expert_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("expert cache lock was poisoned"))?
            .fuse_pair(&pair)
    }

    pub fn load_f32_vector(&self, name: &str) -> Result<Tensor> {
        let tensor = self.load_f32_tensor(name)?;
        ensure!(
            tensor.rank() == 1,
            "GGUF tensor {name:?} has shape {:?}, expected an F32 vector",
            tensor.shape()
        );
        Ok(tensor)
    }

    pub fn load_f32_tensor(&self, name: &str) -> Result<Tensor> {
        let info = self
            .content
            .tensor_infos
            .get(name)
            .with_context(|| format!("GGUF is missing tensor {name:?}"))?;
        ensure!(
            matches!(info.ggml_dtype, GgmlDType::F32 | GgmlDType::BF16),
            "GGUF tensor {name:?} has dtype {:?}, expected F32 or BF16",
            info.ggml_dtype
        );
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("GGUF file lock was poisoned"))?;
        let tensor = info
            .read(&mut *file, self.content.tensor_data_offset, &Device::Cpu)
            .with_context(|| format!("failed to load F32 GGUF tensor {name:?}"))?;
        Ok(tensor.dequantize(&Device::Cpu)?)
    }
}

fn integer(value: &Value) -> Result<usize> {
    match value {
        Value::U8(v) => Ok(*v as usize),
        Value::I8(v) => Ok(*v as usize),
        Value::U16(v) => Ok(*v as usize),
        Value::I16(v) => Ok(*v as usize),
        Value::U32(v) => Ok(*v as usize),
        Value::I32(v) => Ok(*v as usize),
        Value::U64(v) => Ok(*v as usize),
        Value::I64(v) => Ok(*v as usize),
        _ => anyhow::bail!("expected integer GGUF metadata, found {value:?}"),
    }
}

fn get_usize(content: &Content, key: &str) -> Result<usize> {
    integer(
        content
            .metadata
            .get(key)
            .with_context(|| format!("GGUF is missing {key:?}"))?,
    )
}

pub fn inspect_gguf(path: impl AsRef<Path>) -> Result<GgufSummary> {
    let path = path.as_ref();
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let content = Content::read(&mut file)
        .with_context(|| format!("failed to parse GGUF header {}", path.display()))?;
    let architecture = match content.metadata.get("general.architecture") {
        Some(Value::String(value)) => value.clone(),
        other => anyhow::bail!("invalid general.architecture metadata: {other:?}"),
    };
    let metadata_prefix = match architecture.as_str() {
        "qwen3next" => "qwen3next",
        "qwen35moe" => "qwen35moe",
        _ => anyhow::bail!("unsupported GGUF architecture {architecture:?}"),
    };
    let all_layers = get_usize(&content, &format!("{metadata_prefix}.block_count"))?;
    let auxiliary_predictor_layers = content
        .metadata
        .get(&format!("{metadata_prefix}.nextn_predict_layers"))
        .map(integer)
        .transpose()?
        .unwrap_or(0);
    ensure!(
        auxiliary_predictor_layers <= all_layers,
        "next-token predictor layer count exceeds total GGUF block count"
    );
    let layers = all_layers - auxiliary_predictor_layers;
    let interval = get_usize(
        &content,
        &format!("{metadata_prefix}.full_attention_interval"),
    )?;
    ensure!(interval > 0, "full_attention_interval must be positive");
    let full_attention_layers = layers / interval;
    let vocab_size = match content.metadata.get("tokenizer.ggml.tokens") {
        Some(Value::Array(tokens)) => tokens.len(),
        other => anyhow::bail!("invalid tokenizer.ggml.tokens metadata: {other:?}"),
    };
    let dtypes = content
        .tensor_infos
        .values()
        .map(|tensor| format!("{:?}", tensor.ggml_dtype))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    Ok(GgufSummary {
        architecture,
        layers,
        auxiliary_predictor_layers,
        hidden_size: get_usize(&content, &format!("{metadata_prefix}.embedding_length"))?,
        experts_per_layer: get_usize(&content, &format!("{metadata_prefix}.expert_count"))?,
        experts_selected: get_usize(&content, &format!("{metadata_prefix}.expert_used_count"))?,
        vocab_size,
        full_attention_layers,
        linear_attention_layers: layers - full_attention_layers,
        tensor_count: content.tensor_infos.len(),
        dtypes,
        format: "gguf".into(),
    })
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use candle_core::{Device, Tensor, quantized::QTensor};
    use candle_core::{IndexOp, quantized::gguf_file};

    use super::*;

    #[test]
    fn direct_matrix_matches_its_dequantized_reference() {
        let device = Device::Cpu;
        let weights: Vec<f32> = (0..512)
            .map(|index| (index as f32 % 29. - 14.) / 16.)
            .collect();
        let weights = Tensor::from_vec(weights, (2, 256), &device).unwrap();
        let input: Vec<f32> = (0..256)
            .map(|index| (index as f32 % 11. - 5.) / 8.)
            .collect();
        let input = Tensor::from_vec(input, (1, 256), &device).unwrap();

        for dtype in [
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8_0,
            GgmlDType::F32,
        ] {
            let tensor = QTensor::quantize(&weights, dtype).unwrap();
            let expected = input
                .matmul(&tensor.dequantize(&device).unwrap().t().unwrap())
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let matrix = QuantizedMatrix::new(tensor).unwrap();
            let actual = matrix.forward(&input).unwrap().to_vec2::<f32>().unwrap();
            for (actual, expected) in actual[0].iter().zip(&expected[0]) {
                assert!(
                    (actual - expected).abs() < 2e-2,
                    "{dtype:?}: actual {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn direct_matrix_rejects_wrong_input_width() {
        let weights = Tensor::zeros((1, 256), DType::F32, &Device::Cpu).unwrap();
        let matrix =
            QuantizedMatrix::new(QTensor::quantize(&weights, GgmlDType::Q4K).unwrap()).unwrap();
        let input = Tensor::zeros((1, 128), DType::F32, &Device::Cpu).unwrap();
        assert!(matrix.forward(&input).is_err());
    }

    #[test]
    fn concatenated_quantized_rows_match_separate_matrices() {
        let device = Device::Cpu;
        let first_values: Vec<f32> = (0..2 * 256)
            .map(|index| (index as f32 % 23. - 11.) / 16.)
            .collect();
        let second_values: Vec<f32> = (0..3 * 256)
            .map(|index| (index as f32 % 17. - 8.) / 12.)
            .collect();
        let first = QuantizedMatrix::new(
            QTensor::quantize(
                &Tensor::from_vec(first_values, (2, 256), &device).unwrap(),
                GgmlDType::Q8_0,
            )
            .unwrap(),
        )
        .unwrap();
        let second = QuantizedMatrix::new(
            QTensor::quantize(
                &Tensor::from_vec(second_values, (3, 256), &device).unwrap(),
                GgmlDType::Q8_0,
            )
            .unwrap(),
        )
        .unwrap();
        let input = Tensor::from_vec(vec![0.25; 256], (1, 256), &device).unwrap();
        let mut expected = first.forward(&input).unwrap().to_vec2::<f32>().unwrap()[0].clone();
        expected.extend_from_slice(&second.forward(&input).unwrap().to_vec2::<f32>().unwrap()[0]);

        let combined = first.concatenate_rows(&second).unwrap();
        let actual = combined.forward(&input).unwrap().to_vec2::<f32>().unwrap();

        assert_eq!(combined.shape(), [5, 256]);
        assert_eq!(actual[0], expected);
    }

    #[test]
    fn direct_embedding_matches_dequantized_rows() {
        let values: Vec<f32> = (0..4 * 256)
            .map(|index| (index as f32 % 19. - 9.) / 8.)
            .collect();
        let weights = Tensor::from_vec(values, (4, 256), &Device::Cpu).unwrap();
        let tensor = QTensor::quantize(&weights, GgmlDType::Q8_0).unwrap();
        let expected = tensor
            .dequantize(&Device::Cpu)
            .unwrap()
            .index_select(&Tensor::new(&[1u32, 3], &Device::Cpu).unwrap(), 0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let embedding = QuantizedEmbedding::new(tensor).unwrap();
        let ids = Tensor::new(&[1u32, 3], &Device::Cpu).unwrap();
        let actual = embedding.forward(&ids).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn loads_only_one_matrix_from_a_fused_expert_tensor() {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..512)
            .map(|index| (index as f32 % 23. - 11.) / 8.)
            .collect();
        let weights = Tensor::from_vec(values, (2, 1, 256), &device).unwrap();
        let quantized = QTensor::quantize(&weights, GgmlDType::Q4K).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("experts.gguf");
        let mut file = File::create(&path).unwrap();
        gguf_file::write(&mut file, &[], &[("experts", &quantized)]).unwrap();
        drop(file);

        let input = Tensor::ones((1, 256), DType::F32, &device).unwrap();
        let dequantized = quantized.dequantize(&device).unwrap();
        let expected = input
            .matmul(&dequantized.i(1).unwrap().t().unwrap())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let checkpoint = GgufCheckpoint::open(path).unwrap();
        assert_eq!(
            checkpoint.warm_tensor_pages("experts").unwrap(),
            quantized.storage_size_in_bytes()
        );
        checkpoint.configure_expert_cache(1024 * 1024).unwrap();
        let experts = checkpoint.expert_tensor("experts").unwrap();
        assert_eq!(experts.name(), "experts");
        assert_eq!(experts.expert_count(), 2);
        let expert = experts.load(1).unwrap();
        assert!(expert.storage_bytes() < quantized.storage_size_in_bytes());
        let actual = expert.forward(&input).unwrap().to_vec2::<f32>().unwrap();
        assert!((actual[0][0] - expected[0][0]).abs() < 2e-2);
        experts.load(1).unwrap();
        let stats = checkpoint.expert_cache_stats().unwrap();
        assert_eq!(stats.requests, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.bytes_loaded, expert.storage_bytes() as u64);

        checkpoint.mark_expert_cache_fully_resident().unwrap();
        experts.load(1).unwrap();
        let stats = checkpoint.expert_cache_stats().unwrap();
        assert!(stats.fully_resident);
        assert_eq!(stats.hits, 2);

        experts.load(0).unwrap();
        let stats = checkpoint.expert_cache_stats().unwrap();
        assert!(!stats.fully_resident);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.entries, 2);
    }

    #[test]
    fn fuses_resident_expert_pairs_without_changing_bytes_or_outputs() {
        let device = Device::Cpu;
        let first_values: Vec<f32> = (0..2 * 256)
            .map(|index| (index as f32 % 23. - 11.) / 8.)
            .collect();
        let second_values: Vec<f32> = (0..2 * 256)
            .map(|index| (index as f32 % 19. - 9.) / 7.)
            .collect();
        let first = QTensor::quantize(
            &Tensor::from_vec(first_values, (2, 1, 256), &device).unwrap(),
            GgmlDType::Q4K,
        )
        .unwrap();
        let second = QTensor::quantize(
            &Tensor::from_vec(second_values, (2, 1, 256), &device).unwrap(),
            GgmlDType::Q4K,
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("expert-pair.gguf");
        let mut file = File::create(&path).unwrap();
        gguf_file::write(&mut file, &[], &[("first", &first), ("second", &second)]).unwrap();
        drop(file);

        let checkpoint = GgufCheckpoint::open(path).unwrap();
        checkpoint.configure_expert_cache(1024 * 1024).unwrap();
        let first = checkpoint.expert_tensor("first").unwrap();
        let second = checkpoint.expert_tensor("second").unwrap();
        for expert in 0..2 {
            first.warm(expert).unwrap();
            second.warm(expert).unwrap();
        }
        let before = checkpoint.expert_cache_stats().unwrap();
        assert_eq!(before.entries, 4);
        let input = Tensor::ones((1, 256), DType::F32, &device).unwrap();
        let mut expected = first.load(1).unwrap().forward(&input).unwrap();
        expected = Tensor::cat(
            &[expected, second.load(1).unwrap().forward(&input).unwrap()],
            1,
        )
        .unwrap();

        let fused_bytes = checkpoint
            .fuse_cached_expert_pair("first", "second")
            .unwrap();
        let after = checkpoint.expert_cache_stats().unwrap();
        assert_eq!(after.entries, 2);
        assert_eq!(after.resident_bytes, before.resident_bytes);
        assert_eq!(after.bytes_loaded, before.bytes_loaded);
        assert_eq!(fused_bytes, before.resident_bytes);
        let actual = checkpoint
            .expert_pair("first", "second")
            .unwrap()
            .load(1)
            .unwrap()
            .forward(&input)
            .unwrap();
        assert_eq!(
            actual.to_vec2::<f32>().unwrap(),
            expected.to_vec2::<f32>().unwrap()
        );
    }
}
