use std::{
    borrow::Cow,
    collections::BTreeSet,
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
    },
};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct GgufSummary {
    pub architecture: String,
    pub layers: usize,
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

/// An executable GGUF matrix that keeps weights in their on-disk numeric
/// representation during multiplication.
///
/// The API intentionally exposes no whole-matrix dequantization method. F32
/// activations are multiplied directly by Candle's ggml block kernels.
pub struct QuantizedMatrix {
    tensor: Arc<QTensor>,
    dtype: GgmlDType,
    rows: usize,
    columns: usize,
    storage_bytes: usize,
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
                GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K | GgmlDType::Q8_0 | GgmlDType::F32
            ),
            "unsupported executable GGUF matrix dtype {:?}; expected Q4_K, Q5_K, Q6_K, Q8_0, or F32",
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
        Ok(xs.apply_op1_no_bwd(self.tensor.as_ref())?)
    }
}

/// Parsed GGUF metadata plus an owned file handle used to load executable
/// compressed tensors on demand. Loaded matrices own only their compressed
/// bytes; they do not retain a dequantized copy.
pub struct GgufCheckpoint {
    path: PathBuf,
    file: Mutex<File>,
    content: Content,
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
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
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

    /// Load one matrix from a GGUF tensor shaped `[experts, rows, columns]`.
    /// Only the selected expert's compressed byte range is read.
    pub fn load_expert_matrix(&self, name: &str, expert: usize) -> Result<QuantizedMatrix> {
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
        ensure!(
            expert < dims[0],
            "expert index {expert} is outside tensor {name:?} with {} experts",
            dims[0]
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
        let expert_offset = expert
            .checked_mul(expert_bytes)
            .context("GGUF expert matrix offset overflowed")?;
        let start = self
            .content
            .tensor_data_offset
            .saturating_add(info.offset)
            .saturating_add(expert_offset as u64);
        let mut raw = vec![0; expert_bytes];
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("GGUF file lock was poisoned"))?;
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut raw)
            .with_context(|| format!("failed to read expert {expert} from GGUF tensor {name:?}"))?;
        let storage = QStorage::from_data(Cow::Owned(raw), &Device::Cpu, info.ggml_dtype)?;
        QuantizedMatrix::new(QTensor::new(storage, (rows, columns))?)
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
    ensure!(
        architecture == "qwen3next",
        "unsupported GGUF architecture {architecture:?}"
    );
    let layers = get_usize(&content, "qwen3next.block_count")?;
    let interval = get_usize(&content, "qwen3next.full_attention_interval")?;
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
        hidden_size: get_usize(&content, "qwen3next.embedding_length")?,
        experts_per_layer: get_usize(&content, "qwen3next.expert_count")?,
        experts_selected: get_usize(&content, "qwen3next.expert_used_count")?,
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
        let expert = checkpoint.load_expert_matrix("experts", 1).unwrap();
        assert!(expert.storage_bytes() < quantized.storage_size_in_bytes());
        let actual = expert.forward(&input).unwrap().to_vec2::<f32>().unwrap();
        assert!((actual[0][0] - expected[0][0]).abs() < 2e-2);
    }
}
