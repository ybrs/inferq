//! The on-disk representation of one cached prefix.
//!
//! Entries are safetensors files: a JSON header naming every state tensor,
//! followed by the raw little-endian f32 rows. The format was chosen because
//! it is already in this build, because a header read is enough to reject an
//! entry that belongs to another checkpoint, and because an entry can be
//! opened from Python when a cached state has to be inspected.
//!
//! ```text
//! __metadata__  inferq_format, model_fingerprint, position, layer_count, ...
//! tokens        u32[position]        the exact prefix this state represents
//! layer.N.keys  f32[position, ...]   full-attention layers
//! layer.N.values
//! layer.N.conv  f32[...]             linear (DeltaNet) layers
//! layer.N.recurrent
//! mtp.keys      f32[position, ...]   the NextN predictor's own cache
//! mtp.values
//! last_hidden   f32[hidden_size]     target hidden carry for the MTP arm
//! ```

use std::{collections::HashMap, fs::File, path::Path};

use anyhow::{Context, Result, ensure};
use safetensors::tensor::{Dtype, SafeTensors, TensorView, serialize_to_file};

use crate::{
    SessionImage,
    qwen::{
        LayerStateImage, QuantizedAttentionImage, QuantizedDeltaCheckpoint, QuantizedStateImage,
    },
};

/// Bumped whenever the meaning of the stored bytes changes. Entries written by
/// another version are ignored rather than migrated.
pub const FORMAT: &str = "inferq-prompt-cache-1";

pub const EXTENSION: &str = "inferq-prompt";

fn f32_view(values: &[f32]) -> Result<TensorView<'_>> {
    TensorView::new(Dtype::F32, vec![values.len()], bytemuck::cast_slice(values))
        .map_err(|error| anyhow::anyhow!("failed to describe an f32 tensor: {error}"))
}

fn u32_view(values: &[u32]) -> Result<TensorView<'_>> {
    TensorView::new(Dtype::U32, vec![values.len()], bytemuck::cast_slice(values))
        .map_err(|error| anyhow::anyhow!("failed to describe a u32 tensor: {error}"))
}

/// Write an entry, then rename it into place.
///
/// The rename is what makes a half-written entry impossible to read: a reader
/// either sees the complete file under its final name or does not see it.
pub fn write(
    path: &Path,
    image: &SessionImage,
    model_fingerprint: &str,
    quantization: &str,
    created_unix: u64,
) -> Result<u64> {
    let mut tensors: Vec<(String, TensorView<'_>)> =
        vec![("tokens".to_owned(), u32_view(&image.tokens)?)];
    for (index, layer) in image.model.layers.iter().enumerate() {
        match layer {
            LayerStateImage::Full(state) => {
                tensors.push((format!("layer.{index}.keys"), f32_view(&state.keys)?));
                tensors.push((format!("layer.{index}.values"), f32_view(&state.values)?));
            }
            LayerStateImage::Linear(state) => {
                tensors.push((format!("layer.{index}.conv"), f32_view(state.conv())?));
                tensors.push((
                    format!("layer.{index}.recurrent"),
                    f32_view(state.recurrent())?,
                ));
            }
        }
    }
    if let Some(mtp) = &image.mtp {
        tensors.push(("mtp.keys".to_owned(), f32_view(&mtp.keys)?));
        tensors.push(("mtp.values".to_owned(), f32_view(&mtp.values)?));
    }
    if let Some(hidden) = &image.last_target_hidden {
        tensors.push(("last_hidden".to_owned(), f32_view(hidden)?));
    }
    let metadata = HashMap::from([
        ("inferq_format".to_owned(), FORMAT.to_owned()),
        ("model_fingerprint".to_owned(), model_fingerprint.to_owned()),
        ("quantization".to_owned(), quantization.to_owned()),
        ("position".to_owned(), image.position().to_string()),
        (
            "layer_count".to_owned(),
            image.model.layers.len().to_string(),
        ),
        ("created_unix".to_owned(), created_unix.to_string()),
    ]);
    let temporary = path.with_extension("writing");
    serialize_to_file(tensors, Some(metadata), &temporary)
        .map_err(|error| anyhow::anyhow!("{error}"))
        .with_context(|| format!("failed to write {}", temporary.display()))?;
    let bytes = std::fs::metadata(&temporary)
        .with_context(|| format!("failed to stat {}", temporary.display()))?
        .len();
    std::fs::rename(&temporary, path)
        .with_context(|| format!("failed to publish {}", path.display()))?;
    Ok(bytes)
}

/// A mapped entry file. Held only while the image is being rebuilt from it.
struct Mapped {
    map: memmap2::Mmap,
}

impl Mapped {
    fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        // SAFETY: the map is read-only, is dropped before this function's
        // caller returns, and entries are published by rename, so the bytes
        // behind an opened path are never rewritten in place.
        let map = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("failed to map {}", path.display()))?;
        Ok(Self { map })
    }
}

/// Read only what is needed to decide whether an entry is usable: its format,
/// the checkpoint it belongs to, and the exact tokens it represents.
pub struct EntryHeader {
    pub tokens: Vec<u32>,
    pub position: usize,
}

/// The file's `__metadata__` block, which is what identifies the entry.
fn metadata_map(bytes: &[u8], path: &Path) -> Result<HashMap<String, String>> {
    let (_, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|error| anyhow::anyhow!("{} is not a readable entry: {error}", path.display()))?;
    metadata
        .metadata()
        .clone()
        .with_context(|| format!("{} carries no metadata", path.display()))
}

fn metadata_value<'a>(
    metadata: &'a HashMap<String, String>,
    key: &str,
    path: &Path,
) -> Result<&'a str> {
    metadata
        .get(key)
        .map(String::as_str)
        .with_context(|| format!("{} has no `{key}` metadata", path.display()))
}

fn read_f32(tensors: &SafeTensors<'_>, name: &str, path: &Path) -> Result<Vec<f32>> {
    let view = tensors
        .tensor(name)
        .map_err(|error| anyhow::anyhow!("{} is missing `{name}`: {error}", path.display()))?;
    ensure!(
        view.dtype() == Dtype::F32,
        "{} stores `{name}` as {:?}, expected F32",
        path.display(),
        view.dtype()
    );
    let bytes = view.data();
    ensure!(
        bytes.len().is_multiple_of(std::mem::size_of::<f32>()),
        "{} stores `{name}` with a partial float",
        path.display()
    );
    // The header pads to an eight-byte boundary, so tensor data is normally
    // four-byte aligned and casts without copying; the fallback keeps a file
    // written by some other producer readable rather than rejected.
    Ok(match bytemuck::try_cast_slice::<u8, f32>(bytes) {
        Ok(values) => values.to_vec(),
        Err(_) => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    })
}

fn read_tokens(tensors: &SafeTensors<'_>, path: &Path) -> Result<Vec<u32>> {
    let view = tensors
        .tensor("tokens")
        .map_err(|error| anyhow::anyhow!("{} is missing `tokens`: {error}", path.display()))?;
    ensure!(
        view.dtype() == Dtype::U32,
        "{} stores `tokens` as {:?}, expected U32",
        path.display(),
        view.dtype()
    );
    let bytes = view.data();
    ensure!(
        bytes.len().is_multiple_of(std::mem::size_of::<u32>()),
        "{} stores `tokens` with a partial id",
        path.display()
    );
    Ok(match bytemuck::try_cast_slice::<u8, u32>(bytes) {
        Ok(values) => values.to_vec(),
        Err(_) => bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    })
}

fn check_identity(
    metadata: &HashMap<String, String>,
    path: &Path,
    model_fingerprint: &str,
) -> Result<()> {
    let format = metadata_value(metadata, "inferq_format", path)?;
    ensure!(
        format == FORMAT,
        "{} was written in format `{format}`, this build reads `{FORMAT}`",
        path.display()
    );
    let fingerprint = metadata_value(metadata, "model_fingerprint", path)?;
    ensure!(
        fingerprint == model_fingerprint,
        "{} belongs to checkpoint {fingerprint}, not {model_fingerprint}",
        path.display()
    );
    Ok(())
}

/// Read the tokens an entry represents without paging in its state.
pub fn read_header(path: &Path, model_fingerprint: &str) -> Result<EntryHeader> {
    let mapped = Mapped::open(path)?;
    let metadata = metadata_map(&mapped.map, path)?;
    check_identity(&metadata, path, model_fingerprint)?;
    let tensors = SafeTensors::deserialize(&mapped.map)
        .map_err(|error| anyhow::anyhow!("{} is not a readable entry: {error}", path.display()))?;
    let position: usize = metadata_value(&metadata, "position", path)?
        .parse()
        .with_context(|| format!("{} has an unreadable position", path.display()))?;
    let tokens = read_tokens(&tensors, path)?;
    ensure!(
        tokens.len() == position,
        "{} holds {} tokens for position {position}",
        path.display(),
        tokens.len()
    );
    Ok(EntryHeader { tokens, position })
}

/// Rebuild a session image. `layer_kinds` describes the loaded model, so a
/// file whose layer sequence disagrees is rejected before any state is copied.
pub fn read(
    path: &Path,
    model_fingerprint: &str,
    layer_kinds: &[LayerKind],
) -> Result<SessionImage> {
    let mapped = Mapped::open(path)?;
    let metadata = metadata_map(&mapped.map, path)?;
    check_identity(&metadata, path, model_fingerprint)?;
    let tensors = SafeTensors::deserialize(&mapped.map)
        .map_err(|error| anyhow::anyhow!("{} is not a readable entry: {error}", path.display()))?;
    let position: usize = metadata_value(&metadata, "position", path)?
        .parse()
        .with_context(|| format!("{} has an unreadable position", path.display()))?;
    let layer_count: usize = metadata_value(&metadata, "layer_count", path)?
        .parse()
        .with_context(|| format!("{} has an unreadable layer count", path.display()))?;
    ensure!(
        layer_count == layer_kinds.len(),
        "{} holds {layer_count} layers, the model has {}",
        path.display(),
        layer_kinds.len()
    );
    let tokens = read_tokens(&tensors, path)?;
    ensure!(
        tokens.len() == position,
        "{} holds {} tokens for position {position}",
        path.display(),
        tokens.len()
    );
    let mut layers = Vec::with_capacity(layer_count);
    for (index, kind) in layer_kinds.iter().enumerate() {
        layers.push(match kind {
            LayerKind::Full { stride } => {
                let keys = read_f32(&tensors, &format!("layer.{index}.keys"), path)?;
                let values = read_f32(&tensors, &format!("layer.{index}.values"), path)?;
                let expected = position * stride;
                ensure!(
                    keys.len() == expected && values.len() == expected,
                    "{} layer {index} holds {}/{} key/value elements, this model needs {expected}",
                    path.display(),
                    keys.len(),
                    values.len()
                );
                LayerStateImage::Full(QuantizedAttentionImage {
                    keys,
                    values,
                    positions: position,
                })
            }
            LayerKind::Linear => LayerStateImage::Linear(QuantizedDeltaCheckpoint::from_parts(
                read_f32(&tensors, &format!("layer.{index}.conv"), path)?,
                read_f32(&tensors, &format!("layer.{index}.recurrent"), path)?,
            )),
        });
    }
    let model = QuantizedStateImage { layers, position };
    model
        .validate()
        .with_context(|| format!("{} holds inconsistent state", path.display()))?;
    let names = tensors.names();
    let mtp = if names.contains(&"mtp.keys") {
        let keys = read_f32(&tensors, "mtp.keys", path)?;
        let values = read_f32(&tensors, "mtp.values", path)?;
        // The predictor is one full-attention layer, so its rows are the same
        // width as the target's.
        let stride = layer_kinds.iter().find_map(|kind| match kind {
            LayerKind::Full { stride } => Some(*stride),
            LayerKind::Linear => None,
        });
        if let Some(stride) = stride {
            let expected = position * stride;
            ensure!(
                keys.len() == expected && values.len() == expected,
                "{} holds {}/{} MTP key/value elements, this model needs {expected}",
                path.display(),
                keys.len(),
                values.len()
            );
        }
        Some(QuantizedAttentionImage {
            keys,
            values,
            positions: position,
        })
    } else {
        None
    };
    let last_target_hidden = names
        .contains(&"last_hidden")
        .then(|| read_f32(&tensors, "last_hidden", path))
        .transpose()?;
    Ok(SessionImage {
        tokens,
        model,
        mtp,
        last_target_hidden,
    })
}

/// Which state a layer keeps, as the loaded model reports it.
///
/// A full-attention layer carries the number of f32 elements one position
/// occupies, so an entry whose rows are the wrong width is rejected on read
/// rather than restored into a kernel that reads them as a different shape.
/// Linear layers need no width here: restoring one compares its conv and
/// recurrent lengths against the live state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Full { stride: usize },
    Linear,
}

impl LayerKind {
    /// The layer sequence and KV widths of a loaded model.
    pub fn for_config(config: &crate::Qwen3NextConfig) -> Vec<Self> {
        let stride = config.num_key_value_heads * config.head_dim;
        (0..config.num_hidden_layers)
            .map(|layer| match config.layer_type(layer) {
                crate::LayerType::FullAttention => Self::Full { stride },
                crate::LayerType::LinearAttention => Self::Linear,
            })
            .collect()
    }
}

/// Reject an entry whose name does not describe an entry at all.
pub fn is_entry(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension == EXTENSION)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(position: usize) -> SessionImage {
        let keys: Vec<f32> = (0..position * 4).map(|v| v as f32).collect();
        let values: Vec<f32> = (0..position * 4).map(|v| -(v as f32)).collect();
        SessionImage {
            tokens: (0..position as u32).map(|t| t * 7 + 1).collect(),
            model: QuantizedStateImage {
                layers: vec![
                    LayerStateImage::Linear(QuantizedDeltaCheckpoint::from_parts(
                        vec![0.5, -0.25],
                        vec![1.5, 2.5, 3.5],
                    )),
                    LayerStateImage::Full(QuantizedAttentionImage {
                        keys,
                        values,
                        positions: position,
                    }),
                ],
                position,
            },
            mtp: Some(QuantizedAttentionImage {
                // The predictor is one full-attention layer, so its rows are
                // as wide as the target's.
                keys: vec![1.; position * 4],
                values: vec![2.; position * 4],
                positions: position,
            }),
            last_target_hidden: Some(vec![0.125, 0.25, 0.375]),
        }
    }

    fn kinds() -> Vec<LayerKind> {
        vec![LayerKind::Linear, LayerKind::Full { stride: 4 }]
    }

    #[test]
    fn round_trips_an_entry() {
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("entry.inferq-prompt");
        let written = image(6);
        let bytes = write(&path, &written, "fnv1a64:abc", "Q4K", 12).expect("write");
        assert!(bytes > 0);
        assert!(!directory.path().join("entry.writing").exists());
        let read_back = read(&path, "fnv1a64:abc", &kinds()).expect("read");
        assert_eq!(read_back, written);
        let header = read_header(&path, "fnv1a64:abc").expect("header");
        assert_eq!(header.tokens, written.tokens);
        assert_eq!(header.position, 6);
    }

    #[test]
    fn round_trips_without_optional_sections() {
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("entry.inferq-prompt");
        let mut written = image(3);
        written.mtp = None;
        written.last_target_hidden = None;
        write(&path, &written, "fnv1a64:abc", "Q4K", 0).expect("write");
        assert_eq!(read(&path, "fnv1a64:abc", &kinds()).expect("read"), written);
    }

    #[test]
    fn rejects_another_checkpoints_entry() {
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("entry.inferq-prompt");
        write(&path, &image(4), "fnv1a64:abc", "Q4K", 0).expect("write");
        let error = read(&path, "fnv1a64:def", &kinds()).expect_err("wrong checkpoint");
        assert!(format!("{error:#}").contains("belongs to checkpoint"));
        assert!(read_header(&path, "fnv1a64:def").is_err());
    }

    #[test]
    fn rejects_a_different_layer_sequence() {
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("entry.inferq-prompt");
        write(&path, &image(4), "fnv1a64:abc", "Q4K", 0).expect("write");
        assert!(read(&path, "fnv1a64:abc", &[LayerKind::Linear]).is_err());
        let swapped = [LayerKind::Full { stride: 4 }, LayerKind::Linear];
        assert!(read(&path, "fnv1a64:abc", &swapped).is_err());
        // Same layer sequence, different KV width: the entry belongs to a
        // model this one is not.
        let wider = [LayerKind::Linear, LayerKind::Full { stride: 8 }];
        let error = read(&path, "fnv1a64:abc", &wider).expect_err("wrong stride");
        assert!(
            format!("{error:#}").contains("key/value elements"),
            "{error:#}"
        );
    }

    #[test]
    fn rejects_a_truncated_file() {
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("entry.inferq-prompt");
        write(&path, &image(4), "fnv1a64:abc", "Q4K", 0).expect("write");
        let bytes = std::fs::read(&path).expect("read raw");
        std::fs::write(&path, &bytes[..bytes.len() / 2]).expect("truncate");
        assert!(read(&path, "fnv1a64:abc", &kinds()).is_err());
    }

    #[test]
    fn recognises_entry_paths() {
        assert!(is_entry(Path::new("a-1-2.inferq-prompt")));
        assert!(!is_entry(Path::new("a-1-2.writing")));
        assert!(!is_entry(Path::new("notes.txt")));
    }
}
