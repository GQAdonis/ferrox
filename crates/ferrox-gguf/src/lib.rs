//! ferrox-gguf: a reader for the GGUF file format.
//!
//! GGUF is a public, documented binary format originated by the ggml/llama.cpp
//! project (spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
//! See docs/THIRD_PARTY_NOTICES.md for design-credit details.

pub mod sharded;
pub use sharded::{ShardError, ShardName, ShardedGguf};

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};

use byteorder::{LittleEndian, ReadBytesExt};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};

/// Re-exported so downstream crates can name the type in a
/// [`TensorSource`] implementation without taking their own `memmap2`
/// dependency (which would then have to be kept version-compatible with
/// this one for the trait to line up at all).
pub use memmap2::Mmap as MmapHandle;
use thiserror::Error;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("not a GGUF file (bad magic {0:#010x})")]
    BadMagic(u32),
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("unknown metadata value type tag {0}")]
    UnknownValueType(u32),
    #[error("unknown tensor dtype tag {0}")]
    UnknownTensorType(u32),
    #[error("tensor '{0}' not found")]
    TensorNotFound(String),
    #[error("malformed tensor data for '{0}': expected {1} bytes, file has {2}")]
    TruncatedTensor(String, usize, usize),
}

/// A single scalar/array metadata value from the GGUF key-value header.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(v) => Some(*v),
            // Some exporters store bool hparams as 0/1 integers.
            GgufValue::U8(v) => Some(*v != 0),
            GgufValue::U32(v) => Some(*v != 0),
            GgufValue::U64(v) => Some(*v != 0),
            GgufValue::I32(v) => Some(*v != 0),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

/// GGML tensor element type tags (subset actually used by ferrox today).
/// Numeric tag values verified directly against `ggml/include/ggml.h`'s
/// `enum ggml_type` (the public GGUF/ggml tensor-type tag space), not
/// assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    BF16,
    IQ4NL,
    IQ4XS,
    /// The codebook-grid low-bit formats used throughout the published
    /// "Dynamic" low-bit GGUFs of the large MoE targets (see
    /// docs/MODELS.md). Executable on CPU via `ferrox-quant`
    /// (scalar + AVX2 kernels, goldens cross-validated against the
    /// real compiled ggml implementation); no NEON or GPU kernels yet.
    IQ1S,
    IQ2XXS,
    IQ3XXS,
    /// The second tier of codebook-grid formats: the ones the published
    /// `UD-*` recipes reach for when `*_XXS` is too lossy. Same grid
    /// machinery, but each stores its sign bits *literally* (a byte per
    /// 8 elements) or in wider grid codes rather than as a 7-bit index
    /// into `KSIGNS_IQ2XS`, which is why they need their own kernels
    /// and their own grids rather than a parameter on the `_XXS` ones.
    /// IQ3_S in particular is not optional coverage: it is what an
    /// `IQ3_M` mix is largely made of.
    IQ2XS,
    IQ2S,
    IQ3S,
    IQ1M,
    /// GGUF block-MXFP4 (ggml tag 39): 17-byte blocks of 32 elements
    /// (1 E8M0 scale byte + 16 nibble bytes). Distinct from the
    /// safetensors two-buffer MXFP4 layout Kimi K3 uses -- same math,
    /// different byte layout. Executable on CPU via `ferrox-quant`.
    MXFP4,
    /// Plain 32-bit integer tensor (not a quantization format) -- used
    /// by e.g. DeepSeek-V4's token-hash routing tables. Recognized and
    /// sized, no execution path.
    I32,
    Other(u32),
}

impl GgmlType {
    fn from_tag(tag: u32) -> Self {
        match tag {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            16 => GgmlType::IQ2XXS,
            17 => GgmlType::IQ2XS,
            18 => GgmlType::IQ3XXS,
            19 => GgmlType::IQ1S,
            20 => GgmlType::IQ4NL,
            21 => GgmlType::IQ3S,
            22 => GgmlType::IQ2S,
            23 => GgmlType::IQ4XS,
            26 => GgmlType::I32,
            29 => GgmlType::IQ1M,
            39 => GgmlType::MXFP4,
            30 => GgmlType::BF16,
            other => GgmlType::Other(other),
        }
    }

    /// Bytes per contiguous quantization block, and elements per block.
    pub fn block_layout(&self) -> (usize, usize) {
        match self {
            GgmlType::F32 => (4, 1),
            GgmlType::F16 => (2, 1),
            GgmlType::BF16 => (2, 1),
            GgmlType::Q4_0 => (18, 32), // 1x f16 scale + 16 bytes of nibbles
            GgmlType::Q4_1 => (20, 32), // 1x f16 scale + 1x f16 min + 16 bytes of nibbles
            GgmlType::Q5_0 => (22, 32), // 1x f16 scale + 4 bytes of 5th-bit + 16 bytes of nibbles
            GgmlType::Q5_1 => (24, 32), // 1x f16 scale + 1x f16 min + 4 bytes of 5th-bit + 16 bytes of nibbles
            GgmlType::Q8_0 => (34, 32), // 1x f16 scale + 32x i8
            GgmlType::Q8_1 => (36, 32), // 1x f16 scale + 1x f16 sum + 32x i8
            GgmlType::Q2K => (84, 256), // super-block layout (approximate; see ferrox-quant)
            GgmlType::Q3K => (110, 256),
            GgmlType::Q4K => (144, 256),
            GgmlType::Q5K => (176, 256),
            GgmlType::Q6K => (210, 256),
            GgmlType::IQ4NL => (18, 32), // 1x f16 scale + 16 bytes of 4-bit codebook indices
            GgmlType::IQ4XS => (136, 256), // 1x f16 scale + split 6-bit sub-scales + 128 bytes of indices
            // Block sizes for the low-bit IQ formats are cross-checked
            // against each format's published bits-per-weight figure,
            // which the layout must reproduce exactly:
            //   IQ1_S    50 bytes/256 elems * 8 = 1.5625 bpw
            //   IQ1_M    56 bytes/256 elems * 8 = 1.75   bpw
            //   IQ2_XXS  66 bytes/256 elems * 8 = 2.0625 bpw
            //   IQ2_XS   74 bytes/256 elems * 8 = 2.3125 bpw
            //   IQ2_S    82 bytes/256 elems * 8 = 2.5625 bpw
            //   IQ3_XXS  98 bytes/256 elems * 8 = 3.0625 bpw
            //   IQ3_S   110 bytes/256 elems * 8 = 3.4375 bpw
            GgmlType::IQ1S => (50, 256), // 1x f16 scale + 32 bytes grid indices + 16 bytes qh
            GgmlType::IQ2XXS => (66, 256), // 1x f16 scale + 64 bytes of u16 grid/sign codes
            GgmlType::IQ3XXS => (98, 256), // 1x f16 scale + 96 bytes of grid/sign codes
            GgmlType::IQ2XS => (74, 256), // 1x f16 scale + 64 bytes of u16 grid/sign codes + 8 scale bytes
            GgmlType::IQ2S => (82, 256),  // 1x f16 scale + 32 grid + 32 sign + 8 qh + 8 scale bytes
            GgmlType::IQ3S => (110, 256), // 1x f16 scale + 64 grid + 8 qh + 32 sign + 4 scale bytes
            // IQ1_M is the one IQ format with *no* f16 scale field: the
            // block scale is reassembled from the four scale words' top
            // nibbles (see `ferrox-quant`), so all 56 bytes are payload.
            GgmlType::IQ1M => (56, 256), // 32 grid bytes + 16 qh + 8 scale bytes
            GgmlType::MXFP4 => (17, 32), // 1x E8M0 scale byte + 16 nibble bytes
            GgmlType::I32 => (4, 1),
            GgmlType::Other(_) => (0, 1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64, // relative to start of tensor-data section
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    pub fn byte_len(&self) -> usize {
        let (block_bytes, block_elems) = self.dtype.block_layout();
        if block_bytes == 0 {
            return 0;
        }
        let n = self.n_elements();
        (n / block_elems) * block_bytes
    }
}

/// A parsed GGUF file: metadata header plus tensor descriptors, backed by an
/// mmap so multi-hundred-gigabyte weight files are never fully copied into
/// process memory (same read-only mmap strategy llama.cpp uses).
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    mmap: Arc<Mmap>,
    data_start: usize,
}

impl GgufFile {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self, GgufError> {
        let mut cursor = io::Cursor::new(&mmap[..]);

        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = cursor.read_u32::<LittleEndian>()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        let kv_count = cursor.read_u64::<LittleEndian>()?;

        let mut metadata = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = read_gguf_string(&mut cursor)?;
            let value = read_gguf_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64::<LittleEndian>()?);
            }
            let dtype_tag = cursor.read_u32::<LittleEndian>()?;
            let offset = cursor.read_u64::<LittleEndian>()?;
            tensors.push(TensorInfo {
                name,
                shape,
                dtype: GgmlType::from_tag(dtype_tag),
                offset,
            });
        }

        // Tensor data begins at the next `general.alignment` boundary
        // (default 32) after the header. This matches the GGUF spec.
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let pos = cursor.position() as usize;
        let data_start = pos.div_ceil(alignment) * alignment;

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            mmap: Arc::new(mmap),
            data_start,
        })
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let start = self.data_start + info.offset as usize;
        let len = info.byte_len();
        if start + len > self.mmap.len() {
            return Err(GgufError::TruncatedTensor(
                name.to_string(),
                len,
                self.mmap.len().saturating_sub(start),
            ));
        }
        Ok(&self.mmap[start..start + len])
    }

    /// Zero-copy accessor: returns a cheaply-cloneable handle to the
    /// underlying mmap plus the byte range for `name`, instead of
    /// copying the tensor's bytes into a new heap allocation the way
    /// `tensor_bytes` + `.to_vec()` would. Reading blocks directly
    /// against the mmap rather than materializing a full in-memory
    /// copy means a loaded checkpoint's resident memory is the mmap
    /// itself, not a second copy of it.
    pub fn tensor_mapped_range(
        &self,
        name: &str,
    ) -> Result<(Arc<Mmap>, std::ops::Range<usize>), GgufError> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let start = self.data_start + info.offset as usize;
        let len = info.byte_len();
        if start + len > self.mmap.len() {
            return Err(GgufError::TruncatedTensor(
                name.to_string(),
                len,
                self.mmap.len().saturating_sub(start),
            ));
        }
        Ok((Arc::clone(&self.mmap), start..start + len))
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(|v| v.as_u64())
    }

    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }
}

/// One logical GGUF namespace, single-file or split: everything a model
/// loader or tokenizer needs to read metadata and tensor bytes without
/// caring how many physical files back them. Implemented by both
/// [`GgufFile`] (one file) and [`sharded::ShardedGguf`] (a validated
/// shard set); consumers written against this trait accept either.
pub trait TensorSource {
    fn metadata(&self, key: &str) -> Option<&GgufValue>;
    fn find_tensor(&self, name: &str) -> Option<&TensorInfo>;
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError>;
    fn tensor_mapped_range(
        &self,
        name: &str,
    ) -> Result<(Arc<Mmap>, std::ops::Range<usize>), GgufError>;

    fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata(key).and_then(|v| v.as_u64())
    }
    fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata(key).and_then(|v| v.as_str())
    }
    fn metadata_f32(&self, key: &str) -> Option<f32> {
        self.metadata(key).and_then(|v| v.as_f32())
    }
    fn metadata_bool(&self, key: &str) -> Option<bool> {
        self.metadata(key).and_then(|v| v.as_bool())
    }
}

impl TensorSource for GgufFile {
    fn metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }
    fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        GgufFile::find_tensor(self, name)
    }
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        GgufFile::tensor_bytes(self, name)
    }
    fn tensor_mapped_range(
        &self,
        name: &str,
    ) -> Result<(Arc<Mmap>, std::ops::Range<usize>), GgufError> {
        GgufFile::tensor_mapped_range(self, name)
    }
}

fn read_gguf_string(cursor: &mut io::Cursor<&[u8]>) -> Result<String, GgufError> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_gguf_value(cursor: &mut io::Cursor<&[u8]>) -> Result<GgufValue, GgufError> {
    let type_tag = cursor.read_u32::<LittleEndian>()?;
    read_gguf_value_typed(cursor, type_tag)
}

fn read_gguf_value_typed(
    cursor: &mut io::Cursor<&[u8]>,
    type_tag: u32,
) -> Result<GgufValue, GgufError> {
    Ok(match type_tag {
        0 => GgufValue::U8(cursor.read_u8()?),
        1 => GgufValue::I8(cursor.read_i8()?),
        2 => GgufValue::U16(cursor.read_u16::<LittleEndian>()?),
        3 => GgufValue::I16(cursor.read_i16::<LittleEndian>()?),
        4 => GgufValue::U32(cursor.read_u32::<LittleEndian>()?),
        5 => GgufValue::I32(cursor.read_i32::<LittleEndian>()?),
        6 => GgufValue::F32(cursor.read_f32::<LittleEndian>()?),
        7 => GgufValue::Bool(cursor.read_u8()? != 0),
        8 => GgufValue::String(read_gguf_string(cursor)?),
        9 => {
            let elem_tag = cursor.read_u32::<LittleEndian>()?;
            let len = cursor.read_u64::<LittleEndian>()? as usize;
            let mut items = Vec::with_capacity(len);
            for _ in 0..len {
                items.push(read_gguf_value_typed(cursor, elem_tag)?);
            }
            GgufValue::Array(items)
        }
        10 => GgufValue::U64(cursor.read_u64::<LittleEndian>()?),
        11 => GgufValue::I64(cursor.read_i64::<LittleEndian>()?),
        12 => GgufValue::F64(cursor.read_f64::<LittleEndian>()?),
        other => return Err(GgufError::UnknownValueType(other)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;
    use std::io::Write;

    /// Builds a tiny synthetic GGUF byte buffer in memory (no real model
    /// weights involved) so the parser can be exercised without any
    /// external file or network access.
    fn build_synthetic_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(2).unwrap(); // kv_count

        // kv 1: general.alignment = 32 (u32)
        write_string(&mut buf, "general.alignment");
        buf.write_u32::<LittleEndian>(4).unwrap(); // type = u32
        buf.write_u32::<LittleEndian>(32).unwrap();

        // kv 2: general.name = "synthetic-test"
        write_string(&mut buf, "general.name");
        buf.write_u32::<LittleEndian>(8).unwrap(); // type = string
        write_string(&mut buf, "synthetic-test");

        // tensor 0: "tok_embd.weight", shape [4, 8], F32
        write_string(&mut buf, "tok_embd.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
        buf.write_u64::<LittleEndian>(4).unwrap();
        buf.write_u64::<LittleEndian>(8).unwrap();
        buf.write_u32::<LittleEndian>(0).unwrap(); // dtype F32
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        // pad to 32-byte alignment
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        // tensor data: 32 f32 values
        for i in 0..32u32 {
            buf.write_f32::<LittleEndian>(i as f32 * 0.5).unwrap();
        }
        buf
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        buf.write_all(s.as_bytes()).unwrap();
    }

    #[test]
    fn parses_header_and_metadata() {
        let bytes = build_synthetic_gguf();
        let tmp = std::env::temp_dir().join("ferrox_test_synthetic.gguf");
        std::fs::write(&tmp, &bytes).unwrap();
        let f = GgufFile::open(&tmp).unwrap();
        assert_eq!(f.version, 3);
        assert_eq!(f.metadata_str("general.name"), Some("synthetic-test"));
        assert_eq!(f.metadata_u64("general.alignment"), Some(32));
        assert_eq!(f.tensors.len(), 1);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn reads_tensor_bytes_correctly() {
        let bytes = build_synthetic_gguf();
        let tmp = std::env::temp_dir().join("ferrox_test_synthetic2.gguf");
        std::fs::write(&tmp, &bytes).unwrap();
        let f = GgufFile::open(&tmp).unwrap();
        let info = f.find_tensor("tok_embd.weight").unwrap();
        assert_eq!(info.shape, vec![4, 8]);
        assert_eq!(info.n_elements(), 32);
        let raw = f.tensor_bytes("tok_embd.weight").unwrap();
        assert_eq!(raw.len(), 32 * 4);
        let first = f32::from_le_bytes(raw[0..4].try_into().unwrap());
        assert_eq!(first, 0.0);
        let second = f32::from_le_bytes(raw[4..8].try_into().unwrap());
        assert_eq!(second, 0.5);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn recognized_low_bit_types_match_their_published_bits_per_weight() {
        // Each low-bit IQ block size must reproduce that format's
        // published bits-per-weight exactly -- an arithmetic
        // cross-check independent of this file's own table, and the
        // cheapest way to catch a transposed digit in a block size
        // (which would otherwise silently mis-stride a whole tensor).
        for (ty, tag, bpw_x16) in [
            (GgmlType::IQ1S, 19u32, 25u32), // 1.5625 bpw * 16
            (GgmlType::IQ1M, 29, 28),       // 1.75   bpw * 16
            (GgmlType::IQ2XXS, 16, 33),     // 2.0625 bpw * 16
            (GgmlType::IQ2XS, 17, 37),      // 2.3125 bpw * 16
            (GgmlType::IQ2S, 22, 41),       // 2.5625 bpw * 16
            (GgmlType::IQ3XXS, 18, 49),     // 3.0625 bpw * 16
            (GgmlType::IQ3S, 21, 55),       // 3.4375 bpw * 16
        ] {
            assert_eq!(GgmlType::from_tag(tag), ty);
            let (bytes, elems) = ty.block_layout();
            assert_eq!(
                bytes * 8 * 16,
                bpw_x16 as usize * elems,
                "{ty:?} block layout does not reproduce its published bits-per-weight"
            );
        }
        assert_eq!(GgmlType::from_tag(26), GgmlType::I32);
        assert_eq!(GgmlType::I32.block_layout(), (4, 1));
        // An unknown tag still degrades to Other, never a wrong name.
        assert_eq!(GgmlType::from_tag(999), GgmlType::Other(999));
    }

    #[test]
    fn rejects_bad_magic() {
        let tmp = std::env::temp_dir().join("ferrox_test_bad.gguf");
        std::fs::write(&tmp, b"NOPE0000").unwrap();
        match GgufFile::open(&tmp) {
            Err(GgufError::BadMagic(_)) => {}
            Err(other) => panic!("expected BadMagic error, got a different error: {other}"),
            Ok(_) => panic!("expected BadMagic error, got Ok"),
        }
        std::fs::remove_file(&tmp).ok();
    }

    /// Not a correctness test: a one-off inspection tool for dumping
    /// every real tensor name/shape/dtype from a real downloaded GGUF
    /// shard, gated behind an env var so it never runs in CI (needs a
    /// real multi-GB file on disk). Used to verify
    /// `kimi_gguf_loader`'s tensor-name assumptions against a real
    /// downloaded `unsloth/Kimi-K3-GGUF` payload shard.
    #[test]
    #[ignore = "prints real tensor names from a real GGUF file at FERROX_INSPECT_PATH; not a correctness assertion"]
    fn dump_real_gguf_tensor_table() {
        let path = std::env::var("FERROX_INSPECT_PATH")
            .expect("set FERROX_INSPECT_PATH to a real .gguf file path");
        let file = GgufFile::open(&path).expect("real file must parse");
        println!(
            "architecture: {:?}",
            file.metadata_str("general.architecture")
        );
        println!("tensor_count: {}", file.tensors.len());
        for t in &file.tensors {
            println!("{:<45} shape={:?} dtype={:?}", t.name, t.shape, t.dtype);
        }
    }
}
