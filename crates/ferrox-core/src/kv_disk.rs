//! The disk tier for the KV prefix cache: where a block goes so that a
//! prefix survives eviction from RAM, and a process restart.
//!
//! One file per block, named by its
//! [`BlockHash`](crate::kv_block::BlockHash) and sharded into
//! subdirectories by a hex prefix of that hash, so a machine that has
//! cached a million prefixes never puts a million entries in one
//! directory. The payload is the block's per-layer K and V tensors,
//! flattened, in the cache's own dtype -- **no re-encoding, no
//! compression**: a KV block is already dense float data, and a
//! compressor would only spend CPU on the request path to lose.
//!
//! # What a reader is protected from
//!
//! A cache file outlives the process that wrote it, so every failure
//! mode here is "someone else's bytes":
//!
//! - **A torn write.** The publish is temp-file + `fsync` + `rename`,
//!   which is atomic within a directory on every filesystem ferrox
//!   targets -- a reader sees the whole file or no file. But a crash
//!   mid-`write` to the *temp* file, a truncated copy, or a partial
//!   restore from a backup can still leave a short file lying around,
//!   so the format records its own total length and a SHA-256 of its
//!   body. A file that does not match is refused
//!   ([`BlockFormatError`]), never partially deserialized.
//! - **A different build.** The format is versioned with an explicit
//!   readable-set; an unknown version is refused rather than guessed
//!   at.
//! - **A different model or config.** That is
//!   [`kv_signature`](crate::kv_signature)'s job, and this module does
//!   not duplicate it: a decoded file becomes an
//!   [`UnverifiedBlock`], and only
//!   [`UnverifiedBlock::verify`] against the reader's own expectation
//!   produces a usable [`KvBlock`].
//!
//! # Layout
//!
//! ```text
//! <root>/.tmp/<hash>.<pid>.<n>.tmp     in-progress writes
//! <root>/<hh>/<full-hex>.kvb           published blocks (hh = shard prefix)
//! ```
//!
//! # File format (version 1)
//!
//! ```text
//! magic           8   b"FRXKVBLK"
//! format_version  4   u32 LE, checked against READABLE_FORMAT_VERSIONS
//! header_len      4   u32 LE
//! body_len        8   u64 LE
//! digest         32   SHA-256 over header || body
//! header  header_len  block hash, dims, dtype, model identity
//! body      body_len  per layer: all K elements, then all V elements
//! ```
//!
//! The digest covers header and body but not the fixed prefix, so the
//! lengths are checked against the real file size *before* anything is
//! hashed or parsed -- a 4 GB `body_len` on a 200-byte file is rejected
//! by arithmetic, not by allocating.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use sha2::{Digest, Sha256};

use crate::cache::KvCache;
use crate::kv_block::BlockHash;
use crate::kv_signature::{
    CacheSignature, KvBlock, KvDtype, UnverifiedBlock, BLOCK_FORMAT_VERSION,
    READABLE_FORMAT_VERSIONS,
};

const MAGIC: &[u8; 8] = b"FRXKVBLK";
/// magic + version + header_len + body_len + digest.
const PREFIX_LEN: usize = 8 + 4 + 4 + 8 + 32;
/// Extension of a published block file.
pub const BLOCK_FILE_EXT: &str = "kvb";
/// Subdirectory holding in-progress writes. Not a valid shard name --
/// shard directories are lowercase hex, and `.` is not a hex digit --
/// so it can never collide with one.
const TMP_DIR: &str = ".tmp";

const DTYPE_F32: u32 = 0;

fn dtype_code(dtype: KvDtype) -> u32 {
    match dtype {
        KvDtype::F32 => DTYPE_F32,
    }
}

fn dtype_from_code(code: u32) -> Option<KvDtype> {
    match code {
        DTYPE_F32 => Some(KvDtype::F32),
        _ => None,
    }
}

fn dtype_width(dtype: KvDtype) -> usize {
    match dtype {
        KvDtype::F32 => 4,
    }
}

/// Why a block file was refused. Every variant means "these bytes are
/// not a block this build can read", and none of them is recoverable by
/// reading harder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockFormatError {
    /// Shorter than the fixed prefix: there is not even a header to
    /// check.
    TooShort { len: usize },
    /// Not a ferrox block file at all.
    BadMagic,
    /// Written by a build whose layout this one does not know.
    UnsupportedFormat {
        found: u32,
        readable: &'static [u32],
    },
    /// The file's own declared length does not match the bytes present
    /// -- a half-written or truncated file.
    Truncated { expected: u64, actual: u64 },
    /// Right length, wrong bytes: bit rot, an interrupted overwrite, or
    /// a file that was edited.
    ChecksumMismatch,
    /// Structurally impossible content: a dimension of zero, a body
    /// that cannot hold the tensors the header describes.
    Malformed(&'static str),
    /// A dtype code this build has no reader for.
    UnknownDtype(u32),
}

impl std::fmt::Display for BlockFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockFormatError::TooShort { len } => write!(
                f,
                "KV block file is {len} bytes, shorter than the {PREFIX_LEN}-byte header prefix"
            ),
            BlockFormatError::BadMagic => write!(f, "KV block file has the wrong magic"),
            BlockFormatError::UnsupportedFormat { found, readable } => write!(
                f,
                "KV block file format version {found} is not readable by this build (readable: {readable:?})"
            ),
            BlockFormatError::Truncated { expected, actual } => write!(
                f,
                "KV block file declares {expected} bytes but is {actual}; refusing a torn file"
            ),
            BlockFormatError::ChecksumMismatch => {
                write!(f, "KV block file failed its SHA-256 checksum")
            }
            BlockFormatError::Malformed(what) => {
                write!(f, "KV block file is malformed: {what}")
            }
            BlockFormatError::UnknownDtype(code) => {
                write!(f, "KV block file has unknown dtype code {code}")
            }
        }
    }
}

impl std::error::Error for BlockFormatError {}

/// Serializes a block. The `hash` is stored inside the file as well as
/// in its name, so a block found under a wrong or renamed path can
/// still be checked against the identity it claims.
pub fn encode_block(hash: &BlockHash, block: &KvBlock) -> Vec<u8> {
    let sig = block.signature();
    let mut header = Vec::with_capacity(64 + sig.model.len());
    header.extend_from_slice(hash.as_bytes());
    header.extend_from_slice(&(sig.n_layers as u32).to_le_bytes());
    header.extend_from_slice(&(sig.n_kv_heads as u32).to_le_bytes());
    header.extend_from_slice(&(sig.head_dim as u32).to_le_bytes());
    header.extend_from_slice(&(sig.tokens as u32).to_le_bytes());
    header.extend_from_slice(&dtype_code(sig.dtype).to_le_bytes());
    header.extend_from_slice(&(sig.model.len() as u32).to_le_bytes());
    header.extend_from_slice(sig.model.as_bytes());

    let mut body = Vec::with_capacity(body_len(sig) as usize);
    for layer in block.layers() {
        for value in &layer.k {
            body.extend_from_slice(&value.to_le_bytes());
        }
        for value in &layer.v {
            body.extend_from_slice(&value.to_le_bytes());
        }
    }

    let mut digest = Sha256::new();
    digest.update(&header);
    digest.update(&body);
    let digest: [u8; 32] = digest.finalize().into();

    let mut out = Vec::with_capacity(PREFIX_LEN + header.len() + body.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&BLOCK_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(&(body.len() as u64).to_le_bytes());
    out.extend_from_slice(&digest);
    out.extend_from_slice(&header);
    out.extend_from_slice(&body);
    out
}

/// Bytes the body of a block with this signature occupies. Lets the
/// store charge a block against its budget without serializing it
/// first.
fn body_len(sig: &CacheSignature) -> u64 {
    let per_layer = sig.tokens as u64
        * sig.n_kv_heads as u64
        * sig.head_dim as u64
        * dtype_width(sig.dtype) as u64;
    // K and V.
    per_layer * 2 * sig.n_layers as u64
}

/// Total on-disk size of a block with this signature, header included.
pub fn encoded_len(sig: &CacheSignature) -> u64 {
    let header = 32 + 4 * 6 + sig.model.len() as u64;
    PREFIX_LEN as u64 + header + body_len(sig)
}

/// The identity and payload recovered from a block file. The signature
/// is deliberately *unverified*: use
/// [`UnverifiedBlock::verify`](crate::kv_signature::UnverifiedBlock::verify)
/// to turn it into a block this process may use.
#[derive(Debug)]
pub struct DecodedBlock {
    /// The hash the file claims to be stored under.
    pub hash: BlockHash,
    pub block: UnverifiedBlock,
}

/// Parses a block file. Checks, in order: length, magic, format
/// version, declared-vs-actual size, checksum, then structure. Nothing
/// is allocated from a length field until that length has been checked
/// against the bytes actually present.
pub fn decode_block(bytes: &[u8]) -> Result<DecodedBlock, BlockFormatError> {
    if bytes.len() < PREFIX_LEN {
        return Err(BlockFormatError::TooShort { len: bytes.len() });
    }
    if &bytes[..8] != MAGIC {
        return Err(BlockFormatError::BadMagic);
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if !READABLE_FORMAT_VERSIONS.contains(&version) {
        return Err(BlockFormatError::UnsupportedFormat {
            found: version,
            readable: READABLE_FORMAT_VERSIONS,
        });
    }
    let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as u64;
    let body_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let declared = PREFIX_LEN as u64 + header_len + body_len;
    if declared != bytes.len() as u64 {
        return Err(BlockFormatError::Truncated {
            expected: declared,
            actual: bytes.len() as u64,
        });
    }
    let digest_recorded = &bytes[24..PREFIX_LEN];
    let mut digest = Sha256::new();
    digest.update(&bytes[PREFIX_LEN..]);
    let digest: [u8; 32] = digest.finalize().into();
    if digest != digest_recorded {
        return Err(BlockFormatError::ChecksumMismatch);
    }

    let header = &bytes[PREFIX_LEN..PREFIX_LEN + header_len as usize];
    let body = &bytes[PREFIX_LEN + header_len as usize..];
    if header.len() < 32 + 4 * 6 {
        return Err(BlockFormatError::Malformed(
            "header shorter than its fields",
        ));
    }
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&header[..32]);
    let hash = BlockHash::from_bytes(hash);
    let field = |i: usize| u32::from_le_bytes(header[32 + i * 4..36 + i * 4].try_into().unwrap());
    let n_layers = field(0) as usize;
    let n_kv_heads = field(1) as usize;
    let head_dim = field(2) as usize;
    let tokens = field(3) as usize;
    let dtype_code = field(4);
    let model_len = field(5) as usize;
    let dtype = dtype_from_code(dtype_code).ok_or(BlockFormatError::UnknownDtype(dtype_code))?;
    if header.len() != 32 + 4 * 6 + model_len {
        return Err(BlockFormatError::Malformed(
            "model name length disagrees with header",
        ));
    }
    let model = std::str::from_utf8(&header[32 + 4 * 6..])
        .map_err(|_| BlockFormatError::Malformed("model name is not UTF-8"))?
        .to_string();
    if n_layers == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Err(BlockFormatError::Malformed(
            "zero layers, heads, or head dim",
        ));
    }

    let per_layer_elems = tokens
        .checked_mul(n_kv_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or(BlockFormatError::Malformed("layer size overflows"))?;
    let expected_body = (per_layer_elems as u64)
        .checked_mul(2 * n_layers as u64)
        .and_then(|n| n.checked_mul(dtype_width(dtype) as u64))
        .ok_or(BlockFormatError::Malformed("body size overflows"))?;
    if expected_body != body.len() as u64 {
        return Err(BlockFormatError::Malformed(
            "body does not match declared dims",
        ));
    }

    let mut layers = Vec::with_capacity(n_layers);
    let mut offset = 0usize;
    for _ in 0..n_layers {
        let k = read_f32(&body[offset..offset + per_layer_elems * 4]);
        offset += per_layer_elems * 4;
        let v = read_f32(&body[offset..offset + per_layer_elems * 4]);
        offset += per_layer_elems * 4;
        let mut cache = KvCache::new(n_kv_heads, head_dim);
        cache.k = k;
        cache.v = v;
        cache.seq_len = tokens;
        layers.push(cache);
    }

    let signature = CacheSignature {
        format_version: version,
        model,
        n_layers,
        n_kv_heads,
        head_dim,
        dtype,
        tokens,
    };
    Ok(DecodedBlock {
        hash,
        block: UnverifiedBlock::new(Some(signature), layers),
    })
}

fn read_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Something went wrong reaching the disk tier. Corruption and
/// incompatibility are **not** here: those are misses, reported through
/// [`DiskStats`], because a caller's only sane response to either is to
/// recompute the prefix.
#[derive(Clone, Debug)]
pub enum StoreError {
    Io {
        op: &'static str,
        path: PathBuf,
        message: String,
    },
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Io { op, path, message } => {
                write!(
                    f,
                    "KV block store failed to {op} {}: {message}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for StoreError {}

fn io_err(op: &'static str, path: &Path, err: io::Error) -> StoreError {
    StoreError::Io {
        op,
        path: path.to_path_buf(),
        message: err.to_string(),
    }
}

/// How the store is sized and laid out.
#[derive(Clone, Debug)]
pub struct DiskConfig {
    /// Directory the store owns. Created if absent.
    pub root: PathBuf,
    /// Byte budget for published blocks. Eviction keeps the store at or
    /// under this.
    pub max_bytes: u64,
    /// Hex characters of the hash used as the shard subdirectory name.
    /// 2 gives 256 shards, which keeps directory sizes sane well past a
    /// million blocks.
    pub shard_chars: usize,
}

impl DiskConfig {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        DiskConfig {
            root: root.into(),
            max_bytes: 1 << 30,
            shard_chars: 2,
        }
    }

    pub fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    pub fn with_shard_chars(mut self, shard_chars: usize) -> Self {
        self.shard_chars = shard_chars.clamp(1, 8);
        self
    }
}

#[derive(Default)]
struct Stats {
    writes: AtomicU64,
    write_failures: AtomicU64,
    write_nanos: AtomicU64,
    /// Writes whose block was evicted while it was being written, so
    /// the published file was withdrawn again.
    write_raced_eviction: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    /// Index hits for a block whose file has not been published yet.
    pending_misses: AtomicU64,
    /// Files that failed [`decode_block`] and were quarantined.
    corrupt: AtomicU64,
    /// Blocks that decoded cleanly but do not match this reader's
    /// signature expectation.
    incompatible: AtomicU64,
    read_nanos: AtomicU64,
    evictions: AtomicU64,
    evicted_bytes: AtomicU64,
}

/// A snapshot of the tier's behaviour. Note the two time-valued fields:
/// hit *rate* alone cannot tell an operator whether a disk hit was
/// cheaper than recomputing the prefix, which is the only question that
/// decides whether the tier is worth having.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiskStats {
    pub blocks: usize,
    pub bytes: u64,
    pub writes: u64,
    pub write_failures: u64,
    pub write_raced_eviction: u64,
    pub write_nanos: u64,
    pub hits: u64,
    pub misses: u64,
    pub pending_misses: u64,
    pub corrupt: u64,
    pub incompatible: u64,
    pub read_nanos: u64,
    pub evictions: u64,
    pub evicted_bytes: u64,
}

struct Entry {
    bytes: u64,
    last_used: u64,
    /// False between reserving the entry and the file landing under its
    /// final name. A reader treats it as a miss rather than opening a
    /// path that may not exist yet.
    published: bool,
    /// Bumped every time this hash is re-reserved, so a write that
    /// finishes after its entry was evicted and re-created cannot mark
    /// the *new* entry published.
    generation: u64,
}

struct Index {
    entries: HashMap<BlockHash, Entry>,
    bytes: u64,
    clock: u64,
    generation: u64,
}

impl Index {
    fn touch(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }
}

#[cfg(test)]
type Hook = Box<dyn Fn(&BlockHash) + Send + Sync>;

#[cfg(test)]
#[derive(Default)]
struct Hooks {
    /// Called after the rename and before the post-publish eviction
    /// re-check, so a test can evict the block in exactly the window
    /// the re-check exists to cover.
    after_rename: Mutex<Option<Hook>>,
}

struct StoreInner {
    root: PathBuf,
    shard_chars: usize,
    max_bytes: u64,
    index: Mutex<Index>,
    stats: Stats,
    seq: AtomicU64,
    #[cfg(test)]
    hooks: Hooks,
}

/// A content-addressed block store on disk.
///
/// Cheap to clone (it is an `Arc` inside), so the store can be shared
/// by the request threads that read from it and whatever writes into
/// it.
#[derive(Clone)]
pub struct DiskKvStore {
    inner: Arc<StoreInner>,
}

impl DiskKvStore {
    /// Creates the store's directories. Does **not** scan `root` for
    /// pre-existing blocks: reattaching to a store left by a previous
    /// process is [`Self::reindex`], an explicit step, because it costs
    /// a directory walk and a caller may prefer to start cold.
    pub fn open(config: DiskConfig) -> Result<Self, StoreError> {
        let root = config.root.clone();
        fs::create_dir_all(&root).map_err(|e| io_err("create", &root, e))?;
        let tmp = root.join(TMP_DIR);
        fs::create_dir_all(&tmp).map_err(|e| io_err("create", &tmp, e))?;
        Ok(DiskKvStore {
            inner: Arc::new(StoreInner {
                root,
                shard_chars: config.shard_chars.clamp(1, 8),
                max_bytes: config.max_bytes,
                index: Mutex::new(Index {
                    entries: HashMap::new(),
                    bytes: 0,
                    clock: 0,
                    generation: 0,
                }),
                stats: Stats::default(),
                seq: AtomicU64::new(0),
                #[cfg(test)]
                hooks: Hooks::default(),
            }),
        })
    }

    pub fn root(&self) -> &Path {
        &self.inner.root
    }

    /// Adopts the blocks already under `root`, as a restart would.
    /// Every file is admitted on its *name and size* only -- the
    /// contents are checked when the block is read, which is the only
    /// place a check can be trusted anyway, and walking a 100k-block
    /// store to hash every file would make a restart cost a full read
    /// of the tier.
    ///
    /// Leftover temp files are deleted: a temp file that still exists
    /// is by definition a write that never published.
    pub fn reindex(&self) -> Result<usize, StoreError> {
        let tmp = self.inner.root.join(TMP_DIR);
        if let Ok(entries) = fs::read_dir(&tmp) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }
        let mut found = Vec::new();
        let shards =
            fs::read_dir(&self.inner.root).map_err(|e| io_err("read", &self.inner.root, e))?;
        for shard in shards.flatten() {
            if !shard.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            if shard.file_name() == TMP_DIR {
                continue;
            }
            let Ok(files) = fs::read_dir(shard.path()) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if path.extension().and_then(|e| e.to_str()) != Some(BLOCK_FILE_EXT) {
                    continue;
                }
                let Some(hash) = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(parse_hex_hash)
                else {
                    continue;
                };
                let Ok(meta) = file.metadata() else { continue };
                found.push((hash, meta.len()));
            }
        }
        let mut index = self.inner.index.lock().expect("kv disk index poisoned");
        let mut adopted = 0;
        for (hash, bytes) in found {
            if index.entries.contains_key(&hash) {
                continue;
            }
            let last_used = index.touch();
            let generation = {
                index.generation += 1;
                index.generation
            };
            index.entries.insert(
                hash,
                Entry {
                    bytes,
                    last_used,
                    published: true,
                    generation,
                },
            );
            index.bytes += bytes;
            adopted += 1;
        }
        let victims = self.collect_victims(&mut index, None);
        drop(index);
        self.delete(victims);
        Ok(adopted)
    }

    /// Writes a block and publishes it, on the calling thread.
    ///
    /// Ordering: the index entry is reserved **first**, so the block is
    /// subject to eviction for the whole duration of the write; then
    /// the file is written to a temp path, `fsync`ed, and renamed into
    /// place; then the index is re-checked. If the entry was evicted
    /// while the write was in flight, the just-renamed file is deleted
    /// -- otherwise an eviction would silently fail to free the bytes
    /// it accounted for, and the store would drift over budget one
    /// raced write at a time.
    pub fn put_blocking(&self, hash: BlockHash, block: &KvBlock) -> Result<(), StoreError> {
        let started = Instant::now();
        let bytes = encoded_len(block.signature());
        let generation = self.reserve(hash, bytes);
        let result = self.write_and_publish(&hash, block, generation);
        self.inner
            .stats
            .write_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        match result {
            Ok(()) => {
                self.inner.stats.writes.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(err) => {
                self.inner
                    .stats
                    .write_failures
                    .fetch_add(1, Ordering::Relaxed);
                self.forget(&hash, generation);
                Err(err)
            }
        }
    }

    /// Reserves (or re-reserves) an index entry, charges its bytes, and
    /// evicts whatever that pushes over budget. Returns the entry's
    /// generation.
    fn reserve(&self, hash: BlockHash, bytes: u64) -> u64 {
        let mut index = self.inner.index.lock().expect("kv disk index poisoned");
        let last_used = index.touch();
        index.generation += 1;
        let generation = index.generation;
        if let Some(previous) = index.entries.insert(
            hash,
            Entry {
                bytes,
                last_used,
                published: false,
                generation,
            },
        ) {
            index.bytes -= previous.bytes;
        }
        index.bytes += bytes;
        let victims = self.collect_victims(&mut index, Some(&hash));
        drop(index);
        self.delete(victims);
        generation
    }

    /// Drops an entry that will never be published (a failed write).
    fn forget(&self, hash: &BlockHash, generation: u64) {
        let mut index = self.inner.index.lock().expect("kv disk index poisoned");
        if index
            .entries
            .get(hash)
            .is_some_and(|e| e.generation == generation)
        {
            if let Some(entry) = index.entries.remove(hash) {
                index.bytes -= entry.bytes;
            }
        }
    }

    fn write_and_publish(
        &self,
        hash: &BlockHash,
        block: &KvBlock,
        generation: u64,
    ) -> Result<(), StoreError> {
        let bytes = encode_block(hash, block);
        let final_path = self.block_path(hash);
        let shard = final_path.parent().expect("block path has a parent");
        fs::create_dir_all(shard).map_err(|e| io_err("create", shard, e))?;
        let tmp_path = self.tmp_path(hash);
        {
            let mut file =
                fs::File::create(&tmp_path).map_err(|e| io_err("create", &tmp_path, e))?;
            if let Err(e) = file.write_all(&bytes) {
                let _ = fs::remove_file(&tmp_path);
                return Err(io_err("write", &tmp_path, e));
            }
            // Without this the rename can be durable while the contents
            // are not, which is exactly how a zero-length "published"
            // block file appears after a power loss.
            if let Err(e) = file.sync_all() {
                let _ = fs::remove_file(&tmp_path);
                return Err(io_err("sync", &tmp_path, e));
            }
        }
        fs::rename(&tmp_path, &final_path).map_err(|e| {
            let _ = fs::remove_file(&tmp_path);
            io_err("publish", &final_path, e)
        })?;

        #[cfg(test)]
        {
            let hook = self
                .inner
                .hooks
                .after_rename
                .lock()
                .expect("hook lock poisoned");
            if let Some(hook) = hook.as_ref() {
                hook(hash);
            }
        }

        let mut index = self.inner.index.lock().expect("kv disk index poisoned");
        match index.entries.get_mut(hash) {
            Some(entry) if entry.generation == generation => {
                entry.published = true;
                Ok(())
            }
            _ => {
                // Evicted (or superseded) mid-write. The eviction
                // already released this entry's bytes and could not
                // delete a file that did not exist yet, so the file is
                // ours to withdraw.
                drop(index);
                let _ = fs::remove_file(&final_path);
                self.inner
                    .stats
                    .write_raced_eviction
                    .fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
        }
    }

    /// Looks a block up, verifying it against `expected` before
    /// returning it.
    ///
    /// `Ok(None)` covers every "you will have to recompute this"
    /// outcome -- absent, not yet published, corrupt on disk, or built
    /// for a different config -- because they are the same answer to
    /// the caller. The counters in [`DiskStats`] tell them apart.
    /// `Err` is reserved for I/O that failed in a way worth surfacing.
    pub fn get(
        &self,
        hash: &BlockHash,
        expected: &CacheSignature,
    ) -> Result<Option<Arc<KvBlock>>, StoreError> {
        let path = {
            let mut index = self.inner.index.lock().expect("kv disk index poisoned");
            let clock = index.clock + 1;
            match index.entries.get_mut(hash) {
                None => {
                    self.inner.stats.misses.fetch_add(1, Ordering::Relaxed);
                    return Ok(None);
                }
                Some(entry) if !entry.published => {
                    self.inner
                        .stats
                        .pending_misses
                        .fetch_add(1, Ordering::Relaxed);
                    return Ok(None);
                }
                Some(entry) => {
                    entry.last_used = clock;
                    index.clock = clock;
                    self.block_path(hash)
                }
            }
        };
        let started = Instant::now();
        let outcome = self.read_verified(&path, hash, expected);
        self.inner
            .stats
            .read_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        outcome
    }

    fn read_verified(
        &self,
        path: &Path,
        hash: &BlockHash,
        expected: &CacheSignature,
    ) -> Result<Option<Arc<KvBlock>>, StoreError> {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // The file vanished under us -- an eviction between the
                // index lookup and the open, or an external cleaner.
                self.inner.stats.misses.fetch_add(1, Ordering::Relaxed);
                self.drop_entry(hash);
                return Ok(None);
            }
            Err(e) => return Err(io_err("read", path, e)),
        };
        let decoded = match decode_block(&bytes) {
            Ok(decoded) => decoded,
            Err(_) => {
                self.inner.stats.corrupt.fetch_add(1, Ordering::Relaxed);
                self.quarantine(hash);
                return Ok(None);
            }
        };
        if &decoded.hash != hash {
            // The file under this name is some other block. Treat it
            // exactly like corruption: the name is the identity.
            self.inner.stats.corrupt.fetch_add(1, Ordering::Relaxed);
            self.quarantine(hash);
            return Ok(None);
        }
        match decoded.block.verify(expected) {
            Ok(block) => {
                self.inner.stats.hits.fetch_add(1, Ordering::Relaxed);
                Ok(Some(Arc::new(block)))
            }
            Err(_) => {
                self.inner
                    .stats
                    .incompatible
                    .fetch_add(1, Ordering::Relaxed);
                Ok(None)
            }
        }
    }

    /// Removes a block from the index and deletes its file.
    pub fn remove(&self, hash: &BlockHash) {
        self.quarantine(hash);
    }

    fn quarantine(&self, hash: &BlockHash) {
        self.drop_entry(hash);
        let _ = fs::remove_file(self.block_path(hash));
    }

    fn drop_entry(&self, hash: &BlockHash) {
        let mut index = self.inner.index.lock().expect("kv disk index poisoned");
        if let Some(entry) = index.entries.remove(hash) {
            index.bytes -= entry.bytes;
        }
    }

    /// True if the index holds a published entry for `hash`. Says
    /// nothing about whether its contents will verify.
    pub fn contains(&self, hash: &BlockHash) -> bool {
        let index = self.inner.index.lock().expect("kv disk index poisoned");
        index.entries.get(hash).is_some_and(|e| e.published)
    }

    /// Byte ceiling the store evicts against.
    pub fn capacity(&self) -> u64 {
        self.inner.max_bytes
    }

    pub fn stats(&self) -> DiskStats {
        let index = self.inner.index.lock().expect("kv disk index poisoned");
        let stats = &self.inner.stats;
        DiskStats {
            blocks: index.entries.len(),
            bytes: index.bytes,
            writes: stats.writes.load(Ordering::Relaxed),
            write_failures: stats.write_failures.load(Ordering::Relaxed),
            write_raced_eviction: stats.write_raced_eviction.load(Ordering::Relaxed),
            write_nanos: stats.write_nanos.load(Ordering::Relaxed),
            hits: stats.hits.load(Ordering::Relaxed),
            misses: stats.misses.load(Ordering::Relaxed),
            pending_misses: stats.pending_misses.load(Ordering::Relaxed),
            corrupt: stats.corrupt.load(Ordering::Relaxed),
            incompatible: stats.incompatible.load(Ordering::Relaxed),
            read_nanos: stats.read_nanos.load(Ordering::Relaxed),
            evictions: stats.evictions.load(Ordering::Relaxed),
            evicted_bytes: stats.evicted_bytes.load(Ordering::Relaxed),
        }
    }

    /// Picks least-recently-used entries until the store fits its
    /// budget, removing them from the index and returning their paths
    /// for deletion. Index first, file second -- an entry that is gone
    /// from the index is unreachable, whereas a file deleted while the
    /// index still points at it would be a hit that fails to open.
    fn collect_victims(&self, index: &mut Index, protect: Option<&BlockHash>) -> Vec<PathBuf> {
        let budget = self.inner.max_bytes;
        if index.bytes <= budget {
            return Vec::new();
        }
        let mut candidates: Vec<(u64, BlockHash)> = index
            .entries
            .iter()
            .filter(|(hash, _)| Some(*hash) != protect)
            .map(|(hash, entry)| (entry.last_used, *hash))
            .collect();
        candidates.sort_unstable();
        let mut victims = Vec::new();
        for (_, hash) in candidates {
            if index.bytes <= budget {
                break;
            }
            if let Some(entry) = index.entries.remove(&hash) {
                index.bytes -= entry.bytes;
                self.inner.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.inner
                    .stats
                    .evicted_bytes
                    .fetch_add(entry.bytes, Ordering::Relaxed);
                if entry.published {
                    victims.push(self.block_path(&hash));
                }
            }
        }
        victims
    }

    fn delete(&self, paths: Vec<PathBuf>) {
        for path in paths {
            let _ = fs::remove_file(path);
        }
    }

    fn block_path(&self, hash: &BlockHash) -> PathBuf {
        self.inner
            .root
            .join(hash.shard_prefix(self.inner.shard_chars))
            .join(format!("{}.{BLOCK_FILE_EXT}", hash.to_hex()))
    }

    fn tmp_path(&self, hash: &BlockHash) -> PathBuf {
        let n = self.inner.seq.fetch_add(1, Ordering::Relaxed);
        self.inner.root.join(TMP_DIR).join(format!(
            "{}.{}.{n}.tmp",
            hash.shard_prefix(16),
            std::process::id()
        ))
    }
}

fn parse_hex_hash(text: &str) -> Option<BlockHash> {
    if text.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        let hi = text.as_bytes()[i * 2] as char;
        let lo = text.as_bytes()[i * 2 + 1] as char;
        *byte = ((hi.to_digit(16)? << 4) | lo.to_digit(16)?) as u8;
    }
    Some(BlockHash::from_bytes(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_block::BlockHasher;

    /// A throwaway directory that removes itself, so a failing test
    /// does not leave block files behind. `std::env::temp_dir` plus the
    /// pid and a counter: the workspace has no `tempfile` dependency
    /// and this needs eight lines.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static N: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "ferrox-kvdisk-{tag}-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("temp dir");
            TempDir(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn layer(n_kv_heads: usize, head_dim: usize, tokens: usize, fill: f32) -> KvCache {
        let mut cache = KvCache::new(n_kv_heads, head_dim);
        for t in 0..tokens {
            let k = vec![fill + t as f32; n_kv_heads * head_dim];
            let v = vec![fill - t as f32; n_kv_heads * head_dim];
            cache.push(&k, &v).expect("unpooled push cannot fail");
        }
        cache
    }

    fn block(model: &str, n_layers: usize, tokens: usize, fill: f32) -> KvBlock {
        let layers = (0..n_layers)
            .map(|l| layer(2, 4, tokens, fill + l as f32 * 100.0))
            .collect();
        KvBlock::stamp(model, layers).expect("stamp")
    }

    fn expected(model: &str, n_layers: usize, tokens: usize) -> CacheSignature {
        CacheSignature::expected(model, n_layers, 2, 4, tokens)
    }

    fn hash(n: usize) -> BlockHash {
        BlockHasher::new("model-a", &[] as &[&str]).chain(&[n, n + 1], 2)[0]
    }

    fn store(dir: &TempDir, max_bytes: u64) -> DiskKvStore {
        DiskKvStore::open(DiskConfig::new(dir.path()).with_max_bytes(max_bytes)).expect("open")
    }

    #[test]
    fn a_block_round_trips_through_a_file() {
        let dir = TempDir::new("roundtrip");
        let store = store(&dir, 1 << 20);
        let h = hash(1);
        let written = block("model-a", 3, 4, 1.0);
        store.put_blocking(h, &written).expect("put");

        let read = store
            .get(&h, &expected("model-a", 3, 4))
            .expect("get")
            .expect("the block just written must be found");
        assert_eq!(read.layers().len(), 3);
        for (a, b) in read.layers().iter().zip(written.layers()) {
            assert_eq!(a.k, b.k);
            assert_eq!(a.v, b.v);
            assert_eq!(a.seq_len, b.seq_len);
        }
        let stats = store.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.writes, 1);
        assert_eq!(stats.blocks, 1);
        assert!(stats.read_nanos > 0, "a read must be timed");
        assert!(stats.write_nanos > 0, "a write must be timed");
    }

    #[test]
    fn the_accounted_size_is_the_real_file_size() {
        let dir = TempDir::new("size");
        let store = store(&dir, 1 << 20);
        let h = hash(2);
        let written = block("model-a", 2, 8, 0.25);
        let predicted = encoded_len(written.signature());
        store.put_blocking(h, &written).expect("put");
        let on_disk = fs::metadata(store.block_path(&h)).expect("stat").len();
        assert_eq!(
            predicted, on_disk,
            "the budget charges what the file really costs"
        );
        assert_eq!(store.stats().bytes, on_disk);
    }

    #[test]
    fn blocks_are_sharded_by_hash_prefix() {
        let dir = TempDir::new("shard");
        let store =
            DiskKvStore::open(DiskConfig::new(dir.path()).with_shard_chars(2)).expect("open");
        let h = hash(3);
        store.put_blocking(h, &block("model-a", 1, 2, 1.0)).unwrap();
        let path = store.block_path(&h);
        assert_eq!(
            path.parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap(),
            &h.to_hex()[..2]
        );
        assert!(path.exists());
    }

    /// The crash-safety case, stated as the failure it prevents: a file
    /// cut short must be *refused*, not parsed into whatever the
    /// remaining bytes happen to say.
    #[test]
    fn a_truncated_file_is_refused_at_every_cut_point() {
        let h = hash(4);
        let bytes = encode_block(&h, &block("model-a", 2, 4, 3.0));
        assert!(bytes.len() > PREFIX_LEN + 16);

        // Cut inside the fixed prefix: not even a header to read.
        let err = decode_block(&bytes[..PREFIX_LEN - 1]).expect_err("short file");
        assert_eq!(
            err,
            BlockFormatError::TooShort {
                len: PREFIX_LEN - 1
            }
        );

        // Cut inside the body: the declared length no longer matches.
        for cut in [PREFIX_LEN, PREFIX_LEN + 8, bytes.len() - 4, bytes.len() - 1] {
            let err = decode_block(&bytes[..cut]).expect_err("truncated file");
            assert_eq!(
                err,
                BlockFormatError::Truncated {
                    expected: bytes.len() as u64,
                    actual: cut as u64,
                },
                "a file cut at {cut} must be refused"
            );
        }

        // A file that is the right length but has been altered.
        let mut flipped = bytes.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 0xff;
        assert_eq!(
            decode_block(&flipped).expect_err("altered file"),
            BlockFormatError::ChecksumMismatch
        );

        // And a file that is not one of ours at all.
        let mut alien = bytes;
        alien[0] = b'X';
        assert_eq!(
            decode_block(&alien).expect_err("foreign file"),
            BlockFormatError::BadMagic
        );
    }

    /// Truncation through the whole store, not just the decoder: a torn
    /// file is a miss and is removed, so the next request recomputes
    /// instead of tripping over it forever.
    #[test]
    fn a_torn_file_on_disk_is_a_miss_and_is_quarantined() {
        let dir = TempDir::new("torn");
        let store = store(&dir, 1 << 20);
        let h = hash(5);
        store.put_blocking(h, &block("model-a", 2, 4, 1.0)).unwrap();
        let path = store.block_path(&h);

        // Simulate a write that died half way.
        let full = fs::read(&path).expect("read back");
        fs::write(&path, &full[..full.len() / 2]).expect("truncate");

        let got = store.get(&h, &expected("model-a", 2, 4)).expect("get");
        assert!(got.is_none(), "a torn block must not be returned");
        assert_eq!(store.stats().corrupt, 1);
        assert!(!path.exists(), "a torn block must not be left to trip over");
        assert!(!store.contains(&h));
    }

    #[test]
    fn an_unreadable_format_version_is_refused() {
        let h = hash(6);
        let mut bytes = encode_block(&h, &block("model-a", 1, 2, 1.0));
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        // Re-checksum so the version is the only thing wrong.
        let mut digest = Sha256::new();
        digest.update(&bytes[PREFIX_LEN..]);
        let digest: [u8; 32] = digest.finalize().into();
        bytes[24..PREFIX_LEN].copy_from_slice(&digest);
        assert_eq!(
            decode_block(&bytes).expect_err("unknown version"),
            BlockFormatError::UnsupportedFormat {
                found: 99,
                readable: READABLE_FORMAT_VERSIONS,
            }
        );
    }

    /// The signature discipline is not re-implemented here, but it is
    /// enforced here: a block written under one config must not be
    /// handed to a reader expecting another.
    #[test]
    fn a_block_from_a_different_config_is_a_miss_not_a_hit() {
        let dir = TempDir::new("config");
        let store = store(&dir, 1 << 20);
        let h = hash(7);
        store.put_blocking(h, &block("model-a", 2, 4, 1.0)).unwrap();

        assert!(store
            .get(&h, &expected("model-b", 2, 4))
            .expect("get")
            .is_none());
        assert!(store
            .get(&h, &CacheSignature::expected("model-a", 2, 8, 4, 4))
            .expect("get")
            .is_none());
        assert_eq!(store.stats().incompatible, 2);
        assert_eq!(store.stats().hits, 0);
        // Still readable by a reader that does match: an incompatible
        // read is not destructive.
        assert!(store
            .get(&h, &expected("model-a", 2, 4))
            .expect("get")
            .is_some());
    }

    /// A file whose name says one block and whose contents say another
    /// is treated as corruption. The name is the identity; a store that
    /// trusted the contents instead would serve a prefix under the
    /// wrong hash, which is the silent-wrong-answer case the hashing
    /// exists to prevent.
    #[test]
    fn a_file_stored_under_the_wrong_name_is_rejected() {
        let dir = TempDir::new("misfiled");
        let store = store(&dir, 1 << 20);
        let (a, b) = (hash(8), hash(9));
        store.put_blocking(a, &block("model-a", 1, 2, 1.0)).unwrap();
        store.put_blocking(b, &block("model-a", 1, 2, 2.0)).unwrap();
        // Put b's bytes under a's name.
        let bytes = fs::read(store.block_path(&b)).expect("read b");
        fs::write(store.block_path(&a), bytes).expect("misfile");

        assert!(store
            .get(&a, &expected("model-a", 1, 2))
            .expect("get")
            .is_none());
        assert_eq!(store.stats().corrupt, 1);
    }

    #[test]
    fn eviction_keeps_the_store_inside_its_budget() {
        let dir = TempDir::new("evict");
        let one = encoded_len(block("model-a", 1, 4, 1.0).signature());
        // Room for two blocks and change, never three.
        let store = store(&dir, one * 2 + 8);
        let hashes: Vec<BlockHash> = (0..4).map(|i| hash(20 + i)).collect();
        for (i, h) in hashes.iter().enumerate() {
            store
                .put_blocking(*h, &block("model-a", 1, 4, i as f32))
                .expect("put");
            assert!(
                store.stats().bytes <= store.capacity(),
                "the store must never sit over budget"
            );
        }
        let stats = store.stats();
        assert_eq!(stats.blocks, 2);
        assert_eq!(stats.evictions, 2);
        assert!(stats.evicted_bytes >= one * 2);
        // The two oldest are gone, from the index and from the disk.
        for h in &hashes[..2] {
            assert!(!store.contains(h));
            assert!(
                !store.block_path(h).exists(),
                "an evicted file must be deleted"
            );
        }
        for h in &hashes[2..] {
            assert!(store.contains(h));
        }
    }

    #[test]
    fn a_read_makes_a_block_the_least_likely_eviction_victim() {
        let dir = TempDir::new("lru");
        let one = encoded_len(block("model-a", 1, 4, 1.0).signature());
        let store = store(&dir, one * 2 + 8);
        let (a, b, c) = (hash(30), hash(31), hash(32));
        store.put_blocking(a, &block("model-a", 1, 4, 1.0)).unwrap();
        store.put_blocking(b, &block("model-a", 1, 4, 2.0)).unwrap();
        // Touch `a`, so `b` is now the oldest.
        assert!(store.get(&a, &expected("model-a", 1, 4)).unwrap().is_some());
        store.put_blocking(c, &block("model-a", 1, 4, 3.0)).unwrap();

        assert!(store.contains(&a), "a recently read block must survive");
        assert!(!store.contains(&b));
        assert!(store.contains(&c));
    }

    /// The post-rename re-check. The block is evicted in the window
    /// between the rename and the index update -- exactly the window
    /// the re-check exists for. Without it the file stays on disk
    /// forever with nothing accounting for its bytes, and the store
    /// drifts over budget one raced write at a time.
    #[test]
    fn a_block_evicted_mid_write_does_not_leave_its_file_behind() {
        let dir = TempDir::new("raced");
        let store = store(&dir, 1 << 20);
        let h = hash(40);
        {
            let evicting = store.clone();
            let mut hook = store
                .inner
                .hooks
                .after_rename
                .lock()
                .expect("hook lock poisoned");
            *hook = Some(Box::new(move |hash| {
                // Whoever evicts cannot delete a file that does not
                // exist yet; the writer must notice and withdraw it.
                evicting.drop_entry(hash);
            }));
        }
        store
            .put_blocking(h, &block("model-a", 1, 4, 1.0))
            .expect("put");

        assert!(
            !store.block_path(&h).exists(),
            "a file published for an entry that no longer exists must be withdrawn"
        );
        assert!(!store.contains(&h));
        let stats = store.stats();
        assert_eq!(stats.write_raced_eviction, 1);
        assert_eq!(stats.bytes, 0, "no bytes may be left unaccounted");
    }

    #[test]
    fn no_temp_files_survive_a_successful_write() {
        let dir = TempDir::new("tmp");
        let store = store(&dir, 1 << 20);
        for i in 0..4 {
            store
                .put_blocking(hash(50 + i), &block("model-a", 1, 2, i as f32))
                .expect("put");
        }
        let leftovers: Vec<_> = fs::read_dir(dir.path().join(TMP_DIR))
            .expect("tmp dir")
            .flatten()
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp files must not accumulate: {leftovers:?}"
        );
    }

    /// Survival across a restart is the whole point of the tier: a
    /// second store opened on the same directory finds what the first
    /// one wrote, and sweeps away temp files that never published.
    #[test]
    fn a_new_store_reattaches_to_what_the_previous_one_published() {
        let dir = TempDir::new("restart");
        let h = hash(60);
        {
            let store = store(&dir, 1 << 20);
            store.put_blocking(h, &block("model-a", 2, 4, 7.0)).unwrap();
        }
        // A write that died before publishing.
        let orphan = dir.path().join(TMP_DIR).join("dead.tmp");
        fs::write(&orphan, b"half a block").expect("orphan");

        let reopened = store(&dir, 1 << 20);
        assert!(
            !reopened.contains(&h),
            "reattaching must be an explicit step, not a side effect of open()"
        );
        assert_eq!(reopened.reindex().expect("reindex"), 1);
        assert!(reopened.contains(&h));
        assert!(!orphan.exists(), "an unpublished temp file must be swept");

        let read = reopened
            .get(&h, &expected("model-a", 2, 4))
            .expect("get")
            .expect("a block written before the restart must still be readable");
        assert_eq!(read.tokens(), 4);
    }

    #[test]
    fn reindex_evicts_down_to_the_budget() {
        let dir = TempDir::new("reindex-evict");
        let one = encoded_len(block("model-a", 1, 4, 1.0).signature());
        {
            let store = store(&dir, 1 << 20);
            for i in 0..4 {
                store
                    .put_blocking(hash(70 + i), &block("model-a", 1, 4, i as f32))
                    .unwrap();
            }
        }
        let small = store(&dir, one * 2 + 8);
        small.reindex().expect("reindex");
        let stats = small.stats();
        assert_eq!(stats.blocks, 2, "a shrunken budget must bind on restart");
        assert!(stats.bytes <= small.capacity());
    }

    #[test]
    fn an_absent_block_is_a_plain_miss() {
        let dir = TempDir::new("miss");
        let store = store(&dir, 1 << 20);
        assert!(store
            .get(&hash(80), &expected("model-a", 1, 2))
            .expect("get")
            .is_none());
        assert_eq!(store.stats().misses, 1);
        assert_eq!(store.stats().corrupt, 0);
    }

    /// A hit whose file has been deleted behind the store's back (an
    /// external cleaner, a `rm -rf` on the shard) is a miss, and the
    /// stale entry is dropped rather than left to fail forever.
    #[test]
    fn a_file_deleted_behind_the_stores_back_is_a_miss() {
        let dir = TempDir::new("vanished");
        let store = store(&dir, 1 << 20);
        let h = hash(90);
        store.put_blocking(h, &block("model-a", 1, 2, 1.0)).unwrap();
        fs::remove_file(store.block_path(&h)).expect("remove");
        assert!(store
            .get(&h, &expected("model-a", 1, 2))
            .expect("get")
            .is_none());
        assert!(!store.contains(&h));
        assert_eq!(store.stats().bytes, 0);
    }

    #[test]
    fn rewriting_a_block_does_not_double_charge_it() {
        let dir = TempDir::new("rewrite");
        let store = store(&dir, 1 << 20);
        let h = hash(100);
        store.put_blocking(h, &block("model-a", 1, 4, 1.0)).unwrap();
        let once = store.stats().bytes;
        store.put_blocking(h, &block("model-a", 1, 4, 1.0)).unwrap();
        assert_eq!(store.stats().bytes, once);
        assert_eq!(store.stats().blocks, 1);
    }

    #[test]
    fn hex_names_round_trip() {
        let h = hash(110);
        assert_eq!(parse_hex_hash(&h.to_hex()), Some(h));
        assert_eq!(parse_hex_hash("nothex"), None);
        assert_eq!(parse_hex_hash(&"z".repeat(64)), None);
    }
}
