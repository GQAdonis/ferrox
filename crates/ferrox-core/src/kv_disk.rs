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
//! # The write-ordering invariant
//!
//! A write is accepted on one thread and finished on another, so for a
//! while a block is "in the store" without being on disk. The rule that
//! makes that safe, and the one every step below is ordered around:
//!
//! > **buffer -> index -> queue.** A concurrent reader must never see
//! > an index hit for a block that has neither a file nor a buffered
//! > payload.
//!
//! So the payload is reachable *before* anything claims the block
//! exists, and on the way out the file is published *before* the
//! buffered copy is released. A reader holds the index lock while it
//! consults the buffer, because "this block is not on disk yet" and
//! "here is its payload" have to be one decision -- as two, the writer
//! can publish and release in between and the reader finds nothing.
//! When that invariant does break, the reader gets
//! [`StoreError::MissingPayload`] rather than a quiet miss: a
//! correctness bug that degrades into a cache miss is a bug nobody ever
//! finds.
//!
//! The queue is bounded and **never drops**: a full queue makes the
//! caller write the block itself ([`DiskStats::inline_writes`] counts
//! it). Dropping writes silently would be indistinguishable from a cold
//! cache later.
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

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoreError {
    Io {
        op: &'static str,
        path: PathBuf,
        message: String,
    },
    /// The index named a block whose payload is nowhere: not on disk,
    /// not in the write buffer. This is not a cache miss -- it is the
    /// write-ordering invariant (buffer -> index -> queue) having been
    /// violated, i.e. a bug in this module, and it is surfaced rather
    /// than smoothed into a miss so a test can fail on it.
    MissingPayload { hash: BlockHash },
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
            StoreError::MissingPayload { hash } => write!(
                f,
                "KV block store index names {hash:?} but it has neither a file nor a buffered \
                 payload; the write-ordering invariant was violated"
            ),
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

/// How the store is sized, laid out, and how much writing it will do
/// off the calling thread.
#[derive(Clone, Debug)]
pub struct DiskConfig {
    /// Directory the store owns. Created if absent.
    pub root: PathBuf,
    /// Byte budget for blocks the store is accounting for. Eviction
    /// keeps the store at or under this.
    pub max_bytes: u64,
    /// Hex characters of the hash used as the shard subdirectory name.
    /// 2 gives 256 shards, which keeps directory sizes sane well past a
    /// million blocks.
    pub shard_chars: usize,
    /// Writes that may be waiting for a writer thread at once. When it
    /// is full, [`DiskKvStore::put`] writes on the calling thread
    /// instead of dropping the block -- backpressure, not loss.
    pub queue_capacity: usize,
    /// Background writer threads. `0` is legitimate and means every
    /// write happens on the thread that asked for it.
    pub writer_threads: usize,
}

impl DiskConfig {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        DiskConfig {
            root: root.into(),
            max_bytes: 1 << 30,
            shard_chars: 2,
            queue_capacity: 64,
            writer_threads: 1,
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

    pub fn with_queue_capacity(mut self, queue_capacity: usize) -> Self {
        self.queue_capacity = queue_capacity;
        self
    }

    pub fn with_writer_threads(mut self, writer_threads: usize) -> Self {
        self.writer_threads = writer_threads;
        self
    }
}

#[derive(Default)]
struct Stats {
    writes: AtomicU64,
    queued_writes: AtomicU64,
    /// Writes that ran on the calling thread because the queue was
    /// full. The plan's "count the fallbacks": a store that is
    /// permanently inline is a store whose queue is too small or whose
    /// disk is too slow, and that is invisible without this.
    inline_writes: AtomicU64,
    write_failures: AtomicU64,
    /// Queued writes whose block was evicted (or superseded) before a
    /// writer thread reached it.
    write_skipped: AtomicU64,
    write_nanos: AtomicU64,
    /// Writes whose block was evicted while it was being written, so
    /// the published file was withdrawn again.
    write_raced_eviction: AtomicU64,
    hits: AtomicU64,
    /// Reads served from the write buffer, before the block reached
    /// disk. These are what make the write path asynchronous *and*
    /// immediately visible.
    buffer_hits: AtomicU64,
    misses: AtomicU64,
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
    pub queue_depth: usize,
    pub writes: u64,
    pub queued_writes: u64,
    pub inline_writes: u64,
    pub write_failures: u64,
    pub write_skipped: u64,
    pub write_raced_eviction: u64,
    pub write_nanos: u64,
    pub hits: u64,
    pub buffer_hits: u64,
    pub misses: u64,
    pub corrupt: u64,
    pub incompatible: u64,
    pub read_nanos: u64,
    pub evictions: u64,
    pub evicted_bytes: u64,
}

struct Entry {
    bytes: u64,
    last_used: u64,
    /// False between admitting the block and its file landing under its
    /// final name. While it is false the payload lives in the write
    /// buffer, and a reader is served from there.
    published: bool,
    /// Bumped every time this hash is admitted, so a write that
    /// finishes after its entry was evicted and re-created cannot mark
    /// the *new* entry published, and a queued job whose block has been
    /// superseded can tell.
    generation: u64,
}

struct Index {
    entries: HashMap<BlockHash, Entry>,
    bytes: u64,
    clock: u64,
}

impl Index {
    fn touch(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }
}

/// A block that has been accepted but whose file is not on disk yet.
struct Buffered {
    generation: u64,
    block: Arc<KvBlock>,
}

#[derive(Clone, Copy)]
struct WriteJob {
    hash: BlockHash,
    generation: u64,
}

struct QueueState {
    jobs: VecDeque<WriteJob>,
    running: usize,
    shutdown: bool,
}

/// A bounded queue of pending block writes.
///
/// Bounded, and **never lossy**: `try_push` refusing is the caller's
/// signal to write the block itself, not to drop it. A dropped write is
/// indistinguishable from a cache miss later, which is exactly the kind
/// of silent degradation that makes a cache tier impossible to trust.
struct WriteQueue {
    state: Mutex<QueueState>,
    ready: Condvar,
    idle: Condvar,
    capacity: usize,
}

impl WriteQueue {
    fn new(capacity: usize) -> Self {
        WriteQueue {
            state: Mutex::new(QueueState {
                jobs: VecDeque::new(),
                running: 0,
                shutdown: false,
            }),
            ready: Condvar::new(),
            idle: Condvar::new(),
            capacity: capacity.max(1),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, QueueState> {
        self.state.lock().expect("kv disk write queue poisoned")
    }

    /// `false` means "full, or shutting down" -- write it yourself.
    fn try_push(&self, job: WriteJob) -> bool {
        let mut state = self.lock();
        if state.shutdown || state.jobs.len() >= self.capacity {
            return false;
        }
        state.jobs.push_back(job);
        self.ready.notify_one();
        true
    }

    fn pop_blocking(&self) -> Option<WriteJob> {
        let mut state = self.lock();
        loop {
            if let Some(job) = state.jobs.pop_front() {
                state.running += 1;
                return Some(job);
            }
            if state.shutdown {
                return None;
            }
            state = self
                .ready
                .wait(state)
                .expect("kv disk write queue poisoned");
        }
    }

    fn pop_now(&self) -> Option<WriteJob> {
        let mut state = self.lock();
        let job = state.jobs.pop_front()?;
        state.running += 1;
        Some(job)
    }

    fn finish(&self) {
        let mut state = self.lock();
        state.running -= 1;
        self.idle.notify_all();
    }

    fn shutdown(&self) {
        let mut state = self.lock();
        state.shutdown = true;
        self.ready.notify_all();
    }

    fn depth(&self) -> usize {
        self.lock().jobs.len()
    }
}

#[cfg(test)]
type Hook = Arc<dyn Fn(&BlockHash) + Send + Sync>;

/// Which order the write path uses. Production is
/// `BufferThenIndex`; the other two exist so a test can prove the
/// invariant test is not vacuous -- a concurrency test that passes on
/// broken code proves nothing.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum WriteOrder {
    #[default]
    BufferThenIndex,
    /// Index the block before buffering it: a reader can then hit the
    /// index for a block with no file and no payload.
    IndexBeforeBuffer,
    /// Release the buffered payload before marking the file published:
    /// same hole, at the other end of the write.
    DropBufferBeforeMarking,
}

#[cfg(test)]
#[derive(Default)]
struct Hooks {
    order: Mutex<WriteOrder>,
    /// Called after the rename and before the post-publish eviction
    /// re-check, so a test can evict the block in exactly the window
    /// the re-check exists to cover.
    after_rename: Mutex<Option<Hook>>,
    /// Called inside the window between the two steps of admission.
    in_put_window: Mutex<Option<Hook>>,
    /// Called inside the window between the two steps of publication.
    in_publish_window: Mutex<Option<Hook>>,
}

#[cfg(test)]
impl Hooks {
    fn fire(slot: &Mutex<Option<Hook>>, hash: &BlockHash) {
        // Cloned out before calling, so a hook that re-enters the store
        // cannot deadlock on the hook slot itself.
        let hook = slot.lock().expect("kv disk hook poisoned").clone();
        if let Some(hook) = hook {
            hook(hash);
        }
    }
}

/// Everything the store's threads share. Deliberately holds no join
/// handles: the writer threads hold an `Arc<Shared>`, so a `Drop` here
/// that joined them could run *on* a writer thread and deadlock. The
/// handles live in [`DiskKvStore`], which is not `Clone`.
struct Shared {
    root: PathBuf,
    shard_chars: usize,
    max_bytes: u64,
    index: Mutex<Index>,
    /// Blocks accepted but not yet on disk.
    ///
    /// **Lock order: `index`, then `buffer`, never the reverse.** A
    /// reader decides "the index says this block exists" and "here is
    /// its payload" as one atomic act; otherwise the writer could mark
    /// a block published and drop its buffered copy in between, and the
    /// reader would find nothing.
    buffer: Mutex<HashMap<BlockHash, Buffered>>,
    queue: WriteQueue,
    stats: Stats,
    seq: AtomicU64,
    generation: AtomicU64,
    #[cfg(test)]
    hooks: Hooks,
}

/// A content-addressed block store on disk, with its own writer
/// threads.
///
/// Not `Clone` on purpose -- it owns the writer threads and joins them
/// when it drops. Share it as an `Arc<DiskKvStore>`; every method takes
/// `&self`.
pub struct DiskKvStore {
    shared: Arc<Shared>,
    writers: Vec<std::thread::JoinHandle<()>>,
}

impl Drop for DiskKvStore {
    /// Stops accepting queued work and joins the writer threads. Blocks
    /// already queued but not started are **not** written: they were
    /// never durable, and a shutdown that waits for an arbitrarily deep
    /// queue is worse than a cold cache. Call [`Self::flush`] first if
    /// they matter.
    fn drop(&mut self) {
        self.shared.queue.shutdown();
        for writer in self.writers.drain(..) {
            let _ = writer.join();
        }
    }
}

impl DiskKvStore {
    /// Creates the store's directories and starts its writer threads.
    /// Does **not** scan `root` for pre-existing blocks: reattaching to
    /// a store left by a previous process is [`Self::reindex`], an
    /// explicit step, because it costs a directory walk and a caller
    /// may prefer to start cold.
    pub fn open(config: DiskConfig) -> Result<Self, StoreError> {
        let root = config.root.clone();
        fs::create_dir_all(&root).map_err(|e| io_err("create", &root, e))?;
        let tmp = root.join(TMP_DIR);
        fs::create_dir_all(&tmp).map_err(|e| io_err("create", &tmp, e))?;
        let shared = Arc::new(Shared {
            root,
            shard_chars: config.shard_chars.clamp(1, 8),
            max_bytes: config.max_bytes,
            index: Mutex::new(Index {
                entries: HashMap::new(),
                bytes: 0,
                clock: 0,
            }),
            buffer: Mutex::new(HashMap::new()),
            queue: WriteQueue::new(config.queue_capacity),
            stats: Stats::default(),
            seq: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            #[cfg(test)]
            hooks: Hooks::default(),
        });
        let mut writers = Vec::with_capacity(config.writer_threads);
        for n in 0..config.writer_threads {
            let shared = Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name(format!("ferrox-kv-write-{n}"))
                .spawn(move || {
                    while let Some(job) = shared.queue.pop_blocking() {
                        shared.run_job(job);
                        shared.queue.finish();
                    }
                })
                .map_err(|e| io_err("spawn writer for", &config.root, e))?;
            writers.push(handle);
        }
        Ok(DiskKvStore { shared, writers })
    }

    pub fn root(&self) -> &Path {
        &self.shared.root
    }

    /// Accepts a block: buffered, indexed, then queued. Returns as soon
    /// as the block is *visible* -- a reader can have it immediately,
    /// whether or not it has reached disk.
    ///
    /// If the write queue is full the block is written on this thread
    /// rather than dropped, and the fallback is counted.
    pub fn put(&self, hash: BlockHash, block: KvBlock) -> Result<(), StoreError> {
        self.shared.put(hash, block, false)
    }

    /// Like [`Self::put`], but always writes on the calling thread.
    pub fn put_blocking(&self, hash: BlockHash, block: KvBlock) -> Result<(), StoreError> {
        self.shared.put(hash, block, true)
    }

    /// Runs every pending write to completion, on this thread if no
    /// writer thread gets there first. Returns once the queue is empty
    /// and nothing is in flight.
    pub fn flush(&self) {
        loop {
            if let Some(job) = self.shared.queue.pop_now() {
                self.shared.run_job(job);
                self.shared.queue.finish();
                continue;
            }
            let state = self.shared.queue.lock();
            if state.jobs.is_empty() && state.running == 0 {
                return;
            }
            // Timed, so a store with no writer threads and a job pushed
            // by another thread cannot park here forever.
            let _ = self
                .shared
                .queue
                .idle
                .wait_timeout(state, std::time::Duration::from_millis(1));
        }
    }

    /// Looks a block up, verifying it against `expected` before
    /// returning it. See [`Shared::get`] for what `Ok(None)` covers.
    pub fn get(
        &self,
        hash: &BlockHash,
        expected: &CacheSignature,
    ) -> Result<Option<Arc<KvBlock>>, StoreError> {
        self.shared.get(hash, expected)
    }

    /// Adopts the blocks already under `root`, as a restart would.
    pub fn reindex(&self) -> Result<usize, StoreError> {
        self.shared.reindex()
    }

    /// Removes a block from the index, the write buffer, and the disk.
    pub fn remove(&self, hash: &BlockHash) {
        self.shared.quarantine(hash);
    }

    /// True if the index holds an entry for `hash` -- on disk or still
    /// buffered. Says nothing about whether its contents will verify.
    pub fn contains(&self, hash: &BlockHash) -> bool {
        let index = self.shared.index.lock().expect("kv disk index poisoned");
        index.entries.contains_key(hash)
    }

    /// Byte ceiling the store evicts against.
    pub fn capacity(&self) -> u64 {
        self.shared.max_bytes
    }

    /// Where a block's file lives (or would). Useful to an operator
    /// tracing one block; the file may not exist yet, or at all.
    pub fn block_path(&self, hash: &BlockHash) -> PathBuf {
        self.shared.block_path(hash)
    }

    pub fn stats(&self) -> DiskStats {
        self.shared.stats()
    }
}

impl Shared {
    fn next_generation(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// The write path, in the order the plan requires:
    ///
    /// 1. **buffer** -- the payload is reachable before anything claims
    ///    it exists;
    /// 2. **index** -- now it is claimed to exist, and is subject to
    ///    eviction and budget accounting;
    /// 3. **queue** -- only now does the write get scheduled.
    ///
    /// Reversing 1 and 2 lets a reader hit the index for a block with
    /// no file and no payload. Reversing 2 and 3 would be harmless but
    /// pointless: a queued write whose block is not indexed cannot be
    /// evicted or accounted for.
    fn put(&self, hash: BlockHash, block: KvBlock, inline: bool) -> Result<(), StoreError> {
        let bytes = encoded_len(block.signature());
        let block = Arc::new(block);
        let generation = self.next_generation();

        #[cfg(test)]
        let index_first = *self.hooks.order.lock().expect("kv disk hook poisoned")
            == WriteOrder::IndexBeforeBuffer;
        #[cfg(not(test))]
        let index_first = false;

        if index_first {
            self.reserve(hash, bytes, generation);
            #[cfg(test)]
            Hooks::fire(&self.hooks.in_put_window, &hash);
            self.buffer_block(hash, generation, Arc::clone(&block));
        } else {
            self.buffer_block(hash, generation, Arc::clone(&block));
            #[cfg(test)]
            Hooks::fire(&self.hooks.in_put_window, &hash);
            self.reserve(hash, bytes, generation);
        }

        let job = WriteJob { hash, generation };
        if !inline && self.queue.try_push(job) {
            self.stats.queued_writes.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        if !inline {
            self.stats.inline_writes.fetch_add(1, Ordering::Relaxed);
        }
        self.run_write(job, block)
    }

    fn buffer_block(&self, hash: BlockHash, generation: u64, block: Arc<KvBlock>) {
        self.buffer
            .lock()
            .expect("kv disk buffer poisoned")
            .insert(hash, Buffered { generation, block });
    }

    /// Drops a buffered payload, but only if it is still the one this
    /// generation put there -- a newer `put` for the same hash owns the
    /// slot now.
    fn release_buffer(&self, hash: &BlockHash, generation: u64) {
        let mut buffer = self.buffer.lock().expect("kv disk buffer poisoned");
        if buffer.get(hash).is_some_and(|b| b.generation == generation) {
            buffer.remove(hash);
        }
    }

    fn buffered(&self, hash: &BlockHash, generation: u64) -> Option<Arc<KvBlock>> {
        let buffer = self.buffer.lock().expect("kv disk buffer poisoned");
        buffer
            .get(hash)
            .filter(|b| b.generation == generation)
            .map(|b| Arc::clone(&b.block))
    }

    /// Runs a queued write. A block whose buffered payload is gone was
    /// evicted or superseded while it waited, and is skipped rather
    /// than resurrected.
    fn run_job(&self, job: WriteJob) {
        match self.buffered(&job.hash, job.generation) {
            Some(block) => {
                let _ = self.run_write(job, block);
            }
            None => {
                self.stats.write_skipped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn run_write(&self, job: WriteJob, block: Arc<KvBlock>) -> Result<(), StoreError> {
        let started = Instant::now();
        let result = self.write_and_publish(&job.hash, &block, job.generation);
        self.stats
            .write_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        match result {
            Ok(()) => {
                self.stats.writes.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(err) => {
                self.stats.write_failures.fetch_add(1, Ordering::Relaxed);
                self.abandon(&job.hash, job.generation);
                Err(err)
            }
        }
    }

    /// Reserves (or re-reserves) an index entry, charges its bytes, and
    /// evicts whatever that pushes over budget.
    fn reserve(&self, hash: BlockHash, bytes: u64, generation: u64) {
        let mut index = self.index.lock().expect("kv disk index poisoned");
        let last_used = index.touch();
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
        self.discard(victims);
    }

    /// Drops an entry that will never be published (a failed write).
    /// Index first, then the buffer: an entry that is gone from the
    /// index is unreachable, so no reader can be looking for the
    /// payload we are about to free.
    fn abandon(&self, hash: &BlockHash, generation: u64) {
        {
            let mut index = self.index.lock().expect("kv disk index poisoned");
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
        self.release_buffer(hash, generation);
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
        Hooks::fire(&self.hooks.after_rename, hash);

        #[cfg(test)]
        let drop_buffer_first = *self.hooks.order.lock().expect("kv disk hook poisoned")
            == WriteOrder::DropBufferBeforeMarking;
        #[cfg(not(test))]
        let drop_buffer_first = false;

        // Mark published, *then* release the buffered payload. In
        // between, a reader either sees "published" and reads the file
        // (which exists) or sees "buffered" and reads the buffer (which
        // still holds it). Releasing first opens a window where neither
        // is true.
        let survived = if drop_buffer_first {
            self.release_buffer(hash, generation);
            #[cfg(test)]
            Hooks::fire(&self.hooks.in_publish_window, hash);
            self.mark_published(hash, generation)
        } else {
            let survived = self.mark_published(hash, generation);
            #[cfg(test)]
            Hooks::fire(&self.hooks.in_publish_window, hash);
            self.release_buffer(hash, generation);
            survived
        };

        if !survived {
            // Evicted (or superseded) mid-write. The eviction already
            // released this entry's bytes and could not delete a file
            // that did not exist yet, so the file is ours to withdraw.
            let _ = fs::remove_file(&final_path);
            self.stats
                .write_raced_eviction
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn mark_published(&self, hash: &BlockHash, generation: u64) -> bool {
        let mut index = self.index.lock().expect("kv disk index poisoned");
        match index.entries.get_mut(hash) {
            Some(entry) if entry.generation == generation => {
                entry.published = true;
                true
            }
            _ => false,
        }
    }

    /// Looks a block up, verifying it against `expected` before
    /// returning it.
    ///
    /// `Ok(None)` covers every "you will have to recompute this"
    /// outcome -- absent, corrupt on disk, or built for a different
    /// config -- because they are the same answer to the caller. The
    /// counters in [`DiskStats`] tell them apart. `Err` is I/O that
    /// failed, or the write-ordering invariant breaking.
    fn get(
        &self,
        hash: &BlockHash,
        expected: &CacheSignature,
    ) -> Result<Option<Arc<KvBlock>>, StoreError> {
        enum Source {
            Disk(PathBuf),
            Buffer(Arc<KvBlock>),
        }
        let source = {
            let mut index = self.index.lock().expect("kv disk index poisoned");
            let clock = index.clock + 1;
            let Some(entry) = index.entries.get_mut(hash) else {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            };
            entry.last_used = clock;
            let published = entry.published;
            index.clock = clock;
            if published {
                Source::Disk(self.block_path(hash))
            } else {
                // The index lock is deliberately still held: "not
                // published" and "here is the buffered payload" must be
                // one decision. Two decisions leave a gap for the
                // writer to publish and release in between.
                let buffered = self
                    .buffer
                    .lock()
                    .expect("kv disk buffer poisoned")
                    .get(hash)
                    .map(|b| Arc::clone(&b.block));
                match buffered {
                    Some(block) => Source::Buffer(block),
                    None => return Err(StoreError::MissingPayload { hash: *hash }),
                }
            }
        };
        match source {
            Source::Buffer(block) => {
                if block.signature() != expected {
                    self.stats.incompatible.fetch_add(1, Ordering::Relaxed);
                    return Ok(None);
                }
                self.stats.buffer_hits.fetch_add(1, Ordering::Relaxed);
                Ok(Some(block))
            }
            Source::Disk(path) => {
                let started = Instant::now();
                let outcome = self.read_verified(&path, hash, expected);
                self.stats
                    .read_nanos
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                outcome
            }
        }
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
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                self.drop_entry(hash);
                return Ok(None);
            }
            Err(e) => return Err(io_err("read", path, e)),
        };
        let decoded = match decode_block(&bytes) {
            Ok(decoded) => decoded,
            Err(_) => {
                self.stats.corrupt.fetch_add(1, Ordering::Relaxed);
                self.quarantine(hash);
                return Ok(None);
            }
        };
        if &decoded.hash != hash {
            // The file under this name is some other block. Treat it
            // exactly like corruption: the name is the identity.
            self.stats.corrupt.fetch_add(1, Ordering::Relaxed);
            self.quarantine(hash);
            return Ok(None);
        }
        match decoded.block.verify(expected) {
            Ok(block) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                Ok(Some(Arc::new(block)))
            }
            Err(_) => {
                self.stats.incompatible.fetch_add(1, Ordering::Relaxed);
                Ok(None)
            }
        }
    }

    fn quarantine(&self, hash: &BlockHash) {
        self.drop_entry(hash);
        self.buffer
            .lock()
            .expect("kv disk buffer poisoned")
            .remove(hash);
        let _ = fs::remove_file(self.block_path(hash));
    }

    fn drop_entry(&self, hash: &BlockHash) {
        let mut index = self.index.lock().expect("kv disk index poisoned");
        if let Some(entry) = index.entries.remove(hash) {
            index.bytes -= entry.bytes;
        }
    }

    fn reindex(&self) -> Result<usize, StoreError> {
        let tmp = self.root.join(TMP_DIR);
        if let Ok(entries) = fs::read_dir(&tmp) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }
        let mut found = Vec::new();
        let shards = fs::read_dir(&self.root).map_err(|e| io_err("read", &self.root, e))?;
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
        let mut adopted = 0;
        let mut index = self.index.lock().expect("kv disk index poisoned");
        for (hash, bytes) in found {
            if index.entries.contains_key(&hash) {
                continue;
            }
            let last_used = index.touch();
            index.entries.insert(
                hash,
                Entry {
                    bytes,
                    last_used,
                    published: true,
                    generation: self.next_generation(),
                },
            );
            index.bytes += bytes;
            adopted += 1;
        }
        let victims = self.collect_victims(&mut index, None);
        drop(index);
        self.discard(victims);
        Ok(adopted)
    }

    /// Picks least-recently-used entries until the store fits its
    /// budget, removing them from the index and returning them for
    /// disposal. Index first, payload second -- an entry that is gone
    /// from the index is unreachable, whereas a file deleted while the
    /// index still points at it would be a hit that fails to open.
    fn collect_victims(&self, index: &mut Index, protect: Option<&BlockHash>) -> Vec<Victim> {
        let budget = self.max_bytes;
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
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .evicted_bytes
                    .fetch_add(entry.bytes, Ordering::Relaxed);
                victims.push(Victim {
                    hash,
                    published: entry.published,
                });
            }
        }
        victims
    }

    /// Frees what eviction removed from the index: the buffered payload
    /// and, if it got that far, the file.
    fn discard(&self, victims: Vec<Victim>) {
        if victims.is_empty() {
            return;
        }
        {
            let mut buffer = self.buffer.lock().expect("kv disk buffer poisoned");
            for victim in &victims {
                buffer.remove(&victim.hash);
            }
        }
        for victim in victims {
            if victim.published {
                let _ = fs::remove_file(self.block_path(&victim.hash));
            }
        }
    }

    fn stats(&self) -> DiskStats {
        let index = self.index.lock().expect("kv disk index poisoned");
        let stats = &self.stats;
        DiskStats {
            blocks: index.entries.len(),
            bytes: index.bytes,
            queue_depth: self.queue.depth(),
            writes: stats.writes.load(Ordering::Relaxed),
            queued_writes: stats.queued_writes.load(Ordering::Relaxed),
            inline_writes: stats.inline_writes.load(Ordering::Relaxed),
            write_failures: stats.write_failures.load(Ordering::Relaxed),
            write_skipped: stats.write_skipped.load(Ordering::Relaxed),
            write_raced_eviction: stats.write_raced_eviction.load(Ordering::Relaxed),
            write_nanos: stats.write_nanos.load(Ordering::Relaxed),
            hits: stats.hits.load(Ordering::Relaxed),
            buffer_hits: stats.buffer_hits.load(Ordering::Relaxed),
            misses: stats.misses.load(Ordering::Relaxed),
            corrupt: stats.corrupt.load(Ordering::Relaxed),
            incompatible: stats.incompatible.load(Ordering::Relaxed),
            read_nanos: stats.read_nanos.load(Ordering::Relaxed),
            evictions: stats.evictions.load(Ordering::Relaxed),
            evicted_bytes: stats.evicted_bytes.load(Ordering::Relaxed),
        }
    }

    fn block_path(&self, hash: &BlockHash) -> PathBuf {
        self.root
            .join(hash.shard_prefix(self.shard_chars))
            .join(format!("{}.{BLOCK_FILE_EXT}", hash.to_hex()))
    }

    fn tmp_path(&self, hash: &BlockHash) -> PathBuf {
        let n = self.seq.fetch_add(1, Ordering::Relaxed);
        self.root.join(TMP_DIR).join(format!(
            "{}.{}.{n}.tmp",
            hash.shard_prefix(16),
            std::process::id()
        ))
    }
}

/// An entry eviction has already removed from the index, awaiting
/// disposal of its payload.
struct Victim {
    hash: BlockHash,
    published: bool,
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
    use std::sync::atomic::AtomicUsize;

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

    /// Default test store: everything on the calling thread, so a test
    /// that does not care about the writer pool never races it.
    fn store(dir: &TempDir, max_bytes: u64) -> DiskKvStore {
        DiskKvStore::open(
            DiskConfig::new(dir.path())
                .with_max_bytes(max_bytes)
                .with_writer_threads(0),
        )
        .expect("open")
    }

    fn put_now(store: &DiskKvStore, hash: BlockHash, block: KvBlock) {
        store.put_blocking(hash, block).expect("put");
    }

    #[test]
    fn a_block_round_trips_through_a_file() {
        let dir = TempDir::new("roundtrip");
        let store = store(&dir, 1 << 20);
        let h = hash(1);
        let written = block("model-a", 3, 4, 1.0);
        let copy = block("model-a", 3, 4, 1.0);
        put_now(&store, h, written);

        let read = store
            .get(&h, &expected("model-a", 3, 4))
            .expect("get")
            .expect("the block just written must be found");
        assert_eq!(read.layers().len(), 3);
        for (a, b) in read.layers().iter().zip(copy.layers()) {
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
        put_now(&store, h, written);
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
        let store = DiskKvStore::open(
            DiskConfig::new(dir.path())
                .with_shard_chars(2)
                .with_writer_threads(0),
        )
        .expect("open");
        let h = hash(3);
        put_now(&store, h, block("model-a", 1, 2, 1.0));
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
        put_now(&store, h, block("model-a", 2, 4, 1.0));
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
        put_now(&store, h, block("model-a", 2, 4, 1.0));

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
        put_now(&store, a, block("model-a", 1, 2, 1.0));
        put_now(&store, b, block("model-a", 1, 2, 2.0));
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
            put_now(&store, *h, block("model-a", 1, 4, i as f32));
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
        put_now(&store, a, block("model-a", 1, 4, 1.0));
        put_now(&store, b, block("model-a", 1, 4, 2.0));
        // Touch `a`, so `b` is now the oldest.
        assert!(store.get(&a, &expected("model-a", 1, 4)).unwrap().is_some());
        put_now(&store, c, block("model-a", 1, 4, 3.0));

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
            let evicting = Arc::clone(&store.shared);
            let mut hook = store
                .shared
                .hooks
                .after_rename
                .lock()
                .expect("hook lock poisoned");
            *hook = Some(Arc::new(move |hash: &BlockHash| {
                // Whoever evicts cannot delete a file that does not
                // exist yet; the writer must notice and withdraw it.
                evicting.drop_entry(hash);
            }));
        }
        put_now(&store, h, block("model-a", 1, 4, 1.0));

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
            put_now(&store, hash(50 + i), block("model-a", 1, 2, i as f32));
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
            put_now(&store, h, block("model-a", 2, 4, 7.0));
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
                put_now(&store, hash(70 + i), block("model-a", 1, 4, i as f32));
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
        put_now(&store, h, block("model-a", 1, 2, 1.0));
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
        put_now(&store, h, block("model-a", 1, 4, 1.0));
        let once = store.stats().bytes;
        put_now(&store, h, block("model-a", 1, 4, 1.0));
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

    // ---------------------------------------------------------------
    // Write ordering: buffer -> index -> queue
    // ---------------------------------------------------------------

    /// Runs one `put` with a reader firing inside the window between
    /// the two ordered steps of the write path, and reports what the
    /// reader saw. Deterministic on purpose: a sleep-and-hope
    /// concurrency test that passes tells you nothing about a window it
    /// may simply have missed.
    ///
    /// Returns `(missing_payload_errors, served)`.
    fn probe_window(order: WriteOrder, publish_window: bool) -> (usize, usize) {
        let dir = TempDir::new("ordering");
        let store = store(&dir, 1 << 20);
        *store.shared.hooks.order.lock().unwrap() = order;

        let violations = Arc::new(AtomicUsize::new(0));
        let served = Arc::new(AtomicUsize::new(0));
        let reader = Arc::clone(&store.shared);
        let v = Arc::clone(&violations);
        let s = Arc::clone(&served);
        let hook: Hook = Arc::new(move |hash: &BlockHash| {
            match reader.get(hash, &expected("model-a", 1, 4)) {
                Ok(Some(_)) => {
                    s.fetch_add(1, Ordering::Relaxed);
                }
                // Not indexed yet: an honest miss, the reader simply
                // recomputes.
                Ok(None) => {}
                Err(StoreError::MissingPayload { .. }) => {
                    v.fetch_add(1, Ordering::Relaxed);
                }
                Err(other) => panic!("unexpected store error: {other}"),
            }
        });
        let slot = if publish_window {
            &store.shared.hooks.in_publish_window
        } else {
            &store.shared.hooks.in_put_window
        };
        *slot.lock().unwrap() = Some(hook);

        put_now(&store, hash(200), block("model-a", 1, 4, 1.0));
        (
            violations.load(Ordering::Relaxed),
            served.load(Ordering::Relaxed),
        )
    }

    /// The invariant. A reader that looks inside either window -- after
    /// the block is buffered but before it is indexed, and after it is
    /// published but before the buffer is released -- either misses
    /// cleanly or gets the block. It never gets an index hit with
    /// nothing behind it.
    #[test]
    fn a_reader_never_sees_an_index_hit_with_no_payload() {
        let (violations, _) = probe_window(WriteOrder::BufferThenIndex, false);
        assert_eq!(violations, 0, "admission window must be safe");

        let (violations, served) = probe_window(WriteOrder::BufferThenIndex, true);
        assert_eq!(violations, 0, "publication window must be safe");
        assert_eq!(
            served, 1,
            "the reader must actually have reached the block, or this test proves nothing"
        );
    }

    /// The proof that the test above is not vacuous: with the two steps
    /// of admission reversed -- index first, buffer second, which is
    /// the natural way to write it -- the very same reader hits an
    /// index entry for a block with no file and no payload.
    #[test]
    fn indexing_before_buffering_is_caught() {
        let (violations, _) = probe_window(WriteOrder::IndexBeforeBuffer, false);
        assert_eq!(
            violations, 1,
            "index-then-buffer must be detected as an invariant violation"
        );
    }

    /// The other end of the write, and the subtler half: releasing the
    /// buffered payload before marking the file published leaves the
    /// same gap.
    #[test]
    fn releasing_the_buffer_before_publishing_is_caught() {
        let (violations, _) = probe_window(WriteOrder::DropBufferBeforeMarking, true);
        assert_eq!(
            violations, 1,
            "release-then-mark must be detected as an invariant violation"
        );
    }

    /// The same invariant under real concurrency rather than a hook:
    /// writers queueing blocks while readers hammer them. This one can
    /// only ever *sample* the windows, which is why the deterministic
    /// probes above exist -- but it also exercises the writer threads,
    /// the queue, and eviction all at once.
    #[test]
    fn concurrent_readers_never_see_an_index_hit_with_no_payload() {
        let dir = TempDir::new("concurrent");
        let one = encoded_len(block("model-a", 1, 4, 1.0).signature());
        let store = Arc::new(
            DiskKvStore::open(
                DiskConfig::new(dir.path())
                    // Tight enough that eviction runs constantly.
                    .with_max_bytes(one * 8)
                    .with_queue_capacity(4)
                    .with_writer_threads(2),
            )
            .expect("open"),
        );
        let hashes: Vec<BlockHash> = (0..16).map(|i| hash(300 + i)).collect();
        let violations = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let readers: Vec<_> = (0..4)
            .map(|_| {
                let store = Arc::clone(&store);
                let hashes = hashes.clone();
                let violations = Arc::clone(&violations);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let want = expected("model-a", 1, 4);
                    while !stop.load(Ordering::Relaxed) {
                        for h in &hashes {
                            match store.get(h, &want) {
                                Ok(_) => {}
                                Err(StoreError::MissingPayload { .. }) => {
                                    violations.fetch_add(1, Ordering::Relaxed);
                                }
                                Err(other) => panic!("unexpected store error: {other}"),
                            }
                        }
                    }
                })
            })
            .collect();

        for round in 0..4 {
            for (i, h) in hashes.iter().enumerate() {
                store
                    .put(*h, block("model-a", 1, 4, (round * 16 + i) as f32))
                    .expect("put");
            }
        }
        store.flush();
        stop.store(true, Ordering::Relaxed);
        for reader in readers {
            reader.join().expect("reader thread");
        }

        assert_eq!(
            violations.load(Ordering::Relaxed),
            0,
            "no reader may ever see an index hit with no payload"
        );
        let stats = store.stats();
        assert!(
            stats.buffer_hits > 0,
            "readers must have caught blocks still in the write buffer, \
             or this test never entered the window"
        );
        assert!(stats.evictions > 0, "the budget must have bound");
        assert!(stats.bytes <= store.capacity());
    }

    /// A block is readable the instant `put` returns, before any writer
    /// thread has touched it. This is what makes the queue safe to use
    /// on the request path.
    #[test]
    fn a_queued_block_is_readable_before_it_reaches_disk() {
        let dir = TempDir::new("buffered");
        // No writer threads: nothing can publish until we flush.
        let store = store(&dir, 1 << 20);
        let h = hash(400);
        store.put(h, block("model-a", 1, 4, 1.0)).expect("put");

        assert!(
            !store.block_path(&h).exists(),
            "nothing has been written yet"
        );
        let got = store
            .get(&h, &expected("model-a", 1, 4))
            .expect("get")
            .expect("a queued block must be readable immediately");
        assert_eq!(got.tokens(), 4);
        assert_eq!(store.stats().buffer_hits, 1);

        store.flush();
        assert!(store.block_path(&h).exists(), "flush must publish it");
        assert!(store
            .get(&h, &expected("model-a", 1, 4))
            .expect("get")
            .is_some());
        assert_eq!(store.stats().hits, 1, "and now it comes off the disk");
    }

    /// Backpressure, not loss: when the queue is full the block is
    /// written on the calling thread. Nothing is dropped, and the
    /// fallback is counted so an operator can see a queue that is too
    /// small.
    #[test]
    fn a_full_queue_writes_inline_rather_than_dropping_the_block() {
        let dir = TempDir::new("backpressure");
        let store = DiskKvStore::open(
            DiskConfig::new(dir.path())
                .with_queue_capacity(2)
                // Nothing drains the queue, so it stays full.
                .with_writer_threads(0),
        )
        .expect("open");

        let hashes: Vec<BlockHash> = (0..5).map(|i| hash(500 + i)).collect();
        for (i, h) in hashes.iter().enumerate() {
            store
                .put(*h, block("model-a", 1, 4, i as f32))
                .expect("put");
        }
        let stats = store.stats();
        assert_eq!(stats.queued_writes, 2, "the queue holds exactly its cap");
        assert_eq!(stats.inline_writes, 3, "the rest fall back to this thread");
        assert_eq!(stats.writes, 3, "and the fallbacks really wrote");

        // Every block is readable regardless of which path it took --
        // the point of "fall back" instead of "drop".
        let want = expected("model-a", 1, 4);
        for h in &hashes {
            assert!(
                store.get(h, &want).expect("get").is_some(),
                "no block may be lost to a full queue"
            );
        }
        store.flush();
        for h in &hashes {
            assert!(store.block_path(h).exists(), "flush publishes the rest");
        }
    }

    /// A queued write whose block was evicted before a writer reached
    /// it is skipped, not resurrected: eviction has already released
    /// its bytes, so writing it would put the store over budget with a
    /// file nothing accounts for.
    #[test]
    fn a_queued_write_evicted_before_it_runs_is_skipped() {
        let dir = TempDir::new("skipped");
        let store = store(&dir, 1 << 20);
        let h = hash(600);
        store.put(h, block("model-a", 1, 4, 1.0)).expect("put");
        store.remove(&h);
        store.flush();

        let stats = store.stats();
        assert_eq!(stats.write_skipped, 1);
        assert_eq!(stats.writes, 0);
        assert!(!store.block_path(&h).exists());
        assert_eq!(stats.bytes, 0);
    }

    /// A second `put` for the same hash supersedes the first: the
    /// queued job for the older generation finds a payload that is no
    /// longer its own and skips, rather than overwriting the newer
    /// block with the older one.
    #[test]
    fn a_superseded_queued_write_does_not_overwrite_the_newer_block() {
        let dir = TempDir::new("superseded");
        let store = store(&dir, 1 << 20);
        let h = hash(700);
        store.put(h, block("model-a", 1, 4, 1.0)).expect("put");
        store.put(h, block("model-a", 1, 4, 9.0)).expect("put");
        store.flush();

        let got = store
            .get(&h, &expected("model-a", 1, 4))
            .expect("get")
            .expect("hit");
        assert_eq!(
            got.layers()[0].k[0],
            9.0,
            "the newer block must win, not whichever write ran last"
        );
        assert_eq!(store.stats().write_skipped, 1);
        assert_eq!(store.stats().blocks, 1);
    }
}
