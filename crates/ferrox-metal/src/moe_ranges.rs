//! llama.cpp-style Concurrent hazard tracking for MoE encode.
//!
//! Mirrors `ggml_mem_ranges` (`ggml-metal-common.cpp`): two ops may overlap
//! under `MTLDispatchTypeConcurrent` when they don't write memory another
//! op in the set reads or writes. SRC∩SRC is allowed; SRC∩DST and DST∩*
//! require a buffer-scope barrier + range reset.
//!
//! Ferrox tracks whole MTLBuffer identities (llama uses byte ranges on the
//! alloc — equivalent for our non-view scratch buffers).

use crate::gpu::memory_barrier_buffers;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

#[derive(Default)]
pub(crate) struct MoeMemRanges {
    srcs: Vec<usize>,
    dsts: Vec<usize>,
}

#[inline]
fn buf_key(b: &ProtocolObject<dyn MTLBuffer>) -> usize {
    b as *const ProtocolObject<dyn MTLBuffer> as usize
}

impl MoeMemRanges {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn reset(&mut self) {
        self.srcs.clear();
        self.dsts.clear();
    }

    /// Return false if `srcs`/`dsts` conflict with the pending Concurrent set.
    pub(crate) fn check(
        &self,
        srcs: &[&ProtocolObject<dyn MTLBuffer>],
        dsts: &[&ProtocolObject<dyn MTLBuffer>],
    ) -> bool {
        for s in srcs {
            let k = buf_key(s);
            if self.dsts.contains(&k) {
                return false;
            }
        }
        for d in dsts {
            let k = buf_key(d);
            if self.srcs.contains(&k) || self.dsts.contains(&k) {
                return false;
            }
        }
        true
    }

    pub(crate) fn add(
        &mut self,
        srcs: &[&ProtocolObject<dyn MTLBuffer>],
        dsts: &[&ProtocolObject<dyn MTLBuffer>],
    ) {
        for s in srcs {
            let k = buf_key(s);
            if !self.srcs.contains(&k) {
                self.srcs.push(k);
            }
        }
        for d in dsts {
            let k = buf_key(d);
            if !self.dsts.contains(&k) {
                self.dsts.push(k);
            }
        }
    }

    /// llama `concurrency_check` + optional `concurrency_reset` (barrier).
    pub(crate) fn begin_op(
        &mut self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        srcs: &[&ProtocolObject<dyn MTLBuffer>],
        dsts: &[&ProtocolObject<dyn MTLBuffer>],
    ) {
        if !self.check(srcs, dsts) {
            memory_barrier_buffers(encoder);
            self.reset();
        }
    }

    pub(crate) fn end_op(
        &mut self,
        srcs: &[&ProtocolObject<dyn MTLBuffer>],
        dsts: &[&ProtocolObject<dyn MTLBuffer>],
    ) {
        self.add(srcs, dsts);
    }
}

use objc2_metal::MTLComputeCommandEncoder;
