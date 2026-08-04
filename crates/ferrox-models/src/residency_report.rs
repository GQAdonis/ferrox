//! Dry-run residency planning: what a checkpoint would cost to run,
//! computed from its GGUF header alone -- no weights loaded, no
//! allocation. This is the "plan before you allocate" half of the
//! residency workstream; the enforcement half at decode time is the
//! bounded expert store (`ferrox_core::expert_store`) plus the global
//! device plan (`ferrox_moe::ResidencyPlan`).
//!
//! Accounting model (deliberately explicit about what it does NOT yet
//! cover): dense/always-resident weight bytes, routed-expert bytes
//! (resident, or capped by the streaming budget), per-request KV-cache
//! bytes at a stated context length, request concurrency, and a safety
//! headroom fraction. Not yet accounted: activation/scratch buffers,
//! tokenizer/runtime overhead, GPU placement (VRAM budgets are planned
//! separately by `ResidencyPlan`), and Kimi's recurrent state (this
//! report reads GGUF headers; the Kimi safetensors path has no
//! header-only reporter yet).

use ferrox_gguf::ShardedGguf;

use crate::config::ModelConfig;
use crate::loader::LoadError;

/// The inputs a plan is computed against. `expert_cache_bytes: None`
/// means routed experts load resident (mmap); `Some(budget)` means
/// they stream through the bounded store and cost at most `budget`
/// resident bytes.
#[derive(Debug, Clone, Copy)]
pub struct ResidencyAssumptions {
    pub context_tokens: usize,
    pub concurrent_requests: usize,
    pub expert_cache_bytes: Option<u64>,
    /// Fraction of host RAM held back for everything this report does
    /// not model (activations, allocator slack, OS). 0.2 is the
    /// default the CLI uses.
    pub headroom_fraction: f64,
}

/// One line of the plan, with the reason it costs what it costs.
#[derive(Debug, Clone)]
pub struct ResidencyLine {
    pub label: String,
    pub bytes: u64,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct ResidencyReport {
    pub lines: Vec<ResidencyLine>,
    pub required_bytes: u64,
    pub host_ram_bytes: u64,
    pub usable_ram_bytes: u64,
    pub assumptions: ResidencyAssumptions,
}

impl ResidencyReport {
    /// Computes the plan for the checkpoint at `path` (any shard of a
    /// split set, or a single file) against `host_ram_bytes` (pass the
    /// probed value, or a hypothetical machine's size for what-if
    /// planning).
    pub fn from_gguf(
        path: impl AsRef<std::path::Path>,
        assumptions: ResidencyAssumptions,
        host_ram_bytes: u64,
    ) -> Result<Self, LoadError> {
        let file = ShardedGguf::open(path)?;
        let config = ModelConfig::from_gguf(&file)?;

        let mut dense_bytes: u64 = 0;
        let mut routed_bytes: u64 = 0;
        let mut routed_tensors = 0usize;
        for (_, t) in file.tensors() {
            let is_routed_expert = t.name.contains("_exps.weight");
            if is_routed_expert {
                routed_bytes += t.byte_len() as u64;
                routed_tensors += 1;
            } else {
                dense_bytes += t.byte_len() as u64;
            }
        }

        let mut lines = Vec::new();
        lines.push(ResidencyLine {
            label: "dense weights".to_string(),
            bytes: dense_bytes,
            reason: "attention/norms/router/shared-expert/embedding/output tensors, \
                     always resident (quantized in place, mmap or owned)"
                .to_string(),
        });

        let expert_line = match assumptions.expert_cache_bytes {
            Some(budget) => {
                let capped = budget.min(routed_bytes);
                ResidencyLine {
                    label: "routed experts (streamed)".to_string(),
                    bytes: capped,
                    reason: format!(
                        "{routed_tensors} packed expert tensors totalling {routed_bytes} \
                         bytes on disk, streamed through a bounded cache of {budget} bytes \
                         (resident cost = min(budget, total))"
                    ),
                }
            }
            None => ResidencyLine {
                label: "routed experts (resident)".to_string(),
                bytes: routed_bytes,
                reason: format!(
                    "{routed_tensors} packed expert tensors, loaded as zero-copy mmap views \
                     -- resident under memory pressure only via OS page cache eviction; \
                     enable expert streaming to bound this explicitly"
                ),
            },
        };
        lines.push(expert_line);

        // KV cache: keys + values, f32, per layer, per kv-head, at the
        // stated context, per concurrent request.
        let kv_per_request = (config.n_layers as u64)
            * 2
            * (config.n_kv_heads as u64)
            * (config.head_dim as u64)
            * 4
            * (assumptions.context_tokens as u64);
        lines.push(ResidencyLine {
            label: "KV caches".to_string(),
            bytes: kv_per_request * assumptions.concurrent_requests as u64,
            reason: format!(
                "{} layers x 2 (K+V) x {} kv-heads x {} head-dim x 4 bytes x {} context \
                 tokens = {kv_per_request} bytes/request, x {} concurrent requests",
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                assumptions.context_tokens,
                assumptions.concurrent_requests
            ),
        });

        let required_bytes = lines.iter().map(|l| l.bytes).sum();
        let usable_ram_bytes =
            (host_ram_bytes as f64 * (1.0 - assumptions.headroom_fraction)) as u64;
        Ok(ResidencyReport {
            lines,
            required_bytes,
            host_ram_bytes,
            usable_ram_bytes,
            assumptions,
        })
    }

    pub fn fits(&self) -> bool {
        self.required_bytes <= self.usable_ram_bytes
    }

    /// Strict mode: an `Err` with the full plan text when the plan
    /// overcommits usable RAM. Callers refuse to load on `Err`.
    pub fn check_strict(&self) -> Result<(), String> {
        if self.fits() {
            Ok(())
        } else {
            Err(format!(
                "residency plan overcommits: requires {} bytes but only {} usable \
                 ({} host RAM minus {:.0}% headroom)\n{self}",
                self.required_bytes,
                self.usable_ram_bytes,
                self.host_ram_bytes,
                self.assumptions.headroom_fraction * 100.0
            ))
        }
    }
}

fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    format!("{v:.2} {}", UNITS[u])
}

impl std::fmt::Display for ResidencyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "residency plan (context={}, concurrency={}, headroom={:.0}%):",
            self.assumptions.context_tokens,
            self.assumptions.concurrent_requests,
            self.assumptions.headroom_fraction * 100.0
        )?;
        for line in &self.lines {
            writeln!(
                f,
                "  {:<28} {:>12}  {}",
                line.label,
                human(line.bytes),
                line.reason
            )?;
        }
        writeln!(
            f,
            "  {:<28} {:>12}",
            "TOTAL required",
            human(self.required_bytes)
        )?;
        writeln!(
            f,
            "  {:<28} {:>12}  ({} host RAM minus headroom)",
            "usable host RAM",
            human(self.usable_ram_bytes),
            human(self.host_ram_bytes)
        )?;
        write!(
            f,
            "  verdict: {}",
            if self.fits() {
                "FITS"
            } else {
                "DOES NOT FIT (strict mode refuses to load)"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
    }

    fn assumptions(cache: Option<u64>) -> ResidencyAssumptions {
        ResidencyAssumptions {
            context_tokens: 128,
            concurrent_requests: 2,
            expert_cache_bytes: cache,
            headroom_fraction: 0.2,
        }
    }

    #[test]
    fn moe_fixture_plan_accounts_experts_kv_and_streaming_cap() {
        let path = fixture("ferrox_real_moe_test.gguf");

        let resident = ResidencyReport::from_gguf(&path, assumptions(None), 1 << 30).expect("plan");
        let experts_resident = resident.lines[1].bytes;
        assert!(experts_resident > 0, "MoE fixture has routed expert bytes");

        // Streaming with a tiny budget caps the expert line at the
        // budget; everything else is identical.
        let streamed =
            ResidencyReport::from_gguf(&path, assumptions(Some(100)), 1 << 30).expect("plan");
        assert_eq!(streamed.lines[1].bytes, 100);
        assert_eq!(streamed.lines[0].bytes, resident.lines[0].bytes);
        assert_eq!(streamed.lines[2].bytes, resident.lines[2].bytes);
        assert_eq!(
            resident.required_bytes - streamed.required_bytes,
            experts_resident - 100
        );

        // A budget larger than the experts costs only the experts.
        let big =
            ResidencyReport::from_gguf(&path, assumptions(Some(u64::MAX)), 1 << 30).expect("plan");
        assert_eq!(big.lines[1].bytes, experts_resident);

        // KV arithmetic is exactly the documented formula.
        let file = ShardedGguf::open(&path).unwrap();
        let cfg = ModelConfig::from_gguf(&file).unwrap();
        let expected_kv = (cfg.n_layers * 2 * cfg.n_kv_heads * cfg.head_dim * 4 * 128 * 2) as u64;
        assert_eq!(resident.lines[2].bytes, expected_kv);
    }

    #[test]
    fn strict_mode_refuses_overcommit_and_accepts_a_fitting_plan() {
        let path = fixture("ferrox_real_moe_test.gguf");
        let fits = ResidencyReport::from_gguf(&path, assumptions(None), 1 << 30).unwrap();
        assert!(fits.check_strict().is_ok());

        // A "machine" with 1 byte of RAM cannot fit anything.
        let no_fit = ResidencyReport::from_gguf(&path, assumptions(None), 1).unwrap();
        let err = no_fit.check_strict().expect_err("must refuse");
        assert!(err.contains("overcommits"), "{err}");
        assert!(err.contains("DOES NOT FIT"), "{err}");
    }
}
