//! Rendering a server's cache geometry for a human: which pools the
//! model has, how big each one is, and what it costs in VRAM.
//!
//! Shared by every client that reports it, so the same server document
//! reads the same way whichever one you are holding. Everything here is
//! derived from a served geometry document; nothing talks to a server
//! and nothing loads a model, so a control CLI that links this stays
//! dependency-light.
//!
//! The rule that shapes the output: **a column nothing can be said
//! about is dropped whole rather than filled with zeros.** A server that
//! reported no per-unit costs gets no `vram` column, because `0.0 GiB`
//! is a lie and a total that silently omits it is worse. A pool the
//! model does not have is not a row.
//!
//! Ported 1:1 from FreeToken's `python/freetoken/cache_report.py`
//! (Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.

use serde::{Deserialize, Serialize};

/// Every pool a rebuild knows about, and the unit a user types it in.
pub const CACHE_UNITS: [(&str, &str); 4] = [
    ("moe", "slots"),
    ("kv", "tokens"),
    ("mamba", "slots"),
    ("swa", "tokens"),
];

/// The engine's per-unit VRAM costs, as acknowledged at load.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct UnitBytes {
    pub moe_per_expert: u64,
    pub mamba_per_slot: u64,
    pub kv_per_token: u64,
    pub swa_per_token: u64,
}

/// What a rebuild will accept for one pool, in the unit a client types.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Limit {
    pub min: u64,
    pub max: u64,
}

/// Per-pool rebuild bounds. The server already denominates these in the
/// unit a client types -- tokens for the paged pools, slots for the
/// others -- so a range reads as "what you may ask this pool to
/// become".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Limits {
    pub moe_experts: Option<Limit>,
    pub kv_tokens: Option<Limit>,
    pub mamba_slots: Option<Limit>,
    pub swa_tokens: Option<Limit>,
}

/// The served cache geometry: what the engine allocated, what each unit
/// costs, and what a live rebuild would accept.
///
/// Every field defaults to zero/absent, which throughout this module
/// means the same thing -- "nothing to report" -- and never an error. A
/// client talking to an older engine renders what it can and drops the
/// rest.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheGeometry {
    pub num_experts: u64,
    pub num_moe_layers: u64,
    pub moe_cache_size: u64,
    pub moe_cache_policy: Option<String>,
    pub num_pages: u64,
    pub page_size: u64,
    pub num_mamba_slots: u64,
    pub num_swa_pages: u64,
    pub swa_page_size: u64,
    pub cache_budget_bytes: u64,
    pub unit_bytes: UnitBytes,
    pub limits: Limits,
}

/// Which cache pools the served model actually has, read off its
/// geometry.
///
/// Drives what a client offers and accepts: a sparse-attention model
/// has a window pool but no recurrent-state pool, a hybrid model has
/// the reverse, a dense model has no expert cache. Offering a target
/// the model does not have is just a way to hand the user an error from
/// the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachePools {
    pub moe: bool,
    /// Every model has a KV pool; a rebuild always takes `num_pages`.
    pub kv: bool,
    pub mamba: bool,
    pub swa: bool,
}

impl Default for CachePools {
    fn default() -> Self {
        CachePools {
            moe: false,
            kv: true,
            mamba: false,
            swa: false,
        }
    }
}

impl CachePools {
    pub fn from_geometry(geometry: &CacheGeometry) -> Self {
        let experts = geometry.num_experts * geometry.num_moe_layers;
        CachePools {
            moe: experts > 0
                || geometry.moe_cache_size > 0
                || geometry.unit_bytes.moe_per_expert > 0,
            kv: true,
            mamba: geometry.num_mamba_slots > 0 || geometry.unit_bytes.mamba_per_slot > 0,
            // `swa_page_size` is the definitive signal: it is non-zero
            // only for a window pool.
            swa: geometry.swa_page_size > 0 || geometry.unit_bytes.swa_per_token > 0,
        }
    }

    /// The pools a client may name, in report order.
    pub fn targets(&self) -> Vec<&'static str> {
        [
            ("moe", self.moe),
            ("kv", self.kv),
            ("mamba", self.mamba),
            ("swa", self.swa),
        ]
        .iter()
        .filter(|(_, present)| *present)
        .map(|(name, _)| *name)
        .collect()
    }
}

pub fn format_percent(rate: f64) -> String {
    let percent = rate * 100.0;
    if (percent - percent.round()).abs() < 1e-9 {
        format!("{percent:.0}%")
    } else {
        format!("{percent:.1}%")
    }
}

pub fn format_bytes(num_bytes: u64) -> String {
    if num_bytes >= 1 << 30 {
        format!("{:.1} GiB", num_bytes as f64 / (1u64 << 30) as f64)
    } else {
        format!("{:.1} MiB", num_bytes as f64 / (1u64 << 20) as f64)
    }
}

/// `<tokens> tok` for a paged pool, with the page arithmetic spelled
/// out -- a rebuild takes tokens but allocates pages, so the
/// decomposition is what makes a rounded-up request (and the pool's
/// granularity) legible.
pub fn format_tokens(pages: u64, page_size: u64) -> String {
    format!("{} tok ({} x {})", pages * page_size, pages, page_size)
}

/// MoE residency: cached slots over the model's total routed experts
/// (experts per layer x MoE layers, the same basis the engine sizes the
/// cache against). `None` for a non-MoE model, or a server that
/// reported no expert counts.
pub fn cache_rate(cache_size: u64, geometry: &CacheGeometry) -> Option<f64> {
    let total = geometry.num_experts * geometry.num_moe_layers;
    if total == 0 || cache_size == 0 {
        return None;
    }
    Some(cache_size as f64 / total as f64)
}

/// Round a token count up to whole pages. Pools are allocated in pages,
/// so asking for 1000 tokens of a 64-token page gets 16 pages (1024
/// tokens) -- never 15, which would be less than what was asked for.
pub fn pages_for_tokens(tokens: u64, page_size: u64) -> u64 {
    let page_size = page_size.max(1);
    (tokens.div_ceil(page_size)).max(1)
}

/// VRAM a pool holds: its unit count times the engine's per-unit cost.
///
/// This is the *allocated* size, not occupancy -- the pool is reserved
/// up front, so this is what it costs whether or not anything is cached
/// in it. `0` means "unknown": a pool the model does not have, or a
/// server that reported no `unit_bytes` at all. Callers must not print
/// a `0` as a real figure.
///
/// `units` overrides the count from the geometry, for costing a size
/// that is not live yet -- a rebuild target, or the size a rebuild just
/// returned.
pub fn pool_bytes(geometry: &CacheGeometry, pool: &str, units: Option<u64>) -> u64 {
    let unit_bytes = &geometry.unit_bytes;
    match pool {
        "moe" => units.unwrap_or(geometry.moe_cache_size) * unit_bytes.moe_per_expert,
        "mamba" => units.unwrap_or(geometry.num_mamba_slots) * unit_bytes.mamba_per_slot,
        // Units are pages; the cost is per token.
        "kv" => {
            units.unwrap_or(geometry.num_pages)
                * geometry.page_size.max(1)
                * unit_bytes.kv_per_token
        }
        "swa" => {
            units.unwrap_or(geometry.num_swa_pages)
                * geometry.swa_page_size
                * unit_bytes.swa_per_token
        }
        _ => 0,
    }
}

/// `min..max` a rebuild accepts for a pool, in the same unit the size
/// column shows. Empty when the server advertised no usable bounds --
/// an empty range prints as an empty cell rather than a misleading
/// `0..0`.
pub fn format_range(geometry: &CacheGeometry, pool: &str) -> String {
    let limit = match pool {
        "moe" => geometry.limits.moe_experts,
        "kv" => geometry.limits.kv_tokens,
        "mamba" => geometry.limits.mamba_slots,
        "swa" => geometry.limits.swa_tokens,
        _ => None,
    };
    match limit {
        Some(limit) if limit.max > 0 => format!("{}..{}", limit.min, limit.max),
        _ => String::new(),
    }
}

/// One rendered row: pool name, its size in the unit a client types,
/// the bytes it holds (`0` = unknown), and the rebuild range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheRow {
    pub pool: String,
    pub detail: String,
    pub allocated_bytes: u64,
    pub resize_range: String,
}

/// One row per pool the model has, in report order.
pub fn cache_status_rows(geometry: &CacheGeometry) -> Vec<CacheRow> {
    let pools = CachePools::from_geometry(geometry);
    let page_size = geometry.page_size.max(1);
    let mut rows = Vec::new();
    let row = |pool: &str, detail: String| CacheRow {
        pool: pool.to_string(),
        detail,
        allocated_bytes: pool_bytes(geometry, pool, None),
        resize_range: format_range(geometry, pool),
    };

    if pools.moe {
        let moe = geometry.moe_cache_size;
        let policy = geometry.moe_cache_policy.as_deref().unwrap_or("lru");
        let mut detail = format!("{moe} slots ({policy}");
        match cache_rate(moe, geometry) {
            Some(rate) => detail.push_str(&format!(", {})", format_percent(rate))),
            None => detail.push(')'),
        }
        rows.push(row("moe", detail));
    }
    if geometry.num_pages > 0 {
        rows.push(row("kv", format_tokens(geometry.num_pages, page_size)));
    }
    if geometry.num_mamba_slots > 0 {
        // Hybrid (recurrent-state) models only.
        rows.push(row("mamba", format!("{} slots", geometry.num_mamba_slots)));
    }
    if geometry.swa_page_size > 0 && geometry.num_swa_pages > 0 {
        // Window-pool models only.
        rows.push(row(
            "swa",
            format_tokens(geometry.num_swa_pages, geometry.swa_page_size),
        ));
    }
    rows
}

/// The served geometry as a table -- one row per pool the model has,
/// carrying its size, the VRAM it holds, and the range a rebuild would
/// accept -- under a header with the maintenance state and the engine's
/// cache budget.
pub fn format_cache_status(geometry: &CacheGeometry, state: &str, prefix: &str) -> String {
    let rows = cache_status_rows(geometry);
    let known: Vec<u64> = rows
        .iter()
        .map(|r| r.allocated_bytes)
        .filter(|b| *b > 0)
        .collect();
    let budget = geometry.cache_budget_bytes;

    let mut header = format!("{prefix}state={state}");
    if !known.is_empty() {
        let total: u64 = known.iter().sum();
        header.push_str(&format!(", {} allocated", format_bytes(total)));
        if budget > 0 {
            // Named for what it is: the ceiling the rebuild fit-check
            // enforces, NOT a cap on what is already allocated. An
            // auto-sized pool can sit above it (and then every rebuild
            // is rejected), so calling it "of X" would read as an
            // arithmetic error.
            header.push_str(&format!(" (rebuild budget {})", format_bytes(budget)));
        }
    }
    if rows.is_empty() {
        return header;
    }

    let with_vram = !known.is_empty();
    let with_range = rows.iter().any(|r| !r.resize_range.is_empty());
    let mut table: Vec<Vec<String>> = Vec::with_capacity(rows.len() + 1);
    let mut head = vec!["pool".to_string(), "size".to_string()];
    if with_vram {
        head.push("vram".to_string());
    }
    if with_range {
        head.push("resizable to".to_string());
    }
    table.push(head);
    for r in &rows {
        let mut cells = vec![r.pool.clone(), r.detail.clone()];
        if with_vram {
            cells.push(if r.allocated_bytes > 0 {
                format_bytes(r.allocated_bytes)
            } else {
                String::new()
            });
        }
        if with_range {
            cells.push(r.resize_range.clone());
        }
        table.push(cells);
    }

    let widths: Vec<usize> = (0..table[0].len())
        .map(|i| {
            table
                .iter()
                .map(|row| row[i].chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect();
    let mut out = header;
    for row in &table {
        let line: Vec<String> = row
            .iter()
            .zip(widths.iter())
            .map(|(cell, width)| format!("{cell:<width$}"))
            .collect();
        out.push('\n');
        out.push_str("  ");
        out.push_str(line.join("  ").trim_end());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moe_geometry() -> CacheGeometry {
        CacheGeometry {
            num_experts: 128,
            num_moe_layers: 4,
            moe_cache_size: 128,
            num_pages: 64,
            page_size: 16,
            cache_budget_bytes: 4 << 30,
            unit_bytes: UnitBytes {
                moe_per_expert: 1 << 20,
                kv_per_token: 1024,
                ..UnitBytes::default()
            },
            limits: Limits {
                moe_experts: Some(Limit { min: 8, max: 512 }),
                kv_tokens: Some(Limit {
                    min: 256,
                    max: 65536,
                }),
                ..Limits::default()
            },
            ..CacheGeometry::default()
        }
    }

    #[test]
    fn pools_are_read_off_the_geometry() {
        let pools = CachePools::from_geometry(&moe_geometry());
        assert_eq!(pools.targets(), vec!["moe", "kv"]);

        let dense = CacheGeometry {
            num_pages: 8,
            page_size: 1,
            ..CacheGeometry::default()
        };
        assert_eq!(CachePools::from_geometry(&dense).targets(), vec!["kv"]);
    }

    /// A window pool is signalled by its page size, not by a page
    /// count: a model with a window has one even before anything is
    /// allocated in it.
    #[test]
    fn a_window_pool_is_detected_by_its_page_size() {
        let swa = CacheGeometry {
            swa_page_size: 1,
            ..CacheGeometry::default()
        };
        assert!(CachePools::from_geometry(&swa).swa);
    }

    #[test]
    fn a_rebuild_never_rounds_a_request_down() {
        assert_eq!(pages_for_tokens(1000, 64), 16); // 1024 tokens, not 960
        assert_eq!(pages_for_tokens(1024, 64), 16);
        assert_eq!(pages_for_tokens(0, 64), 1);
        assert_eq!(pages_for_tokens(5, 0), 5);
    }

    #[test]
    fn pool_cost_is_units_times_the_engines_per_unit_price() {
        let g = moe_geometry();
        assert_eq!(pool_bytes(&g, "moe", None), 128 << 20);
        assert_eq!(pool_bytes(&g, "kv", None), 64 * 16 * 1024);
        // A costing for a size that is not live yet.
        assert_eq!(pool_bytes(&g, "moe", Some(256)), 256 << 20);
        // Unknown pools and unpriced pools both cost "unknown".
        assert_eq!(pool_bytes(&g, "mamba", None), 0);
        assert_eq!(pool_bytes(&g, "nonsense", None), 0);
    }

    #[test]
    fn residency_is_slots_over_all_routed_experts() {
        let g = moe_geometry();
        assert_eq!(cache_rate(g.moe_cache_size, &g), Some(0.25));
        assert_eq!(format_percent(0.25), "25%");
        assert_eq!(format_percent(0.255), "25.5%");
        assert_eq!(cache_rate(0, &g), None);
        assert_eq!(cache_rate(4, &CacheGeometry::default()), None);
    }

    #[test]
    fn a_status_table_names_every_pool_and_prices_it() {
        let text = format_cache_status(&moe_geometry(), "serving", "cache: ");
        assert!(text.starts_with("cache: state=serving, "), "{text}");
        assert!(text.contains("rebuild budget 4.0 GiB"), "{text}");
        assert!(text.contains("128 slots (lru, 25%)"), "{text}");
        assert!(text.contains("1024 tok (64 x 16)"), "{text}");
        assert!(text.contains("8..512"), "{text}");
    }

    /// An engine that reported no per-unit costs must not be rendered
    /// as costing 0.0 GiB -- the column is dropped instead.
    #[test]
    fn an_unpriced_server_gets_no_vram_column() {
        let g = CacheGeometry {
            num_pages: 8,
            page_size: 4,
            ..CacheGeometry::default()
        };
        let text = format_cache_status(&g, "serving", "cache: ");
        assert!(!text.contains("vram"), "{text}");
        assert!(!text.contains("GiB"), "{text}");
        assert!(!text.contains("allocated"), "{text}");
        assert!(text.contains("32 tok (8 x 4)"), "{text}");
    }

    #[test]
    fn a_server_with_no_pools_renders_only_the_header() {
        let text = format_cache_status(&CacheGeometry::default(), "loading", "cache: ");
        assert_eq!(text, "cache: state=loading");
    }

    #[test]
    fn an_unbounded_pool_prints_no_range_rather_than_zero_to_zero() {
        let g = CacheGeometry {
            num_pages: 4,
            page_size: 1,
            limits: Limits {
                kv_tokens: Some(Limit { min: 0, max: 0 }),
                ..Limits::default()
            },
            ..CacheGeometry::default()
        };
        assert_eq!(format_range(&g, "kv"), "");
        assert!(!format_cache_status(&g, "serving", "").contains("0..0"));
    }
}
