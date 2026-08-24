---
name: one binary, `ferrox serve` behind a feature
overview: "GOAL: stop shipping two binaries for one product. `ferrox` and `ferrox-server` are separate executables today, and a user reasonably asks why. Fold the server into the CLI as a `ferrox serve` subcommand gated on an OPTIONAL `serve` Cargo feature, so the download path gets one binary while `cargo install ferrox-cli` stays small. THE MEASUREMENT THAT DECIDES THE DESIGN (taken 2026-08-20 on Host B, release build): ferrox 7.8 MB / 78 crates, ferrox-server 14 MB / 175 crates, and the server's dependency set is a strict SUPERSET of the CLI's, diffing them, the server adds 98 crates and the CLI adds exactly one (`ferrox-cli` itself). So an unconditional merge costs the server user nothing and costs the CLI-only user +98 crates, +6 MB, and a C toolchain, because `aws-lc-sys` (C and assembly, via rustls via axum-server and ureq) enters the tree. That asymmetry is why this is a feature flag and not a plain merge. NOT A GOAL: deleting the `ferrox-server` crate, it is published on crates.io as of v0.8.0 and stays as a thin shim."
todos:
  - id: server-lib-target
    content: "PREREQUISITE, nothing else can start before it. `crates/ferrox-server` is a bin-only crate: `[[bin]] name = \"ferrox-server\" path = \"src/main.rs\"` (Cargo.toml:13-15) and everything lives in `main.rs` behind `fn main()` at main.rs:2585 with the real entry point `async fn run(...)` at main.rs:2661. Nothing outside the binary can call it. Add a `[lib]` target: move the module tree and `run()` into `lib.rs`, keep `main.rs` as a thin `fn main() -> anyhow::Result<()>` that calls it. Public surface should be small and deliberate, a `ServerArgs`-equivalent config struct plus one entry function, not `pub mod` on all 22 modules. Purely mechanical, no behaviour change, and the existing tests must keep passing unmoved"
    status: pending
  - id: serve-feature
    content: "Add an OPTIONAL `serve` feature to `ferrox-cli`: `serve = [\"dep:ferrox-server\"]`, default OFF. Gate a `Commands::Serve` arm on `#[cfg(feature = \"serve\")]`. `cargo install ferrox-cli` stays 78 crates with no C toolchain; `cargo install ferrox-cli --features serve` gets one binary that does both. Default MUST stay off, see `serve-default-off-rationale` in the body for why the obvious choice (default on) is wrong"
    status: pending
  - id: serve-subcommand-parity
    content: "`ferrox serve` must accept exactly what `ferrox-server` accepts today, or it is a downgrade dressed as a simplification. From `ServerArgs` (crates/ferrox-server/src/main.rs:90-160): -m/--model, --host, --port (including `0` for a kernel-assigned port), -t/--threads, -dev/--device, -ngl/--n-gpu-layers, --exit-on-stdin-close, --list-devices, --ui-server, --mcp-config, --allow-multiple-instances. Note docs/CLI.md:231-233 currently lists only eight of these and reads as exhaustive, fix that while you are here. Also carry the `ferrox.server.ready` stdout line (main.rs:2439) unchanged, since supervisors parse it"
    status: pending
  - id: argv-rewriter-collision
    content: "`ferrox` rewrites llama.cpp-style argv before clap sees it, against a hardcoded `SUBCOMMANDS` list (crates/ferrox-cli/src/main.rs:344-361). `serve` must be added to that list or `ferrox serve -m model.gguf` will be rewritten as an implicit `run`. This is a silent failure, it would START A COMPLETION instead of a server, so it needs a test asserting `serve` survives the rewriter, in both the feature-on and feature-off builds"
    status: pending
  - id: tls-backend-cost
    content: "DONE, and it was one feature flag. `aws-lc-sys` (C + assembly) had exactly one source: axum-server's `tls-rustls`, which is defined as `[\"tls-rustls-no-provider\", \"rustls/aws-lc-rs\"]`. ureq was innocent, it already asks rustls for `ring` with default-features off, but cargo FEATURE UNIFICATION meant axum-server's choice was imposed on ureq's rustls too, which is why `cargo tree -i aws-lc-sys` listed ureq as a parent and made it look like two independent sources. Switched to `tls-rustls-no-provider` + an explicit `rustls` dep on `ring`, and `install_ring_crypto_provider()` runs unconditionally at startup because `no-provider` moves the failure to ACCEPT time. Measured on Host B: 175 -> 173 crates, release binary 14 MB -> 13 MB, aws-lc-rs/aws-lc-sys gone entirely (`cargo tree -i aws-lc-sys` now errors with no match). aws-lc-rs alone took 18.0s to compile. FOUND AND FIXED A SEPARATE PRE-EXISTING BUG while smoke-testing this: the TLS arm passed a BLOCKING `std::net::TcpListener` to `axum_server::from_tcp_rustls`, and tokio panics on registering one. TLS was therefore completely broken on main, it bound, printed its `ferrox.server.ready` line, then panicked on first accept, so it looked like a healthy start followed by a server that answered nothing. `set_nonblocking(true)` before handover. Verified end to end: `GET /health` over HTTPS returns 200 on TLS 1.3 with AEAD-CHACHA20-POLY1305, which is a ring suite and so also proves the provider is live"
    status: completed
  - id: server-shim-crate
    content: "Keep `ferrox-server` as a published binary crate, now a thin shim over `ferrox_server::run()`, or over `ferrox-cli --features serve`, so `cargo install ferrox-server` keeps working. It went to crates.io at v0.8.0 today, and README, docs/AGENTS_COOKBOOK.md, docs/API.md, docs/CLI.md, docs/CONFIG.md, docs/MODELS.md, scripts/install.sh and all three GitHub workflows name it. Deleting it is a breaking change for every one of those; deprecate in the description and keep publishing for at least two minor versions"
    status: pending
  - id: release-and-install-path
    content: "Decide what the DOWNLOAD path ships, which is the whole point of this plan, most users never run `cargo install`. Options: (a) one `ferrox` built with `--features serve` and drop `ferrox-server` from the tarball, (b) ship both for a transition. scripts/install.sh:91 loops over `for bin in ferrox ferrox-server`, and .github/workflows/release.yml copies both into the archive. If (a), install.sh must still not leave a stale `ferrox-server` on an upgrading user's PATH shadowing the new one, that is the failure mode a user would report as `--ui-server` silently doing nothing"
    status: pending
  - id: one-instance-registry
    content: "`ferrox` already refuses to be the second process holding the same model (crates/ferrox-cli/src/main.rs:387-423, `--allow-multiple-instances`). Under one binary, `ferrox serve` and `ferrox run` become the same executable, so check the registry still distinguishes a serving instance from a completion run and does not refuse a legitimate pair. Its rationale is a benchmarking one (docs/CLI.md:288-292) and it must survive intact"
    status: pending
  - id: docs-single-binary
    content: "Rewrite the install and usage surface once the shape is decided: README, docs/CLI.md (the flag list above), docs/API.md, docs/AGENTS_COOKBOOK.md, docs/CONFIG.md, crates/ferrox-cli/README.md, crates/ferrox-server/README.md. State the feature explicitly, a reader who runs `cargo install ferrox-cli` and finds no `serve` subcommand has hit exactly the confusion this plan exists to remove, and the error for the missing subcommand should say `rebuild with --features serve`, not `unrecognized subcommand`"
    status: pending
  - id: hygiene-generate-tmp
    content: "DONE. `crates/ferrox-server/src/generate.rs.tmp`, a 7-byte checked-in stray, deleted"
    status: completed
isProject: false
---

# One binary, `ferrox serve` behind a feature

> Prompted by a fair question: *why two binaries for one product?*
> The answer below is measured, not argued: the merge is worth doing,
> but not unconditionally, and the reason is a C compiler.

## The measurement

Host B, release build, 2026-08-20:

| | `ferrox` | `ferrox-server` |
|---|---|---|
| Binary | 7.8 MB | 14 MB |
| Unique crates | 78 | 175 |

The number that decides the design is the **diff**, not the totals:

- crates in `ferrox-server` but not `ferrox`: **98**
- crates in `ferrox` but not `ferrox-server`: **1** (`ferrox-cli` itself)

The server's dependency set is a strict superset of the CLI's. So a
plain merge is **asymmetric**: it costs a server user nothing and costs a
CLI-only user everything. Concretely, +98 crates, +6 MB, and this:

```
aws-lc-sys v0.43.0
└── aws-lc-rs v1.17.3
    ├── rustls v0.23.43
    │   ├── axum-server v0.8.0  ← ferrox-server
    │   └── tokio-rustls v0.26.4 ← axum-server
    └── ureq v2.12.1            ← ferrox-server
```

`aws-lc-sys` is a C-and-assembly crypto library. Merging unconditionally
means somebody who types `cargo install ferrox-cli` to run one prompt
compiles a TLS stack, and needs a C toolchain to do it. That is a worse
first experience than the two binaries they were confused by.

Hence: **a feature, not a merge.**

## The shape

```bash
cargo install ferrox-cli                    # 78 crates, no C toolchain, no `serve`
cargo install ferrox-cli --features serve   # one binary, does both
```

The prebuilt tarballs and the install script ship the `--features serve`
build, so for the majority of users, who download rather than compile, there is exactly one binary and the original question disappears.

This fits what `ferrox` already is. It has fifteen subcommands, and one
of them (`chat --url`) is already a *client* of the server. A `serve`
sibling is the natural completion of that surface, not a new idea bolted
onto it.

### `serve-default-off-rationale`

The obvious choice is to default the feature ON so `cargo install
ferrox-cli` "just works". It is the wrong one, for a reason that only
shows up in someone else's build log: a default-on feature is what a
dependent gets unless they remember `default-features = false`, so every
crate that ever depends on `ferrox-cli` inherits a TLS stack it did not
ask for. Default off, documented loudly, and shipped on in the artifacts
people actually download.

## What could go wrong, ranked

1. **The argv rewriter silently swallows `serve`.** `ferrox` accepts
   llama.cpp-style flags by rewriting argv before clap sees it, keyed off
   a hardcoded `SUBCOMMANDS` list (`crates/ferrox-cli/src/main.rs:344-361`).
   Miss `serve` there and `ferrox serve -m model.gguf` is rewritten into
   an implicit `run`, it would start a *completion* instead of a server,
   print tokens, and exit. No error, no clue. This is the one that needs
   a test before it needs code.
2. **Flag parity quietly regressing.** `ServerArgs`
   (`crates/ferrox-server/src/main.rs:90-160`) carries eleven flags;
   `docs/CLI.md:231-233` documents eight and reads as exhaustive. A
   reimplementation that follows the docs instead of the struct ships a
   downgrade. Port from the struct.
3. **A stale `ferrox-server` shadowing the new binary.** If the tarball
   stops shipping it, an upgrading user still has the old one on PATH.
   `scripts/install.sh:91` loops `for bin in ferrox ferrox-server`. The
   symptom would be reported as "`--ui-server` does nothing".
4. **The one-instance registry misfiring.** `ferrox` refuses to be the
   second process holding a model
   (`crates/ferrox-cli/src/main.rs:387-423`). Under one binary a server
   and a completion run are the same executable; the registry must still
   tell them apart.

## What this is not

Not a deletion of `ferrox-server`. It was published to crates.io at
v0.8.0 on 2026-08-20, and it is named in README, five files under
`docs/`, `scripts/install.sh`, and all three GitHub workflows. It stays
as a thin shim over the library target for at least two minor versions,
with the deprecation stated in its crate description rather than
discovered.

## Order of work

1. `server-lib-target`, mechanical, unblocks everything, no behaviour change.
2. `tls-backend-cost`, independent, and it shrinks the worst case for
   everything after it. Worth doing even if the rest is abandoned.
3. `serve-feature` + `serve-subcommand-parity` + `argv-rewriter-collision`.
   The actual change, and the rewriter test comes first.
4. `one-instance-registry`, `server-shim-crate`.
5. `release-and-install-path`, then `docs-single-binary` once the shape
   is settled rather than while it is moving.

## Definition of done

- `cargo install ferrox-cli` builds with no C toolchain and has no
  `serve` subcommand, and says how to get one.
- `cargo install ferrox-cli --features serve` serves an API identical to
  today's `ferrox-server`, including the `ferrox.server.ready` stdout
  line supervisors parse.
- `cargo install ferrox-server` still works.
- The release tarball's story about how many binaries it contains
  matches what it actually contains.
- Crate count and binary size recorded before and after, on the same
  host, in the same session.
