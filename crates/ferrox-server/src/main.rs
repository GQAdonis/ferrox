//! Everything the server is lives in `lib.rs`; this binary exists so
//! `cargo install ferrox-server` keeps producing a `ferrox-server`
//! executable. The same library backs ferrox-cli's optional `serve`
//! feature, which is why there is no logic here to drift.

fn main() -> anyhow::Result<()> {
    ferrox_server::run_server(ferrox_server::ServerArgs::parse_llama_style(
        std::env::args(),
    ))
}
