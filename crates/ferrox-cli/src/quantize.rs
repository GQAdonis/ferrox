//! `ferrox quantize`: read a GGUF, write a GGUF whose eligible tensors
//! are re-encoded to a quantization ferrox can actually produce.
//!
//! Today that is Q8_0 and nothing else. See [`policy`] for why the
//! other targets refuse by name instead of being approximated, and for
//! the tensor-eligibility rules this shares with llama.cpp.
//!
//! The pass is streaming: the input is mmap'd, tensors are re-encoded
//! one at a time, and the output is written through a `BufWriter`. A
//! 70B checkpoint costs the output's page cache plus one tensor's f32
//! expansion, not the model.

pub mod policy;

use std::collections::BTreeMap;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use ferrox_gguf::{GgmlType, GgufFile, GgufValue, GgufWriter, TensorPlan};
use rayon::prelude::*;

use policy::{disposition, parse_target, Disposition, Target};

/// `general.quantization_version`, ggml's `GGML_QNT_VERSION`. Every
/// tool in the ecosystem reads it; a file without it looks pre-2023.
const GGML_QNT_VERSION: u32 = 2;

/// Rows re-encoded per rayon task. Big enough that the per-task
/// overhead disappears, small enough that a 128k-row embedding table
/// does not become 128k allocations.
const ROWS_PER_TASK: usize = 64;

#[derive(Parser, Debug)]
pub struct QuantizeArgs {
    /// Source GGUF. Its quantizable tensors must be F32, F16 or BF16:
    /// re-quantizing an already-quantized checkpoint compounds two
    /// roundings and is refused rather than done quietly.
    pub input: PathBuf,

    /// Destination GGUF. Defaults to `<input stem>-<TYPE>.gguf` beside
    /// the input. (llama-quantize's default is `ggml-model-<TYPE>.gguf`
    /// in the same directory, which collides the moment two models
    /// share one; this one does not.)
    pub output: Option<PathBuf>,

    /// Quantization to write. ferrox can write Q8_0. Every other
    /// llama.cpp target is refused BY NAME -- ferrox reads them all and
    /// encodes one, and a subcommand that pretended otherwise would
    /// hand back a file that loads and is worse.
    #[arg(long = "type", default_value = "Q8_0")]
    pub ty: String,

    /// Print the plan (per tensor: quantize or copy, and why) and the
    /// resulting size, without writing anything.
    #[arg(long)]
    pub dry_run: bool,

    /// Overwrite the output if it already exists.
    #[arg(long)]
    pub force: bool,
}

/// One tensor's decided fate, and the numbers that follow from it.
struct Planned {
    name: String,
    shape: Vec<u64>,
    source_dtype: GgmlType,
    out_dtype: GgmlType,
    source_bytes: usize,
    out_bytes: usize,
    /// `None` for a tensor being copied; `Some(reason)` is the reason.
    copy_reason: Option<&'static str>,
}

pub fn run(args: QuantizeArgs) -> Result<()> {
    let target = parse_target(&args.ty).map_err(|e| anyhow::anyhow!("{e}"))?;

    let file =
        GgufFile::open(&args.input).with_context(|| format!("opening {}", args.input.display()))?;

    // A split checkpoint's shards each carry a slice of the tensors and
    // a `split.*` header that would be a lie on a single output file.
    // Refuse by name rather than silently quantizing one thirteenth of
    // a model.
    if file
        .metadata_u64(ferrox_gguf::sharded::SPLIT_COUNT_KEY)
        .is_some_and(|n| n > 1)
    {
        bail!(
            "{} is one shard of a split GGUF. `ferrox quantize` writes a single file and has no \
             --keep-split; merge the shards first, or quantize the unsplit source.",
            args.input.display()
        );
    }

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| default_output_path(&args.input, target));
    if !args.dry_run {
        if output == args.input {
            bail!("output would overwrite the input ({})", output.display());
        }
        if output.exists() && !args.force {
            bail!(
                "{} already exists (pass --force to overwrite)",
                output.display()
            );
        }
    }

    let planned = plan(&file, target)?;

    let mut metadata: BTreeMap<String, GgufValue> = file
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    metadata.insert(
        "general.file_type".to_string(),
        GgufValue::U32(target.llama_ftype()),
    );
    metadata.insert(
        "general.quantization_version".to_string(),
        GgufValue::U32(GGML_QNT_VERSION),
    );

    let src_total: u64 = planned.iter().map(|p| p.source_bytes as u64).sum();
    let out_total: u64 = planned.iter().map(|p| p.out_bytes as u64).sum();
    let n_quantized = planned.iter().filter(|p| p.copy_reason.is_none()).count();

    println!(
        "quantize: {} -> {}",
        args.input.display(),
        if args.dry_run {
            "(dry run, nothing written)".to_string()
        } else {
            output.display().to_string()
        }
    );
    println!("  target: {}", target.name());
    for (i, p) in planned.iter().enumerate() {
        match p.copy_reason {
            Some(reason) => println!(
                "  [{:>4}/{}] {:<44} {:?} {:>10} B  copy ({reason})",
                i + 1,
                planned.len(),
                p.name,
                p.source_dtype,
                p.source_bytes
            ),
            None => println!(
                "  [{:>4}/{}] {:<44} {:?} {:>10} B -> {:?} {:>10} B",
                i + 1,
                planned.len(),
                p.name,
                p.source_dtype,
                p.source_bytes,
                p.out_dtype,
                p.out_bytes
            ),
        }
    }
    println!(
        "  {n_quantized}/{} tensors quantized; {:.2} MiB -> {:.2} MiB ({:.2}x)",
        planned.len(),
        src_total as f64 / (1024.0 * 1024.0),
        out_total as f64 / (1024.0 * 1024.0),
        src_total as f64 / out_total.max(1) as f64,
    );

    if args.dry_run {
        return Ok(());
    }

    let plan_entries: Vec<TensorPlan> = planned
        .iter()
        .map(|p| TensorPlan {
            name: p.name.clone(),
            shape: p.shape.clone(),
            dtype: p.out_dtype,
            byte_len: p.out_bytes,
        })
        .collect();

    let out_file =
        std::fs::File::create(&output).with_context(|| format!("creating {}", output.display()))?;
    let mut writer = GgufWriter::create(
        BufWriter::with_capacity(4 << 20, out_file),
        &metadata,
        plan_entries,
    )?;

    for p in &planned {
        let src = file.tensor_bytes(&p.name)?;
        if p.copy_reason.is_some() {
            writer.write_tensor(&p.name, src)?;
        } else {
            let encoded = encode_tensor(p, src, target)?;
            writer.write_tensor(&p.name, &encoded)?;
        }
    }
    writer.finish()?.into_inner()?;

    println!("wrote {}", output.display());
    Ok(())
}

fn default_output_path(input: &Path, target: Target) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "ggml-model".to_string());
    input.with_file_name(format!("{stem}-{}.gguf", target.name()))
}

/// Decides every tensor's fate and sizes the output, refusing before a
/// single byte is written. The refusals here are the point: a tensor
/// this build cannot encode must stop the run, not be quietly copied
/// through at F16 into a file labelled Q8_0.
fn plan(file: &GgufFile, target: Target) -> Result<Vec<Planned>> {
    let block_elems = target.block_elems();
    let mut out = Vec::with_capacity(file.tensors.len());

    for t in &file.tensors {
        let source_bytes = t.byte_len().ok_or_else(|| {
            anyhow::anyhow!(
                "tensor '{}' has dtype {:?}, whose block layout this build does not know, so it \
                 cannot even be copied through",
                t.name,
                t.dtype
            )
        })?;

        match disposition(&t.name, &t.shape, t.dtype, target) {
            Disposition::Copy(reason) => out.push(Planned {
                name: t.name.clone(),
                shape: t.shape.clone(),
                source_dtype: t.dtype,
                out_dtype: t.dtype,
                source_bytes,
                out_bytes: source_bytes,
                copy_reason: Some(reason),
            }),
            Disposition::Quantize => {
                if !matches!(t.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
                    bail!(
                        "tensor '{}' is {:?}. `ferrox quantize` reads F32/F16/BF16 sources only: \
                         re-quantizing an already-quantized tensor stacks a second rounding on \
                         the first, and the result is worse than quantizing the original \
                         checkpoint once. Convert from the original weights instead.",
                        t.name,
                        t.dtype
                    );
                }
                let n_cols = t.shape[0] as usize;
                if !n_cols.is_multiple_of(block_elems) {
                    bail!(
                        "tensor '{}' has {n_cols} columns, which is not a multiple of {}'s block \
                         size ({block_elems}). llama.cpp has no fallback type for {} either -- it \
                         stops here too.",
                        t.name,
                        target.name(),
                        target.name()
                    );
                }
                let n_elements = t.element_count().ok_or_else(|| {
                    anyhow::anyhow!("tensor '{}' declares an unrepresentable shape", t.name)
                })?;
                let (block_bytes, _) = target.ggml_type().block_layout();
                out.push(Planned {
                    name: t.name.clone(),
                    shape: t.shape.clone(),
                    source_dtype: t.dtype,
                    out_dtype: target.ggml_type(),
                    source_bytes,
                    out_bytes: n_elements / block_elems * block_bytes,
                    copy_reason: None,
                });
            }
        }
    }
    Ok(out)
}

/// Re-encodes one tensor, row by row, in parallel over row groups.
///
/// Row-wise and not whole-tensor because a Q8_0 block must not straddle
/// two rows: the reader walks a row at a time, so a block spanning the
/// boundary would be decoded with the wrong scale for half its values.
/// `n_cols % block_elems == 0` (checked in `plan`) is what makes the
/// row-wise and flat tilings identical -- but relying on that instead
/// of tiling per row is how the next format, with a 256-element block,
/// would silently break.
fn encode_tensor(p: &Planned, src: &[u8], target: Target) -> Result<Vec<u8>> {
    let n_cols = p.shape[0] as usize;
    let n_rows = (p.shape.iter().product::<u64>() as usize)
        .checked_div(n_cols)
        .unwrap_or(0);
    let src_row_bytes = source_bytes_per_element(p.source_dtype) * n_cols;
    let out_row_bytes = p.out_bytes / n_rows.max(1);

    let groups: Vec<Vec<u8>> = (0..n_rows)
        .collect::<Vec<_>>()
        .par_chunks(ROWS_PER_TASK)
        .map(|rows| {
            let mut buf = Vec::with_capacity(rows.len() * out_row_bytes);
            let mut scratch = vec![0f32; n_cols];
            for &r in rows {
                let row = &src[r * src_row_bytes..(r + 1) * src_row_bytes];
                decode_source_row(p.source_dtype, row, &mut scratch)?;
                match target {
                    Target::Q8_0 => {
                        ferrox_quant::encode_row_q8_0(&scratch, &mut buf).ok_or_else(|| {
                            anyhow::anyhow!(
                                "tensor '{}' row length {n_cols} is not a whole number of Q8_0 \
                                 blocks",
                                p.name
                            )
                        })?
                    }
                }
            }
            Ok(buf)
        })
        .collect::<Result<Vec<Vec<u8>>>>()?;

    let mut out = Vec::with_capacity(p.out_bytes);
    for g in groups {
        out.extend_from_slice(&g);
    }
    debug_assert_eq!(out.len(), p.out_bytes);
    Ok(out)
}

fn source_bytes_per_element(dtype: GgmlType) -> usize {
    match dtype {
        GgmlType::F32 => 4,
        GgmlType::F16 | GgmlType::BF16 => 2,
        // `plan` refuses every other source dtype before this is
        // reached; 0 would produce a zero-length row and a silently
        // wrong file, so make it impossible to compute one.
        other => unreachable!("source dtype {other:?} reached the encoder"),
    }
}

fn decode_source_row(dtype: GgmlType, row: &[u8], out: &mut [f32]) -> Result<()> {
    match dtype {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(row.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*c);
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                *o = half::f16::from_le_bytes(*c).to_f32();
            }
        }
        GgmlType::BF16 => {
            // BF16 is f32's top 16 bits, so the widening is a shift,
            // not a format conversion.
            for (o, c) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                *o = f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16);
            }
        }
        other => bail!("source dtype {other:?} reached the encoder"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ferrox-quantize-{}-{}-{:?}",
            tag,
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes a tiny but structurally real F16 GGUF: a 2-D weight, a
    /// 1-D norm, and a tensor llama.cpp keeps at source precision.
    fn write_f16_source(path: &Path) -> Vec<f32> {
        let n = 64usize;
        let values: Vec<f32> = (0..n * 2)
            .map(|i| ((i as f32) * 0.037).sin() * 0.8)
            .collect();
        let f16_bytes = |vals: &[f32]| -> Vec<u8> {
            vals.iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect()
        };
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".into()),
        );
        metadata.insert("general.file_type".to_string(), GgufValue::U32(1));

        let w = f16_bytes(&values);
        let norm = f16_bytes(&values[..64]);
        let gate = f16_bytes(&values[..128]);
        let plan = vec![
            TensorPlan {
                name: "blk.0.attn_q.weight".into(),
                shape: vec![64, 2],
                dtype: GgmlType::F16,
                byte_len: w.len(),
            },
            TensorPlan {
                name: "blk.0.attn_norm.weight".into(),
                shape: vec![64],
                dtype: GgmlType::F16,
                byte_len: norm.len(),
            },
            TensorPlan {
                name: "blk.0.ffn_gate_inp.weight".into(),
                shape: vec![64, 2],
                dtype: GgmlType::F16,
                byte_len: gate.len(),
            },
        ];
        let f = std::fs::File::create(path).unwrap();
        let mut wr = GgufWriter::create(BufWriter::new(f), &metadata, plan).unwrap();
        wr.write_tensor("blk.0.attn_q.weight", &w).unwrap();
        wr.write_tensor("blk.0.attn_norm.weight", &norm).unwrap();
        wr.write_tensor("blk.0.ffn_gate_inp.weight", &gate).unwrap();
        wr.finish().unwrap().into_inner().unwrap();
        values
    }

    /// End to end: an F16 GGUF in, a Q8_0 GGUF out, read back by the
    /// same reader the engine loads models with. The tensors that
    /// llama.cpp keeps at source precision are still F16, the one it
    /// quantizes is Q8_0, and the values survive within Q8_0's error.
    #[test]
    fn an_f16_gguf_round_trips_through_quantize_and_reads_back_as_q8_0() {
        let dir = tmp_dir("roundtrip");
        let src = dir.join("src.gguf");
        let dst = dir.join("dst.gguf");
        let values = write_f16_source(&src);

        run(QuantizeArgs {
            input: src.clone(),
            output: Some(dst.clone()),
            ty: "Q8_0".into(),
            dry_run: false,
            force: true,
        })
        .unwrap();

        let out = GgufFile::open(&dst).unwrap();
        assert_eq!(out.metadata_u64("general.file_type"), Some(7));
        assert_eq!(out.metadata_u64("general.quantization_version"), Some(2));
        assert_eq!(out.metadata_str("general.architecture"), Some("llama"));

        let q = out.find_tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(q.dtype, GgmlType::Q8_0);
        assert_eq!(q.shape, vec![64, 2]);
        // The norm and the router gate are llama.cpp's keep-list.
        assert_eq!(
            out.find_tensor("blk.0.attn_norm.weight").unwrap().dtype,
            GgmlType::F16
        );
        assert_eq!(
            out.find_tensor("blk.0.ffn_gate_inp.weight").unwrap().dtype,
            GgmlType::F16
        );

        let back =
            ferrox_quant::dequant_q8_0(out.tensor_bytes("blk.0.attn_q.weight").unwrap()).unwrap();
        assert_eq!(back.len(), values.len());
        for (i, (&want, &have)) in values.iter().zip(back.iter()).enumerate() {
            assert!((want - have).abs() < 0.01, "element {i}: {want} -> {have}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The quantized bytes are the ones llama.cpp's encoder would have
    /// written for the same row, not merely bytes that decode to
    /// similar numbers. `ferrox-quant`'s golden pins the encoder; this
    /// pins the pipeline that feeds it -- row tiling, F16 decode, and
    /// the writer's data-section layout all have to be right for these
    /// bytes to land where the reader looks.
    #[test]
    fn the_quantized_tensor_bytes_are_what_the_encoder_produces_for_those_rows() {
        let dir = tmp_dir("bytes");
        let src = dir.join("src.gguf");
        let dst = dir.join("dst.gguf");
        let values = write_f16_source(&src);
        run(QuantizeArgs {
            input: src,
            output: Some(dst.clone()),
            ty: "q8_0".into(),
            dry_run: false,
            force: true,
        })
        .unwrap();

        // Re-encode independently, from the f16 the source stored (not
        // from `values`, which is f32 and would round differently).
        let mut want = Vec::new();
        let f16_roundtrip: Vec<f32> = values
            .iter()
            .map(|v| half::f16::from_f32(*v).to_f32())
            .collect();
        for row in f16_roundtrip.chunks(64) {
            ferrox_quant::encode_row_q8_0(row, &mut want).unwrap();
        }
        let out = GgufFile::open(&dst).unwrap();
        assert_eq!(out.tensor_bytes("blk.0.attn_q.weight").unwrap(), &want[..]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The refusal this subcommand is scoped around, exercised through
    /// the real entry point rather than only through `parse_target`.
    #[test]
    fn asking_for_a_k_quant_refuses_before_touching_the_filesystem() {
        let dir = tmp_dir("refuse");
        let src = dir.join("src.gguf");
        let dst = dir.join("dst.gguf");
        write_f16_source(&src);
        let err = run(QuantizeArgs {
            input: src,
            output: Some(dst.clone()),
            ty: "Q4_K_M".into(),
            dry_run: false,
            force: true,
        })
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("cannot WRITE Q4_K_M"), "{msg}");
        assert!(msg.contains("Q8_0"), "{msg}");
        assert!(!dst.exists(), "a refused run must not leave a file behind");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A source tensor that is already quantized is refused by name.
    /// Requantizing stacks two roundings, and the file would carry no
    /// sign that it happened.
    #[test]
    fn an_already_quantized_source_tensor_is_refused_by_name() {
        let dir = tmp_dir("requant");
        let src = dir.join("src.gguf");
        let bytes = vec![0u8; 144 * 2]; // two Q4_K super-blocks
        let plan = vec![TensorPlan {
            name: "blk.0.attn_q.weight".into(),
            shape: vec![256, 2],
            dtype: GgmlType::Q4K,
            byte_len: bytes.len(),
        }];
        let f = std::fs::File::create(&src).unwrap();
        let mut wr = GgufWriter::create(BufWriter::new(f), &BTreeMap::new(), plan).unwrap();
        wr.write_tensor("blk.0.attn_q.weight", &bytes).unwrap();
        wr.finish().unwrap().into_inner().unwrap();

        let err = run(QuantizeArgs {
            input: src,
            output: Some(dir.join("dst.gguf")),
            ty: "Q8_0".into(),
            dry_run: true,
            force: true,
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("F32/F16/BF16 sources only"),
            "{err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A row length that is not a whole number of blocks stops the run.
    /// The alternative -- padding the row -- writes more elements than
    /// the shape declares, and every later row decodes shifted.
    #[test]
    fn a_row_length_that_is_not_a_multiple_of_the_block_size_is_refused() {
        let dir = tmp_dir("ragged");
        let src = dir.join("src.gguf");
        let bytes = vec![0u8; 33 * 2 * 2];
        let plan = vec![TensorPlan {
            name: "blk.0.attn_q.weight".into(),
            shape: vec![33, 2],
            dtype: GgmlType::F16,
            byte_len: bytes.len(),
        }];
        let f = std::fs::File::create(&src).unwrap();
        let mut wr = GgufWriter::create(BufWriter::new(f), &BTreeMap::new(), plan).unwrap();
        wr.write_tensor("blk.0.attn_q.weight", &bytes).unwrap();
        wr.finish().unwrap().into_inner().unwrap();

        let err = run(QuantizeArgs {
            input: src,
            output: Some(dir.join("dst.gguf")),
            ty: "Q8_0".into(),
            dry_run: true,
            force: true,
        })
        .unwrap_err();
        assert!(err.to_string().contains("not a multiple of"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_existing_output_is_not_overwritten_without_force() {
        let dir = tmp_dir("clobber");
        let src = dir.join("src.gguf");
        let dst = dir.join("dst.gguf");
        write_f16_source(&src);
        std::fs::write(&dst, b"precious").unwrap();
        let err = run(QuantizeArgs {
            input: src,
            output: Some(dst.clone()),
            ty: "Q8_0".into(),
            dry_run: false,
            force: false,
        })
        .unwrap_err();
        assert!(err.to_string().contains("--force"), "{err}");
        assert_eq!(std::fs::read(&dst).unwrap(), b"precious");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_default_output_name_carries_the_target_and_sits_beside_the_input() {
        assert_eq!(
            default_output_path(Path::new("/m/Llama-3.2-1B-F16.gguf"), Target::Q8_0),
            PathBuf::from("/m/Llama-3.2-1B-F16-Q8_0.gguf")
        );
    }
}
