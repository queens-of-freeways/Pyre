//! CLI additions for G1: `perplexity` and `inference` over real .m models.

use dllama_exec::Inference;
use dllama_ir::ModelMeta;
use dllama_model::{open as open_m, ModelFile, Weights};
use dllama_tokenizer::{DecodeState, Tokenizer};
use std::collections::HashMap;
use std::time::Instant;

/// A loaded model from either container (.m or .gguf) — the format is
/// sniffed from the magic bytes; both surface the same ModelMeta + Weights.
pub enum ModelSource {
    M(ModelFile),
    Gguf(dllama_gguf::GgufFile),
}

impl ModelSource {
    pub fn meta(&self, _max_seq_len: u32) -> Result<ModelMeta, String> {
        match self {
            ModelSource::M(f) => Ok(f.meta.clone()),
            ModelSource::Gguf(f) => f.meta(_max_seq_len),
        }
    }
    pub fn weights(&self) -> Result<Weights<'_>, String> {
        match self {
            ModelSource::M(f) => f.weights(),
            ModelSource::Gguf(f) => f.weights(),
        }
    }
    /// Load the tokenizer: prefer the GGUF-embedded tokenizer when the
    /// model is a GGUF and no .t file is given; otherwise load the .t.
    pub fn tokenizer(&self, path: Option<&str>) -> Result<Tokenizer, String> {
        match (self, path) {
            (ModelSource::Gguf(f), None) => f.tokenizer(),
            (_, Some(p)) => Tokenizer::load(p),
            (ModelSource::M(_), None) => Err("no --tokenizer provided for .m model".into()),
        }
    }
}

/// Parse `--buffer-float-type` (`q80` default | `f32`) — q80 matches the C++
/// engine's activation-cast semantics; f32 skips the cast (GGUF fidelity mode).
pub fn parse_buffer_type(s: Option<&str>) -> dllama_ir::FloatType {
    if s == Some("f32") {
        dllama_ir::FloatType::F32
    } else {
        dllama_ir::FloatType::Q80
    }
}

pub fn open_model(path: &str, max_seq_len: u32) -> Result<ModelSource, String> {
    use std::io::Read;
    let mut magic = [0u8; 4];
    std::fs::File::open(path)
        .map_err(|e| format!("cannot open model file ({path}): {e}"))?
        .read_exact(&mut magic)
        .map_err(|e| format!("cannot read model file ({path}): {e}"))?;
    if &magic == b"GGUF" {
        Ok(ModelSource::Gguf(dllama_gguf::GgufFile::open(path)?))
    } else {
        Ok(ModelSource::M(open_m(path, max_seq_len)?))
    }
}

pub fn run_perplexity(args: &[String]) -> i32 {
    apply_threads_flag(args);
    let model_path = match flag(args, "--model") {
        Some(p) => p.to_string(),
        None => {
            eprintln!("--model is required");
            return 1;
        }
    };
    let tokenizer_path = flag(args, "--tokenizer").unwrap_or_default().to_string();
    let prompt = flag(args, "--prompt").unwrap_or("").to_string();
    let max_seq_len: u32 = flag(args, "--max-seq-len").unwrap_or("0").parse().unwrap_or(0);
    if prompt.is_empty() {
        eprintln!("--prompt is required");
        return 1;
    }

    let source: ModelSource = match open_model(&model_path, max_seq_len) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    let mut meta = match source.meta(max_seq_len) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    meta.sync_type = parse_buffer_type(flag(args, "--buffer-float-type"));
    print_header(&meta);
    let weights: Weights = match source.weights() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    let tokenizer = match source.tokenizer(if tokenizer_path.is_empty() { None } else { Some(&tokenizer_path) }) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };

    let input = match tokenizer.encode(&prompt, true, true) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    let n = input.len();
    println!("Evaluating {n} tokens...");

    let mut inf = match Inference::new(&meta, &weights) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };

    let mut total_log_prob = 0.0f32;
    for pos in 0..n - 1 {
        let logits = match inf.forward(input[pos] as u32, pos as u32) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("🚨 {e}");
                return 1;
            }
        };
        // C++ perplexity: softmax over logits, prob of the actual next token
        let probs: Vec<f32> = {
            let mut p = logits.to_vec();
            let len = p.len();
            dllama_kernel::cpu::softmax(&mut p, len);
            p
        };
        let target = input[pos + 1];
        let prob = probs[target as usize];
        total_log_prob += prob.max(1e-30).ln();
        if std::env::var("DLLAMA_RS_DEBUG").is_ok() {
            let mut top: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut s = String::new();
            for (i, p) in top.iter().take(5) {
                let piece = tokenizer
                    .vocab
                    .get(*i)
                    .map(|v| String::from_utf8_lossy(v).into_owned())
                    .unwrap_or_default();
                s.push_str(&format!("  [{i}] {piece:?} p={p:.5} | "));
            }
            println!("    pos {pos} target={target} p={prob:.7} ::{s}");
        }
        println!("{:5} / {}, prob={:.6}", pos + 1, n - 1, prob);
    }

    let avg_log_prob = total_log_prob / (n - 1) as f32;
    let ppl = (-avg_log_prob).exp();
    println!();
    println!("Results");
    println!("   perplexity: {:.6} (lower = better)", ppl);
    println!("   avgLogProb: {:.6}", avg_log_prob);
    println!("   bitPerToken: {:.6}", -avg_log_prob / std::f32::consts::LN_2);
    0
}

pub fn run_inference(args: &[String]) -> i32 {
    apply_threads_flag(args);
    let model_path = match flag(args, "--model") {
        Some(p) => p.to_string(),
        None => {
            eprintln!("--model is required");
            return 1;
        }
    };
    let tokenizer_path = flag(args, "--tokenizer").unwrap_or_default().to_string();
    let prompt = flag(args, "--prompt").unwrap_or("").to_string();
    let steps: usize = flag(args, "--steps").unwrap_or("64").parse().unwrap_or(64);
    let max_seq_len: u32 = flag(args, "--max-seq-len").unwrap_or("0").parse().unwrap_or(0);
    if prompt.is_empty() {
        eprintln!("--prompt is required");
        return 1;
    }

    let source = match open_model(&model_path, max_seq_len) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    let mut meta = match source.meta(max_seq_len) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    meta.sync_type = parse_buffer_type(flag(args, "--buffer-float-type"));
    print_header(&meta);
    let weights = match source.weights() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    let tokenizer = match source.tokenizer(if tokenizer_path.is_empty() { None } else { Some(&tokenizer_path) }) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    let input = match tokenizer.encode(&prompt, true, true) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    let n = input.len();
    println!("{prompt}");

    let mut inf = match Inference::new(&meta, &weights) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };

    // prompt evaluation (batch 1): feed all but the last token
    let eval_start = Instant::now();
    for pos in 0..n - 1 {
        if let Err(e) = inf.forward(input[pos] as u32, pos as u32) {
            eprintln!("🚨 {e}");
            return 1;
        }
    }
    let eval_ms = eval_start.elapsed().as_secs_f32() * 1000.0;
    let n_eval = (n - 1).max(1);

    // greedy generation
    let mut token = input[n - 1];
    let mut state = DecodeState::default();
    let pred_start = Instant::now();
    let max_pos = (meta.seq_len as usize).min(n - 1 + steps);
    let mut generated = 0usize;
    let mut pos = n - 1;
    while pos < max_pos {
        let logits = match inf.forward(token as u32, pos as u32) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("🚨 {e}");
                return 1;
            }
        };
        token = dllama_exec::argmax(logits) as i32;
        if let Some(piece) = tokenizer.decode(&mut state, token) {
            print!("{piece}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        if tokenizer.is_eos(token) {
            break;
        }
        generated += 1;
        pos += 1;
    }
    let pred_ms = pred_start.elapsed().as_secs_f32() * 1000.0;
    println!();
    println!();
    println!("Evaluation");
    println!("    nTokens: {n_eval}");
    println!("   tokens/s: {:.2} ({:.2} ms/tok)", n_eval as f32 / (eval_ms / 1000.0), eval_ms / n_eval as f32);
    println!("Prediction");
    println!("    nTokens: {generated}");
    if generated > 0 {
        println!("   tokens/s: {:.2} ({:.2} ms/tok)", generated as f32 / (pred_ms / 1000.0), pred_ms / generated as f32);
    }
    0
}

/// Sanity checks for GGUF loading: dump the tensor directory and
/// cross-check the integer-path matmuls against dequant reference dots on
/// REAL tensor bytes (the synthetic unit tests can miss real-data layouts).
pub fn run_gguf_check(args: &[String]) -> i32 {
    let model_path = match flag(args, "--model") {
        Some(p) => p.to_string(),
        None => {
            eprintln!("--model is required");
            return 1;
        }
    };
    let g = match dllama_gguf::GgufFile::open(&model_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    let meta = match g.meta(0) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    println!("gguf v{}: arch {:?}, dim {}, {} layers, vocab {}", g.version, meta.arch, meta.dim, meta.n_layers, meta.vocab_size);
    let weights = match g.weights() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };

    // cross-check every quant kind present in the file
    let k = meta.dim as usize;
    let x: Vec<f32> = (0..k).map(|i| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0).collect();
    let mut x80 = vec![0u8; dllama_quant::q80_bytes(k)];
    dllama_quant::quantize_row_q80(&x, &mut x80);
    let mut xq = vec![0.0f32; k];
    dllama_quant::dequantize_row_q80(&x80, &mut xq, k);

    for name in ["token_embd", "blk.0.attn_q", "blk.0.attn_v", "blk.0.ffn_gate", "blk.0.ffn_down", "output"] {
        let t = match weights.get(name) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let row_bytes = t.kind.row_bytes(k) as usize;
        let rows = (t.bytes.len() / row_bytes).min(8);
        let mut out_int = vec![0.0f32; rows];
        let mut out_ref = vec![0.0f32; rows];
        let mut wrow = vec![0.0f32; k];
        for di in 0..rows {
            let row = &t.bytes[di * row_bytes..(di + 1) * row_bytes];
            dllama_kernel::gguf::dequant_row(t.kind, row, &mut wrow, k);
            out_ref[di] = dllama_kernel::avx2::dot_product(&xq, &wrow);
        }
        match t.kind {
            dllama_model::QuantKind::F32 | dllama_model::QuantKind::F16 => {}
            dllama_model::QuantKind::DllamaQ40 | dllama_model::QuantKind::GgufQ4_0 => {
                dllama_kernel::avx2::matmul_q80_q40(&mut out_int, &x80, t.bytes, rows, k);
            }
            _ => {
                dllama_kernel::gguf::matmul_q80(&mut out_int, &x80, t.kind, t.bytes, rows, k);
            }
        }
        let max_diff = out_int
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (*a - *b).abs() / out_ref.iter().fold(0.0f32, |m, v| m.max(v.abs())))
            .fold(0.0f32, |a, b| a.max(b));
        println!("{name:22} kind={:?} rows={:6} max_rel_diff(int vs ref)={:e}", t.kind, t.bytes.len() / row_bytes, max_diff);
    }

    // vocab comparison: GGUF tokenizer.ggml.tokens vs the .t tokenizer file
    if let Some(t_path) = flag(args, "--tokenizer") {
        let tok = match Tokenizer::load(t_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("🚨 {e}");
                return 1;
            }
        };
        let gguf_tokens = g.get("tokenizer.ggml.tokens").and_then(|v| match v {
            dllama_gguf::Value::Array(items) => Some(items),
            _ => None,
        });
        match gguf_tokens {
            Some(items) => {
                let mut mismatches = 0;
                let n = items.len().min(tok.vocab.len());
                for i in 0..n {
                    let gs = match &items[i] {
                        dllama_gguf::Value::Str(s) => s.as_bytes(),
                        _ => b"",
                    };
                    let ts = tok.vocab[i].as_slice();
                    // gguf vocab pieces are plain; .t pieces may be byte-encoded
                    if gs != ts && gs != String::from_utf8_lossy(ts).as_bytes() {
                        if mismatches < 10 {
                            println!("vocab mismatch at id {i}: gguf={:?} t={:?}", String::from_utf8_lossy(gs), String::from_utf8_lossy(ts));
                        }
                        mismatches += 1;
                    }
                }
                println!("vocab: {} ids compared, {} mismatches (gguf n={})", n, mismatches, items.len());
                // sample dump for alignment inspection
                for &i in &[0, 1, 2, 3, 4, 5, 10, 14, 90, 94, 100, 1000, 5000, 50000, 130000, 151000, 151600, 151640] {
                    if i < items.len() && i < tok.vocab.len() {
                        let gs = match &items[i] {
                            dllama_gguf::Value::Str(s) => s.clone(),
                            _ => String::new(),
                        };
                        let ts = tok.vocab[i].clone();
                        println!("id {i:6}: gguf={gs:?}  t={:?}", String::from_utf8_lossy(&ts));
                    }
                }
            }
            None => println!("no tokenizer.ggml.tokens in this file"),
        }
    }

    // cross-mode weight comparison: dequantize the same tensors from the GGUF
    // and the dllama .m file — they approximate the same original weights, so
    // correlation must be ~1. A permutation bug inside blocks shows up as a
    // clearly depressed correlation.
    if let Some(m_path) = flag(args, "--compare-m") {
        let mf = match open_m(m_path, 0) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("🚨 {e}");
                return 1;
            }
        };
        let mw = match mf.weights() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("🚨 {e}");
                return 1;
            }
        };
        let k = meta.dim as usize;
        for name in ["token_embd", "blk.0.attn_q", "blk.0.attn_k", "blk.0.attn_v", "blk.0.attn_output",
                      "blk.0.attn_norm", "blk.0.ffn_norm", "blk.0.attn_q_norm", "blk.0.attn_k_norm",
                      "blk.0.ffn_gate", "blk.0.ffn_up", "blk.0.ffn_down", "output_norm", "output"] {
            let (gt, mt) = match (weights.get(name), mw.get(name)) {
                (Ok(a), Ok(b)) => (a, b),
                _ => continue,
            };
            let row_bytes_g = gt.kind.row_bytes(k) as usize;
            let row_bytes_m = mt.kind.row_bytes(k) as usize;
            let mut gr = vec![0.0f32; k];
            let mut mr = vec![0.0f32; k];
            let mut cors = Vec::new();
            let rows_avail = (gt.bytes.len() / row_bytes_g).min(mt.bytes.len() / row_bytes_m);
            let rows_check = if name == "token_embd" || name == "output" {
                rows_avail.min(3000)
            } else if name.contains("norm") {
                rows_avail
            } else {
                rows_avail.min(512)
            };
            for di in 0..rows_check {
                dllama_kernel::gguf::dequant_row(gt.kind, &gt.bytes[di * row_bytes_g..][..row_bytes_g], &mut gr, k);
                dllama_kernel::gguf::dequant_row(mt.kind, &mt.bytes[di * row_bytes_m..][..row_bytes_m], &mut mr, k);
                let (mut sxy, mut sx, mut sy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for i in 0..k {
                    let (a, b) = (gr[i] as f64, mr[i] as f64);
                    sxy += a * b; sx += a; sy += b; sxx += a * a; syy += b * b;
                }
                let n = k as f64;
                let cov = sxy - sx * sy / n;
                let var = ((sxx - sx * sx / n) * (syy - sy * sy / n)).sqrt();
                if var > 0.0 {
                    cors.push(cov / var);
                }
            }
            if !cors.is_empty() {
                let avg = cors.iter().sum::<f64>() / cors.len() as f64;
                let min = cors.iter().cloned().fold(1.0f64, |a, b| a.min(b));
                println!("{name:22} corr(gguf, m) avg={avg:.4} min={min:.4} over {} rows", cors.len());
            }
        }
    }
    0
}

/// Convert a GGUF file into the dllama `.m` container (requantized q40) so the
/// G1-validated .m engine path can run the same weights — the decisive
/// weights-vs-engine isolation experiment.
pub fn run_gguf_to_m(args: &[String]) -> i32 {
    use std::io::Write;
    let model_path = match flag(args, "--model") {
        Some(p) => p.to_string(),
        None => {
            eprintln!("--model is required");
            return 1;
        }
    };
    let out_path = flag(args, "--out").unwrap_or("converted.m").to_string();
    let g = match dllama_gguf::GgufFile::open(&model_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    let meta = match g.meta(0) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    let weights = match g.weights() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    println!("converting {} ({} layers) to {out_path} ...", meta.name, meta.n_layers);

    // header
    let mut hdr: Vec<u8> = Vec::new();
    let mut kv = |key: i32, val: i32, hdr: &mut Vec<u8>| {
        hdr.extend_from_slice(&key.to_le_bytes());
        hdr.extend_from_slice(&val.to_le_bytes());
    };
    hdr.extend_from_slice(&0xA00ABCDi32.to_le_bytes());
    let header_placeholder = hdr.len();
    hdr.extend_from_slice(&0i32.to_le_bytes()); // headerSize placeholder
    kv(1, 0xABCD01, &mut hdr); // ARCH_TYPE = QWEN3
    kv(2, meta.dim as i32, &mut hdr);
    kv(3, meta.hidden_dim as i32, &mut hdr);
    kv(4, meta.n_layers as i32, &mut hdr);
    kv(5, meta.n_heads as i32, &mut hdr);
    kv(6, meta.n_kv_heads as i32, &mut hdr);
    kv(9, meta.vocab_size as i32, &mut hdr);
    kv(10, meta.seq_len as i32, &mut hdr);
    kv(11, 1, &mut hdr); // HIDDEN_ACT = SILU
    kv(12, meta.rope_theta as i32, &mut hdr);
    kv(13, 2, &mut hdr); // WEIGHT_FLOAT_TYPE = Q40
    kv(18, 1, &mut hdr); // ROPE_TYPE = FALCON
    kv(19, meta.head_dim as i32, &mut hdr);
    kv(20, 6, &mut hdr); // NORM_EPSILON = 1e-6
    let header_size = hdr.len() as i32;
    hdr[header_placeholder..header_placeholder + 4].copy_from_slice(&header_size.to_le_bytes());

    let file = std::fs::File::create(&out_path).map_err(|e| e.to_string());
    let mut file = match file {
        Ok(f) => f,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    if let Err(e) = file.write_all(&hdr) {
        eprintln!("🚨 {e}");
        return 1;
    }

    let dim = meta.dim as usize;
    let f32s = |name: &str| -> Vec<f32> {
        let t = weights.get(name).expect("tensor");
        let mut v = vec![0.0f32; t.bytes.len() / t.kind.block_bytes() * t.kind.block_elems()];
        let len = v.len();
        dllama_kernel::gguf::dequant_row(t.kind, t.bytes, &mut v, len);
        v
    };
    let write_f32 = |file: &mut std::fs::File, v: &[f32]| {
        for x in v {
            let _ = file.write_all(&x.to_le_bytes());
        }
    };
    let write_q40 = |file: &mut std::fs::File, v: &[f32], rows: usize, k: usize| {
        assert_eq!(v.len(), rows * k);
        let mut row = vec![0.0f32; k];
        let mut buf = vec![0u8; dllama_quant::q40_bytes(k)];
        for di in 0..rows {
            row.copy_from_slice(&v[di * k..(di + 1) * k]);
            dllama_quant::quantize_row_q40(&row, &mut buf);
            let _ = file.write_all(&buf);
        }
    };

    // token embedding (f32)
    write_f32(&mut file, &f32s("token_embd"));
    for l in 0..meta.n_layers {
        let layer = l as usize;
        // .m order per layer: q, k, v, wo, w1(gate), w2(down), w3(up)
        let q_dim = meta.q_dim() as usize;
        let kv_dim = meta.kv_dim() as usize;
        let ffn = meta.ffn_dim() as usize;
        let shapes: &[(&str, usize, usize)] = &[
            ("attn_q", q_dim, dim),
            ("attn_k", kv_dim, dim),
            ("attn_v", kv_dim, dim),
            ("attn_output", dim, q_dim),
            ("ffn_gate", ffn, dim),
            ("ffn_down", dim, ffn),
            ("ffn_up", ffn, dim),
        ];
        for (w, rows, k) in shapes {
            let name = format!("blk.{layer}.{w}");
            let v = f32s(&name);
            write_q40(&mut file, &v, *rows, *k);
        }
        for n in ["attn_q_norm", "attn_k_norm"] {
            let name = format!("blk.{layer}.{n}");
            if weights.contains(&name) {
                write_f32(&mut file, &f32s(&name));
            }
        }
        write_f32(&mut file, &f32s(&format!("blk.{layer}.attn_norm")));
        write_f32(&mut file, &f32s(&format!("blk.{layer}.ffn_norm")));
    }
    write_f32(&mut file, &f32s("output_norm"));
    {
        let v = f32s("output");
        write_q40(&mut file, &v, meta.vocab_size as usize, dim);
    }
    println!("done: {out_path}");
    0
}

/// Convert a dllama `.m` model to a GGUF file with F32 tensors — the trace-diff
/// instrument: both engines then run identical effective weight values, so
/// the first divergent op in a side-by-side trace reveals engine-path bugs.
pub fn run_m_to_gguf(args: &[String]) -> i32 {
    let model_path = match flag(args, "--model") {
        Some(p) => p.to_string(),
        None => {
            eprintln!("--model is required");
            return 1;
        }
    };
    let out_path = flag(args, "--out").unwrap_or("converted.gguf").to_string();
    let mf = match open_m(&model_path, 0) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    let meta = &mf.meta;
    let weights = match mf.weights() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("🚨 {e}");
            return 1;
        }
    };
    let f32s = |name: &str| -> Option<Vec<f32>> {
        weights.get(name).ok().map(|t| {
            let mut v = vec![0.0f32; t.bytes.len() / t.kind.block_bytes() * t.kind.block_elems()];
            let len = v.len();
            dllama_kernel::gguf::dequant_row(t.kind, t.bytes, &mut v, len);
            v
        })
    };

    // collect tensors: (gguf name, values, row_len, rows)
    let dim = meta.dim as usize;
    let q_dim = meta.q_dim() as usize;
    let kv_dim = meta.kv_dim() as usize;
    let ffn = meta.ffn_dim() as usize;
    let mut tensors: Vec<(String, Vec<f32>, usize, usize)> = Vec::new();
    if let Some(v) = f32s("token_embd") {
        tensors.push(("token_embd.weight".into(), v, dim, meta.vocab_size as usize));
    }
    for l in 0..meta.n_layers {
        let layer = l as usize;
        let shapes: &[(&str, &str, usize, usize)] = &[
            ("attn_q", "attn_q", q_dim, dim),
            ("attn_k", "attn_k", kv_dim, dim),
            ("attn_v", "attn_v", kv_dim, dim),
            ("attn_output", "attn_output", dim, q_dim),
            ("ffn_gate", "ffn_gate", ffn, dim),
            ("ffn_down", "ffn_down", dim, ffn),
            ("ffn_up", "ffn_up", ffn, dim),
        ];
        for (w, name, rows, k) in shapes {
            if let Some(v) = f32s(&format!("blk.{layer}.{name}")) {
                tensors.push((format!("blk.{layer}.{w}.weight"), v, *k, *rows));
            }
        }
        if meta.qk_norm {
            for (n, len) in [("attn_q_norm", meta.head_dim as usize), ("attn_k_norm", meta.head_dim as usize)] {
                if let Some(v) = f32s(&format!("blk.{layer}.{n}")) {
                    tensors.push((format!("blk.{layer}.{n}.weight"), v, len, 1));
                }
            }
        }
        for n in ["attn_norm", "ffn_norm"] {
            if let Some(v) = f32s(&format!("blk.{layer}.{n}")) {
                tensors.push((format!("blk.{layer}.{n}.weight"), v, dim, 1));
            }
        }
    }
    if let Some(v) = f32s("output_norm") {
        tensors.push(("output_norm.weight".into(), v, dim, 1));
    }
    if let Some(v) = f32s("output") {
        tensors.push(("output.weight".into(), v, dim, meta.vocab_size as usize));
    }
    println!("{} tensors", tensors.len());

    // serialize
    let mut buf: Vec<u8> = Vec::with_capacity(16 << 20);
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    let nkv = 11u64;
    buf.extend_from_slice(&nkv.to_le_bytes());
    let mut put_kv_str = |key: &str, val: &str, buf: &mut Vec<u8>| {
        let b = key.as_bytes();
        buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
        buf.extend_from_slice(b);
        buf.extend_from_slice(&8u32.to_le_bytes()); // T_STR
        let v = val.as_bytes();
        buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
        buf.extend_from_slice(v);
    };
    let mut put_kv_u32 = |key: &str, val: u32, buf: &mut Vec<u8>| {
        let b = key.as_bytes();
        buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
        buf.extend_from_slice(b);
        buf.extend_from_slice(&4u32.to_le_bytes()); // T_U32
        buf.extend_from_slice(&val.to_le_bytes());
    };
    let mut put_kv_f32 = |key: &str, val: f32, buf: &mut Vec<u8>| {
        let b = key.as_bytes();
        buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
        buf.extend_from_slice(b);
        buf.extend_from_slice(&6u32.to_le_bytes()); // T_F32
        buf.extend_from_slice(&val.to_le_bytes());
    };
    put_kv_str("general.architecture", "qwen3", &mut buf);
    put_kv_str("general.name", &meta.name, &mut buf);
    put_kv_u32("qwen3.block_count", meta.n_layers, &mut buf);
    put_kv_u32("qwen3.embedding_length", meta.dim, &mut buf);
    put_kv_u32("qwen3.feed_forward_length", meta.hidden_dim, &mut buf);
    put_kv_u32("qwen3.attention.head_count", meta.n_heads, &mut buf);
    put_kv_u32("qwen3.attention.head_count_kv", meta.n_kv_heads, &mut buf);
    put_kv_u32("qwen3.attention.key_length", meta.head_dim_or_derived(), &mut buf);
    put_kv_f32("qwen3.attention.layer_norm_rms_epsilon", meta.norm_epsilon, &mut buf);
    put_kv_f32("qwen3.rope.freq_base", meta.rope_theta, &mut buf);
    put_kv_u32("qwen3.context_length", meta.seq_len, &mut buf);

    // tensor directory (offsets computed after, aligned to 32)
    let dir_len: usize = tensors
        .iter()
        .map(|(n, v, _, _)| 8 + n.len() + 4 + 8 + 4 + 8)
        .sum();
    let mut data_off = buf.len() + dir_len;
    data_off = data_off.div_ceil(32) * 32;
    let mut off = 0usize;
    for (name, v, k, rows) in &tensors {
        let b = name.as_bytes();
        buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
        buf.extend_from_slice(b);
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&(*k as u64).to_le_bytes()); // ne[0] = row length
        buf.extend_from_slice(&(*rows as u64).to_le_bytes()); // ne[1] = rows
        buf.extend_from_slice(&0u32.to_le_bytes()); // ggml_type = F32
        buf.extend_from_slice(&(off as u64).to_le_bytes());
        let sz = v.len() * 4;
        off += sz.div_ceil(32) * 32;
    }
    let _ = data_off;
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    for (_, v, _, _) in &tensors {
        for x in v {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
    }
    if let Err(e) = std::fs::write(&out_path, &buf) {
        eprintln!("🚨 {e}");
        return 1;
    }
    println!("done: {out_path} ({} MB)", buf.len() / (1024 * 1024));
    0
}

/// Distributed perplexity: root + TCP workers (--workers "host:port host:port").
pub fn run_distributed_perplexity(args: &[String]) -> i32 {
    apply_threads_flag(args);
    let model_path = match flag(args, "--model") {
        Some(p) => p.to_string(),
        None => {
            eprintln!("--model is required");
            return 1;
        }
    };
    let tokenizer_path = flag(args, "--tokenizer").unwrap_or_default().to_string();
    let prompt = flag(args, "--prompt").unwrap_or("").to_string();
    let max_seq_len: u32 = flag(args, "--max-seq-len").unwrap_or("0").parse().unwrap_or(0);
    let workers_str = flag(args, "--workers").unwrap_or("").to_string();
    let worker_addrs: Vec<(String, u16)> = if workers_str.trim() == "auto" {
        // G3.2: auto node discovery (UDP probe; see dllama-cluster)
        let timeout: u64 = flag(args, "--timeout").and_then(|v| v.parse().ok()).unwrap_or(1000);
        match dllama_cluster::discover_nodes(timeout) {
            Ok(list) if list.is_empty() => {
                eprintln!("no nodes discovered — start workers (dllama-rs worker) or pass --workers host:port");
                return 1;
            }
            Ok(list) => {
                println!("?? auto-discovered {} node(s)", list.len());
                for n in &list {
                    println!("   {n}");
                }
                list.iter()
                    .filter_map(|a| {
                        let (h, p) = a.rsplit_once(':')?;
                        Some((h.to_string(), p.parse().ok()?))
                    })
                    .collect()
            }
            Err(e) => {
                eprintln!("discovery failed: {e}");
                return 1;
            }
        }
    } else {
        workers_str
            .split_whitespace()
            .filter_map(|a| {
                let (h, p) = a.rsplit_once(':')?;
                Some((h.to_string(), p.parse().ok()?))
            })
            .collect()
    };
    if prompt.is_empty() || worker_addrs.is_empty() {
        eprintln!("--prompt and --workers are required");
        return 1;
    }
    let n_nodes = worker_addrs.len() as u32 + 1;

    // open model + validate sharding
    let source = match open_model(&model_path, max_seq_len) {
        Ok(f) => f,
        Err(e) => { eprintln!("🚨 {e}"); return 1; }
    };
    let meta = match source.meta(max_seq_len) {
        Ok(m) => m,
        Err(e) => { eprintln!("🚨 {e}"); return 1; }
    };
    if let Err(e) = dllama_ir::check_sharding(&meta, n_nodes) {
        eprintln!("🚨 sharding: {e}");
        return 1;
    }
    let weights = match source.weights() {
        Ok(w) => w,
        Err(e) => { eprintln!("🚨 {e}"); return 1; }
    };
    println!("💿 {n_nodes} nodes (1 root + {} workers)", worker_addrs.len());

    // connect + send setup to each worker
    use std::net::TcpStream;
    let mut worker_streams: Vec<TcpStream> = Vec::with_capacity(worker_addrs.len());
    for (i, (host, port)) in worker_addrs.iter().enumerate() {
        let node_index = i as u32 + 1;
        let mut stream = match TcpStream::connect(format!("{host}:{port}")) {
            Ok(s) => s,
            Err(e) => { eprintln!("🚨 connect {host}:{port}: {e}"); return 1; }
        };
        // handshake
        use std::io::Write;
        if stream.write_all(&dllama_cluster::MAGIC).is_err()
            || stream.write_all(&dllama_cluster::PROTOCOL_VERSION.to_le_bytes()).is_err() {
            eprintln!("🚨 handshake failed for {host}:{port}");
            return 1;
        }
        dllama_cluster::write_meta(&mut stream, &meta).unwrap();
        use std::io::Read;
        stream.write_all(&node_index.to_le_bytes()).unwrap();
        stream.write_all(&n_nodes.to_le_bytes()).unwrap();

        // send the node's weight slices
        let shard = match dllama_ir::shard::shard_node(&meta, n_nodes, node_index) {
            Ok(s) => s,
            Err(e) => { eprintln!("🚨 {e}"); return 1; }
        };
        let slices = match dllama_cluster::node_tensor_slices(&meta, &weights, &shard) {
            Ok(s) => s,
            Err(e) => { eprintln!("🚨 {e}"); return 1; }
        };
        stream.write_all(&(slices.len() as u32).to_le_bytes()).unwrap();
        for (name, kind, bytes) in &slices {
            // string name
            stream.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
            stream.write_all(name.as_bytes()).unwrap();
            // kind
            let kid = match kind {
                dllama_model::QuantKind::F32 => 0u8,
                dllama_model::QuantKind::F16 => 1,
                dllama_model::QuantKind::DllamaQ40 => 2,
                dllama_model::QuantKind::GgufQ4_0 => 3,
                dllama_model::QuantKind::GgufQ8_0 => 4,
                dllama_model::QuantKind::GgufQ4K => 5,
                dllama_model::QuantKind::GgufQ6K => 6,
            };
            stream.write_all(&[kid]).unwrap();
            // bytes
            stream.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
            stream.write_all(bytes).unwrap();
        }
        // read ack
        let mut ack = [0u8; 4];
        if stream.read_exact(&mut ack).is_err() {
            eprintln!("🚨 no ack from {host}:{port}");
            return 1;
        }
        println!("🔗 worker {node_index} at {host}:{port} ({:?} tensors)", slices.len());
        worker_streams.push(stream);
    }

    // build root's own Inference from its shard slices (owned arena)
    let root_shard = match dllama_ir::shard::shard_node(&meta, n_nodes, 0) {
        Ok(s) => s,
        Err(e) => { eprintln!("🚨 {e}"); return 1; }
    };
    let root_slices = match dllama_cluster::node_tensor_slices(&meta, &weights, &root_shard) {
        Ok(s) => s,
        Err(e) => { eprintln!("🚨 {e}"); return 1; }
    };
    let mut arena: Vec<Vec<u8>> = Vec::with_capacity(root_slices.len());
    let mut map: HashMap<String, dllama_model::Tensor> = HashMap::with_capacity(root_slices.len());
    for (name, kind, bytes) in root_slices {
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        map.insert(name, dllama_model::Tensor { kind, bytes: leaked });
    }
    let root_weights = dllama_model::Weights::from_map(map);
    // sanity check: no empty tensors
    for name in ["token_embd", "blk.0.attn_q", "blk.0.attn_k", "blk.0.attn_v", "blk.0.attn_output", "blk.0.ffn_gate", "blk.0.ffn_down", "blk.0.attn_norm", "output"] {
        match root_weights.get(name) {
            Ok(t) => println!("  root tensor {name}: {} bytes ({:?})", t.bytes.len(), t.kind),
            Err(e) => println!("  root tensor {name}: MISSING ({e})"),
        }
    }
    let mut inf = match dllama_exec::Inference::new_distributed(&meta, &root_weights, Some(root_shard), true) {
        Ok(i) => i,
        Err(e) => { eprintln!("🚨 {e}"); return 1; }
    };

    // tokenize
    let tokenizer = match source.tokenizer(if tokenizer_path.is_empty() { None } else { Some(&tokenizer_path) }) {
        Ok(t) => t,
        Err(e) => { eprintln!("🚨 {e}"); return 1; }
    };
    let input = match tokenizer.encode(&prompt, true, true) {
        Ok(t) => t,
        Err(e) => { eprintln!("🚨 {e}"); return 1; }
    };
    let n = input.len();
    println!("Evaluating {n} tokens ({n_nodes} nodes)...");

    // perplexity loop
    let mut total_log_prob = 0.0f32;
    for pos in 0..n - 1 {
        let token = input[pos] as u32;
        // broadcast control packet
        let pkt = dllama_cluster::ControlPacket { token, position: pos as u32, batch_size: 1 };
        let pkt_bytes = pkt.encode();
        for w in &mut worker_streams {
            use std::io::Write;
            if w.write_all(&pkt_bytes).is_err() {
                eprintln!("🚨 worker disconnected");
                return 1;
            }
        }
        // root forward
        let mut sctx = dllama_exec::distributed::SyncCtx::Root { workers: &mut worker_streams };
        let logits = match inf.forward_with_stream(token, pos as u32, &mut sctx) {
            Ok(l) => l,
            Err(e) => { eprintln!("🚨 {e}"); return 1; }
        };
        let probs: Vec<f32> = {
            let mut p = logits.to_vec();
            dllama_kernel::avx2::softmax(&mut p);
            p
        };
        let target = input[pos + 1];
        let prob = probs[target as usize];
        total_log_prob += prob.max(1e-30).ln();
        println!("{:5} / {}, prob={:.6}", pos + 1, n - 1, prob);
    }
    let ppl = (-(total_log_prob / (n - 1) as f32)).exp();
    println!("\nResults ({n_nodes} nodes)");
    println!("   perplexity: {:.6} (lower = better)", ppl);
    0
}


/// Distributed generation (G6.5): the root drives sampling, every node keeps
/// its KV cache in sync via the same control packets the perplexity driver
/// uses. Workers just evaluate; only the root needs the tokenizer output.
/// Usage: dllama-rs inference-dist --model <name> --prompt "..." --steps N
///        [--workers auto|host:port ...] [--temperature ...] [--seed ...]
pub fn run_distributed_inference(args: &[String]) -> i32 {
    apply_threads_flag(args);
    let model_path = match flag(args, "--model") {
        Some(p) => p.to_string(),
        None => { eprintln!("--model is required"); return 1; }
    };
    let tokenizer_path = flag(args, "--tokenizer").unwrap_or_default().to_string();
    let prompt = flag(args, "--prompt").unwrap_or("").to_string();
    let steps: usize = flag(args, "--steps").unwrap_or("64").parse().unwrap_or(64);
    let max_seq_len: u32 = flag(args, "--max-seq-len").unwrap_or("0").parse().unwrap_or(0);
    let workers_str = flag(args, "--workers").unwrap_or("").to_string();
    let worker_addrs: Vec<(String, u16)> = if workers_str.trim() == "auto" {
        let timeout: u64 = flag(args, "--timeout").and_then(|v| v.parse().ok()).unwrap_or(1000);
        match dllama_cluster::discover_nodes(timeout) {
            Ok(list) if list.is_empty() => {
                eprintln!("no nodes discovered — start workers (dllama-rs worker) or pass --workers host:port");
                return 1;
            }
            Ok(list) => {
                println!("auto-discovered {} node(s)", list.len());
                for n in &list { println!("   {n}"); }
                list.iter().filter_map(|a| {
                    let (h, p) = a.rsplit_once(':')?;
                    Some((h.to_string(), p.parse().ok()?))
                }).collect()
            }
            Err(e) => { eprintln!("discovery failed: {e}"); return 1; }
        }
    } else {
        workers_str.split_whitespace().filter_map(|a| {
            let (h, p) = a.rsplit_once(':')?;
            Some((h.to_string(), p.parse().ok()?))
        }).collect()
    };
    if prompt.is_empty() || worker_addrs.is_empty() {
        eprintln!("--prompt and --workers are required");
        return 1;
    }
    let n_nodes = worker_addrs.len() as u32 + 1;

    // open model + validate sharding (same setup shape as the ppl driver)
    let source = match open_model(&model_path, max_seq_len) {
        Ok(f) => f,
        Err(e) => { eprintln!("error: {e}"); return 1; }
    };
    let meta = match source.meta(max_seq_len) {
        Ok(m) => m,
        Err(e) => { eprintln!("error: {e}"); return 1; }
    };
    if let Err(e) = dllama_ir::check_sharding(&meta, n_nodes) {
        eprintln!("error sharding: {e}");
        return 1;
    }
    let weights = match source.weights() {
        Ok(w) => w,
        Err(e) => { eprintln!("error: {e}"); return 1; }
    };
    println!("{n_nodes} nodes (1 root + {} workers)", worker_addrs.len());

    use std::io::{Read, Write};
    use std::net::TcpStream;
    let mut worker_streams: Vec<TcpStream> = Vec::with_capacity(worker_addrs.len());
    for (i, (host, port)) in worker_addrs.iter().enumerate() {
        let node_index = i as u32 + 1;
        let mut stream = match TcpStream::connect(format!("{host}:{port}")) {
            Ok(s) => s,
            Err(e) => { eprintln!("error connect {host}:{port}: {e}"); return 1; }
        };
        if stream.write_all(&dllama_cluster::MAGIC).is_err()
            || stream.write_all(&dllama_cluster::PROTOCOL_VERSION.to_le_bytes()).is_err() {
            eprintln!("handshake failed for {host}:{port}");
            return 1;
        }
        dllama_cluster::write_meta(&mut stream, &meta).unwrap();
        stream.write_all(&node_index.to_le_bytes()).unwrap();
        stream.write_all(&n_nodes.to_le_bytes()).unwrap();

        let shard = match dllama_ir::shard::shard_node(&meta, n_nodes, node_index) {
            Ok(s) => s,
            Err(e) => { eprintln!("error: {e}"); return 1; }
        };
        let slices = match dllama_cluster::node_tensor_slices(&meta, &weights, &shard) {
            Ok(s) => s,
            Err(e) => { eprintln!("error: {e}"); return 1; }
        };
        stream.write_all(&(slices.len() as u32).to_le_bytes()).unwrap();
        for (name, kind, bytes) in &slices {
            stream.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
            stream.write_all(name.as_bytes()).unwrap();
            let kid = match kind {
                dllama_model::QuantKind::F32 => 0u8,
                dllama_model::QuantKind::F16 => 1,
                dllama_model::QuantKind::DllamaQ40 => 2,
                dllama_model::QuantKind::GgufQ4_0 => 3,
                dllama_model::QuantKind::GgufQ8_0 => 4,
                dllama_model::QuantKind::GgufQ4K => 5,
                dllama_model::QuantKind::GgufQ6K => 6,
            };
            stream.write_all(&[kid]).unwrap();
            stream.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
            stream.write_all(bytes).unwrap();
        }
        let mut ack = [0u8; 4];
        if stream.read_exact(&mut ack).is_err() {
            eprintln!("no ack from {host}:{port}");
            return 1;
        }
        println!("worker {node_index} at {host}:{port} ({:?} tensors)", slices.len());
        worker_streams.push(stream);
    }

    // root's own shard (owned arena, leaked — CLI lifetime)
    let root_shard = match dllama_ir::shard::shard_node(&meta, n_nodes, 0) {
        Ok(s) => s,
        Err(e) => { eprintln!("error: {e}"); return 1; }
    };
    let root_slices = match dllama_cluster::node_tensor_slices(&meta, &weights, &root_shard) {
        Ok(s) => s,
        Err(e) => { eprintln!("error: {e}"); return 1; }
    };
    let mut map: HashMap<String, dllama_model::Tensor> = HashMap::with_capacity(root_slices.len());
    for (name, kind, bytes) in root_slices {
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        map.insert(name, dllama_model::Tensor { kind, bytes: leaked });
    }
    let root_weights = dllama_model::Weights::from_map(map);
    let mut inf = match dllama_exec::Inference::new_distributed(&meta, &root_weights, Some(root_shard), true) {
        Ok(i) => i,
        Err(e) => { eprintln!("error: {e}"); return 1; }
    };

    let tokenizer = match source.tokenizer(if tokenizer_path.is_empty() { None } else { Some(&tokenizer_path) }) {
        Ok(t) => t,
        Err(e) => { eprintln!("error: {e}"); return 1; }
    };
    let mut sampler = crate::sampler::sampler_from_args(args);
    let input = match tokenizer.encode(&prompt, true, true) {
        Ok(t) => t,
        Err(e) => { eprintln!("error: {e}"); return 1; }
    };
    let n = input.len();
    let seq_len = meta.seq_len as usize;
    println!("Evaluating {n} prompt tokens ({n_nodes} nodes)...");

    // one distributed forward: broadcast the control packet + root forward
    macro_rules! dist_forward {
        ($tok:expr, $pos:expr) => {{
            let pkt = dllama_cluster::ControlPacket { token: $tok, position: $pos, batch_size: 1 };
            let pkt_bytes = pkt.encode();
            for w in &mut worker_streams {
                if w.write_all(&pkt_bytes).is_err() {
                    eprintln!("worker disconnected");
                    return 1;
                }
            }
            let mut sctx = dllama_exec::distributed::SyncCtx::Root { workers: &mut worker_streams };
            inf.forward_with_stream($tok, $pos, &mut sctx)
        }};
    }

    // feed the prompt (positions 0..n-1); the last forward's logits start
    let mut logits: Vec<f32> = Vec::new();
    let t0 = std::time::Instant::now();
    for pos in 0..n {
        let token = input[pos] as u32;
        logits = match dist_forward!(token, pos as u32) {
            Ok(l) => l.to_vec(),
            Err(e) => { eprintln!("error: {e}"); return 1; }
        };
    }
    let mut pos = n;
    let mut token = sampler.next(&logits);

    // generation loop — each sampled token is broadcast to every node so all
    // KV caches stay identical; only the root samples and decodes.
    let mut state = dllama_tokenizer::DecodeState::default();
    let mut generated = 0usize;
    while pos < seq_len && generated < steps {
        if tokenizer.is_eos(token) { break; }
        if let Some(piece) = tokenizer.decode(&mut state, token) {
            print!("{piece}");
            use std::io::Write as _;
            let _ = std::io::stdout().flush();
        }
        let tok32 = token as u32;
        logits = match dist_forward!(tok32, pos as u32) {
            Ok(l) => l.to_vec(),
            Err(e) => { eprintln!("error: {e}"); return 1; }
        };
        pos += 1;
        generated += 1;
        let next = sampler.next(&logits);
        if tokenizer.is_eos(next) { break; }
        token = next;
    }
    println!();
    let dt = t0.elapsed().as_secs_f64();
    let eval_s = dt - generated as f64 * 0.0; // (prompt+gen share the clock; report total)
    println!("\nPrediction ({n_nodes} nodes)");
    if generated > 0 {
        println!("   tokens/s: {:.2} ({:.0} ms/tok)", generated as f64 / eval_s, eval_s * 1000.0 / generated as f64);
    }
    println!("   prompt tokens: {n}, generated: {generated}");

    // clean stop so workers end their session gracefully (batch_size: 0)
    let stop = dllama_cluster::ControlPacket { token: 0, position: pos as u32, batch_size: 0 };
    let stop_bytes = stop.encode();
    for w in &mut worker_streams {
        let _ = w.write_all(&stop_bytes);
    }
    0
}

/// Interactive chat: ChatML template, GGUF tokenizer, streaming output.
pub fn run_chat(args: &[String]) -> i32 {
    apply_threads_flag(args);
    let model_path = match flag(args, "--model") {
        Some(p) => p.to_string(),
        None => {
            eprintln!("--model is required");
            return 1;
        }
    };
    let tokenizer_path = flag(args, "--tokenizer").unwrap_or("").to_string();
    let max_seq_len: u32 = flag(args, "--max-seq-len").unwrap_or("0").parse().unwrap_or(0);

    let source = match open_model(&model_path, max_seq_len) {
        Ok(f) => f,
        Err(e) => { eprintln!("🚨 {e}"); return 1; }
    };
    let mut meta = match source.meta(max_seq_len) {
        Ok(m) => m,
        Err(e) => { eprintln!("🚨 {e}"); return 1; }
    };
    meta.sync_type = parse_buffer_type(flag(args, "--buffer-float-type"));
    print_header(&meta);
    let weights = match source.weights() {
        Ok(w) => w,
        Err(e) => { eprintln!("🚨 {e}"); return 1; }
    };
    let tokenizer = match source.tokenizer(if tokenizer_path.is_empty() { None } else { Some(&tokenizer_path) }) {
        Ok(t) => t,
        Err(e) => { eprintln!("🚨 {e}"); return 1; }
    };
    println!("💡 Vocab: {} tokens, BOS: {}, EOS: {:?}", tokenizer.vocab_size(), tokenizer.bos_id, tokenizer.eos_token_ids);

    let mut inf = match dllama_exec::Inference::new(&meta, &weights) {
        Ok(i) => i,
        Err(e) => { eprintln!("🚨 {e}"); return 1; }
    };

    let seq_len = meta.seq_len as usize;
    let mut pos: usize = 0;
    let mut delta_items: Vec<(String, String)> = Vec::new();
    let mut state = dllama_tokenizer::DecodeState::default();
    let mut first_turn = true;

    loop {
        // read user input
        print!("\n👱 > ");
        use std::io::Write;
        std::io::stdout().flush().unwrap();
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() || input.trim().is_empty() {
            println!("👋 goodbye");
            return 0;
        }
        let input = input.trim().to_string();
        delta_items.push(("user".into(), input));

        // format with ChatML template
        let mut prompt = String::new();
        if first_turn {
            prompt.push_str("You are a helpful assistant.\n");
            first_turn = false;
        }
        for (role, content) in &delta_items {
            prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        prompt.push_str("<|im_start|>assistant\n");

        // tokenize
        let tokens = match tokenizer.encode(&prompt, pos == 0, true) {
            Ok(t) => t,
            Err(e) => { eprintln!("🚨 {e}"); return 1; }
        };
        let n = tokens.len();
        if pos + n >= seq_len {
            println!("⚠ context exhausted");
            return 0;
        }

        // feed prompt tokens[0..n-2] (all but the last — autoregressive pattern)
        // on multi-turn: only feed the delta (tokens after the previous pos)
        let prev_pos = pos;
        let feed_start = prev_pos; // absolute position to start feeding
        let feed_count = n - 1 - prev_pos; // tokens to feed (excluding last)
        if dllama_exec::prefill::prefill_enabled() && feed_count > 0 {
            let chunk: Vec<i32> = tokens[feed_start..feed_start + feed_count].to_vec();
            if let Err(e) = inf.prefill(&chunk, feed_start) {
                eprintln!("error: {e}");
                return 1;
            }
        } else {
            for i in 0..feed_count {
                let _ = inf.forward(tokens[feed_start + i] as u32, (feed_start + i) as u32);
            }
        }
        pos = feed_start + feed_count;

        // the last prompt token starts generation
        let mut token = tokens[n - 1];
        let mut generated = 0;
        while pos < seq_len {
            let logits = match inf.forward(token as u32, pos as u32) {
                Ok(l) => l,
                Err(e) => { eprintln!("🚨 {e}"); return 1; }
            };
            token = dllama_exec::argmax(logits) as i32;
            if tokenizer.is_eos(token) {
                break;
            }
            if let Some(piece) = tokenizer.decode(&mut state, token) {
                print!("{piece}");
                std::io::stdout().flush().unwrap();
            }
            pos += 1;
            generated += 1;
        }
        println!();
        delta_items.clear();
    }
}

fn print_header(meta: &ModelMeta) {
    println!("💡 Arch: {:?}", meta.arch);
    println!("💡 Dim: {}", meta.dim);
    println!("💡 HeadDim: {}", meta.head_dim_or_derived());
    println!("💡 HiddenDim: {}", meta.hidden_dim);
    println!("💡 VocabSize: {}", meta.vocab_size);
    println!("💡 nLayers: {}", meta.n_layers);
    println!("💡 nHeads: {}", meta.n_heads);
    println!("💡 nKvHeads: {}", meta.n_kv_heads);
    println!("💡 SeqLen: {}", meta.seq_len);
    println!("💡 NormEpsilon: {:.0e}", meta.norm_epsilon);
    println!("💡 RopeType: {:?}", meta.rope_type);
    println!("💡 RopeTheta: {:.0}", meta.rope_theta);
    if let Some(s) = &meta.rope_scaling {
        println!("💡 RopeScaling: f={:.1}, l={:.1}, h={:.1}, o={}", s.factor, s.low_freq_factor, s.high_freq_factor, s.orig_max_seq_len);
    }
}

/// Apply the `--threads N` CLI flag (G6). Call before building Inference.
pub fn apply_threads_flag(args: &[String]) {
    if let Some(t) = flag(args, "--threads") {
        if let Ok(n) = t.parse::<usize>() {
            dllama_kernel::set_threads(n);
        }
    }
    eprintln!(
        "[kernel] threads: {} (matmul rows; --threads N or DLLAMA_RS_THREADS)",
        dllama_kernel::threads()
    );
}

pub fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            return it.next().map(|s| s.as_str());
        }
    }
    None
}
