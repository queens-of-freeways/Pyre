//! dllama-model: loader for the `.m` model container.
//!
//! Byte-compatible port of `src/llm.cpp` (`loadLlmHeader` + `loadLlmNetWeight`):
//! - magic `0xA00ABCD`, `i32 headerSize`, then `(headerSize-8)` bytes of
//!   `i32` key/value pairs; weights start at file offset `headerSize`.
//! - weight order in the file (dense): embedding(f32) | per layer:
//!   q, k, v, wo, w1, w2, w3 (q40) [+ q_norm, k_norm (f32) for qwen3]
//!   + attn_norm, ffn_norm (f32) | final_norm (f32) | output head (q40).
//! - matmul weights are stored `[out][in]` row-major, q40 blocks along `in`.

use dllama_ir::{Arch, FloatType, HiddenAct, ModelMeta, RopeScaling, RopeType};
use dllama_quant::{q40_bytes, Q40_BLOCK_SIZE};
use std::collections::HashMap;

pub const MAGIC: i32 = 0xA00ABCD;

// Header keys — mirror `LlmHeaderKey` (llm.hpp).
const KEY_VERSION: i32 = 0;
const KEY_ARCH_TYPE: i32 = 1;
const KEY_DIM: i32 = 2;
const KEY_HIDDEN_DIM: i32 = 3;
const KEY_N_LAYERS: i32 = 4;
const KEY_N_HEADS: i32 = 5;
const KEY_N_KV_HEADS: i32 = 6;
const KEY_N_EXPERTS: i32 = 7;
const KEY_N_ACTIVE_EXPERTS: i32 = 8;
const KEY_VOCAB_SIZE: i32 = 9;
const KEY_SEQ_LEN: i32 = 10;
const KEY_HIDDEN_ACT: i32 = 11;
const KEY_ROPE_THETA: i32 = 12;
const KEY_WEIGHT_FLOAT_TYPE: i32 = 13;
const KEY_ROPE_SCALING_FACTOR: i32 = 14;
const KEY_ROPE_SCALING_LOW_FREQ_FACTOR: i32 = 15;
const KEY_ROPE_SCALING_HIGH_FREQ_FACTORY: i32 = 16;
const KEY_ROPE_SCALING_ORIG_MAX_SEQ_LEN: i32 = 17;
const KEY_ROPE_TYPE: i32 = 18;
const KEY_HEAD_DIM: i32 = 19;
const KEY_NORM_EPSILON: i32 = 20;
const KEY_MOE_HIDDEN_DIM: i32 = 21;

const ARCH_LLAMA: i32 = 0xABCD00;
const ARCH_QWEN3: i32 = 0xABCD01;
const ARCH_QWEN3_MOE: i32 = 0xABCD02;

fn i32_at(b: &[u8], off: usize) -> Result<i32, String> {
    if off + 4 > b.len() {
        return Err(format!("model file truncated at offset {off}"));
    }
    Ok(i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]))
}

/// Backing storage for model bytes: memory-mapped (large models stream
/// through the page cache — a 30B model cannot be `fs::read` into RAM) or
/// owned (tests, converters).
pub enum DataSource {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl DataSource {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            DataSource::Owned(v) => v,
            DataSource::Mapped(m) => m,
        }
    }
}

/// An opened model file: raw bytes + parsed metadata.
pub struct ModelFile {
    pub source: DataSource,
    pub meta: ModelMeta,
}

impl ModelFile {
    pub fn data(&self) -> &[u8] {
        self.source.as_slice()
    }
}

/// Named weight tensors, borrowed from the model file's weight region.
/// Each tensor carries its quant kind (per-tensor types for GGUF sources).
pub use dllama_quant::QuantKind;

#[derive(Clone, Copy, Debug)]
pub struct Tensor<'a> {
    pub kind: QuantKind,
    pub bytes: &'a [u8],
}

pub struct Weights<'a> {
    map: HashMap<String, Tensor<'a>>,
}

impl<'a> Weights<'a> {
    pub fn get(&self, name: &str) -> Result<Tensor<'a>, String> {
        self.map
            .get(name)
            .copied()
            .ok_or_else(|| format!("weight tensor not found: {name}"))
    }
    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
    /// construct from a pre-built name -> tensor map (GGUF adapter)
    pub fn from_map(map: HashMap<String, Tensor<'a>>) -> Self {
        Weights { map }
    }
    /// total number of values in a tensor (all rows)
    pub fn len_of(&self, name: &str) -> Result<usize, String> {
        let t = self.get(name)?;
        Ok(t.bytes.len() / t.kind.block_bytes() * t.kind.block_elems())
    }
}

/// Open a `.m` file: parse header, keep bytes for weight loading.
pub fn open(path: &str, max_seq_len: u32) -> Result<ModelFile, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("cannot open model file ({path}): {e}"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| format!("cannot mmap model file ({path}): {e}"))?;
    let meta = parse_header(&mmap, max_seq_len)?;
    Ok(ModelFile { source: DataSource::Mapped(mmap), meta })
}

impl ModelFile {
    /// Load the weights view (weights start at file offset `headerSize`).
    pub fn weights(&self) -> Result<Weights<'_>, String> {
        let data = self.data();
        let header_size = i32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        load_weights(&self.meta, &data[header_size..])
    }
}

fn norm_epsilon(value: i32) -> Result<f32, String> {
    match value {
        5 => Ok(1e-5),
        6 => Ok(1e-6),
        other => Err(format!("unsupported norm epsilon: {other}")),
    }
}

/// Parse the `.m` header — port of `loadLlmHeader` (llm.cpp:40).
fn parse_header(data: &[u8], max_seq_len: u32) -> Result<ModelMeta, String> {
    let magic = i32_at(data, 0)?;
    if magic == 0xABCD00 || magic == 0xABCD01 {
        return Err("old model format is not supported".into());
    }
    if magic != MAGIC {
        return Err(format!("unsupported magic number: {magic:#x}"));
    }
    let header_size = i32_at(data, 4)? as usize;
    if header_size < 8 || header_size > data.len() {
        return Err(format!("invalid header size: {header_size}"));
    }
    let n_kv = (header_size - 8) / 4; // ints of KV data

    let mut meta = ModelMeta {
        name: String::new(),
        arch: Arch::Llama,
        dim: 0,
        hidden_dim: 0,
        moe_hidden_dim: 0,
        n_layers: 0,
        n_heads: 0,
        n_kv_heads: 0,
        head_dim: 0,
        n_experts: 0,
        n_active_experts: 0,
        vocab_size: 0,
        seq_len: 0,
        orig_seq_len: 0,
        hidden_act: HiddenAct::Silu,
        norm_epsilon: 1e-5,
        rope_theta: 10000.0,
        rope_type: RopeType::Llama,
        rope_scaling: None,
        qk_norm: false,
        weight_type: FloatType::F32,
        sync_type: FloatType::Q80,
    };

    let mut rope_scaling_factor = 1.0f32;
    let mut low = 0.0f32;
    let mut high = 0.0f32;
    let mut orig_max: i32 = 0;
    let mut rope_type_raw: i32 = 0;
    let mut has_weight_type = false;

    let mut off = 8usize;
    while off + 8 <= header_size {
        let key = i32_at(data, off)?;
        let value = i32_at(data, off + 4)?;
        match key {
            KEY_VERSION => {} // version 1+ assumed
            KEY_ARCH_TYPE => {
                meta.arch = match value {
                    ARCH_LLAMA => Arch::Llama,
                    ARCH_QWEN3 => Arch::Qwen3,
                    ARCH_QWEN3_MOE => Arch::Qwen3Moe,
                    other => return Err(format!("unsupported architecture: {other:#x}")),
                };
            }
            KEY_DIM => meta.dim = value as u32,
            KEY_HIDDEN_DIM => meta.hidden_dim = value as u32,
            KEY_N_LAYERS => meta.n_layers = value as u32,
            KEY_N_HEADS => meta.n_heads = value as u32,
            KEY_N_KV_HEADS => meta.n_kv_heads = value as u32,
            KEY_N_EXPERTS => meta.n_experts = value as u32,
            KEY_N_ACTIVE_EXPERTS => meta.n_active_experts = value as u32,
            KEY_VOCAB_SIZE => meta.vocab_size = value as u32,
            KEY_SEQ_LEN => {
                meta.seq_len = value as u32;
                meta.orig_seq_len = value as u32;
            }
            KEY_HIDDEN_ACT => {
                meta.hidden_act = match value {
                    0 => HiddenAct::Gelu,
                    1 => HiddenAct::Silu,
                    other => return Err(format!("unsupported hidden act: {other}")),
                }
            }
            KEY_ROPE_THETA => meta.rope_theta = value as f32,
            KEY_WEIGHT_FLOAT_TYPE => {
                meta.weight_type = FloatType::from_wire(value as u32)
                    .ok_or_else(|| format!("unsupported weight float type: {value}"))?;
                has_weight_type = true;
            }
            KEY_ROPE_SCALING_FACTOR => rope_scaling_factor = value as f32,
            KEY_ROPE_SCALING_LOW_FREQ_FACTOR => low = value as f32,
            KEY_ROPE_SCALING_HIGH_FREQ_FACTORY => high = value as f32,
            KEY_ROPE_SCALING_ORIG_MAX_SEQ_LEN => orig_max = value,
            KEY_ROPE_TYPE => rope_type_raw = value,
            KEY_HEAD_DIM => meta.head_dim = value as u32,
            KEY_NORM_EPSILON => meta.norm_epsilon = norm_epsilon(value)?,
            KEY_MOE_HIDDEN_DIM => meta.moe_hidden_dim = value as u32,
            other => return Err(format!("unsupported header key: {other}")),
        }
        off += 8;
    }
    let _ = n_kv;
    let _ = KEY_VERSION;

    if !has_weight_type {
        return Err("model does not specify weight type".into());
    }

    // qwen3 arch forces the Falcon rope variant (llm.cpp:113)
    meta.rope_type = match meta.arch {
        Arch::Qwen3 | Arch::Qwen3Moe => RopeType::Falcon,
        _ => RopeType::from_wire(rope_type_raw).unwrap_or(RopeType::Llama),
    };
    if meta.rope_type == RopeType::Llama31 {
        meta.rope_scaling = Some(RopeScaling {
            factor: rope_scaling_factor,
            low_freq_factor: low,
            high_freq_factor: high,
            orig_max_seq_len: orig_max.max(0) as u32,
        });
    }
    meta.qk_norm = matches!(meta.arch, Arch::Qwen3 | Arch::Qwen3Moe);

    // effective context length (--max-seq-len)
    if max_seq_len > 0 && meta.seq_len > max_seq_len {
        meta.seq_len = max_seq_len;
    }
    if meta.head_dim == 0 {
        if meta.n_heads == 0 || meta.dim % meta.n_heads != 0 {
            return Err(format!(
                "invalid head config: dim={} n_heads={}",
                meta.dim, meta.n_heads
            ));
        }
        meta.head_dim = meta.dim / meta.n_heads;
    }
    if meta.dim == 0 || meta.n_layers == 0 || meta.vocab_size == 0 {
        return Err("incomplete model header".into());
    }
    if meta.weight_type == FloatType::Q40 && meta.dim % Q40_BLOCK_SIZE as u32 != 0 {
        return Err("q40 model with dim not divisible by 32".into());
    }
    meta.sync_type = if meta.weight_type == FloatType::Q40 {
        FloatType::Q80
    } else {
        FloatType::F32
    };
    Ok(meta)
}

/// Bytes of one `[rows][cols]` weight tensor with per-row blocked packing.
fn tensor_bytes(t: FloatType, rows: u32, cols: u32) -> Result<u64, String> {
    let cols = cols as usize;
    Ok(match t {
        FloatType::F32 => rows as u64 * cols as u64 * 4,
        FloatType::F16 => rows as u64 * cols as u64 * 2,
        FloatType::Q40 => {
            if cols % Q40_BLOCK_SIZE != 0 {
                return Err(format!("q40 tensor with {cols} columns (not multiple of 32)"));
            }
            rows as u64 * q40_bytes(cols) as u64
        }
        FloatType::Q80 => return Err("q80 is not a valid weight type".into()),
    })
}

/// Load all weight tensors by name, consuming the weight region in the exact
/// order the C++ loader reads it (llm.cpp:604 `loadLlmNetWeight`).
pub fn load_weights<'a>(meta: &ModelMeta, bytes: &'a [u8]) -> Result<Weights<'a>, String> {
    let dim = meta.dim;
    let q_dim = meta.q_dim();
    let kv_dim = meta.kv_dim();
    let vocab = meta.vocab_size;
    let ffn = if meta.is_moe() { meta.moe_hidden_dim } else { meta.hidden_dim };
    let wt = meta.weight_type;

    let mut map: HashMap<String, Tensor<'a>> = HashMap::new();
    let mut pos: usize = 0;
    let matmul_kind = if meta.weight_type == FloatType::F32 {
        QuantKind::F32
    } else {
        QuantKind::DllamaQ40
    };
    let mut take = |n: u64, kind: QuantKind, name: String, map: &mut HashMap<String, Tensor<'a>>| -> Result<(), String> {
        let n = n as usize;
        if pos + n > bytes.len() {
            return Err(format!(
                "weight file truncated: need {n} bytes for '{name}' at {pos}, have {}",
                bytes.len() - pos
            ));
        }
        map.insert(name, Tensor { kind, bytes: &bytes[pos..pos + n] });
        pos += n;
        Ok(())
    };

    // token embedding — always f32 ([vocab][dim])
    take(vocab as u64 * dim as u64 * 4, QuantKind::F32, "token_embd".into(), &mut map)?;

    for l in 0..meta.n_layers {
        let layer = l as usize;
        take(tensor_bytes(wt, q_dim, dim)?, matmul_kind, format!("blk.{layer}.attn_q"), &mut map)?;
        take(tensor_bytes(wt, kv_dim, dim)?, matmul_kind, format!("blk.{layer}.attn_k"), &mut map)?;
        take(tensor_bytes(wt, kv_dim, dim)?, matmul_kind, format!("blk.{layer}.attn_v"), &mut map)?;
        take(tensor_bytes(wt, dim, q_dim)?, matmul_kind, format!("blk.{layer}.attn_output"), &mut map)?;

        if meta.n_experts > 0 {
            // gate router (f32, [dim][n_experts]) then per-expert w1/w2/w3 blocks
            take(dim as u64 * meta.n_experts as u64 * 4, QuantKind::F32, format!("blk.{layer}.ffn_gate_inp"), &mut map)?;
            let w1 = tensor_bytes(wt, ffn, dim)?;
            let w2 = tensor_bytes(wt, dim, ffn)?;
            let w3 = tensor_bytes(wt, ffn, dim)?;
            let per_expert = w1 + w2 + w3;
            let total = meta.n_experts as u64 * per_expert;
            take(total, matmul_kind, format!("blk.{layer}.moe_exps"), &mut map)?;
        } else {
            take(tensor_bytes(wt, ffn, dim)?, matmul_kind, format!("blk.{layer}.ffn_gate"), &mut map)?;
            take(tensor_bytes(wt, dim, ffn)?, matmul_kind, format!("blk.{layer}.ffn_down"), &mut map)?;
            take(tensor_bytes(wt, ffn, dim)?, matmul_kind, format!("blk.{layer}.ffn_up"), &mut map)?;
        }

        if meta.qk_norm {
            take(meta.head_dim as u64 * 4, QuantKind::F32, format!("blk.{layer}.attn_q_norm"), &mut map)?;
            take(meta.head_dim as u64 * 4, QuantKind::F32, format!("blk.{layer}.attn_k_norm"), &mut map)?;
        }
        take(dim as u64 * 4, QuantKind::F32, format!("blk.{layer}.attn_norm"), &mut map)?;
        take(dim as u64 * 4, QuantKind::F32, format!("blk.{layer}.ffn_norm"), &mut map)?;
    }

    take(dim as u64 * 4, QuantKind::F32, "output_norm".into(), &mut map)?;
    take(tensor_bytes(wt, vocab, dim)?, matmul_kind, "output".into(), &mut map)?;

    if pos != bytes.len() {
        return Err(format!(
            "missing bytes in weight file: {} (consumed {} of {})",
            bytes.len() as i64 - pos as i64,
            pos,
            bytes.len()
        ));
    }
    Ok(Weights { map })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal synthetic dense .m file (1 layer, dim 64, q40 weights).
    fn synthetic_model() -> Vec<u8> {
        let dim: i32 = 64;
        let q_dim: i32 = 4 * 16; // 4 heads x 16
        let kv_dim: i32 = 2 * 16;
        let ffn: i32 = 128;
        let vocab: i32 = 64;
        let kv: &[(i32, i32)] = &[
            (KEY_VERSION, 1),
            (KEY_ARCH_TYPE, ARCH_QWEN3),
            (KEY_DIM, dim),
            (KEY_HIDDEN_DIM, ffn),
            (KEY_N_LAYERS, 1),
            (KEY_N_HEADS, 4),
            (KEY_N_KV_HEADS, 2),
            (KEY_VOCAB_SIZE, vocab),
            (KEY_SEQ_LEN, 512),
            (KEY_HIDDEN_ACT, 1),
            (KEY_ROPE_THETA, 1000000),
            (KEY_WEIGHT_FLOAT_TYPE, 2), // q40
            (KEY_ROPE_TYPE, 0),
            (KEY_HEAD_DIM, 16),
            (KEY_NORM_EPSILON, 6),
        ];
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        let header_bytes = 8 + kv.len() * 8;
        buf.extend_from_slice(&(header_bytes as i32).to_le_bytes());
        for (k, v) in kv {
            buf.extend_from_slice(&k.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // embedding f32: vocab*dim*4
        buf.extend(vec![0u8; (vocab * dim * 4) as usize]);
        let q40_row = |cols: i32| (cols / 32 * 18) as usize;
        // q: q_dim rows x dim; k,v: kv_dim x dim; wo: dim x q_dim
        buf.extend(vec![1u8; (q_dim as usize) * q40_row(dim)]);
        buf.extend(vec![2u8; (kv_dim as usize) * q40_row(dim)]);
        buf.extend(vec![3u8; (kv_dim as usize) * q40_row(dim)]);
        buf.extend(vec![4u8; (dim as usize) * q40_row(q_dim)]);
        // w1: ffn x dim; w2: dim x ffn; w3: ffn x dim
        buf.extend(vec![5u8; (ffn as usize) * q40_row(dim)]);
        buf.extend(vec![6u8; (dim as usize) * q40_row(ffn)]);
        buf.extend(vec![7u8; (ffn as usize) * q40_row(dim)]);
        // qwen3: q_norm, k_norm f32 headDim
        buf.extend(vec![8u8; (16 * 4) as usize]);
        buf.extend(vec![9u8; (16 * 4) as usize]);
        // attn_norm, ffn_norm f32 dim
        buf.extend(vec![10u8; (dim * 4) as usize]);
        buf.extend(vec![11u8; (dim * 4) as usize]);
        // final_norm + head q40 vocab x dim
        buf.extend(vec![12u8; (dim * 4) as usize]);
        buf.extend(vec![13u8; (vocab as usize) * q40_row(dim)]);
        buf
    }

    #[test]
    fn parses_synthetic_model() {
        let bytes = synthetic_model();
        let meta = parse_header(&bytes, 0).unwrap();
        assert_eq!(meta.arch, Arch::Qwen3);
        assert_eq!(meta.dim, 64);
        assert_eq!(meta.n_layers, 1);
        assert_eq!(meta.head_dim, 16);
        assert_eq!(meta.norm_epsilon, 1e-6);
        assert_eq!(meta.weight_type, FloatType::Q40);
        assert!(meta.qk_norm);
        assert_eq!(meta.rope_type, RopeType::Falcon); // forced for qwen3
        assert_eq!(meta.seq_len, 512);
        // max-seq-len cap
        let meta2 = parse_header(&bytes, 128).unwrap();
        assert_eq!(meta2.seq_len, 128);
        assert_eq!(meta2.orig_seq_len, 512);

        // weights: header_size offset, all tensors by name, exact byte count
        let header_size = i32_at(&bytes, 4).unwrap() as usize;
        let w = load_weights(&meta, &bytes[header_size..]).unwrap();
        assert_eq!(w.get("token_embd").unwrap().bytes.len(), 64 * 64 * 4);
        assert_eq!(w.get("blk.0.attn_q").unwrap().bytes.len(), 64 * (64 / 32 * 18));
        assert_eq!(w.get("blk.0.attn_k").unwrap().bytes.len(), 32 * (64 / 32 * 18));
        assert_eq!(w.get("blk.0.attn_output").unwrap().bytes.len(), 64 * (64 / 32 * 18));
        assert_eq!(w.get("blk.0.ffn_gate").unwrap().bytes.len(), 128 * (64 / 32 * 18));
        assert_eq!(w.get("blk.0.ffn_down").unwrap().bytes.len(), 64 * (128 / 32 * 18));
        assert_eq!(w.get("blk.0.attn_q_norm").unwrap().bytes.len(), 16 * 4);
        assert_eq!(w.get("output").unwrap().bytes.len(), 64 * (64 / 32 * 18));
        // kinds: matmuls are dllama-q40, norms/embedding f32
        assert_eq!(w.get("blk.0.attn_q").unwrap().kind, QuantKind::DllamaQ40);
        assert_eq!(w.get("token_embd").unwrap().kind, QuantKind::F32);
        // first byte of each tensor region is distinct (sanity of ordering)
        assert_eq!(w.get("blk.0.attn_q").unwrap().bytes[0], 1);
        assert_eq!(w.get("output").unwrap().bytes[0], 13);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = synthetic_model();
        bytes[0] = 0;
        assert!(parse_header(&bytes, 0).is_err());
    }

    #[test]
    fn detects_truncated_weights() {
        let bytes = synthetic_model();
        let meta = parse_header(&bytes, 0).unwrap();
        let header_size = i32_at(&bytes, 4).unwrap() as usize;
        assert!(load_weights(&meta, &bytes[header_size..header_size + 100]).is_err());
    }
}
