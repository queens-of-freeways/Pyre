//! dllama-gguf: GGUF container loader — the model-agnostic front door (G2).
//!
//! Parses the GGUF format (magic `GGUF`, KV metadata, tensor directory, data
//! section) and converts it through a **rule table** (no per-model code):
//! - metadata keyed by `<arch>.*` → `ModelMeta` flags,
//! - tensor names follow GGUF conventions (our IR already uses them),
//! - per-tensor quant kinds map to `dllama-quant::QuantKind`.
//!
//! Supported: llama-family architectures (llama, mistral, qwen2, qwen3, ...)
//! with F32/F16/Q4_0/Q8_0/Q4_K/Q6_K tensors. MoE metadata is parsed; the
//! executor lands in G2.5. Tied embeddings (missing `output` tensor) reuse
//! `token_embd`.

use dllama_ir::{Arch, FloatType, HiddenAct, ModelMeta, RopeScaling, RopeType};
use dllama_model::{DataSource, QuantKind, Tensor, Weights};
use std::collections::HashMap;

const GGUF_MAGIC: [u8; 4] = *b"GGUF";

// ggml tensor type ids (ggml.h ggml_type)
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_Q4_0: u32 = 2;
const GGML_Q4_1: u32 = 3;
const GGML_Q5_0: u32 = 6;
const GGML_Q5_1: u32 = 7;
const GGML_Q8_0: u32 = 8;
const GGML_Q2_K: u32 = 10;
const GGML_Q3_K: u32 = 11;
const GGML_Q4_K: u32 = 12;
const GGML_Q5_K: u32 = 13;
const GGML_Q6_K: u32 = 14;
const GGML_Q8_K: u32 = 15;
const GGML_BF16: u32 = 30;

// metadata value types (gguf spec)
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STR: u32 = 8;
const T_ARR: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

#[derive(Clone, Debug)]
pub enum Value {
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::U32(v) => Some(*v),
            Value::U64(v) => u32::try_from(*v).ok(),
            Value::I32(v) => u32::try_from(*v).ok(),
            Value::I64(v) => u32::try_from(*v).ok(),
            Value::F32(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            Value::U32(v) => Some(*v as f32),
            Value::I32(v) => Some(*v as f32),
            Value::U64(v) => Some(*v as f32),
            Value::I64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

pub struct TensorInfo {
    pub name: String,
    pub ggml_type: u32,
    pub ne: Vec<u64>, // ne[0] = row length (in), ne[1] = rows (out)
    pub offset: u64,  // relative to the data section
}

/// An opened GGUF file.
pub struct GgufFile {
    pub source: DataSource,
    pub version: u32,
    kvs: HashMap<String, Value>,
    tensors: Vec<TensorInfo>,
    data_start: usize,
    alignment: u32,
}

impl GgufFile {
    pub fn data(&self) -> &[u8] {
        self.source.as_slice()
    }

    pub fn open(path: &str) -> Result<GgufFile, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("cannot open gguf file ({path}): {e}"))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| format!("cannot mmap gguf file ({path}): {e}"))?;
        Self::parse_impl(&mmap).map(|(version, kvs, tensors, data_start, alignment)| GgufFile {
            source: DataSource::Mapped(mmap),
            version,
            kvs,
            tensors,
            data_start,
            alignment,
        })
    }

    pub fn parse(data: Vec<u8>) -> Result<GgufFile, String> {
        let (version, kvs, tensors, data_start, alignment) = Self::parse_impl(&data)?;
        Ok(GgufFile {
            source: DataSource::Owned(data),
            version,
            kvs,
            tensors,
            data_start,
            alignment,
        })
    }

    fn parse_impl(data: &[u8]) -> Result<(u32, HashMap<String, Value>, Vec<TensorInfo>, usize, u32), String> {
        parse_impl(data)
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err(format!("gguf truncated at {} (need {n})", self.pos));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64, String> {
        Ok(i64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let len = self.u64()? as usize;
        let b = self.bytes(len)?;
        String::from_utf8(b.to_vec()).map_err(|_| "invalid utf-8 in gguf string".into())
    }
    fn value(&mut self, vtype: u32) -> Result<Value, String> {
        Ok(match vtype {
            T_U8 => Value::U32(self.bytes(1)?[0] as u32),
            T_I8 => Value::I32(self.bytes(1)?[0] as i8 as i32),
            T_U16 => Value::U32(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()) as u32),
            T_I16 => Value::I32(i16::from_le_bytes(self.bytes(2)?.try_into().unwrap()) as i32),
            T_U32 => Value::U32(self.u32()?),
            T_I32 => Value::I32(self.i32()?),
            T_F32 => Value::F32(self.f32()?),
            T_BOOL => Value::Bool(self.bytes(1)?[0] != 0),
            T_STR => Value::Str(self.string()?),
            T_U64 => Value::U64(self.u64()?),
            T_I64 => Value::I64(self.i64()?),
            T_F64 => Value::F64(self.f64()?),
            T_ARR => {
                let elem = self.u32()?;
                let count = self.u64()? as usize;
                let mut items = Vec::with_capacity(count.min(1 << 16));
                for _ in 0..count {
                    items.push(self.value(elem)?);
                }
                Value::Array(items)
            }
            other => return Err(format!("unsupported gguf value type: {other}")),
        })
    }
}

/// Parse the container from raw bytes (shared by `open` and `parse`).
fn parse_impl(
    data: &[u8],
) -> Result<(u32, HashMap<String, Value>, Vec<TensorInfo>, usize, u32), String> {
    let mut r = Reader { data, pos: 0 };
    let magic = r.bytes(4)?;
    if magic != GGUF_MAGIC {
        return Err(format!("not a gguf file (magic {:?})", magic));
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(format!("unsupported gguf version: {version}"));
    }
    let tensor_count = r.u64()? as usize;
    let kv_count = r.u64()? as usize;

    let mut kvs = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = r.string()?;
        let vtype = r.u32()?;
        let value = r.value(vtype)?;
        kvs.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()? as usize;
        if n_dims == 0 || n_dims > 4 {
            return Err(format!("bad tensor dims for {name}: {n_dims}"));
        }
        let mut ne = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            ne.push(r.u64()?);
        }
        let ggml_type = r.u32()?;
        let offset = r.u64()?;
        tensors.push(TensorInfo { name, ggml_type, ne, offset });
    }

    let alignment = kvs
        .get("general.alignment")
        .and_then(|v| v.as_u32())
        .unwrap_or(32)
        .max(1) as usize;
    let data_start = r.pos.div_ceil(alignment) * alignment;

    if data_start > data.len() {
        return Err("gguf data section out of range".into());
    }
    Ok((version, kvs, tensors, data_start, alignment as u32))
}

impl GgufFile {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kvs.get(key)
    }
    pub fn str_of(&self, key: &str) -> Result<String, String> {
        self.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| format!("missing string metadata: {key}"))
    }
    pub fn u32_of(&self, key: &str) -> Result<u32, String> {
        self.get(key)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| format!("missing u32 metadata: {key}"))
    }
    pub fn f32_of(&self, key: &str) -> Result<f32, String> {
        self.get(key)
            .and_then(|v| v.as_f32())
            .ok_or_else(|| format!("missing f32 metadata: {key}"))
    }
    pub fn f32_or(&self, key: &str, default: f32) -> f32 {
        self.get(key).and_then(|v| v.as_f32()).unwrap_or(default)
    }
    pub fn u32_or(&self, key: &str, default: u32) -> u32 {
        self.get(key).and_then(|v| v.as_u32()).unwrap_or(default)
    }

    fn tensor_kind(&self, t: &TensorInfo) -> Result<QuantKind, String> {
        Ok(match t.ggml_type {
            GGML_F32 => QuantKind::F32,
            GGML_F16 => QuantKind::F16,
            GGML_Q4_0 => QuantKind::GgufQ4_0,
            GGML_Q8_0 => QuantKind::GgufQ8_0,
            GGML_Q4_K => QuantKind::GgufQ4K,
            GGML_Q6_K => QuantKind::GgufQ6K,
            GGML_Q4_1 | GGML_Q5_0 | GGML_Q5_1 | GGML_Q2_K | GGML_Q3_K | GGML_Q5_K | GGML_Q8_K => {
                return Err(format!(
                    "tensor '{}' uses quant type {} — not supported yet (G2 covers Q4_0/Q8_0/Q4_K/Q6_K/F16/F32)",
                    t.name, t.ggml_type
                ));
            }
            GGML_BF16 => return Err(format!("tensor '{}': bf16 not supported yet", t.name)),
            other => return Err(format!("tensor '{}': unknown ggml type {other}", t.name)),
        })
    }

    /// Byte size of one tensor (rows x row-bytes).
    pub fn tensor_size(&self, t: &TensorInfo, kind: QuantKind) -> Result<usize, String> {
        if t.ne.is_empty() {
            return Err(format!("tensor '{}' has no dims", t.name));
        }
        let row_len = t.ne[0] as usize;
        let rows: usize = t.ne[1..].iter().product::<u64>() as usize;
        if row_len % kind.block_elems() != 0 {
            return Err(format!(
                "tensor '{}' row length {row_len} not aligned to {}",
                t.name,
                kind.block_elems()
            ));
        }
        Ok(kind.row_bytes(row_len) as usize * rows)
    }

    /// Build the named-weights view (IR names) + validate every tensor slice.
    pub fn weights(&self) -> Result<Weights<'_>, String> {
        let mut map: HashMap<String, Tensor<'_>> = HashMap::with_capacity(self.tensors.len());
        for t in &self.tensors {
            let kind = self.tensor_kind(t)?;
            let size = self.tensor_size(t, kind)?;
            let start = self.data_start + t.offset as usize;
            if start + size > self.data().len() {
                return Err(format!(
                    "tensor '{}' data out of range (offset {}, size {size})",
                    t.name, t.offset
                ));
            }
            let name = t.name.strip_suffix(".weight").unwrap_or(&t.name).to_string();
            map.insert(
                name,
                Tensor { kind, bytes: &self.data()[start..start + size] },
            );
        }
        // tied embeddings: missing output head -> reuse token_embd
        if !map.contains_key("output") {
            if let Some(embd) = map.get("token_embd").copied() {
                map.insert("output".into(), embd);
            }
        }
        Ok(Weights::from_map(map))
    }

    /// Metadata -> ModelMeta rule table (the model-agnostic core).
    pub fn meta(&self, max_seq_len: u32) -> Result<ModelMeta, String> {
        let arch = self.str_of("general.architecture")?;
        let family = match arch.as_str() {
            "llama" | "mistral" | "qwen2" | "qwen3" | "qwen3moe" | "smollm" | "exa" => arch,
            other => {
                return Err(format!(
                    "unsupported GGUF architecture '{other}' (G2 covers the llama family)"
                ));
            }
        };
        let k = |key: &str| format!("{family}.{key}");

        let dim = self.u32_of(&k("embedding_length"))?;
        let heads = self.u32_of(&k("attention.head_count"))?;
        let kv_heads = self.u32_or(&k("attention.head_count_kv"), heads);
        let head_dim = self.u32_or(&k("attention.key_length"), if dim % heads == 0 { dim / heads } else { 0 });
        if head_dim == 0 {
            return Err("cannot derive head_dim".into());
        }
        let n_layers = self.u32_of(&k("block_count"))?;
        let hidden = self.u32_or(&k("feed_forward_length"), 0);
        let vocab = self
            .u32_or(&k("vocab_size"), 0)
            .max(self.tensor_rows("token_embd")? as u32);
        let seq_len = self.u32_or(&k("context_length"), 4096);
        let eps = self.f32_or(&k("attention.layer_norm_rms_epsilon"), 1e-5);
        let theta = self.f32_or(&k("rope.freq_base"), 10000.0);
        let n_experts = self.u32_or(&k("expert_count"), 0);
        let moe_hidden = self.u32_or(&k("expert_feed_forward_length"), 0);

        // rope variant per family + scaling metadata
        let (rope_type, rope_scaling) = match family.as_str() {
            "qwen2" | "qwen3" | "qwen3moe" => (RopeType::Falcon, None),
            _ => {
                let scaling_type = self
                    .get(&k("rope.scaling.type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("none")
                    .to_string();
                if scaling_type == "llama3" {
                    (
                        RopeType::Llama31,
                        Some(RopeScaling {
                            factor: self.f32_or(&k("rope.scaling.factor"), 8.0),
                            low_freq_factor: self.f32_or(&k("rope.scaling.low_freq_factor"), 1.0),
                            high_freq_factor: self.f32_or(&k("rope.scaling.high_freq_factor"), 4.0),
                            orig_max_seq_len: self.u32_or(
                                &k("rope.scaling.original_max_position_embeddings"),
                                8192,
                            ),
                        }),
                    )
                } else {
                    (RopeType::Llama, None)
                }
            }
        };

        // qk-norm is data-driven: presence of the qwen3-style tensors
        let qk_norm = self.tensors.iter().any(|t| t.name == "blk.0.attn_q_norm.weight");

        let name = self
            .get("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or("gguf model")
            .to_string();

        let arch_enum = match family.as_str() {
            "qwen3" => {
                if n_experts > 0 {
                    Arch::Qwen3Moe
                } else {
                    Arch::Qwen3
                }
            }
            "qwen3moe" => Arch::Qwen3Moe,
            _ => Arch::Llama,
        };

        let mut meta = ModelMeta {
            name,
            arch: arch_enum,
            dim,
            hidden_dim: hidden,
            moe_hidden_dim: moe_hidden,
            n_layers,
            n_heads: heads,
            n_kv_heads: kv_heads,
            head_dim,
            n_experts,
            n_active_experts: self.u32_or(&k("expert_used_count"), 0),
            vocab_size: vocab,
            seq_len,
            orig_seq_len: seq_len,
            hidden_act: HiddenAct::Silu,
            norm_epsilon: eps,
            rope_theta: theta,
            rope_type,
            rope_scaling,
            qk_norm,
            // marker: quantized model -> q80 activation/sync semantics
            weight_type: FloatType::Q40,
            sync_type: FloatType::Q80,
        };
        if max_seq_len > 0 && meta.seq_len > max_seq_len {
            meta.seq_len = max_seq_len;
        }
        Ok(meta)
    }

    fn tensor_rows(&self, name: &str) -> Result<u64, String> {
        let t = self
            .tensors
            .iter()
            .find(|t| t.name == name || t.name == format!("{name}.weight"))
            .ok_or_else(|| format!("missing tensor: {name}"))?;
        Ok(t.ne.get(1).copied().unwrap_or(1))
    }

    /// Build a Tokenizer from the GGUF-embedded tokenizer metadata.
    /// Reads `tokenizer.ggml.tokens`, `scores`, `bos_token_id`, `eos_token_id`,
    /// `add_bos_token`, and `tokenizer.chat_template`.
    pub fn tokenizer(&self) -> Result<dllama_tokenizer::Tokenizer, String> {
        let tokens = self
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                Value::Array(items) => Some(items),
                _ => None,
            })
            .ok_or("GGUF has no tokenizer.ggml.tokens metadata")?;

        let vocab: Vec<Vec<u8>> = tokens
            .iter()
            .map(|t| match t {
                Value::Str(s) => s.as_bytes().to_vec(),
                _ => vec![],
            })
            .collect();        let scores: Vec<f32> = self
            .get("tokenizer.ggml.scores")
            .and_then(|v| match v {
                Value::Array(items) => Some(items),
                _ => None,
            })
            .map(|items| {
                items
                    .iter()
                    .map(|s| match s {
                        Value::F32(v) => *v,
                        Value::F64(v) => *v as f32,
                        Value::I32(v) => *v as f32,
                        Value::I64(v) => *v as f32,
                        _ => 0.0,
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![0.0; vocab.len()]);

        let bos_id = self.u32_or("tokenizer.ggml.bos_token_id", 0) as i32;
        let eos_id = self.u32_or("tokenizer.ggml.eos_token_id", bos_id as u32) as i32;
        let add_bos = self
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| match v {
                Value::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);

        let chat_template = self
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        dllama_tokenizer::Tokenizer::from_gguf_data(
            vocab,
            scores,
            bos_id,
            vec![eos_id],
            add_bos,
            chat_template,
        )
    }
}
