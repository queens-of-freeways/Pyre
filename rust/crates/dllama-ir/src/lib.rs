//! dllama-ir: model-agnostic computation-graph IR for the dllama Rust engine.
//!
//! The key idea (see ../DESIGN.md §4): ~all modern decoder LLMs are built from a
//! bounded op set; everything model-specific is *data* (`ModelMeta` flags + weight
//! names), never code. The C++ engine's per-arch switches (`LlmArchType`) die here.

pub mod builder;
pub mod shard;

pub use builder::build_decoder;
pub use shard::{check_sharding, per_token_sync_bytes, shard_all, shard_node, sync_bytes};

/// Pipe (preallocated buffer) index, mirroring the C++ executor model.
pub type Pipe = usize;

/// Float types — wire values mirror `NnFloatType` in `src/nn/nn-quants.hpp`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FloatType {
    F32,
    F16,
    Q40,
    Q80,
}

impl FloatType {
    pub const fn wire(self) -> u32 {
        match self {
            FloatType::F32 => 0,
            FloatType::F16 => 1,
            FloatType::Q40 => 2,
            FloatType::Q80 => 3,
        }
    }

    pub fn from_wire(v: u32) -> Option<Self> {
        match v {
            0 => Some(FloatType::F32),
            1 => Some(FloatType::F16),
            2 => Some(FloatType::Q40),
            3 => Some(FloatType::Q80),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            FloatType::F32 => "f32",
            FloatType::F16 => "f16",
            FloatType::Q40 => "q40",
            FloatType::Q80 => "q80",
        }
    }

    /// Bytes per value for a *blocked* type over a row of `n` values
    /// (n multiple of 32). Q40 -> 4.5 B/val, Q80 -> 8.5 B/val... expressed per row.
    pub const fn row_bytes(self, n: usize) -> usize {
        match self {
            FloatType::F32 => n * 4,
            FloatType::F16 => n * 2,
            FloatType::Q40 => n / 32 * 18,
            FloatType::Q80 => n / 32 * 34,
        }
    }
}

/// Family/arch flags — *not* a hard switch: the builder reads them as data.
/// Phase 2 replaces these with GGUF-metadata-derived flags.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arch {
    Llama,
    Qwen3,
    Qwen3Moe,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HiddenAct {
    Silu,
    Gelu,
}

/// Llama 3.1-style partial RoPE scaling (low/high freq factors).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeScaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub orig_max_seq_len: u32,
}

/// Rotation variant — mirrors `NnRopeType` (nn-core.hpp:125).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RopeType {
    /// Adjacent-pair rotation (C++ ROPE_LLAMA = 0).
    Llama,
    /// rotate-half: cos/sin in split halves (C++ ROPE_FALCON = 1 — used by Qwen3).
    Falcon,
    /// Llama adjacent pairs + llama3.1 partial scaling (C++ ROPE_LLAMA3_1 = 2).
    Llama31,
}

impl RopeType {
    pub const fn wire(self) -> i32 {
        match self {
            RopeType::Llama => 0,
            RopeType::Falcon => 1,
            RopeType::Llama31 => 2,
        }
    }
    pub fn from_wire(v: i32) -> Option<Self> {
        match v {
            0 => Some(RopeType::Llama),
            1 => Some(RopeType::Falcon),
            2 => Some(RopeType::Llama31),
            _ => None,
        }
    }
}

/// Model metadata — the Rust replacement for the fixed `LlmHeader` key enum.
/// Every field is data-driven; unknown models differ only in values/flags.
#[derive(Clone, Debug)]
pub struct ModelMeta {
    pub name: String,
    pub arch: Arch,
    pub dim: u32,
    /// Dense FFN intermediate size (0 for MoE-only models).
    pub hidden_dim: u32,
    /// Per-expert FFN intermediate size (0 for dense models).
    pub moe_hidden_dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    /// 0 => derived as dim / n_heads.
    pub head_dim: u32,
    pub n_experts: u32,
    pub n_active_experts: u32,
    pub vocab_size: u32,
    /// Effective context (--max-seq-len).
    pub seq_len: u32,
    pub orig_seq_len: u32,
    pub hidden_act: HiddenAct,
    pub norm_epsilon: f32,
    pub rope_theta: f32,
    pub rope_type: RopeType,
    pub rope_scaling: Option<RopeScaling>,
    /// Qwen3-style per-head q/k RMSNorm.
    pub qk_norm: bool,
    pub weight_type: FloatType,
    pub sync_type: FloatType,
}

impl ModelMeta {
    pub fn head_dim_or_derived(&self) -> u32 {
        if self.head_dim != 0 {
            self.head_dim
        } else {
            self.dim / self.n_heads
        }
    }
    pub fn q_dim(&self) -> u32 {
        self.n_heads * self.head_dim_or_derived()
    }
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim_or_derived()
    }
    pub const fn is_moe(&self) -> bool {
        self.n_experts > 0
    }
    pub fn ffn_dim(&self) -> u32 {
        if self.is_moe() {
            self.moe_hidden_dim
        } else {
            self.hidden_dim
        }
    }
}

/// How a matmul weight is split across nodes.
/// Generalizes the C++ `NnRowMatmulSlice` / `NnColMatmulSlice` pair:
///   OutSharded == C++ "row matmul" (Megatron column-parallel),
///   InSharded  == C++ "col matmul" (Megatron row-parallel).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Parallel {
    /// W's *output rows* are sharded; each node emits only its slice.
    OutSharded,
    /// W's *input columns* are sharded; every node computes a partial
    /// full-length output that must be summed (all-reduce).
    InSharded,
    /// Full copy on every node.
    Replicated,
}

/// Communication ops inserted between graph nodes (star topology, see DESIGN.md §4.4).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SyncKind {
    /// Root collects partials, sums, broadcasts the full vector.
    AllReduce,
    /// Root collects OutSharded slices and concatenates them.
    AllGather,
}

/// Pipe description (allocation plan for the executor).
#[derive(Clone, Debug)]
pub struct PipeSpec {
    pub name: &'static str,
    pub dtype: FloatType,
    pub len: u32,
}

/// The bounded op registry (DESIGN.md §4.2).
/// Weights are referenced by GGUF-convention names (without `.weight`).
#[derive(Clone, Debug)]
pub enum Op {
    /// out = embedding row for token (rows sharded by vocab).
    Embedding {
        w: String,
        token_pipe: Pipe,
        out: Pipe,
    },
    /// out = x · Wᵀ where W is [n, k] row-major.
    MatMul {
        w: String,
        parallel: Parallel,
        input: Pipe,
        output: Pipe,
    },
    RmsNorm {
        w: String,
        eps: f32,
        input: Pipe,
        output: Pipe,
    },
    /// In-place per-head RMSNorm of q and k (Qwen3-style `attn_q_norm`/`attn_k_norm`).
    QkRmsNorm {
        qw: String,
        kw: String,
        eps: f32,
        q_pipe: Pipe,
        k_pipe: Pipe,
    },
    /// In-place RoPE on q/k at the current position.
    Rope {
        q_pipe: Pipe,
        k_pipe: Pipe,
        position_pipe: Pipe,
    },
    /// Fused GQA attention with the per-layer KV cache.
    Attention {
        layer: u32,
        q_pipe: Pipe,
        k_pipe: Pipe,
        v_pipe: Pipe,
        output: Pipe,
    },
    /// out = act(gate) ⊙ up.
    ActivatedMul {
        act: HiddenAct,
        gate: Pipe,
        up: Pipe,
        out: Pipe,
    },
    /// Fused MoE: route -> top-k -> expert matmuls -> weighted combine.
    Moe {
        layer: u32,
        gate_w: String,
        w1: String,
        w2: String,
        w3: String,
        n_experts: u32,
        n_active: u32,
        input: Pipe,
        output: Pipe,
    },
    Softmax {
        input: Pipe,
        output: Pipe,
    },
    /// accumulator += addend.
    ResidualAdd {
        accumulator: Pipe,
        addend: Pipe,
        output: Pipe,
    },
    /// Logits head (rows sharded by vocab).
    Head {
        w: String,
        input: Pipe,
        output: Pipe,
    },
    Argmax {
        input: Pipe,
        output: Pipe,
    },
    /// Communication op inserted by the builder/splitter (n_nodes > 1 only).
    Sync {
        kind: SyncKind,
        pipe: Pipe,
        dtype: FloatType,
    },
}

impl Op {
    /// Short human-readable kind label (for graph-dump).
    pub fn kind(&self) -> &'static str {
        match self {
            Op::Embedding { .. } => "Embedding",
            Op::MatMul { .. } => "MatMul",
            Op::RmsNorm { .. } => "RmsNorm",
            Op::QkRmsNorm { .. } => "QkRmsNorm",
            Op::Rope { .. } => "Rope",
            Op::Attention { .. } => "Attention",
            Op::ActivatedMul { .. } => "ActivatedMul",
            Op::Moe { .. } => "Moe",
            Op::Softmax { .. } => "Softmax",
            Op::ResidualAdd { .. } => "ResidualAdd",
            Op::Head { .. } => "Head",
            Op::Argmax { .. } => "Argmax",
            Op::Sync { .. } => "Sync",
        }
    }
}

/// A full decoder graph: metadata + pipe allocation + ordered op list.
#[derive(Clone, Debug)]
pub struct Graph {
    pub meta: ModelMeta,
    pub pipes: Vec<PipeSpec>,
    pub ops: Vec<Op>,
}

impl Graph {
    pub fn count_syncs(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, Op::Sync { .. }))
            .count()
    }
}
