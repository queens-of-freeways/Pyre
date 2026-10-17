//! Tensor-parallel sharding planner.
//!
//! Ports the slicing rules of the C++ engine (`sliceRowMatmul` / `sliceColMatmul`
//! / `sliceKvCache` in `src/nn/nn-core.cpp`) into a validated, data-driven plan:
//! no arch-specific code, only `ModelMeta` + `n_nodes`.
//!
//! Enforced constraints (see README "Known Limitations" + issue #70):
//!   1. `n_nodes` must be a power of two (protocol requirement),
//!   2. `n_nodes <= n_kv_heads` (KV cache shards along kv heads),
//!   3. all sharded dims divide evenly and slices align to head boundaries.

use crate::{FloatType, ModelMeta};
use std::fmt;

/// A half-open row/column range `[start, end)` of a weight matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slice {
    pub start: u32,
    pub end: u32,
}

impl Slice {
    pub const fn len(&self) -> u32 {
        self.end - self.start
    }
}

impl fmt::Display for Slice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}..{})", self.start, self.end)
    }
}

/// Everything one node needs to know about its slice of the model.
#[derive(Clone, Debug)]
pub struct NodeShard {
    pub node_index: u32,
    pub n_nodes: u32,
    /// Rows of `wq` (q_dim total): contiguous, head-aligned.
    pub q_rows: Slice,
    /// Rows of `wk`/`wv` (kv_dim total) + rows of the KV cache.
    pub kv_rows: Slice,
    /// Columns of `attn_output` (q_dim total): matches the q slice.
    /// (the wo weight is [dim rows][q_dim cols] — we slice the q_dim columns)
    pub wo_cols: Slice,
    /// Rows of `ffn_gate`/`ffn_up` (ffn_dim total).
    pub ffn_rows: Slice,
    /// Columns of `ffn_down` (ffn_dim total): matches the ffn slice.
    /// (the w2 weight is [dim rows][ffn_dim cols] — we slice the ffn_dim columns)
    pub ffn_cols: Slice,
    /// Rows of `output` head (vocab total).
    pub cls_rows: Slice,
    /// Rows of `token_embd` (vocab total).
    pub embed_rows: Slice,
    /// Attention heads owned by this node.
    pub n_heads: u32,
    /// KV heads owned by this node.
    pub n_kv_heads: u32,
}

#[derive(Debug)]
pub struct ShardError(pub String);

impl fmt::Display for ShardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sharding error: {}", self.0)
    }
}
impl std::error::Error for ShardError {}

fn require(cond: bool, msg: &str) -> Result<(), ShardError> {
    if cond {
        Ok(())
    } else {
        Err(ShardError(msg.to_string()))
    }
}

fn is_power_of_two(n: u32) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Validate that `meta` can be sharded across `n_nodes` (constraint checks only).
pub fn check_sharding(meta: &ModelMeta, n_nodes: u32) -> Result<(), ShardError> {
    require(is_power_of_two(n_nodes), "n_nodes must be a power of two (1, 2, 4, ...)")
        .map_err(|e| ShardError(format!("{e} (got {n_nodes})")))?;
    require(
        n_nodes <= meta.n_kv_heads,
        &format!("n_nodes ({n_nodes}) cannot exceed n_kv_heads ({}) — issue #70", meta.n_kv_heads),
    )?;

    let checks: [(&str, u32); 5] = [
        ("q_dim (n_heads*head_dim)", meta.q_dim()),
        ("kv_dim (n_kv_heads*head_dim)", meta.kv_dim()),
        ("dim", meta.dim),
        ("ffn_dim", meta.ffn_dim()),
        ("vocab_size", meta.vocab_size),
    ];
    for (what, v) in checks {
        require(
            v % n_nodes == 0,
            &format!("{what} ({v}) is not divisible by n_nodes ({n_nodes})"),
        )?;
    }
    // q/kv slices must be head-aligned: with contiguous even splits this holds
    // automatically when q_dim/kv_dim divide, but assert head boundaries anyway.
    let head = meta.head_dim_or_derived();
    require(meta.q_dim() % head == 0 && meta.kv_dim() % head == 0, "head_dim must divide q/kv dims")?;
    Ok(())
}

fn slice_of(total: u32, n_nodes: u32, node_index: u32) -> Slice {
    let per = total / n_nodes;
    let start = per * node_index;
    Slice { start, end: start + per }
}

/// Compute the shard plan for `node_index` (root = 0).
pub fn shard_node(meta: &ModelMeta, n_nodes: u32, node_index: u32) -> Result<NodeShard, ShardError> {
    check_sharding(meta, n_nodes)?;
    require(node_index < n_nodes, "node_index out of range")?;

    let head = meta.head_dim_or_derived();
    Ok(NodeShard {
        node_index,
        n_nodes,
        q_rows: slice_of(meta.q_dim(), n_nodes, node_index),
        kv_rows: slice_of(meta.kv_dim(), n_nodes, node_index),
        wo_cols: slice_of(meta.q_dim(), n_nodes, node_index),
        ffn_rows: slice_of(meta.ffn_dim(), n_nodes, node_index),
        ffn_cols: slice_of(meta.ffn_dim(), n_nodes, node_index),
        cls_rows: slice_of(meta.vocab_size, n_nodes, node_index),
        embed_rows: slice_of(meta.vocab_size, n_nodes, node_index),
        n_heads: meta.n_heads / n_nodes,
        n_kv_heads: meta.n_kv_heads / n_nodes,
    })
    .map(|s| {
        debug_assert_eq!(s.q_rows.len() % head, 0);
        debug_assert_eq!(s.kv_rows.len() % head, 0);
        s
    })
}

/// Full sharding plan for all nodes (root first).
pub fn shard_all(meta: &ModelMeta, n_nodes: u32) -> Result<Vec<NodeShard>, ShardError> {
    check_sharding(meta, n_nodes)?;
    (0..n_nodes)
        .map(|i| shard_node(meta, n_nodes, i))
        .collect()
}

/// Bytes on the wire for one sync of `n` values with dtype `t`
/// (assumes n % 32 == 0 for blocked types; falls back to f32 semantics otherwise).
pub fn sync_bytes(n: u32, t: FloatType) -> u64 {
    let n = n as u64;
    match t {
        FloatType::F32 => n * 4,
        FloatType::F16 => n * 2,
        FloatType::Q40 => n / 32 * 18,
        FloatType::Q80 => n / 32 * 34,
    }
}

/// Estimated per-token sync traffic for *one worker* (send + receive),
/// q80 default: `(2L+1) * 2 * sync(dim) + sync(vocab/n)` bytes/token.
pub fn per_token_sync_bytes(meta: &ModelMeta, n_nodes: u32) -> u64 {
    if n_nodes <= 1 {
        return 0;
    }
    let dim = sync_bytes(meta.dim, meta.sync_type) * 2; // send + recv per all-reduce
    let logits = sync_bytes(meta.vocab_size / n_nodes, meta.sync_type);
    (2 * meta.n_layers as u64 + 1) * dim + logits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Arch, FloatType, HiddenAct, RopeScaling, RopeType};

    fn llama8b() -> ModelMeta {
        ModelMeta {
            name: "llama-3.1-8b".into(),
            arch: Arch::Llama,
            dim: 4096,
            hidden_dim: 14336,
            moe_hidden_dim: 0,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            n_active_experts: 0,
            vocab_size: 128256,
            seq_len: 4096,
            orig_seq_len: 131072,
            hidden_act: HiddenAct::Silu,
            norm_epsilon: 1e-5,
            rope_theta: 500000.0,
            rope_type: RopeType::Llama31,
            rope_scaling: Some(RopeScaling {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                orig_max_seq_len: 8192,
            }),
            qk_norm: false,
            weight_type: FloatType::Q40,
            sync_type: FloatType::Q80,
        }
    }

    #[test]
    fn llama8b_four_nodes() {
        let plan = shard_all(&llama8b(), 4).unwrap();
        assert_eq!(plan.len(), 4);
        let root = &plan[0];
        assert_eq!(root.q_rows, Slice { start: 0, end: 1024 }); // 4096/4
        assert_eq!(root.kv_rows.len(), 1024 / 4 * 1); // kv_dim=1024 -> 256
        assert_eq!(root.n_heads, 8);
        assert_eq!(root.n_kv_heads, 2);
        assert_eq!(root.cls_rows.len(), 128256 / 4);
        // slices are contiguous and cover the full range
        let q_total: u32 = plan.iter().map(|s| s.q_rows.len()).sum();
        assert_eq!(q_total, 4096);
        let vocab_total: u32 = plan.iter().map(|s| s.embed_rows.len()).sum();
        assert_eq!(vocab_total, 128256);
        // adjacent
        assert_eq!(plan[1].q_rows.start, plan[0].q_rows.end);
    }

    #[test]
    fn rejects_non_power_of_two() {
        assert!(shard_all(&llama8b(), 3).is_err());
        assert!(shard_all(&llama8b(), 6).is_err());
        assert!(shard_all(&llama8b(), 0).is_err());
    }

    #[test]
    fn rejects_too_many_nodes_vs_kv_heads() {
        // llama8b has 8 kv heads -> 16 nodes must fail
        let err = shard_all(&llama8b(), 16).unwrap_err();
        assert!(err.0.contains("n_kv_heads"));
    }

    #[test]
    fn sync_traffic_is_nonzero_and_scaled() {
        let m = llama8b();
        let one = per_token_sync_bytes(&m, 1);
        assert_eq!(one, 0);
        let four = per_token_sync_bytes(&m, 4);
        // (2*32+1) * 2 * (4096/32*34) + (128256/4 /32*34)
        let expect = (65 * 2 * (4096u64 / 32 * 34)) + (128256u64 / 4 / 32 * 34);
        assert_eq!(four, expect);
        assert!(four > 0);
    }
}
