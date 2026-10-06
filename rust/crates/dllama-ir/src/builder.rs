//! The generic decoder-graph builder.
//!
//! ONE builder covers the whole llama-family: Llama 2/3.x, Qwen 3 (dense + MoE),
//! Mistral, Gemma-style decoders. What the C++ engine encodes as `archType`
//! switches (`src/llm.cpp:156,181,322,425`) is here just `ModelMeta` data:
//! `qk_norm`, `is_moe()`, `hidden_act`, rope params.
//!
//! Weight names follow GGUF conventions so the Phase-2 rule table maps
//! directly onto them.

use crate::{FloatType, Graph, ModelMeta, Op, Parallel, PipeSpec, Pipe, SyncKind};

/// Pipe indices (single-token step; batching handled by the executor dimension).
pub mod pipes {
    use crate::Pipe;
    pub const TOKEN: Pipe = 0;
    pub const POSITION: Pipe = 1;
    pub const X: Pipe = 2; // residual stream (dim)
    pub const XB: Pipe = 3; // normed activations (dim)
    pub const XB2: Pipe = 4; // attention/FFN output before residual add (dim)
    pub const Q: Pipe = 5;
    pub const K: Pipe = 6;
    pub const V: Pipe = 7;
    pub const ATT: Pipe = 8; // attention output (q_dim)
    pub const HB: Pipe = 9; // ffn gate projection (ffn_dim)
    pub const HB2: Pipe = 10; // ffn up projection (ffn_dim)
    pub const LOGITS: Pipe = 11; // vocab
    pub const OUT: Pipe = 12; // argmax result
    pub const COUNT: usize = 13;
}

/// Build the full decoder graph for `meta` across `n_nodes` nodes.
///
/// Sync ops are inserted at the canonical points (DESIGN.md §4.4) only when
/// `n_nodes > 1`; a single-node graph has no communication ops.
pub fn build_decoder(meta: &ModelMeta, n_nodes: u32) -> Graph {
    let dim = meta.dim;
    let q_dim = meta.q_dim();
    let kv_dim = meta.kv_dim();
    let ffn_dim = meta.ffn_dim();
    let vocab = meta.vocab_size;
    let multi_node = n_nodes > 1;

    let pipes = vec![
        PipeSpec { name: "token", dtype: FloatType::F32, len: 1 },
        PipeSpec { name: "position", dtype: FloatType::F32, len: 1 },
        PipeSpec { name: "x", dtype: FloatType::F32, len: dim },
        PipeSpec { name: "xb", dtype: FloatType::F32, len: dim },
        PipeSpec { name: "xb2", dtype: FloatType::F32, len: dim },
        PipeSpec { name: "q", dtype: FloatType::F32, len: q_dim },
        PipeSpec { name: "k", dtype: FloatType::F32, len: kv_dim },
        PipeSpec { name: "v", dtype: FloatType::F32, len: kv_dim },
        PipeSpec { name: "att", dtype: FloatType::F32, len: q_dim },
        PipeSpec { name: "hb", dtype: FloatType::F32, len: ffn_dim },
        PipeSpec { name: "hb2", dtype: FloatType::F32, len: ffn_dim },
        PipeSpec { name: "logits", dtype: FloatType::F32, len: vocab },
        PipeSpec { name: "out", dtype: FloatType::F32, len: 1 },
    ];
    debug_assert_eq!(pipes.len(), pipes::COUNT);

    let mut ops: Vec<Op> = Vec::new();
    let sync = |ops: &mut Vec<Op>, kind: SyncKind, pipe: Pipe| {
        if multi_node {
            ops.push(Op::Sync { kind, pipe, dtype: meta.sync_type });
        }
    };

    // --- token embedding: vocab rows are OutSharded -> all-reduce partial rows ---
    ops.push(Op::Embedding {
        w: "token_embd".into(),
        token_pipe: pipes::TOKEN,
        out: pipes::X,
    });
    sync(&mut ops, SyncKind::AllReduce, pipes::X);

    for l in 0..meta.n_layers {
        let layer = l as u32;

        // ---- attention block ----
        ops.push(Op::RmsNorm {
            w: format!("blk.{l}.attn_norm"),
            eps: meta.norm_epsilon,
            input: pipes::X,
            output: pipes::XB,
        });
        ops.push(Op::MatMul { w: format!("blk.{l}.attn_q"), parallel: Parallel::OutSharded, input: pipes::XB, output: pipes::Q });
        ops.push(Op::MatMul { w: format!("blk.{l}.attn_k"), parallel: Parallel::OutSharded, input: pipes::XB, output: pipes::K });
        ops.push(Op::MatMul { w: format!("blk.{l}.attn_v"), parallel: Parallel::OutSharded, input: pipes::XB, output: pipes::V });
        if meta.qk_norm {
            ops.push(Op::QkRmsNorm {
                qw: format!("blk.{l}.attn_q_norm"),
                kw: format!("blk.{l}.attn_k_norm"),
                eps: meta.norm_epsilon,
                q_pipe: pipes::Q,
                k_pipe: pipes::K,
            });
        }
        ops.push(Op::Rope { q_pipe: pipes::Q, k_pipe: pipes::K, position_pipe: pipes::POSITION });
        ops.push(Op::Attention { layer, q_pipe: pipes::Q, k_pipe: pipes::K, v_pipe: pipes::V, output: pipes::ATT });
        ops.push(Op::MatMul { w: format!("blk.{l}.attn_output"), parallel: Parallel::InSharded, input: pipes::ATT, output: pipes::XB2 });
        sync(&mut ops, SyncKind::AllReduce, pipes::XB2); // sum partial attention outputs
        ops.push(Op::ResidualAdd { accumulator: pipes::X, addend: pipes::XB2, output: pipes::X });

        // ---- feed-forward block ----
        ops.push(Op::RmsNorm {
            w: format!("blk.{l}.ffn_norm"),
            eps: meta.norm_epsilon,
            input: pipes::X,
            output: pipes::XB,
        });
        if meta.is_moe() {
            ops.push(Op::Moe {
                layer,
                gate_w: format!("blk.{l}.ffn_gate_inp"),
                w1: format!("blk.{l}.ffn_gate_exps"),
                w2: format!("blk.{l}.ffn_down_exps"),
                w3: format!("blk.{l}.ffn_up_exps"),
                n_experts: meta.n_experts,
                n_active: meta.n_active_experts,
                input: pipes::XB,
                output: pipes::XB2,
            });
        } else {
            ops.push(Op::MatMul { w: format!("blk.{l}.ffn_gate"), parallel: Parallel::OutSharded, input: pipes::XB, output: pipes::HB });
            ops.push(Op::MatMul { w: format!("blk.{l}.ffn_up"), parallel: Parallel::OutSharded, input: pipes::XB, output: pipes::HB2 });
            ops.push(Op::ActivatedMul { act: meta.hidden_act, gate: pipes::HB, up: pipes::HB2, out: pipes::HB });
            ops.push(Op::MatMul { w: format!("blk.{l}.ffn_down"), parallel: Parallel::InSharded, input: pipes::HB, output: pipes::XB2 });
        }
        sync(&mut ops, SyncKind::AllReduce, pipes::XB2); // sum partial FFN outputs
        ops.push(Op::ResidualAdd { accumulator: pipes::X, addend: pipes::XB2, output: pipes::X });
    }

    // --- output head ---
    ops.push(Op::RmsNorm { w: "output_norm".into(), eps: meta.norm_epsilon, input: pipes::X, output: pipes::XB });
    ops.push(Op::Head { w: "output".into(), input: pipes::XB, output: pipes::LOGITS });
    sync(&mut ops, SyncKind::AllGather, pipes::LOGITS); // gather vocab slices at root
    ops.push(Op::Argmax { input: pipes::LOGITS, output: pipes::OUT });

    Graph { meta: meta.clone(), pipes, ops }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard::check_sharding;
    use crate::HiddenAct;

    fn llama8b() -> ModelMeta {
        ModelMeta {
            name: "llama-3.1-8b".into(),
            arch: crate::Arch::Llama,
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
            rope_type: crate::RopeType::Llama31,
            rope_scaling: Some(crate::RopeScaling {
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
    fn single_node_graph_has_no_syncs() {
        let g = build_decoder(&llama8b(), 1);
        assert_eq!(g.count_syncs(), 0);
        // 1 embed + 32 layers * 14 ops (no syncs, no qk-norm) + norm + head + argmax
        assert_eq!(g.ops.len(), 1 + 32 * 14 + 3);
    }

    #[test]
    fn four_node_graph_inserts_canonical_syncs() {
        let g = build_decoder(&llama8b(), 4);
        // 2 per layer + 1 after embedding + 1 logits gather
        assert_eq!(g.count_syncs(), 2 * 32 + 2);
        assert_eq!(g.ops.len(), 1 + 32 * 16 + 2 + 3);
        check_sharding(&g.meta, 4).expect("llama8b sharding over 4 nodes must be valid");
    }

    #[test]
    fn moe_and_qk_norm_flags_change_wiring() {
        let mut m = llama8b();
        m.qk_norm = true;
        m.n_experts = 128;
        m.n_active_experts = 8;
        m.moe_hidden_dim = 512;
        m.hidden_dim = 0;
        m.arch = crate::Arch::Qwen3Moe;
        let g = build_decoder(&m, 2);
        assert!(g.ops.iter().any(|op| matches!(op, Op::Moe { .. })));
        assert!(g.ops.iter().any(|op| matches!(op, Op::QkRmsNorm { .. })));
        // dense ffn ops must not appear
        assert!(!g.ops.iter().any(|op| matches!(op, Op::ActivatedMul { .. })));
    }

    #[test]
    fn weights_use_gguf_names() {
        let g = build_decoder(&llama8b(), 1);
        let names: Vec<&str> = g
            .ops
            .iter()
            .filter_map(|op| match op {
                Op::MatMul { w, .. } | Op::RmsNorm { w, .. } | Op::Head { w, .. } => Some(w.as_str()),
                Op::Embedding { w, .. } => Some(w.as_str()),
                _ => None,
            })
            .collect();
        assert!(names.contains(&"token_embd"));
        assert!(names.contains(&"output"));
        assert!(names.contains(&"blk.0.attn_q"));
        assert!(names.contains(&"blk.31.ffn_down"));
    }
}
