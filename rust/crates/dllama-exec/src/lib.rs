//! dllama-exec: single-node executor that walks the `dllama-ir` graph with
//! numerics matching the C++ engine:
//!
//! - activations are **q80-cast before every matmul** (C++ `block_cast_y`,
//!   applied whenever the sync/buffer type is q80 — llm.cpp:209),
//! - rope is applied from a precomputed cache (`fullfillRopeCache`,
//!   nn-core.cpp:376 — Falcon split-halves for Qwen3, adjacent pairs for
//!   llama + llama3.1 partial scaling),
//! - attention scores are `dot/sqrt(headDim)` with softmax over `[0..=pos]`
//!   (nn-cpu-ops.cpp:771).
//!
//! G2: weights are `Tensor { kind, bytes }` — the matmul dispatch is
//! per-tensor quant kind (dllama q40, GGUF Q4_0/Q8_0/Q4_K/Q6_K, F16/F32).

use dllama_ir::{build_decoder, FloatType, Graph, HiddenAct, ModelMeta, Op, RopeType};
use dllama_kernel::{Kernels, KvCache};
use dllama_model::{QuantKind, Tensor, Weights};
use dllama_quant::{dequantize_row_q80, q80_bytes, quantize_row_q80};
use std::collections::HashMap;

/// Per-expert weight views for one MoE layer.
/// - Split: GGUF layout (ffn_gate_exps / ffn_down_exps / ffn_up_exps tensors,
///   each [n_experts][rows][row_len]).
/// - Lumped: dllama `.m` layout (per expert w1|w2|w3 contiguous).
enum ExpertTensors<'a> {
    Split { w1: Tensor<'a>, w2: Tensor<'a>, w3: Tensor<'a>, s1: usize, s2: usize, s3: usize },
    Lumped { t: Tensor<'a>, s1: usize, s2: usize, s3: usize, stride: usize },
}

impl<'a> ExpertTensors<'a> {
    /// (w1, w2, w3) byte slices for expert `e` — each contiguous.
    fn expert(&self, e: usize) -> (Tensor<'a>, Tensor<'a>, Tensor<'a>) {
        match self {
            ExpertTensors::Split { w1, w2, w3, s1, s2, s3 } => (
                Tensor { kind: w1.kind, bytes: &w1.bytes[e * s1..(e + 1) * s1] },
                Tensor { kind: w2.kind, bytes: &w2.bytes[e * s2..(e + 1) * s2] },
                Tensor { kind: w3.kind, bytes: &w3.bytes[e * s3..(e + 1) * s3] },
            ),
            ExpertTensors::Lumped { t, s1, s2, s3, stride } => {
                let base = e * stride;
                (
                    Tensor { kind: t.kind, bytes: &t.bytes[base..base + s1] },
                    Tensor { kind: t.kind, bytes: &t.bytes[base + s1..base + s1 + s2] },
                    Tensor { kind: t.kind, bytes: &t.bytes[base + s1 + s2..base + stride] },
                )
            }
        }
    }
}

pub struct Inference<'w> {
    pub meta: ModelMeta,
    graph: Graph,
    weights: &'w Weights<'w>,
    /// op-level tracing (matches the C++ DEBUG_OP_INPUT_OUTPUT output format)
    debug: bool,
    /// f32 activations per pipe (batch = 1).
    pipes: Vec<Vec<f32>>,
    /// per-layer KV caches ([pos][kv_dim] layout)
    kv: Vec<KvCache>,
    /// rope cache ([pos][stride]; stride = head_dim for Falcon, q_dim for llama)
    rope_cache: Vec<f32>,
    rope_stride: usize,
    /// pre-extracted f32 norm/embedding weights by tensor name
    f32_weights: HashMap<String, Vec<f32>>,
    /// scratch: q80 bytes for the matmul/residual input cast
    q80_buf: Vec<u8>,
    /// G6.3: reusable f32 scratch (residual round-trip, zero-clone ops)
    scratch: Vec<f32>,
    /// G3: shard info (None = single node)
    shard: Option<dllama_ir::shard::NodeShard>,
    /// G3: this node is the root (head gather + argmax)
    is_root: bool,
    /// G5: resolved kernel backend (C-ABI table; see dllama-kernel::Kernels)
    pub k: Kernels,
}

impl<'w> Inference<'w> {
    pub fn new(meta: &ModelMeta, weights: &'w Weights<'w>) -> Result<Inference<'w>, String> {
        let graph = build_decoder(meta, 1); // single node: no Sync ops
        let seq_len = meta.seq_len as usize;
        let kv_dim = meta.kv_dim() as usize;
        let kv = (0..meta.n_layers).map(|_| KvCache::new(seq_len, kv_dim)).collect();

        let (rope_cache, rope_stride) = build_rope_cache(meta);

        // pre-extract f32 norm weights (any quant kind -> f32 vectors)
        let mut f32_weights = HashMap::new();
        for l in 0..meta.n_layers {
            for name in [
                format!("blk.{l}.attn_norm"),
                format!("blk.{l}.ffn_norm"),
                format!("blk.{l}.attn_q_norm"),
                format!("blk.{l}.attn_k_norm"),
            ] {
                if let Ok(t) = weights.get(&name) {
                    f32_weights.insert(name, dequant_vec(t.kind, t.bytes));
                }
            }
        }
        if let Ok(t) = weights.get("output_norm") {
            f32_weights.insert("output_norm".into(), dequant_vec(t.kind, t.bytes));
        }
        if meta.weight_type == FloatType::F32 {
            // pre-convert matmul weights for f32-buffer models (q40 is the fast path)
            let mut names: Vec<String> = vec!["token_embd".into(), "output".into()];
            for l in 0..meta.n_layers {
                for w in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_down", "ffn_up"] {
                    names.push(format!("blk.{l}.{w}"));
                }
            }
            for name in names {
                if let Ok(t) = weights.get(&name) {
                    f32_weights.insert(name, dequant_vec(t.kind, t.bytes));
                }
            }
        }

        let pipes: Vec<Vec<f32>> = graph
            .pipes
            .iter()
            .map(|p| vec![0.0f32; p.len as usize])
            .collect();
        let max_in = graph.pipes.iter().map(|p| p.len as usize).max().unwrap_or(0);
        // round up to a whole q80 block (vocab need not be 32-divisible)
        let q80_len = (max_in + 31) / 32 * 34;

        Ok(Inference {
            meta: meta.clone(),
            graph,
            weights,
            debug: std::env::var("DLLAMA_RS_DEBUG").is_ok(),
            pipes,
            kv,
            rope_cache,
            rope_stride,
            f32_weights,
            q80_buf: vec![0u8; q80_len],
            scratch: vec![0.0f32; max_in],
            shard: None,
            is_root: true,
            k: Kernels::load(),
        })
    }

    /// Run one forward step; returns the logits pipe.
    ///
    /// G6.3: zero per-token clones — the graph and meta are borrowed, and the
    /// ops mutate the pipes in place through disjoint field borrows.
    pub fn forward(&mut self, token: u32, pos: u32) -> Result<&[f32], String> {
        use dllama_ir::builder::pipes as P;
        self.pipes[P::TOKEN][0] = token as f32;
        self.pipes[P::POSITION][0] = pos as f32;

        // destructure `self` once: the op loop borrows graph/meta immutably
        // while ops take the mutable pieces as independent parameters (which
        // is what makes multi-pipe ops borrow-check without cloning).
        let Inference {
            pipes,
            graph,
            meta,
            weights,
            f32_weights,
            q80_buf,
            scratch,
            kv,
            rope_cache,
            rope_stride,
            k,
            debug,
            shard: _,
            is_root: _,
        } = self;
        for op in &graph.ops {
            exec_op(
                op, meta, pipes, weights, f32_weights, q80_buf, scratch, kv,
                rope_cache, *rope_stride, k, *debug,
            )?;
        }
        Ok(&pipes[P::LOGITS])
    }
}

/// One graph op, clone-free (G6.3). Output pipes are `mem::take`-n and put
/// back; inputs are shared borrows; scratch work reuses `q80_buf`/`scratch`.
/// The op set and dbg trace line order match the previous cloned executor
/// exactly (trace-diff verified byte-identical).
#[allow(clippy::too_many_arguments)]
fn exec_op(
    op: &Op,
    meta: &ModelMeta,
    pipes: &mut Vec<Vec<f32>>,
    weights: &Weights<'_>,
    f32_weights: &HashMap<String, Vec<f32>>,
    q80_buf: &mut Vec<u8>,
    scratch: &mut Vec<f32>,
    kv: &mut [KvCache],
    rope_cache: &[f32],
    rope_stride: usize,
    k: &Kernels,
    debug: bool,
) -> Result<(), String> {
    use dllama_ir::builder::pipes as P;
    match op {
        Op::Embedding { w, out, .. } => {
            let token = pipes[P::TOKEN][0] as usize;
            let dim = meta.dim as usize;
            let t = weights.get(w)?;
            let row_bytes = t.kind.row_bytes(dim) as usize;
            if (token + 1) * row_bytes > t.bytes.len() {
                return Err(format!("token {token} out of embedding range").into());
            }
            let row = &t.bytes[token * row_bytes..(token + 1) * row_bytes];
            let mut o = std::mem::take(&mut pipes[*out]);
            k.dequant_row(t.kind, row, &mut o, dim);
            if debug {
                dbg_vec("embedding", "output", &o);
            }
            pipes[*out] = o;
            Ok(())
        }
        Op::MatMul { w, input, output, .. } | Op::Head { w, input, output } => {
            let mut out_buf = std::mem::take(&mut pipes[*output]);
            let x: &[f32] = &pipes[*input];
            if debug {
                dbg_vec(cpp_name(w), "input", x);
            }
            matmul_impl(&mut out_buf, x, w, meta, weights, f32_weights, q80_buf, k)?;
            if debug {
                dbg_vec(cpp_name(w), "output", &out_buf);
            }
            pipes[*output] = out_buf;
            Ok(())
        }
        Op::RmsNorm { w, input, output, eps } => {
            let wv = f32_weights
                .get(w)
                .ok_or_else(|| format!("missing f32 norm weights: {w}"))?;
            let mut out_buf = std::mem::take(&mut pipes[*output]);
            let x: &[f32] = &pipes[*input];
            if debug {
                dbg_vec(cpp_name(w), "input", x);
            }
            k.rmsnorm(&mut out_buf, x, wv, *eps);
            if debug {
                dbg_vec(cpp_name(w), "output", &out_buf);
            }
            pipes[*output] = out_buf;
            Ok(())
        }
        Op::QkRmsNorm { qw, kw, eps, q_pipe, k_pipe } => {
            static EMPTY: &[f32] = &[];
            let wq = f32_weights.get(qw).map(|v| &v[..]).unwrap_or(EMPTY);
            let wk = f32_weights.get(kw).map(|v| &v[..]).unwrap_or(EMPTY);
            let hd = meta.head_dim_or_derived() as usize;
            qk_rmsnorm(&mut pipes[*q_pipe], wq, hd, *eps, meta.n_heads as usize, k);
            qk_rmsnorm(&mut pipes[*k_pipe], wk, hd, *eps, meta.n_kv_heads as usize, k);
            if debug {
                dbg_vec("block_norm_q", "output", &pipes[*q_pipe]);
                dbg_vec("block_norm_k", "output", &pipes[*k_pipe]);
            }
            Ok(())
        }
        Op::Rope { q_pipe, k_pipe, position_pipe } => {
            let pos = pipes[*position_pipe][0] as usize;
            let hd = meta.head_dim_or_derived() as usize;
            apply_rope(&mut pipes[*q_pipe], rope_cache, rope_stride, pos, hd);
            apply_rope(&mut pipes[*k_pipe], rope_cache, rope_stride, pos, hd);
            if debug {
                dbg_vec("block_rope_q", "output", &pipes[*q_pipe]);
                dbg_vec("block_rope_k", "output", &pipes[*k_pipe]);
            }
            Ok(())
        }
        Op::Attention { layer, output, .. } => {
            let pos = pipes[P::POSITION][0] as usize;
            let mut out_buf = std::mem::take(&mut pipes[*output]);
            let q: &[f32] = &pipes[P::Q];
            let kq: &[f32] = &pipes[P::K];
            let v: &[f32] = &pipes[P::V];
            let cache = &mut kv[*layer as usize];
            let kv_dim = cache.kv_dim;
            k.attention(
                &mut cache.k,
                &mut cache.v,
                kv_dim,
                q,
                kq,
                v,
                pos,
                meta.n_heads as usize,
                meta.n_kv_heads as usize,
                meta.head_dim_or_derived() as usize,
                &mut out_buf,
            );
            if debug {
                dbg_vec("block_multihead_att", "output", &out_buf);
            }
            pipes[*output] = out_buf;
            Ok(())
        }
        Op::ActivatedMul { act, gate, up, out } => {
            // The builder emits `out == gate` (silu writes back into the gate
            // pipe). The op is element-wise, so in-place is numerically safe,
            // but the take-pattern needs the aliased content snapshotted to
            // `scratch` first (memcpy into the reusable buffer — no alloc).
            let mut out_buf = std::mem::take(&mut pipes[*out]);
            let n = out_buf.len();
            let g: &[f32];
            let u: &[f32];
            if *out == *gate || *out == *up {
                if scratch.len() < n {
                    scratch.resize(n, 0.0);
                }
                scratch[..n].copy_from_slice(&out_buf);
                g = if *out == *gate { &scratch[..n] } else { &pipes[*gate] };
                u = if *out == *up { &scratch[..n] } else { &pipes[*up] };
            } else {
                g = &pipes[*gate];
                u = &pipes[*up];
            }
            k.activated_mul(&mut out_buf, g, u, *act == HiddenAct::Gelu);
            if debug {
                dbg_vec("block_mul", "output", &out_buf);
            }
            pipes[*out] = out_buf;
            Ok(())
        }
        Op::ResidualAdd { accumulator, addend, output } => {
            // C++ parity: the addend travels through the q80 `zq` pipe
            // (`block_cast_d` / `block_cast_d3`) before `merge_add` — the
            // residual stream only ever sees q80-quantized block outputs
            // when the buffer type is q80 (llm.cpp:536,581).
            let len = pipes[*addend].len();
            if scratch.len() < len {
                scratch.resize(len, 0.0);
            }
            if meta.sync_type != FloatType::F32 {
                let need = q80_bytes(len);
                if q80_buf.len() < need {
                    q80_buf.resize(need, 0);
                }
                quantize_row_q80(&pipes[*addend], &mut q80_buf[..need]);
                dequantize_row_q80(&q80_buf[..need], &mut scratch[..len], len);
            } else {
                scratch[..len].copy_from_slice(&pipes[*addend]);
            }
            if debug {
                dbg_vec("block_merge_add", "input", &scratch[..len]);
            }
            for (a, b) in pipes[*accumulator].iter_mut().zip(scratch[..len].iter()) {
                *a += *b;
            }
            if debug {
                dbg_vec("block_merge_add", "output", &pipes[*accumulator]);
            }
            if *output != *accumulator {
                let acc = std::mem::take(&mut pipes[*accumulator]);
                pipes[*output].clone_from_slice(&acc);
                pipes[*accumulator] = acc;
            }
            Ok(())
        }
        Op::Softmax { input, output } => {
            let n = pipes[*input].len();
            let mut out_buf = std::mem::take(&mut pipes[*output]);
            out_buf.copy_from_slice(&pipes[*input]);
            k.softmax(&mut out_buf, n);
            pipes[*output] = out_buf;
            Ok(())
        }
        Op::Argmax { input, output } => {
            pipes[*output][0] = argmax(&pipes[*input]) as f32;
            Ok(())
        }
        Op::Moe { layer, gate_w, w1, w2, w3, n_experts, n_active, input, output } => {
            let x: &[f32] = &pipes[*input];
            let dim = meta.dim as usize;
            let ffn = meta.moe_hidden_dim as usize;
            // resolve split (GGUF) or lumped (.m) expert storage
            let experts = match (weights.get(w1), weights.get(w2), weights.get(w3)) {
                (Ok(a), Ok(b), Ok(c)) => ExpertTensors::Split {
                    s1: a.kind.row_bytes(dim) as usize * ffn,
                    s2: b.kind.row_bytes(ffn) as usize * dim,
                    s3: c.kind.row_bytes(dim) as usize * ffn,
                    w1: a,
                    w2: b,
                    w3: c,
                },
                _ => {
                    let lumped = weights.get(&format!("blk.{layer}.moe_exps"))?;
                    let s1 = lumped.kind.row_bytes(dim) as usize * ffn;
                    let s2 = lumped.kind.row_bytes(ffn) as usize * dim;
                    ExpertTensors::Lumped { t: lumped, s1, s2, s3: s1, stride: s1 + s2 + s1 }
                },
            };
            let out = moe_ffn(
                x,
                &weights.get(gate_w)?,
                &experts,
                *n_experts as usize,
                *n_active as usize,
                meta.sync_type != FloatType::F32,
                k,
            )?;
            if debug {
                dbg_vec("block_moe_merge_sum", "output", &out);
            }
            pipes[*output].copy_from_slice(&out);
            Ok(())
        }
        Op::Sync { .. } => Ok(()), // unreachable for n_nodes == 1
    }
}

/// out = x · Wᵀ with the engine's q80-activation-cast semantics
/// (C++ `block_cast_y`, llm.cpp:209) whenever buffers are q80.
#[allow(clippy::too_many_arguments)]
fn matmul_impl(
    out: &mut [f32],
    x: &[f32],
    w_name: &str,
    meta: &ModelMeta,
    weights: &Weights<'_>,
    f32_weights: &HashMap<String, Vec<f32>>,
    q80_buf: &mut Vec<u8>,
    k: &Kernels,
) -> Result<(), String> {
    let n = out.len();
    let k_len = x.len();
    let w = weights.get(w_name)?;
    if meta.sync_type != FloatType::F32 {
        // q80-cast activations; integer-exact per-kind kernels
        let need = q80_bytes(k_len);
        if q80_buf.len() < need {
            q80_buf.resize(need, 0);
        }
        quantize_row_q80(x, &mut q80_buf[..need]);
        k.matmul_q80(out, &q80_buf[..need], w.kind, w.bytes, n, k_len);
    } else {
        match w.kind {
            QuantKind::F32 => {
                let wv = f32_weights
                    .get(w_name)
                    .ok_or_else(|| format!("missing f32 weights: {w_name}"))?;
                k.matmul_f32(out, x, wv, 1, n, k_len);
            }
            QuantKind::DllamaQ40 => {
                k.matmul_q40(out, x, w.bytes, 1, n, k_len);
            }
            // GGUF quants with f32 buffers: dequant rows + f32 dot
            other => {
                let row_bytes = other.row_bytes(k_len) as usize;
                let mut wrow = vec![0.0f32; k_len];
                for (di, o) in out.iter_mut().enumerate() {
                    let row = &w.bytes[di * row_bytes..(di + 1) * row_bytes];
                    if row.len() < row_bytes {
                        return Err("weight row out of range".into());
                    }
                    k.dequant_row(other, row, &mut wrow, k_len);
                    *o = k.dot(x, &wrow);
                }
            }
        }
    }
    Ok(())
}


/// dequantize a whole byte buffer (k = bytes/block_bytes * block_elems)
fn dequant_vec(kind: QuantKind, bytes: &[u8]) -> Vec<f32> {
    let k = bytes.len() / kind.block_bytes() * kind.block_elems();
    let mut out = vec![0.0f32; k];
    dllama_kernel::gguf::dequant_row(kind, bytes, &mut out, k);
    out
}

/// dequant a weight row and dot with f32 activations (f32-buffer mode)
fn dequant_dot(out: &mut [f32], x: &[f32], w: &Tensor, n: usize, kk: usize, k: &Kernels) {
    let row_bytes = w.kind.row_bytes(kk) as usize;
    debug_assert!(w.bytes.len() >= n * row_bytes);
    let mut wrow = vec![0.0f32; kk];
    for (i, o) in out.iter_mut().enumerate() {
        k.dequant_row(w.kind, &w.bytes[i * row_bytes..][..row_bytes], &mut wrow, kk);
        *o = k.dot(x, &wrow);
    }
}

/// MoE FFN — exact port of the C++ op chain (llm.cpp QWEN3_MOE branch +
/// nn-cpu-ops.cpp moeGateForward/scale/mergeSum):
/// 1. router gate: x · gate^T over f32 weights (no q80 cast — C++ `block_moe_gate`)
/// 2. softmax over all experts
/// 3. top-k selection with **renormalized weights** (normTopk=1: p/sum(topk))
/// 4. per selected expert: SwiGLU with q80-cast activations (`repeat_z`,
///    silu via the expf_avx2 polynomial, mul, cast_d2) and w2 via q80
/// 5. scale by the expert weight, merge-sum into the output
fn moe_ffn(
    x: &[f32],
    gate: &Tensor,
    experts: &ExpertTensors,
    n_experts: usize,
    n_active: usize,
    q80_buffers: bool,
    k: &Kernels,
) -> Result<Vec<f32>, String> {
    let dim = x.len();
    let ffn = match experts {
        ExpertTensors::Split { w1, s1, .. } => {
            debug_assert_eq!(w1.bytes.len(), n_experts * s1);
            s1 / (w1.kind.row_bytes(dim) as usize)
        }
        ExpertTensors::Lumped { t, s1, stride, .. } => {
            debug_assert_eq!(t.bytes.len(), n_experts * stride);
            s1 / (t.kind.row_bytes(dim) as usize)
        }
    };
    if dim % 32 != 0 || ffn % 32 != 0 {
        return Err(format!("moe dims not 32-aligned: dim={dim} ffn={ffn}").into());
    }
    let topk = n_active.min(n_experts).max(1);

    // 1. router gate (f32 weights, no activation cast)
    let mut gt = vec![0.0f32; n_experts];
    {
        let row_bytes = gate.kind.row_bytes(dim) as usize;
        if gate.bytes.len() < n_experts * row_bytes {
            return Err("moe gate tensor too small".into());
        }
        let mut wrow = vec![0.0f32; dim];
        for (e, o) in gt.iter_mut().enumerate() {
            k.dequant_row(gate.kind, &gate.bytes[e * row_bytes..][..row_bytes], &mut wrow, dim);
            *o = k.dot(x, &wrow);
        }
    }
    // 2. softmax
    let gt_len = gt.len();
    k.softmax(&mut gt, gt_len);
    // 3. top-k + renormalize (moeGateForward, normTopk == 1)
    //    NOTE: ties are broken by index (std::sort is unstable in the C++;
    //    exact f32 ties are astronomically rare — documented deviation)
    let mut order: Vec<usize> = (0..n_experts).collect();
    order.sort_by(|&a, &b| {
        gt[b].partial_cmp(&gt[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    let top: Vec<usize> = order[..topk].to_vec();
    let sum: f32 = top.iter().map(|&p| gt[p]).sum();
    let scales: Vec<f32> = top.iter().map(|&p| gt[p] / sum).collect();

    // 4. per-expert SwiGLU — G6: experts are independent, so they compute in
    // parallel into disjoint buffers; the merge below stays serial in `top`
    // order, which keeps the result bit-exact with the scalar reference.
    let mut xq80 = vec![0u8; q80_bytes(dim)];
    if q80_buffers {
        quantize_row_q80(x, &mut xq80);
    }
    let mut ys = vec![0.0f32; topk * dim];
    dllama_kernel::for_task_slices(&mut ys, topk, dim, |rank, y_e| {
        let e = top[rank];
        let (w1e, w2e, w3e) = experts.expert(e);
        let mut d = vec![0.0f32; ffn];
        let mut l = vec![0.0f32; ffn];
        if q80_buffers {
            k.matmul_q80(&mut d, &xq80, w1e.kind, w1e.bytes, ffn, dim);
            k.matmul_q80(&mut l, &xq80, w3e.kind, w3e.bytes, ffn, dim);
        } else {
            dequant_dot(&mut d, x, &w1e, ffn, dim, k);
            dequant_dot(&mut l, x, &w3e, ffn, dim, k);
        }
        // silu(d) then d*l — exact C++ op split + rounding order
        for i in 0..ffn {
            let g = d[i] / (1.0 + k.expf(-d[i]));
            d[i] = g * l[i];
        }
        if q80_buffers {
            let mut dq80 = vec![0u8; q80_bytes(ffn)];
            quantize_row_q80(&d, &mut dq80);
            k.matmul_q80(y_e, &dq80, w2e.kind, w2e.bytes, dim, ffn);
        } else {
            dequant_dot(y_e, &d, &w2e, dim, ffn, k);
        }
    });
    // 5. scale_F32 then mergeSum (serial, in top order — bit-exact)
    let mut out = vec![0.0f32; dim];
    for rank in 0..topk {
        let s = scales[rank];
        let y_e = &ys[rank * dim..(rank + 1) * dim];
        for i in 0..dim {
            out[i] += s * y_e[i];
        }
    }
    Ok(out)
}

pub mod prefill;
pub mod distributed;

pub use dllama_ir::shard::NodeShard;

/// Public wrapper for the distributed module.
pub fn qk_rmsnorm_pub(x: &mut [f32], w: &[f32], head_dim: usize, eps: f32, n_heads: usize, k: &Kernels) {
    qk_rmsnorm(x, w, head_dim, eps, n_heads, k);
}

/// Public wrapper for the distributed module.
pub fn apply_rope_pub(x: &mut [f32], cache: &[f32], stride: usize, pos: usize, head_dim: usize) {
    apply_rope(x, cache, stride, pos, head_dim);
}

/// C++-style op tracing (format matches DEBUG_OP_INPUT_OUTPUT in nn-cpu-ops.cpp).
fn dbg_vec(name: &str, kind: &str, v: &[f32]) {
    print!("{name:>20}.{kind:>6}: ");
    for x in v.iter().take(32) {
        print!("{x:.6} ");
    }
    println!();
}

/// map our GGUF-ish tensor suffixes to the C++ op names for trace alignment
fn cpp_name(w: &str) -> &str {
    match w.rsplit('.').next().unwrap_or("") {
        "attn_q" => "block_matmul_q",
        "attn_k" => "block_matmul_k",
        "attn_v" => "block_matmul_v",
        "attn_output" => "block_matmul_wo",
        "ffn_gate" => "block_matmul_w1",
        "ffn_up" => "block_matmul_w3",
        "ffn_down" => "block_matmul_w2",
        "output" => "final_matmul_logits",
        "attn_norm" => "block_norm_0",
        "ffn_norm" => "block_norm_1",
        "output_norm" => "final_norm",
        other => other,
    }
}

/// first-occurrence argmax (matches C++ sample_argmax: strict >)
pub fn argmax(x: &[f32]) -> usize {
    let mut max_i = 0usize;
    let mut max_p = x[0];
    for (i, &v) in x.iter().enumerate().skip(1) {
        if v > max_p {
            max_i = i;
            max_p = v;
        }
    }
    max_i
}

fn qk_rmsnorm(x: &mut [f32], w: &[f32], head_dim: usize, eps: f32, n_heads: usize, k: &Kernels) {
    // exact port: per-head invRms_F32 + rmsNorm_F32 association w*(inv*x)
    for h in 0..n_heads {
        let o = h * head_dim;
        let inv = {
            let seg = &x[o..o + head_dim];
            k.inv_rms(seg, eps)
        };
        for i in 0..head_dim {
            x[o + i] = w[i] * (inv * x[o + i]);
        }
    }
}

/// Port of `fullfillRopeLlamaCache` / `fullfillRopeFalconCache` (nn-core.cpp).
fn build_rope_cache(meta: &ModelMeta) -> (Vec<f32>, usize) {
    let seq = meta.seq_len as usize;
    let hd = meta.head_dim_or_derived() as usize;
    match meta.rope_type {
        RopeType::Falcon => {
            let mut cache = vec![0.0f32; seq * hd];
            for pos in 0..seq {
                for j in 0..hd / 2 {
                    let freq = 1.0f32 / unsafe {
                        dllama_kernel::avx2::libm::powf(meta.rope_theta, 2.0 * (j as f32) / hd as f32)
                    };
                    let val = pos as f32 * freq;
                    cache[pos * hd + j] = unsafe { dllama_kernel::avx2::libm::cosf(val) };
                    cache[pos * hd + j + hd / 2] = unsafe { dllama_kernel::avx2::libm::sinf(val) };
                }
            }
            (cache, hd)
        }
        RopeType::Llama | RopeType::Llama31 => {
            // C++ strides the cache by qDim (nNodes=1: qDimEnd - kvDimStart).
            let q_dim = meta.q_dim() as usize;
            let mut cache = vec![0.0f32; seq * q_dim];
            let scaling = if meta.rope_type == RopeType::Llama31 {
                meta.rope_scaling
            } else {
                None
            };
            for pos in 0..seq {
                for h in 0..meta.n_heads as usize {
                    let base = pos * q_dim + h * hd;
                    let mut i = 0usize;
                    while i < hd {
                        let fi = i as f32;
                        let mut freq = 1.0f32
                            / unsafe { dllama_kernel::avx2::libm::powf(meta.rope_theta, fi / hd as f32) };
                        if let Some(s) = scaling.filter(|s: &dllama_ir::RopeScaling| s.factor != 1.0) {
                            freq = scale_frequency_llama3(freq, &s);
                        }
                        let val = pos as f32 * freq;
                        cache[base + i] = unsafe { dllama_kernel::avx2::libm::cosf(val) };
                        cache[base + i + 1] = unsafe { dllama_kernel::avx2::libm::sinf(val) };
                        i += 2;
                    }
                }
            }
            (cache, q_dim)
        }
    }
}

/// Port of `scaleFrequencyLlama3` (nn-core.cpp:311).
fn scale_frequency_llama3(freq: f32, s: &dllama_ir::RopeScaling) -> f32 {
    let wave_len = 2.0 * std::f32::consts::PI / freq;
    let high_freq_wavelen = s.orig_max_seq_len as f32 / s.high_freq_factor;
    if wave_len < high_freq_wavelen {
        return freq;
    }
    let low_freq_wavelen = s.orig_max_seq_len as f32 / s.low_freq_factor;
    if wave_len > low_freq_wavelen {
        return freq / s.factor;
    }
    let smooth = (s.orig_max_seq_len as f32 / wave_len - s.low_freq_factor)
        / (s.high_freq_factor - s.low_freq_factor);
    (1.0 - smooth) * freq / s.factor + smooth * freq
}

/// Apply rope in place. For Falcon the cache stride is head_dim (split-halves
/// rotation, ropeFalcon_F32); for llama types the stride is q_dim with
/// adjacent-pair rotation.
fn apply_rope(x: &mut [f32], cache: &[f32], stride: usize, pos: usize, head_dim: usize) {
    let pos_cache = &cache[pos * stride..pos * stride + stride];
    let n_heads = x.len() / head_dim;
    if stride == head_dim {
        // Falcon: rotate-half
        for h in 0..n_heads {
            let o = h * head_dim;
            for j in 0..head_dim / 2 {
                let cos = pos_cache[j];
                let sin = pos_cache[j + head_dim / 2];
                let q0 = x[o + j];
                let q1 = x[o + j + head_dim / 2];
                x[o + j] = q0 * cos - q1 * sin;
                x[o + j + head_dim / 2] = q0 * sin + q1 * cos;
            }
        }
    } else {
        // Llama / Llama3.1: adjacent pairs
        for h in 0..n_heads {
            let o = h * head_dim;
            let mut i = 0usize;
            while i < head_dim {
                let cos = pos_cache[h * head_dim + i];
                let sin = pos_cache[h * head_dim + i + 1];
                let q0 = x[o + i];
                let q1 = x[o + i + 1];
                x[o + i] = q0 * cos - q1 * sin;
                x[o + i + 1] = q0 * sin + q1 * cos;
                i += 2;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dllama_kernel::avx2;

    /// Synthetic MoE: 3 experts (F32 weights), top-2 routing, identity
    /// projections — verified against a naive reimplementation.
    #[test]
    fn moe_ffn_matches_naive() {
        let dim = 32usize;
        let ffn = 32usize;
        let n_experts = 3usize;
        let k = 2usize;
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 % 7.0) - 3.0).collect();

        // gate rows: r0 = 3*e0, r1 = e1, r2 = e0 -> gt = [-9, -2, -3]
        let mut gate = vec![0.0f32; n_experts * dim];
        gate[0] = 3.0;
        gate[dim + 1] = 1.0;
        gate[2 * dim] = 1.0;
        let gate_bytes: Vec<u8> = gate.iter().flat_map(|v| v.to_le_bytes()).collect();
        let gate_t = Tensor { kind: QuantKind::F32, bytes: gate_bytes.as_slice() };

        // w1 = w3 = identity (per expert); w2_e = c_e * identity
        let mut ident = vec![0.0f32; dim * dim];
        for i in 0..dim {
            ident[i * dim + i] = 1.0;
        }
        let one: Vec<u8> = (0..n_experts)
            .flat_map(|_| ident.iter().flat_map(|v| v.to_le_bytes()))
            .collect();
        let cs = [1.0f32, -2.0, 5.0];
        let w2_bytes: Vec<u8> = (0..n_experts)
            .flat_map(|e| {
                ident
                    .iter()
                    .map(|v| v * cs[e])
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>()
            })
            .collect();
        let w1_t = Tensor { kind: QuantKind::F32, bytes: one.as_slice() };
        let w3_t = Tensor { kind: QuantKind::F32, bytes: one.as_slice() };
        let w2_t = Tensor { kind: QuantKind::F32, bytes: w2_bytes.as_slice() };
        let experts = ExpertTensors::Split {
            s1: ffn * dim * 4,
            s2: dim * ffn * 4,
            s3: ffn * dim * 4,
            w1: w1_t,
            w2: w2_t,
            w3: w3_t,
        };

        let out = moe_ffn(&x, &gate_t, &experts, n_experts, k, false, &Kernels::builtin()).unwrap();
        assert_eq!(out.len(), dim);

        // naive reference (independent code, same math)
        let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(p, q)| p * q).sum() };
        let gt: Vec<f32> = (0..n_experts)
            .map(|e| dot(&x, &gate[e * dim..(e + 1) * dim]))
            .collect();
        let mut sprob = gt.clone();
        avx2::softmax(&mut sprob); // softmax accumulation order was validated vs C++ in G1.5
        let mut order: Vec<usize> = (0..n_experts).collect();
        order.sort_by(|&a, &b| sprob[b].partial_cmp(&sprob[a]).unwrap());
        let top: Vec<usize> = order[..k].to_vec();
        let sum: f32 = top.iter().map(|&p| sprob[p]).sum();
        let scales: Vec<f32> = top.iter().map(|&p| sprob[p] / sum).collect();

        let mut expected = vec![0.0f32; dim];
        for (rank, &e) in top.iter().enumerate() {
            let s = scales[rank];
            for i in 0..dim {
                let h = x[i] / (1.0 + avx2::expf_avx2_scalar(-x[i])) * x[i];
                expected[i] += s * (cs[e] * h);
            }
        }
        // top-2 must select experts 1 and 2 (gt = [-9, -2, -3])
        assert!(top.contains(&1) && top.contains(&2), "top={top:?}");
        for i in 0..dim {
            let scale = expected[i].abs().max(1.0);
            assert!(
                (out[i] - expected[i]).abs() < scale * 1e-4 + 1e-6,
                "i={i} out={} expected={}",
                out[i],
                expected[i]
            );
        }
    }
}
