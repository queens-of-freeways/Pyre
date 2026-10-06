//! G3: distributed forward pass — shard-aware execution with TCP sync.
//!
//! This is a separate implementation from the single-node `exec_op` path to
//! keep the G1.5 bit-parity intact. The distributed path walks the same IR
//! graph but applies slice offsets per `NodeShard` and handles `Sync` ops
//! through the TCP stream.

use crate::{Inference, argmax, dbg_vec, dequant_vec, moe_ffn, ExpertTensors};
use dllama_ir::{builder::pipes as P, FloatType, ModelMeta, Op, RopeType};
use dllama_ir::shard::NodeShard;
use dllama_kernel::KvCache;
use dllama_model::{QuantKind, Tensor, Weights};
use dllama_quant::{dequantize_row_q80, q80_bytes, quantize_row_q80};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;

/// Sync context: how Sync ops communicate.
pub enum SyncCtx<'a> {
    /// single node: no sync needed (graph has no Sync ops)
    Single,
    /// worker: one stream to the root
    Worker { stream: &'a mut TcpStream },
    /// root: streams to all workers
    Root { workers: &'a mut [TcpStream] },
}

// stream helpers (duplicated from dllama-cluster to avoid a circular dep)
trait Net: Read + Write {
    fn nu32(&mut self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(u32::from_le_bytes(b))
    }
    fn pu32(&mut self, v: u32) -> Result<(), String> {
        self.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())
    }
    fn nu64(&mut self) -> Result<u64, String> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(u64::from_le_bytes(b))
    }
    fn pu64(&mut self, v: u64) -> Result<(), String> {
        self.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())
    }
    fn nblob(&mut self) -> Result<Vec<u8>, String> {
        let len = self.nu64()? as usize;
        let mut b = vec![0u8; len];
        self.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(b)
    }
    fn pblob(&mut self, v: &[u8]) -> Result<(), String> {
        self.pu64(v.len() as u64)?;
        self.write_all(v).map_err(|e| e.to_string())
    }
}
impl Net for TcpStream {}

impl<'w> Inference<'w> {
    /// Build a distributed executor (n_nodes > 1 graph with Sync ops).
    pub fn new_distributed(
        meta: &ModelMeta,
        weights: &'w Weights<'w>,
        shard: Option<NodeShard>,
        is_root: bool,
    ) -> Result<Inference<'w>, String> {
        let n_nodes = shard.as_ref().map(|s| s.n_nodes).unwrap_or(1);
        let graph = dllama_ir::build_decoder(meta, n_nodes);

        let seq_len = meta.seq_len as usize;
        // KV cache: local kv heads only in sharded mode
        let kv_dim_local = shard
            .as_ref()
            .map(|s| (s.kv_rows.end - s.kv_rows.start) as usize)
            .unwrap_or(meta.kv_dim() as usize);
        let kv = (0..meta.n_layers).map(|_| KvCache::new(seq_len, kv_dim_local)).collect();

        let (rope_cache, rope_stride) = crate::build_rope_cache(meta);

        // pre-extract f32 norm weights
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

        let pipes: Vec<Vec<f32>> = graph.pipes.iter().map(|p| vec![0.0f32; p.len as usize]).collect();
        let max_in = graph.pipes.iter().map(|p| p.len as usize).max().unwrap_or(0);
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
            shard,
            is_root,
            k: dllama_kernel::Kernels::load(),
        })
    }

    /// shard slice helpers: (offset, len) for each pipe
    fn sl(&self, pipe: usize) -> (usize, usize) {
        match self.shard.as_ref() {
            None => (0, self.pipes[pipe].len()),
            Some(s) => match pipe {
                P::Q | P::ATT => (s.q_rows.start as usize, (s.q_rows.end - s.q_rows.start) as usize),
                P::K | P::V => (s.kv_rows.start as usize, (s.kv_rows.end - s.kv_rows.start) as usize),
                P::HB | P::HB2 => (s.ffn_rows.start as usize, (s.ffn_rows.end - s.ffn_rows.start) as usize),
                P::LOGITS => (s.cls_rows.start as usize, (s.cls_rows.end - s.cls_rows.start) as usize),
                P::X | P::XB | P::XB2 => (0, self.pipes[pipe].len()), // global
                _ => (0, self.pipes[pipe].len()),
            },
        }
    }

    /// local head counts
    fn local_heads(&self) -> (usize, usize) {
        match self.shard.as_ref() {
            None => (self.meta.n_heads as usize, self.meta.n_kv_heads as usize),
            Some(s) => (s.n_heads as usize, s.n_kv_heads as usize),
        }
    }

    /// Distributed forward pass with sync through TCP streams.
    /// Returns the logits pipe (full on root, partial on workers).
    pub fn forward_with_stream(
        &mut self,
        token: u32,
        pos: u32,
        sync: &mut SyncCtx<'_>,
    ) -> Result<&[f32], String> {
        let meta = self.meta.clone();
        let dim = meta.dim as usize;
        let hd = meta.head_dim_or_derived() as usize;
        let (n_heads, n_kv_heads) = self.local_heads();
        let (q_off, q_len) = self.sl(P::Q);
        let (kv_off, kv_len) = self.sl(P::K);
        let (ffn_off, ffn_len) = self.sl(P::HB);
        let (cls_off, cls_len) = self.sl(P::LOGITS);

        self.pipes[P::TOKEN][0] = token as f32;
        self.pipes[P::POSITION][0] = pos as f32;

        // ---- 1. embedding (this node's vocab slice) ----
        {
            let embd = self.weights.get("token_embd")?;
            let row_bytes = embd.kind.row_bytes(dim) as usize;
            let er = self.shard.as_ref().map(|s| s.embed_rows).unwrap_or(dllama_ir::shard::Slice { start: 0, end: meta.vocab_size });
            let local_start = er.start as usize;
            let local_end = er.end as usize;
            self.pipes[P::X].fill(0.0);
            if token as usize >= local_start && (token as usize) < local_end {
                let row_idx = token as usize - local_start;
                let row = &embd.bytes[row_idx * row_bytes..(row_idx + 1) * row_bytes];
                let f = dequant_vec(embd.kind, row);
                self.pipes[P::X].copy_from_slice(&f);
            }
        }

        // ---- sync: embedding all-reduce ----
        self.sync_all_reduce(P::X, sync)?;

        // ---- per-layer ----
        for l in 0..meta.n_layers {
            // == attention block ==
            self.norm_layer(l, "attn_norm", P::X, P::XB)?;

            // q/k/v matmuls (OutSharded: local rows, full-dim input)
            self.matmul_shard(&format!("blk.{l}.attn_q"), P::XB, P::Q, q_off, q_len, dim)?;
            self.matmul_shard(&format!("blk.{l}.attn_k"), P::XB, P::K, kv_off, kv_len, dim)?;
            self.matmul_shard(&format!("blk.{l}.attn_v"), P::XB, P::V, kv_off, kv_len, dim)?;

            if meta.qk_norm {
                self.qk_norm_shard(l, P::Q, P::K, q_off, kv_off, q_len, kv_len, hd, n_heads, n_kv_heads, meta.norm_epsilon)?;
            }

            self.rope_shard(P::Q, P::K, q_off, kv_off, q_len, kv_len, hd, pos as usize)?;

            // attention (local heads, local kv cache)
            {
                let q: Vec<f32> = self.pipes[P::Q][q_off..q_off + q_len].to_vec();
                let k: Vec<f32> = self.pipes[P::K][kv_off..kv_off + kv_len].to_vec();
                let v: Vec<f32> = self.pipes[P::V][kv_off..kv_off + kv_len].to_vec();
                let cache = &mut self.kv[l as usize];
                let out = std::mem::take(&mut self.pipes[P::ATT]);
                let mut out = out;
                // write to the local slice of att
                let att_slice = &mut out[q_off..q_off + q_len];
                let kv_dim = cache.kv_dim;
                self.k.attention(
                    &mut cache.k,
                    &mut cache.v,
                    kv_dim,
                    &q,
                    &k,
                    &v,
                    pos as usize,
                    n_heads,
                    n_kv_heads,
                    hd,
                    att_slice,
                );
                self.pipes[P::ATT] = out;
            }

            // wo: InSharded cols — input = local att slice, output = full dim partial
            {
                let x: Vec<f32> = self.pipes[P::ATT][q_off..q_off + q_len].to_vec();
                self.matmul_to_global(&format!("blk.{l}.attn_output"), &x, P::XB2, dim)?;
            }

            // sync: attention output all-reduce
            self.sync_all_reduce(P::XB2, sync)?;

            // residual
            for i in 0..dim {
                self.pipes[P::X][i] += self.pipes[P::XB2][i];
            }

            // == feed-forward block ==
            self.norm_layer(l, "ffn_norm", P::X, P::XB)?;

            if meta.is_moe() {
                // MoE distributed: router gate replicated, expert FFN sharded,
                // output all-reduced (same pattern as dense w2)
                let x: Vec<f32> = self.pipes[P::XB].clone();
                let gate = self.weights.get(&format!("blk.{l}.ffn_gate_inp"))?;
                let w1 = self.weights.get(&format!("blk.{l}.ffn_gate_exps"))?;
                let w3 = self.weights.get(&format!("blk.{l}.ffn_up_exps"))?;
                let w2 = self.weights.get(&format!("blk.{l}.ffn_down_exps"))?;
                let local_ffn = (self.shard.as_ref().unwrap().ffn_rows.end
                    - self.shard.as_ref().unwrap().ffn_rows.start) as usize;
                let experts = ExpertTensors::Split {
                    s1: local_ffn * w1.kind.row_bytes(dim) as usize,
                    s2: dim * w2.kind.row_bytes(local_ffn) as usize,
                    s3: local_ffn * w3.kind.row_bytes(dim) as usize,
                    w1,
                    w2,
                    w3,
                };
                let out = moe_ffn(
                    &x, &gate, &experts,
                    meta.n_experts as usize,
                    meta.n_active_experts as usize,
                    self.meta.sync_type != FloatType::F32,
                    &self.k,
                )?;
                self.pipes[P::XB2].copy_from_slice(&out);
            } else {

            // w1/w3: OutSharded rows
            self.matmul_shard(&format!("blk.{l}.ffn_gate"), P::XB, P::HB, ffn_off, ffn_len, dim)?;
            self.matmul_shard(&format!("blk.{l}.ffn_up"), P::XB, P::HB2, ffn_off, ffn_len, dim)?;

            // silu * up (local slice)
            {
                let g: Vec<f32> = self.pipes[P::HB][ffn_off..ffn_off + ffn_len].to_vec();
                let u: Vec<f32> = self.pipes[P::HB2][ffn_off..ffn_off + ffn_len].to_vec();
                let o = &mut self.pipes[P::HB][ffn_off..ffn_off + ffn_len];
                self.k.activated_mul(o, &g, &u, meta.hidden_act == dllama_ir::HiddenAct::Gelu);
            }

            // w2: InSharded cols — input = local ffn slice, output = full dim partial
            {
                let x: Vec<f32> = self.pipes[P::HB][ffn_off..ffn_off + ffn_len].to_vec();
                self.matmul_to_global(&format!("blk.{l}.ffn_down"), &x, P::XB2, dim)?;
            }

            } // close else (dense path)
            // sync: FFN output all-reduce
            self.sync_all_reduce(P::XB2, sync)?;

            // residual
            for i in 0..dim {
                self.pipes[P::X][i] += self.pipes[P::XB2][i];
            }
        }

        // ---- output head ----
        self.norm_layer_output()?;
        {
            let xb: Vec<f32> = self.pipes[P::XB].clone();
            self.matmul_shard("output", P::XB, P::LOGITS, cls_off, cls_len, dim)?;
        }

        // sync: logits all-gather (root collects, workers send)
        self.sync_all_gather(cls_off, cls_len, sync)?;

        // argmax only on root
        if self.is_root {
            let logits = &self.pipes[P::LOGITS];
            self.pipes[P::OUT][0] = argmax(logits) as f32;
        }

        Ok(&self.pipes[P::LOGITS])
    }

    // ---- helper methods ----

    fn norm_layer(&mut self, l: u32, name: &str, input: usize, output: usize) -> Result<(), String> {
        let wv = self
            .f32_weights
            .get(&format!("blk.{l}.{name}"))
            .ok_or_else(|| format!("missing norm: blk.{l}.{name}"))?
            .clone();
        let x: Vec<f32> = self.pipes[input].clone();
        self.k.rmsnorm(&mut self.pipes[output], &x, &wv, self.meta.norm_epsilon);
        Ok(())
    }

    fn norm_layer_output(&mut self) -> Result<(), String> {
        let wv = self.f32_weights.get("output_norm").cloned().unwrap_or_default();
        let x: Vec<f32> = self.pipes[P::X].clone();
        self.k.rmsnorm(&mut self.pipes[P::XB], &x, &wv, self.meta.norm_epsilon);
        Ok(())
    }

    /// OutSharded matmul: full-dim input → local-row output written at pipe[off..off+len]
    fn matmul_shard(
        &mut self,
        w_name: &str,
        input_pipe: usize,
        output_pipe: usize,
        out_off: usize,
        out_len: usize,
        k: usize,
    ) -> Result<(), String> {
        let x: Vec<f32> = self.pipes[input_pipe].clone();
        let w = self.weights.get(w_name)?;
        let mut out = vec![0.0f32; out_len];
        self.do_matmul(&mut out, &x, &w, out_len, k)?;
        self.pipes[output_pipe][out_off..out_off + out_len].copy_from_slice(&out);
        Ok(())
    }

    /// InSharded matmul: local-slice input → full-dim partial output
    fn matmul_to_global(
        &mut self,
        w_name: &str,
        x: &[f32],
        output_pipe: usize,
        dim: usize,
    ) -> Result<(), String> {
        let w = self.weights.get(w_name)?;
        let mut out_buf = std::mem::take(&mut self.pipes[output_pipe]);
        self.do_matmul(&mut out_buf, x, &w, dim, x.len())?;
        self.pipes[output_pipe] = out_buf;
        Ok(())
    }

    fn do_matmul(&mut self, out: &mut [f32], x: &[f32], w: &Tensor, n: usize, k: usize) -> Result<(), String> {
        if self.meta.sync_type != FloatType::F32 {
            let need = q80_bytes(k);
            if self.q80_buf.len() < need {
                self.q80_buf.resize(need, 0);
            }
            quantize_row_q80(x, &mut self.q80_buf[..need]);
            self.k.matmul_q80(out, &self.q80_buf[..need], w.kind, w.bytes, n, k);
        } else {
            match w.kind {
                QuantKind::F32 => {
                    let wv = dequant_vec(w.kind, w.bytes);
                    self.k.matmul_f32(out, x, &wv, 1, n, k);
                }
                _ => {
                    let row_bytes = w.kind.row_bytes(k) as usize;
                    let mut wrow = vec![0.0f32; k];
                    for (di, o) in out.iter_mut().enumerate() {
                        if (di + 1) * row_bytes > w.bytes.len() {
                            return Err("weight row out of range".into());
                        }
                        self.k.dequant_row(w.kind, &w.bytes[di * row_bytes..(di + 1) * row_bytes], &mut wrow, k);
                        *o = self.k.dot(x, &wrow);
                    }
                }
            }
        }
        Ok(())
    }

    fn qk_norm_shard(
        &mut self,
        _l: u32,
        q_pipe: usize,
        k_pipe: usize,
        q_off: usize,
        kv_off: usize,
        q_len: usize,
        kv_len: usize,
        hd: usize,
        n_heads: usize,
        n_kv_heads: usize,
        eps: f32,
    ) -> Result<(), String> {
        let wq = self
            .f32_weights
            .get(&format!("blk.{_l}.attn_q_norm"))
            .cloned()
            .unwrap_or_default();
        let wk = self
            .f32_weights
            .get(&format!("blk.{_l}.attn_k_norm"))
            .cloned()
            .unwrap_or_default();
        crate::qk_rmsnorm_pub(&mut self.pipes[q_pipe][q_off..q_off + q_len], &wq, hd, eps, n_heads, &self.k);
        crate::qk_rmsnorm_pub(&mut self.pipes[k_pipe][kv_off..kv_off + kv_len], &wk, hd, eps, n_kv_heads, &self.k);
        Ok(())
    }

    fn rope_shard(
        &mut self,
        q_pipe: usize,
        k_pipe: usize,
        q_off: usize,
        kv_off: usize,
        q_len: usize,
        kv_len: usize,
        hd: usize,
        pos: usize,
    ) -> Result<(), String> {
        // rope operates on the local slice — frequencies are per-head-element,
        // head index doesn't affect the rotation
        crate::apply_rope_pub(
            &mut self.pipes[q_pipe][q_off..q_off + q_len],
            &self.rope_cache,
            self.rope_stride,
            pos,
            hd,
        );
        crate::apply_rope_pub(
            &mut self.pipes[k_pipe][kv_off..kv_off + kv_len],
            &self.rope_cache,
            self.rope_stride,
            pos,
            hd,
        );
        Ok(())
    }

    // ---- sync operations ----

    /// AllReduce: every node sends its q80 partial, receives all slices, sums locally.
    fn sync_all_reduce(&mut self, pipe: usize, sync: &mut SyncCtx<'_>) -> Result<(), String> {
        if matches!(sync, SyncCtx::Single) {
            // single-node: apply the q80 round-trip (C++ zq pipe semantics)
            let data = &mut self.pipes[pipe];
            let len = data.len();
            if self.meta.sync_type != FloatType::F32 && len % 32 == 0 {
                let need = q80_bytes(len);
                quantize_row_q80(data, &mut self.q80_buf[..need]);
                dequantize_row_q80(&self.q80_buf[..need], data, len);
            }
            return Ok(());
        }

        let len = self.pipes[pipe].len();
        if self.meta.sync_type != FloatType::F32 {
            if len % 32 != 0 {
                return Err("all-reduce on non-32-aligned pipe".into());
            }
            let need = q80_bytes(len);
            if self.q80_buf.len() < need {
                self.q80_buf.resize(need, 0);
            }
            quantize_row_q80(&self.pipes[pipe], &mut self.q80_buf[..need]);
            let my_q80 = self.q80_buf[..need].to_vec();

            match sync {
                SyncCtx::Single => unreachable!(),
                SyncCtx::Worker { stream } => {
                    // send my partial
                    stream.pblob(&my_q80)?;
                    // receive all slices (including my own) — sum them
                    let n_slices = stream.nu32()?;
                    let mut sum = vec![0.0f32; len];
                    let mut temp = vec![0.0f32; len];
                    for _ in 0..n_slices {
                        let slice = stream.nblob()?;
                        dequantize_row_q80(&slice, &mut temp, len);
                        for i in 0..len {
                            sum[i] += temp[i];
                        }
                    }
                    self.pipes[pipe].copy_from_slice(&sum);
                }
                SyncCtx::Root { workers } => {
                    // receive all worker slices
                    let mut all_slices: Vec<Vec<u8>> = Vec::with_capacity(workers.len() + 1);
                    for w in workers.iter_mut() {
                        let slice = w.nblob()?;
                        all_slices.push(slice);
                    }
                    all_slices.push(my_q80); // root's own

                    // broadcast all slices to each worker
                    for w in workers.iter_mut() {
                        w.pu32(all_slices.len() as u32)?;
                        for s in &all_slices {
                            w.pblob(s)?;
                        }
                    }

                    // sum locally (root) — accumulate, don't overwrite
                    let mut sum = vec![0.0f32; len];
                    let mut temp = vec![0.0f32; len];
                    for s in &all_slices {
                        dequantize_row_q80(s, &mut temp, len);
                        for i in 0..len {
                            sum[i] += temp[i];
                        }
                    }
                    self.pipes[pipe].copy_from_slice(&sum);
                }
            }
        } else {
            // f32 sync: simple sum (less common; .m f32 models)
            match sync {
                SyncCtx::Single => {}
                SyncCtx::Worker { stream } => {
                    let bytes: Vec<u8> = self.pipes[pipe].iter().flat_map(|v| v.to_le_bytes()).collect();
                    stream.pblob(&bytes)?;
                    let result = stream.nblob()?;
                    for (i, b) in result.chunks_exact(4).enumerate() {
                        self.pipes[pipe][i] = f32::from_le_bytes(b.try_into().unwrap());
                    }
                }
                SyncCtx::Root { workers } => {
                    let mut sum = self.pipes[pipe].clone();
                    for w in workers.iter_mut() {
                        let slice = w.nblob()?;
                        for (i, b) in slice.chunks_exact(4).enumerate() {
                            sum[i] += f32::from_le_bytes(b.try_into().unwrap());
                        }
                    }
                    // broadcast
                    let bytes: Vec<u8> = sum.iter().flat_map(|v| v.to_le_bytes()).collect();
                    for w in workers.iter_mut() {
                        w.pblob(&bytes)?;
                    }
                    self.pipes[pipe].copy_from_slice(&sum);
                }
            }
        }
        Ok(())
    }

    /// AllGather: workers send logits slices to root, root writes at offsets.
    fn sync_all_gather(
        &mut self,
        cls_off: usize,
        cls_len: usize,
        sync: &mut SyncCtx<'_>,
    ) -> Result<(), String> {
        match sync {
            SyncCtx::Single => return Ok(()),
            SyncCtx::Worker { stream } => {
                // send my logits slice (f32)
                let bytes: Vec<u8> = self.pipes[P::LOGITS][cls_off..cls_off + cls_len]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                stream.pblob(&bytes)?;
            }
            SyncCtx::Root { workers } => {
                // receive worker slices, write at their offsets
                let shard = self.shard.as_ref().ok_or("root must have shard")?;
                let stride = cls_len; // this node's own slice length
                for (i, w) in workers.iter_mut().enumerate() {
                    let slice = w.nblob()?;
                    let worker_node = i as u32 + 1; // workers are nodes 1..n-1
                    let w_shard = dllama_ir::shard::shard_node(&self.meta, shard.n_nodes, worker_node)
                        .map_err(|e| e.to_string())?;
                    let w_off = w_shard.cls_rows.start as usize;
                    for (j, b) in slice.chunks_exact(4).enumerate() {
                        if w_off + j < self.pipes[P::LOGITS].len() {
                            self.pipes[P::LOGITS][w_off + j] = f32::from_le_bytes(b.try_into().unwrap());
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
