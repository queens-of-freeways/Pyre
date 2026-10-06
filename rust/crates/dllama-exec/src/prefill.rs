//! Prompt prefill (G6.4): batched prompt processing.
//!
//! `Inference::forward` runs one token through all layers — a 100-token
//! prompt costs 100 full sequential passes. `prefill` runs the prompt chunk
//! through a layer-major batched path instead: each weight matrix is read
//! once and dotted against all `m` activation rows (weight-stationary), the
//! KV cache gets the whole block appended, and causal attention is computed
//! per row over `[0 ..= start+i]`.
//!
//! **Bit-exactness**: every per-row float chain matches the sequential
//! reference kernels exactly — integer block dots are order-free, the fma /
//! mul+add updates stay in the same per-(row, block) order, attention reuses
//! the same `dot`/`softmax` primitives with the same widths. Only the KV
//! cache persists from a prefill call; the caller then runs `forward` on the
//! final prompt token as usual (which recomputes the head). `Op::Head` and
//! `Op::Argmax` are skipped entirely.
//!
//! Layout notes: batch matmuls compute **column-major** (`out[di * m + mi]`,
//! one `m`-chunk per output row) so the threaded kernels pass mutable state
//! through arguments (`for_chunk_segments_mut`, same pattern as G6's
//! `for_rows`); the caller transposes into the row-major pipes. MoE runs
//! per-row in v1.

use crate::{Inference, apply_rope, moe_ffn, qk_rmsnorm, ExpertTensors};
use dllama_ir::{builder::pipes as PP, FloatType, HiddenAct, Op};
use dllama_kernel::Kernels;
use dllama_model::QuantKind;
use dllama_quant::{dequantize_row_q80, f16_to_f32, q80_bytes, quantize_row_q80};

// ---------------------------------------------------------------------------
// weight-stationary batched matmuls — column-major out (d chunks of m)
// ---------------------------------------------------------------------------

#[inline]
fn sext(u: u8) -> i32 {
    (u ^ 0x80) as i32 - 128
}

#[inline]
fn f16le(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// q80 x rows (m × q80_bytes(n)) · W(kind)ᵀ → out_col[d × m] (chunk di holds
/// all m results for output row di). Bit-exact with the sequential table
/// kernels per row (same integer dots, same float chain order).
#[allow(clippy::too_many_arguments)]
pub fn matmul_q80_batch(
    out_col: &mut [f32],
    xq80: &[u8],
    m: usize,
    kind: QuantKind,
    w: &[u8],
    d: usize,
    n: usize,
    k: &Kernels,
) {
    debug_assert_eq!(out_col.len(), d * m);
    let x_row = q80_bytes(n);
    debug_assert_eq!(xq80.len(), m * x_row);
    match kind {
        QuantKind::DllamaQ40 | QuantKind::GgufQ4_0 => {
            let n_blocks = n / 32;
            dllama_kernel::for_chunk_segments_mut(out_col, m, |base, seg| {
                for (i, chunk) in seg.chunks_mut(m).enumerate() {
                    let di = base + i;
                    let row = &w[di * n_blocks * 18..][..n_blocks * 18];
                    chunk.fill(0.0);
                    for j in 0..n_blocks {
                        let wb = &row[j * 18..][..18];
                        let sw = f16le(wb, 0);
                        for (mi, o) in chunk.iter_mut().enumerate() {
                            let xb = &xq80[mi * x_row + j * 34..][..34];
                            let s = sw * f16le(xb, 0);
                            let mut acc: i32 = 0;
                            for kk in 0..16 {
                                let w0 = (wb[2 + kk] & 0x0F) as i32 - 8;
                                let w1 = (wb[2 + kk] >> 4) as i32 - 8;
                                acc += w0 * sext(xb[2 + kk]) + w1 * sext(xb[2 + kk + 16]);
                            }
                            *o = (acc as f32).mul_add(s, *o);
                        }
                    }
                }
            });
        }
        QuantKind::GgufQ8_0 => {
            let n_blocks = n / 32;
            dllama_kernel::for_chunk_segments_mut(out_col, m, |base, seg| {
                let mut wq = vec![0i32; 32];
                for (i, chunk) in seg.chunks_mut(m).enumerate() {
                    let di = base + i;
                    let row = &w[di * n_blocks * 34..][..n_blocks * 34];
                    chunk.fill(0.0);
                    for j in 0..n_blocks {
                        let wb = &row[j * 34..][..34];
                        let sw = f16le(wb, 0);
                        for kk in 0..32 {
                            wq[kk] = sext(wb[2 + kk]);
                        }
                        for (mi, o) in chunk.iter_mut().enumerate() {
                            let xb = &xq80[mi * x_row + j * 34..][..34];
                            let s = sw * f16le(xb, 0);
                            let mut acc: i32 = 0;
                            for kk in 0..32 {
                                acc += wq[kk] * sext(xb[2 + kk]);
                            }
                            *o = (acc as f32).mul_add(s, *o);
                        }
                    }
                }
            });
        }
        QuantKind::GgufQ4K => {
            let row_blocks = n / 256;
            dllama_kernel::for_chunk_segments_mut(out_col, m, |base, seg| {
                let mut nibs = vec![0i32; 32];
                for (i, chunk) in seg.chunks_mut(m).enumerate() {
                    let di = base + i;
                    let b_all = &w[di * row_blocks * 144..][..row_blocks * 144];
                    chunk.fill(0.0);
                    for bi in 0..row_blocks {
                        let b = &b_all[bi * 144..][..144];
                        let d_all = f16le(b, 0);
                        let dmin = f16le(b, 2);
                        let scales = &b[4..16];
                        let qs = &b[16..144];
                        for sb in 0..8 {
                            let (sc, mm): (u8, u8) = if sb < 4 {
                                (scales[sb] & 63, scales[sb + 4] & 63)
                            } else {
                                (
                                    (scales[sb + 4] & 0x0F) | ((scales[sb - 4] >> 6) << 4),
                                    (scales[sb + 4] >> 4) | ((scales[sb] >> 6) << 4),
                                )
                            };
                            let ds_v = d_all * sc as f32;
                            let ms_v = dmin * mm as f32;
                            let lo = sb % 2 == 0;
                            let qs_off = 32 * (sb / 2);
                            for kk in 0..32 {
                                nibs[kk] = if lo {
                                    (qs[qs_off + kk] & 0x0F) as i32
                                } else {
                                    (qs[qs_off + kk] >> 4) as i32
                                };
                            }
                            for (mi, o) in chunk.iter_mut().enumerate() {
                                let xb = &xq80[mi * x_row + (bi * 8 + sb) * 34..][..34];
                                let dx = f16le(xb, 0);
                                let mut dot: i32 = 0;
                                let mut sumx: i32 = 0;
                                for kk in 0..32 {
                                    let xq = sext(xb[2 + kk]);
                                    dot += nibs[kk] * xq;
                                    sumx += xq;
                                }
                                *o += dx * (ds_v * dot as f32 - ms_v * sumx as f32);
                            }
                        }
                    }
                }
            });
        }
        QuantKind::GgufQ6K => {
            let row_blocks = n / 256;
            dllama_kernel::for_chunk_segments_mut(out_col, m, |base, seg| {
                let mut qv = vec![0i32; 32];
                for (i, chunk) in seg.chunks_mut(m).enumerate() {
                    let di = base + i;
                    let b_all = &w[di * row_blocks * 210..][..row_blocks * 210];
                    chunk.fill(0.0);
                    for bi in 0..row_blocks {
                        let b = &b_all[bi * 210..][..210];
                        let d_all = f16le(b, 208);
                        let ql = &b[0..128];
                        let qh = &b[128..192];
                        let sc = &b[192..208];
                        for h in 0..8 {
                            let grp = h / 4;
                            let r = h % 4;
                            for l in 0..32 {
                                let base_off = grp * 64 + if r % 2 == 1 { 32 } else { 0 } + l;
                                let shift = 2 * r;
                                let lo4 = if r < 2 { ql[base_off] & 0x0F } else { ql[base_off] >> 4 };
                                let hi = (qh[grp * 32 + l] >> shift) & 3;
                                qv[l] = ((lo4 | (hi << 4)) as i32) - 32;
                            }
                            let sa = d_all * sext(sc[grp * 8 + r * 2]) as f32;
                            let sb_v = d_all * sext(sc[grp * 8 + r * 2 + 1]) as f32;
                            for (mi, o) in chunk.iter_mut().enumerate() {
                                let xb = &xq80[mi * x_row + (bi * 8 + h) * 34..][..34];
                                let dx = f16le(xb, 0);
                                let mut dot_a: i32 = 0;
                                let mut dot_b: i32 = 0;
                                for l in 0..32 {
                                    let xq = sext(xb[2 + l]);
                                    if l < 16 {
                                        dot_a += qv[l] * xq;
                                    } else {
                                        dot_b += qv[l] * xq;
                                    }
                                }
                                *o += dx * (sa * dot_a as f32 + sb_v * dot_b as f32);
                            }
                        }
                    }
                }
            });
        }
        QuantKind::F16 | QuantKind::F32 => {
            // the reference `matmul_q80_dequant` semantics, batched: q80 rows
            // dequantized once, each weight row dequantized once, table `dot`
            let row_bytes = kind.row_bytes(n) as usize;
            let mut xf = vec![0.0f32; m * n];
            for mi in 0..m {
                dequantize_row_q80(&xq80[mi * x_row..][..x_row], &mut xf[mi * n..][..n], n);
            }
            let rows: Vec<&[f32]> = (0..m).map(|mi| &xf[mi * n..][..n]).collect();
            dllama_kernel::for_chunk_segments_mut(out_col, m, |base, seg| {
                let mut wrow = vec![0.0f32; n];
                for (i, chunk) in seg.chunks_mut(m).enumerate() {
                    let di = base + i;
                    let row = &w[di * row_bytes..][..row_bytes];
                    k.dequant_row(kind, row, &mut wrow, n);
                    for (mi, o) in chunk.iter_mut().enumerate() {
                        *o = k.dot(rows[mi], &wrow);
                    }
                }
            });
        }
    }
}

/// f32-activation batched matmul (f32-buffer models): X[m × k] · Wᵀ →
/// out_col[d × m]. Weight rows dequantized once; table `dot` per (row, m).
#[allow(clippy::too_many_arguments)]
pub fn matmul_f32_batch(
    out_col: &mut [f32],
    x: &[f32],
    m: usize,
    kind: QuantKind,
    w: &[u8],
    d: usize,
    n: usize,
    k: &Kernels,
) {
    debug_assert_eq!(out_col.len(), d * m);
    debug_assert_eq!(x.len(), m * n);
    let row_bytes = kind.row_bytes(n) as usize;
    let rows: Vec<&[f32]> = (0..m).map(|mi| &x[mi * n..][..n]).collect();
    dllama_kernel::for_chunk_segments_mut(out_col, m, |base, seg| {
        let mut wrow = vec![0.0f32; n];
        for (i, chunk) in seg.chunks_mut(m).enumerate() {
            let di = base + i;
            let row = &w[di * row_bytes..][..row_bytes];
            k.dequant_row(kind, row, &mut wrow, n);
            for (mi, o) in chunk.iter_mut().enumerate() {
                *o = k.dot(rows[mi], &wrow);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// prefill executor
// ---------------------------------------------------------------------------

impl<'w> Inference<'w> {
    /// Process `tokens` (positions `start .. start+len`) through the batched
    /// prompt path. Appends to the KV caches; everything else is local. The
    /// caller continues with `forward(last_token, start + len)`.
    pub fn prefill(&mut self, tokens: &[i32], start: usize) -> Result<(), String> {
        let m = tokens.len();
        if m == 0 {
            return Ok(());
        }
        if start + m + 1 >= self.meta.seq_len as usize {
            return Err("prefill: context window exhausted".into());
        }
        let meta = self.meta.clone();
        let graph = &self.graph;
        let k = &self.k;
        let max_in = graph.pipes.iter().map(|p| p.len as usize).max().unwrap_or(0);
        // batch pipes (row-major [m × len]) + shared scratch
        let mut batch: Vec<Vec<f32>> = graph
            .pipes
            .iter()
            .map(|p| vec![0.0f32; m * p.len as usize])
            .collect();
        let mut xq80 = vec![0u8; m * q80_bytes(max_in)];
        let mut scratch = vec![0.0f32; max_in];
        let mut col = vec![0.0f32; m * max_in]; // matmul column-major staging
        let debug = self.debug;

        for op in &graph.ops {
            match op {
                Op::Embedding { w, out, .. } => {
                    let t = self.weights.get(w)?;
                    let dim = meta.dim as usize;
                    let rb = t.kind.row_bytes(dim) as usize;
                    for (mi, &tok) in tokens.iter().enumerate() {
                        let row = &t.bytes[tok as usize * rb..][..rb];
                        let o = &mut batch[*out][mi * dim..][..dim];
                        k.dequant_row(t.kind, row, o, dim);
                    }
                }
                Op::MatMul { w, input, output, .. } => {
                    let xw = self.weights.get(w)?;
                    let n_in = graph.pipes[*input].len as usize;
                    let n_out = graph.pipes[*output].len as usize;
                    if meta.sync_type != FloatType::F32 {
                        let need = q80_bytes(n_in);
                        for mi in 0..m {
                            let xr = &batch[*input][mi * n_in..][..n_in];
                            quantize_row_q80(xr, &mut xq80[mi * need..][..need]);
                        }
                        let c = &mut col[..m * n_out];
                        matmul_q80_batch(c, &xq80[..m * need], m, xw.kind, xw.bytes, n_out, n_in, k);
                    } else {
                        let xf: &[f32] = &batch[*input][..m * n_in];
                        let c = &mut col[..m * n_out];
                        matmul_f32_batch(c, xf, m, xw.kind, xw.bytes, n_out, n_in, k);
                    }
                    // transpose column-major staging into the row-major pipe
                    let o = &mut batch[*output];
                    for mi in 0..m {
                        for di in 0..n_out {
                            o[mi * n_out + di] = col[di * m + mi];
                        }
                    }
                }
                Op::Head { .. } | Op::Argmax { .. } => {
                    // logits/argmax are recomputed by the caller's forward on
                    // the final token — skipped in prefill entirely
                }
                Op::RmsNorm { w, input, output, eps } => {
                    let wv = self
                        .f32_weights
                        .get(w)
                        .ok_or_else(|| format!("missing f32 norm weights: {w}"))?;
                    let len = graph.pipes[*input].len as usize;
                    let mut o = std::mem::take(&mut batch[*output]);
                    let x = &batch[*input];
                    for mi in 0..m {
                        k.rmsnorm(&mut o[mi * len..][..len], &x[mi * len..][..len], wv, *eps);
                    }
                    batch[*output] = o;
                }
                Op::QkRmsNorm { qw, kw, eps, q_pipe, k_pipe } => {
                    static EMPTY: &[f32] = &[];
                    let wq = self.f32_weights.get(qw).map(|v| &v[..]).unwrap_or(EMPTY);
                    let wk = self.f32_weights.get(kw).map(|v| &v[..]).unwrap_or(EMPTY);
                    let hd = meta.head_dim_or_derived() as usize;
                    let qlen = graph.pipes[*q_pipe].len as usize;
                    let mut qp = std::mem::take(&mut batch[*q_pipe]);
                    for mi in 0..m {
                        qk_rmsnorm(&mut qp[mi * qlen..][..qlen], wq, hd, *eps, meta.n_heads as usize, k);
                    }
                    batch[*q_pipe] = qp;
                    let klen = graph.pipes[*k_pipe].len as usize;
                    let mut kp = std::mem::take(&mut batch[*k_pipe]);
                    for mi in 0..m {
                        qk_rmsnorm(&mut kp[mi * klen..][..klen], wk, hd, *eps, meta.n_kv_heads as usize, k);
                    }
                    batch[*k_pipe] = kp;
                }
                Op::Rope { q_pipe, k_pipe, .. } => {
                    let hd = meta.head_dim_or_derived() as usize;
                    let qlen = graph.pipes[*q_pipe].len as usize;
                    let mut qp = std::mem::take(&mut batch[*q_pipe]);
                    for mi in 0..m {
                        apply_rope(&mut qp[mi * qlen..][..qlen], &self.rope_cache, self.rope_stride, start + mi, hd);
                    }
                    batch[*q_pipe] = qp;
                    let klen = graph.pipes[*k_pipe].len as usize;
                    let mut kp = std::mem::take(&mut batch[*k_pipe]);
                    for mi in 0..m {
                        apply_rope(&mut kp[mi * klen..][..klen], &self.rope_cache, self.rope_stride, start + mi, hd);
                    }
                    batch[*k_pipe] = kp;
                }
                Op::Attention { layer, .. } => {
                    let cache = &mut self.kv[*layer as usize];
                    let kv_dim = cache.kv_dim;
                    let hd = meta.head_dim_or_derived() as usize;
                    let n_heads = meta.n_heads as usize;
                    let n_kv_heads = meta.n_kv_heads as usize;
                    // append the whole K/V block first (causal rows see
                    // earlier rows of the same batch)
                    for mi in 0..m {
                        let p = start + mi;
                        cache.k[p * kv_dim..(p + 1) * kv_dim]
                            .copy_from_slice(&batch[PP::K][mi * kv_dim..][..kv_dim]);
                        cache.v[p * kv_dim..(p + 1) * kv_dim]
                            .copy_from_slice(&batch[PP::V][mi * kv_dim..][..kv_dim]);
                    }
                    // per-row causal attention over [0 ..= start+mi]
                    let att_len = graph.pipes[PP::ATT].len as usize;
                    let q_per_kv = n_heads / n_kv_heads;
                    let hd_root = (hd as f32).sqrt();
                    let mut scores = vec![0.0f32; start + m];
                    let mut att = std::mem::take(&mut batch[PP::ATT]);
                    for mi in 0..m {
                        let pos_i = start + mi;
                        for h in 0..n_heads {
                            let kv_h = h / q_per_kv;
                            let qh = &batch[PP::Q][mi * att_len + h * hd..][..hd];
                            let srow = &mut scores[..pos_i + 1];
                            for t in 0..=pos_i {
                                let kh = &cache.k[t * kv_dim + kv_h * hd..][..hd];
                                srow[t] = k.dot(qh, kh) / hd_root;
                            }
                            k.softmax(srow, pos_i + 1);
                            let oh = &mut att[mi * att_len + h * hd..][..hd];
                            oh.fill(0.0);
                            for t in 0..=pos_i {
                                let vh = &cache.v[t * kv_dim + kv_h * hd..][..hd];
                                let sv = scores[t];
                                for i in 0..hd {
                                    oh[i] = sv.mul_add(vh[i], oh[i]);
                                }
                            }
                        }
                    }
                    batch[PP::ATT] = att;
                }
                Op::ActivatedMul { act, gate, up, out } => {
                    // out == gate in the builder: per-row snapshot to scratch,
                    // then in-place mul (element-wise — numerically identical)
                    let len = graph.pipes[*out].len as usize;
                    let gelu = *act == HiddenAct::Gelu;
                    let mut o = std::mem::take(&mut batch[*out]);
                    let u = &batch[*up];
                    for mi in 0..m {
                        scratch[..len].copy_from_slice(&o[mi * len..][..len]);
                        let ur = &u[mi * len..][..len];
                        let orow = &mut o[mi * len..][..len];
                        k.activated_mul(orow, &scratch[..len], ur, gelu);
                    }
                    batch[*out] = o;
                }
                Op::ResidualAdd { accumulator, addend, output } => {
                    // C++ parity: the addend passes through the q80 pipe
                    let len = graph.pipes[*addend].len as usize;
                    let mut acc = std::mem::take(&mut batch[*accumulator]);
                    for mi in 0..m {
                        if meta.sync_type != FloatType::F32 {
                            let need = q80_bytes(len);
                            if self.q80_buf.len() < need {
                                self.q80_buf.resize(need, 0);
                            }
                            quantize_row_q80(&batch[*addend][mi * len..][..len], &mut self.q80_buf[..need]);
                            dequantize_row_q80(&self.q80_buf[..need], &mut scratch[..len], len);
                        } else {
                            scratch[..len].copy_from_slice(&batch[*addend][mi * len..][..len]);
                        }
                        let arow = &mut acc[mi * len..][..len];
                        for i in 0..len {
                            arow[i] += scratch[i];
                        }
                    }
                    batch[*accumulator] = acc;
                    if *output != *accumulator {
                        let src = std::mem::take(&mut batch[*accumulator]);
                        batch[*output].copy_from_slice(&src);
                        batch[*accumulator] = src;
                    }
                }
                Op::Softmax { input, output } => {
                    let len = graph.pipes[*input].len as usize;
                    let mut o = std::mem::take(&mut batch[*output]);
                    for mi in 0..m {
                        o[mi * len..][..len].copy_from_slice(&batch[*input][mi * len..][..len]);
                        k.softmax(&mut o[mi * len..][..len], len);
                    }
                    batch[*output] = o;
                }
                Op::Moe { layer, gate_w, w1, w2, w3, n_experts, n_active, input, output } => {
                    // v1: per-row MoE (expert matmuls not batched yet)
                    let dim = meta.dim as usize;
                    let ffn = meta.moe_hidden_dim as usize;
                    let experts = match (self.weights.get(w1), self.weights.get(w2), self.weights.get(w3)) {
                        (Ok(a), Ok(b), Ok(c)) => ExpertTensors::Split {
                            s1: a.kind.row_bytes(dim) as usize * ffn,
                            s2: b.kind.row_bytes(ffn) as usize * dim,
                            s3: c.kind.row_bytes(dim) as usize * ffn,
                            w1: a,
                            w2: b,
                            w3: c,
                        },
                        _ => {
                            let lumped = self.weights.get(&format!("blk.{layer}.moe_exps"))?;
                            let s1 = lumped.kind.row_bytes(dim) as usize * ffn;
                            let s2 = lumped.kind.row_bytes(ffn) as usize * dim;
                            ExpertTensors::Lumped { t: lumped, s1, s2, s3: s1, stride: s1 + s2 + s1 }
                        },
                    };
                    let out_len = graph.pipes[*output].len as usize;
                    let mut o = std::mem::take(&mut batch[*output]);
                    for mi in 0..m {
                        let xr = &batch[*input][mi * dim..][..dim];
                        let row = moe_ffn(
                            xr,
                            &self.weights.get(gate_w)?,
                            &experts,
                            *n_experts as usize,
                            *n_active as usize,
                            meta.sync_type != FloatType::F32,
                            k,
                        )?;
                        o[mi * out_len..][..out_len].copy_from_slice(&row);
                    }
                    batch[*output] = o;
                }
                Op::Sync { .. } => {}
            }
        }
        if debug {
            eprintln!("[prefill] {m} tokens at pos {start} (batched)");
        }
        Ok(())
    }
}

/// Env kill-switch for the prefill path (A/B validation).
pub fn prefill_enabled() -> bool {
    !matches!(
        std::env::var("DLLAMA_PREFILL").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use dllama_quant::quantize_row_q80;

    /// Batch kernels must be bit-identical with the sequential table
    /// kernels (the reference) for every quant kind.
    #[test]
    fn batch_matmuls_match_sequential_bits() {
        let (d, n, m) = (16usize, 256usize, 5usize);
        let k = Kernels::builtin();
        let mut rng: u64 = 0xfeed_beef;
        let mut byte = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) as u8
        };
        let mut xq80_m = vec![0u8; m * q80_bytes(n)];
        for mi in 0..m {
            let xr: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 12.0 + mi as f32).collect();
            quantize_row_q80(&xr, &mut xq80_m[mi * q80_bytes(n)..][..q80_bytes(n)]);
        }
        for kind in [
            QuantKind::DllamaQ40,
            QuantKind::GgufQ8_0,
            QuantKind::GgufQ4K,
            QuantKind::GgufQ6K,
        ] {
            let rb = kind.row_bytes(n) as usize;
            let (bstride, blocks) = match kind {
                QuantKind::DllamaQ40 => (18usize, n / 32),
                QuantKind::GgufQ8_0 => (34usize, n / 32),
                _ => (rb, n / 256),
            };
            let mut w = vec![0u8; d * rb];
            for di in 0..d {
                for b in w[di * rb..(di + 1) * rb].iter_mut() {
                    *b = byte();
                }
                let row = &mut w[di * rb..(di + 1) * rb];
                for blk in 0..blocks {
                    // per-kind f16 scale offsets (q6_k keeps it at 208)
                    let at = blk * bstride + if bstride == 210 { 208 } else { 0 };
                    row[at] = 0x1F;
                    row[at + 1] = 0x21; // f16 0.01
                    if bstride == 144 && kind == QuantKind::GgufQ4K {
                        row[at + 2] = 0x14;
                        row[at + 3] = 0x21; // dmin ~0.002
                    }
                }
            }
            // sequential reference: one call per x row
            let mut seq = vec![0.0f32; m * d];
            for mi in 0..m {
                let xr = &xq80_m[mi * q80_bytes(n)..][..q80_bytes(n)];
                k.matmul_q80(&mut seq[mi * d..][..d], xr, kind, &w, d, n);
            }
            // batch: column-major staging, transposed for comparison
            let mut col = vec![0.0f32; d * m];
            matmul_q80_batch(&mut col, &xq80_m, m, kind, &w, d, n, &k);
            for mi in 0..m {
                for di in 0..d {
                    assert_eq!(
                        seq[mi * d + di].to_bits(),
                        col[di * m + mi].to_bits(),
                        "{kind:?} [{mi}][{di}]"
                    );
                }
            }
            // the GPU weight cache keys by (host_ptr, len) — valid for the
            // engine (mmap'd weights never move), so keep the test blobs
            // alive to avoid allocator address reuse hitting a stale entry.
            std::mem::forget(w);
        }
    }
}
