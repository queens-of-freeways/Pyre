//! Rust CPU reference kernels — the scalar/blocked paths behind the AVX2
//! suite (avx2.rs); the ABI surface (abi.rs) forwards into this module and
//! gguf/avx2 per quant kind.

use dllama_quant::{dequantize_row_q40, q40_bytes};

/// Blocked dot kernel with 4-way unroll (baseline for the G0 benchmark).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc = [0.0f32; 4];
    let mut i = 0;
    while i + 4 <= n {
        acc[0] += a[i] * b[i];
        acc[1] += a[i + 1] * b[i + 1];
        acc[2] += a[i + 2] * b[i + 2];
        acc[3] += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut tail = 0.0f32;
    while i < n {
        tail += a[i] * b[i];
        i += 1;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

pub fn matmul_f32(out: &mut [f32], x: &[f32], w: &[f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(x.len(), m * k);
    debug_assert_eq!(w.len(), n * k);
    debug_assert_eq!(out.len(), m * n);
    for mi in 0..m {
        let xr = &x[mi * k..][..k];
        let or = &mut out[mi * n..][..n];
        for (ni, o) in or.iter_mut().enumerate() {
            *o = dot(xr, &w[ni * k..][..k]);
        }
    }
}

/// Q40 matmul with per-row on-the-fly dequantization (honest bandwidth-bound path).
pub fn matmul_q40(out: &mut [f32], x: &[f32], w: &[u8], m: usize, n: usize, k: usize) {
    let row_bytes = q40_bytes(k);
    debug_assert_eq!(x.len(), m * k);
    debug_assert!(w.len() >= n * row_bytes);
    debug_assert_eq!(out.len(), m * n);
    let mut row = vec![0.0f32; k];
    for mi in 0..m {
        let xr = &x[mi * k..][..k];
        let or = &mut out[mi * n..][..n];
        for (ni, o) in or.iter_mut().enumerate() {
            dequantize_row_q40(&w[ni * row_bytes..][..row_bytes], &mut row, k);
            *o = dot(xr, &row);
        }
    }
}

/// Q40 matmul with weights pre-dequantized to f32 (upper-bound path for the bench).
pub fn matmul_q40_predequant(out: &mut [f32], x: &[f32], w_f32: &[f32], m: usize, n: usize, k: usize) {
    matmul_f32(out, x, w_f32, m, n, k);
}

pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    // exact port of invRms_F32 + rmsNorm_F32 (see avx2.rs for the numerics)
    let inv = crate::avx2::inv_rms(x, eps);
    crate::avx2::rms_norm(out, x, w, inv);
}

pub fn activated_mul(out: &mut [f32], gate: &[f32], up: &[f32], gelu: bool) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(out.len(), gate.len());
    if gelu {
        // NOTE(gelu): tanh approximation pending bit-exact port (nn-cpu-ops.cpp:453)
        for i in 0..gate.len() {
            let v = gate[i];
            let g = 0.5 * v * (1.0 + (2.0f32 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).exp();
            out[i] = g * up[i];
        }
    } else {
        // exact C++ sequence: silu in place (silu_F32), then mul by up (mul_F32)
        for i in 0..gate.len() {
            let g = gate[i] / (1.0 + crate::avx2::expf_avx2_scalar(-gate[i]));
            out[i] = g * up[i];
        }
    }
}

pub fn softmax(x: &mut [f32], n: usize) {
    debug_assert_eq!(x.len() % n, 0);
    for chunk in x.chunks_mut(n) {
        crate::avx2::softmax(chunk);
    }
}

pub fn rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: u32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    theta: f32,
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert_eq!(k.len(), n_kv_heads * head_dim);
    debug_assert!(head_dim % 2 == 0);
    let half = head_dim / 2;
    for h in 0..n_kv_heads {
        for i in 0..half {
            let freq = (pos as f32) / theta.powf((2 * i) as f32 / head_dim as f32);
            let (sin, cos) = freq.sin_cos();
            // k (all kv heads)
            let ki = h * head_dim + 2 * i;
            let (k0, k1) = (k[ki], k[ki + 1]);
            k[ki] = k0 * cos - k1 * sin;
            k[ki + 1] = k0 * sin + k1 * cos;
            // q (map kv head -> group of query heads)
            let q_per_kv = n_heads / n_kv_heads;
            for g in 0..q_per_kv {
                let qi = (h * q_per_kv + g) * head_dim + 2 * i;
                let (q0, q1) = (q[qi], q[qi + 1]);
                q[qi] = q0 * cos - q1 * sin;
                q[qi + 1] = q0 * sin + q1 * cos;
            }
        }
    }
}

/// Fused GQA attention over caller-owned cache slices: appends k/v at
/// `pos`, reads all cached positions, writes `out` [n_heads * head_dim].
#[allow(clippy::too_many_arguments)]
pub fn attention(
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    kv_dim: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    debug_assert!(k_cache.len() >= (pos + 1) * kv_dim);
    debug_assert!(v_cache.len() >= (pos + 1) * kv_dim);
    debug_assert_eq!(k.len(), kv_dim);
    debug_assert_eq!(v.len(), kv_dim);
    k_cache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(k);
    v_cache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(v);

    let q_per_kv = n_heads / n_kv_heads;
    // C++: headDimRoot computed once (multiheadAtt_F32, nn-cpu-ops.cpp:760)
    let head_dim_root = (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; pos + 1];
    for h in 0..n_heads {
        let kv_h = h / q_per_kv;
        let qh = &q[h * head_dim..][..head_dim];
        // scores over all cached positions
        for t in 0..=pos {
            let kh = &k_cache[t * kv_dim + kv_h * head_dim..][..head_dim];
            scores[t] = crate::avx2::dot_product(qh, kh) / head_dim_root;
        }
        crate::avx2::softmax(&mut scores[..pos + 1]);
        // weighted sum of values
        let oh = &mut out[h * head_dim..][..head_dim];
        oh.fill(0.0);
        for t in 0..=pos {
            let vh = &v_cache[t * kv_dim + kv_h * head_dim..][..head_dim];
            let s = scores[t];
            for i in 0..head_dim {
                // C++ `hY[i] += posA * posV[i]` is fma-contracted by GCC
                oh[i] = s.mul_add(vh[i], oh[i]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvCache;
    use dllama_quant::{dequantize_q40_row, quantize_row_q40, dequantize_q80_row};

    fn approx(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "a={a} b={b} tol={tol}");
    }

    #[test]
    fn matmul_matches_naive() {
        let (m, n, k) = (2usize, 3usize, 4usize);
        let x: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1 - 0.5).collect();
        let w: Vec<f32> = (0..n * k).map(|i| i as f32 * 0.05 - 0.3).collect();
        let mut out = vec![0.0f32; m * n];
        matmul_f32(&mut out, &x, &w, m, n, k);
        for mi in 0..m {
            for ni in 0..n {
                let expect: f32 = (0..k)
                    .map(|ki| x[mi * k + ki] * w[ni * k + ki])
                    .sum();
                approx(out[mi * n + ni], expect, 1e-4);
            }
        }
    }

    #[test]
    fn q40_matmul_matches_matmul_over_dequantized_weights() {
        // Kernel-correctness test: q40 on-the-fly matmul must equal an f32
        // matmul over the *dequantized* weights (quant error is covered by the
        // dllama-quant round-trip tests, not here).
        let (m, n, k) = (1usize, 64usize, 128usize);
        let x: Vec<f32> = (0..m * k).map(|i| ((i * 37) % 11) as f32 - 5.0).collect();
        let w_f32: Vec<f32> = (0..n * k).map(|i| (((i * 13) % 17) as f32 - 8.0) / 8.0).collect();
        let row_bytes = q40_bytes(k);
        let mut w_q = vec![0u8; n * row_bytes];
        for ni in 0..n {
            let mut tmp = vec![0u8; row_bytes];
            quantize_row_q40(&w_f32[ni * k..][..k], &mut tmp);
            w_q[ni * row_bytes..][..row_bytes].copy_from_slice(&tmp);
        }
        // dequantize back -> the exact weights the q40 kernel will use
        let w_deq: Vec<f32> = (0..n)
            .flat_map(|ni| dequantize_q40_row(&w_q[ni * row_bytes..], k))
            .collect();
        let mut out_ref = vec![0.0f32; m * n];
        let mut out_q40 = vec![0.0f32; m * n];
        matmul_f32(&mut out_ref, &x, &w_deq, m, n, k);
        matmul_q40(&mut out_q40, &x, &w_q, m, n, k);
        for i in 0..m * n {
            assert!(
                (out_ref[i] - out_q40[i]).abs() < 1e-3,
                "i={i} ref={} q40={}",
                out_ref[i],
                out_q40[i]
            );
        }
    }

    #[test]
    fn rmsnorm_unit_vector() {
        let x = vec![3.0f32; 64];
        let w = vec![1.0f32; 64];
        let mut out = vec![0.0f32; 64];
        rmsnorm(&mut out, &x, &w, 1e-5);
        for &o in &out {
            approx(o, 3.0 / 3.0, 1e-4); // rms(3)=3 -> 3*1/3 = 1... check: x*w/rms = 3/3 = 1
        }
    }

    #[test]
    fn silu_mul_and_softmax() {
        let gate = vec![1.0f32, -1.0, 0.0];
        let up = vec![2.0f32, 2.0, 2.0];
        let mut out = vec![0.0f32; 3];
        activated_mul(&mut out, &gate, &up, false);
        // exact silu uses the C++ expf_avx2 polynomial (G1.5 bit-parity)
        let silu = |v: f32| v / (1.0 + crate::avx2::expf_avx2_scalar(-v));
        for i in 0..3 {
            approx(out[i], silu(gate[i]) * up[i], 1e-6);
        }
        let mut s = vec![1.0f32, 2.0, 3.0];
        crate::avx2::softmax(&mut s);
        approx(s.iter().sum::<f32>(), 1.0, 1e-5);
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn q80_sync_roundtrip() {
        let x: Vec<f32> = (0..64).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let mut bytes = vec![0u8; dllama_quant::q80_bytes(64)];
        dllama_quant::quantize_row_q80(&x, &mut bytes);
        let back = dequantize_q80_row(&bytes, 64);
        for (a, b) in x.iter().zip(back.iter()) {
            approx(*b, *a, 6.0 / 127.0 + 1e-3);
        }
    }

    #[test]
    fn rope_rotates_first_pair() {
        let (n_heads, n_kv, hd) = (2usize, 1usize, 4usize);
        let mut q = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let mut k = vec![1.0f32, 0.0, 0.0, 0.0];
        rope(&mut q, &mut k, 1, n_heads, n_kv, hd, 10000.0);
        // position 1, i=0: freq = 1/theta^0 = 1 -> full rotation: (1,0) -> (cos, sin)
        approx(q[0], 1.0f32.cos(), 1e-4);
        approx(q[1], 1.0f32.sin(), 1e-4);
        approx(k[0], 1.0f32.cos(), 1e-4);
    }

    #[test]
    fn attention_attends_first_position() {
        let (n_heads, n_kv, hd) = (2usize, 1usize, 2usize);
        let mut cache = KvCache::new(4, n_kv * hd);
        let q = vec![1.0f32, 0.0, 1.0, 0.0];
        let k = vec![1.0f32, 0.0];
        let v = vec![5.0f32, -5.0];
        let mut out = vec![0.0f32; n_heads * hd];
        attention(&mut cache.k, &mut cache.v, cache.kv_dim, &q, &k, &v, 0, n_heads, n_kv, hd, &mut out);
        // only position 0: attention output == v (broadcast to each head)
        approx(out[0], 5.0, 1e-4);
        approx(out[1], -5.0, 1e-4);
    }
}
