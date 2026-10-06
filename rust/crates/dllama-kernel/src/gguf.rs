//! GGUF quantization formats — dequantization + q80-input matmuls.
//!
//! Layouts verified against ggml reference sources (ggml-common.h +
//! ggml-quants.c, fetched 2026-10-02):
//! - Q4_0 / Q8_0 are byte-identical to dllama q40/q80.
//! - K-quants: 256-value super-blocks; Q4_K uses 8 sub-blocks of 32 with
//!   6-bit-packed scales/mins (x = d*sc*q - dmin*m); Q6_K uses 16 sub-blocks
//!   of 16 with i8 scales (x = d*sc*q).

use crate::avx2;
use dllama_quant::{f16_to_f32, QuantKind};

// ---------------------------------------------------------------------------
// dequantization (reference ports)
// ---------------------------------------------------------------------------

pub fn dequant_row_f32(bytes: &[u8], out: &mut [f32], k: usize) {
    debug_assert_eq!(bytes.len(), k * 4);
    for (i, o) in out.iter_mut().take(k).enumerate() {
        *o = f32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]);
    }
}

pub fn dequant_row_f16(bytes: &[u8], out: &mut [f32], k: usize) {
    debug_assert_eq!(bytes.len(), k * 2);
    for (i, o) in out.iter_mut().take(k).enumerate() {
        *o = f16_to_f32(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]));
    }
}

/// GGUF Q4_0 — same layout as dllama q40.
pub fn dequant_row_q4_0(bytes: &[u8], out: &mut [f32], k: usize) {
    dllama_quant::dequantize_row_q40(bytes, out, k);
}

/// GGUF Q8_0 — same layout as dllama q80.
pub fn dequant_row_q8_0(bytes: &[u8], out: &mut [f32], k: usize) {
    dllama_quant::dequantize_row_q80(bytes, out, k);
}

/// GGUF Q4_K (144 B per 256 values) — port of dequantize_row_q4_K.
pub fn dequant_row_q4_k(bytes: &[u8], out: &mut [f32], k: usize) {
    debug_assert_eq!(k % 256, 0);
    let nb = k / 256;
    for i in 0..nb {
        let b = &bytes[i * 144..][..144];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let scales = &b[4..16];
        let qs = &b[16..];
        let y = &mut out[i * 256..][..256];
        // reference walks y sequentially: 4 iterations x (32 low + 32 high)
        let mut y_off = 0usize;
        let mut q_off = 0usize;
        let mut is = 0usize;
        for _it in 0..4 {
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;
            for l in 0..32 {
                y[y_off + l] = d1 * (qs[q_off + l] & 0xF) as f32 - m1;
            }
            for l in 0..32 {
                y[y_off + 32 + l] = d2 * (qs[q_off + l] >> 4) as f32 - m2;
            }
            y_off += 64;
            q_off += 32;
            is += 2;
        }
    }
}

/// port of get_scale_min_k4 (ggml-quants.c:880) — 6-bit packed scale/min.
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// GGUF Q6_K (210 B per 256 values) — port of dequantize_row_q6_K.
pub fn dequant_row_q6_k(bytes: &[u8], out: &mut [f32], k: usize) {
    debug_assert_eq!(k % 256, 0);
    let nb = k / 256;
    for i in 0..nb {
        let b = &bytes[i * 210..][..210];
        let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
        let ql = &b[0..128];
        let qh = &b[128..192];
        let sc = &b[192..208];
        let y = &mut out[i * 256..][..256];
        let mut y_off = 0usize;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        for _n in 0..2 {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;
                y[y_off + l] = d * sc[sc_off + is] as i8 as f32 * q1 as f32;
                y[y_off + l + 32] = d * sc[sc_off + is + 2] as i8 as f32 * q2 as f32;
                y[y_off + l + 64] = d * sc[sc_off + is + 4] as i8 as f32 * q3 as f32;
                y[y_off + l + 96] = d * sc[sc_off + is + 6] as i8 as f32 * q4 as f32;
            }
            y_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

/// Generic row dequant dispatch.
pub fn dequant_row(kind: QuantKind, bytes: &[u8], out: &mut [f32], k: usize) {
    match kind {
        QuantKind::F32 => dequant_row_f32(bytes, out, k),
        QuantKind::F16 => dequant_row_f16(bytes, out, k),
        QuantKind::DllamaQ40 | QuantKind::GgufQ4_0 => dequant_row_q4_0(bytes, out, k),
        QuantKind::GgufQ8_0 => dequant_row_q8_0(bytes, out, k),
        QuantKind::GgufQ4K => dequant_row_q4_k(bytes, out, k),
        QuantKind::GgufQ6K => dequant_row_q6_k(bytes, out, k),
    }
}

// ---------------------------------------------------------------------------
// matmuls: out[d] = dequant(x_q80) . dequant(W)^T with integer-exact cores
// ---------------------------------------------------------------------------

/// q80 activations x GGUF Q8_0 weights: integer dot per 32-block, one fma
/// per block (mirrors the dllama q40 kernel scheme).
pub fn matmul_q80_q8_0(out: &mut [f32], x: &[u8], w: &[u8], d: usize, n: usize) {
    debug_assert_eq!(n % 32, 0);
    let n_blocks = n / 32;
    for di in 0..d {
        let row = &w[di * n_blocks * 34..][..n_blocks * 34];
        let mut sum = 0.0f32;
        for (j, xb) in x.chunks_exact(34).enumerate() {
            let wb = &row[j * 34..][..34];
            let s = f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]))
                * f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
            let mut acc: i32 = 0;
            for k in 0..32 {
                acc += (wb[2 + k] as i8 as i32) * (xb[2 + k] as i8 as i32);
            }
            sum = (acc as f32).mul_add(s, sum);
        }
        out[di] = sum;
    }
}

/// q80 activations x GGUF Q4_K weights.
/// Per 32-value sub-block: w = ds*nibble - ms, so
/// x.w = dx * (ds * sum(nib*xq) - ms * sum(xq)) — two integer accumulators.
pub fn matmul_q80_q4_k(out: &mut [f32], x: &[u8], w: &[u8], d: usize, n: usize) {
    debug_assert_eq!(n % 256, 0);
    let row_blocks = n / 256;
    for di in 0..d {
        let row = &w[di * row_blocks * 144..][..row_blocks * 144];
        let mut sum = 0.0f32;
        for bi in 0..row_blocks {
            let b = &row[bi * 144..][..144];
            let d_all = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
            let scales = &b[4..16];
            let qs = &b[16..];
            for sb in 0..8 {
                let (sc, m) = scale_min_k4(sb, scales);
                let ds = d_all * sc as f32;
                let ms = dmin * m as f32;
                let xb = &x[(bi * 8 + sb) * 34..][..34];
                let dx = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
                let lo = sb % 2 == 0;
                let qs_off = 32 * (sb / 2);
                let (mut dot, mut sumx) = (0i32, 0i32);
                for k in 0..32 {
                    let nib = if lo { qs[qs_off + k] & 0xF } else { qs[qs_off + k] >> 4 } as i32;
                    let xq = xb[2 + k] as i8 as i32;
                    dot += nib * xq;
                    sumx += xq;
                }
                sum += dx * (ds * dot as f32 - ms * sumx as f32);
            }
        }
        out[di] = sum;
    }
}

/// q80 activations x GGUF Q6_K weights.
/// Per x block (32 values): group = h/4, range r = h%4 (see dequant);
/// two 16-value sub-dots with scales sc[grp*8 + r*2 + is].
pub fn matmul_q80_q6_k(out: &mut [f32], x: &[u8], w: &[u8], d: usize, n: usize) {
    debug_assert_eq!(n % 256, 0);
    let row_blocks = n / 256;
    for di in 0..d {
        let row = &w[di * row_blocks * 210..][..row_blocks * 210];
        let mut sum = 0.0f32;
        for bi in 0..row_blocks {
            let b = &row[bi * 210..][..210];
            let d_all = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
            let ql = &b[0..128];
            let qh = &b[128..192];
            let sc = &b[192..208];
            for h in 0..8 {
                let grp = h / 4;
                let r = h % 4;
                let xb = &x[(bi * 8 + h) * 34..][..34];
                let dx = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
                let (mut dot_a, mut dot_b) = (0i32, 0i32);
                for l in 0..32 {
                    let q = match r {
                        0 => ((ql[grp * 64 + l] & 0xF) | (((qh[grp * 32 + l] >> 0) & 3) << 4)) as i32 - 32,
                        1 => ((ql[grp * 64 + l + 32] & 0xF) | (((qh[grp * 32 + l] >> 2) & 3) << 4)) as i32 - 32,
                        2 => ((ql[grp * 64 + l] >> 4) | (((qh[grp * 32 + l] >> 4) & 3) << 4)) as i32 - 32,
                        _ => ((ql[grp * 64 + l + 32] >> 4) | (((qh[grp * 32 + l] >> 6) & 3) << 4)) as i32 - 32,
                    };
                    let xq = xb[2 + l] as i8 as i32;
                    if l < 16 {
                        dot_a += q * xq;
                    } else {
                        dot_b += q * xq;
                    }
                }
                let sa = d_all * sc[grp * 8 + r * 2] as i8 as f32;
                let sb = d_all * sc[grp * 8 + r * 2 + 1] as i8 as f32;
                sum += dx * (sa * dot_a as f32 + sb * dot_b as f32);
            }
        }
        out[di] = sum;
    }
}

/// Generic dequant + f32 dot fallback (F16/F32 weight tensors), with the
/// engine's q80 activation-cast semantics.
pub fn matmul_q80_dequant(out: &mut [f32], x: &[u8], kind: QuantKind, w: &[u8], d: usize, n: usize) {
    debug_assert_eq!(n % 32, 0);
    let mut xq = vec![0.0f32; n];
    dllama_quant::dequantize_row_q80(x, &mut xq, n);
    let mut wrow = vec![0.0f32; n];
    let row_bytes = kind.row_bytes(n) as usize;
    for di in 0..d {
        dequant_row(kind, &w[di * row_bytes..][..row_bytes], &mut wrow, n);
        out[di] = avx2::dot_product(&xq, &wrow);
    }
}

/// Dispatch a q80-input matmul over all supported weight kinds.
/// G6: output rows are independent, so they are distributed across threads
/// (bit-exact by construction); each kind picks its fastest kernel. Below
/// ~8M MACs (~1.3 ms single-core) the matmul runs serial — thread spawn
/// overhead would dominate (per-matmul work on small models is µs-scale).
const PARALLEL_WORK_MACS: usize = 8_000_000;

pub fn matmul_q80(out: &mut [f32], x: &[u8], kind: QuantKind, w: &[u8], d: usize, n: usize) {
    let row_bytes = kind.row_bytes(n) as usize;
    debug_assert!(w.len() >= d * row_bytes);
    let run_rows = |start: usize, end: usize, out_rows: &mut [f32]| {
        let w_rows = &w[start * row_bytes..end * row_bytes];
        let d_rows = end - start;
        run_matmul_rows(out_rows, x, kind, w_rows, d_rows, n)
    };
    if d.saturating_mul(n) < PARALLEL_WORK_MACS {
        run_rows(0, d, out);
    } else {
        crate::for_rows(out, d, run_rows);
    }
}

fn run_matmul_rows(out_rows: &mut [f32], x: &[u8], kind: QuantKind, w_rows: &[u8], d_rows: usize, n: usize) {
    match kind {
            // Q4_0 shares the dllama q40 layout — reuse the AVX2 kernel
            QuantKind::DllamaQ40 | QuantKind::GgufQ4_0 => avx2::matmul_q80_q40(out_rows, x, w_rows, d_rows, n),
            QuantKind::GgufQ8_0 => {
                // scalar + LLVM auto-vectorization beats the hand AVX2 impl on
                // this pattern (bench: 6.0 vs 4.0 GFLOP/s) — and it IS the
                // bit-exact reference; the AVX2 impl stays for the unit test.
                matmul_q80_q8_0(out_rows, x, w_rows, d_rows, n);
            }
            QuantKind::GgufQ4K => {
                #[cfg(target_arch = "x86_64")]
                {
                    if avx2::supported() {
                        unsafe { matmul_q80_q4_k_impl(out_rows, x, w_rows, d_rows, n) };
                    } else {
                        matmul_q80_q4_k(out_rows, x, w_rows, d_rows, n);
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    matmul_q80_q4_k(out_rows, x, w_rows, d_rows, n);
                }
            }
            QuantKind::GgufQ6K => {
                #[cfg(target_arch = "x86_64")]
                {
                    if avx2::supported() {
                        unsafe { matmul_q80_q6_k_impl(out_rows, x, w_rows, d_rows, n) };
                    } else {
                        matmul_q80_q6_k(out_rows, x, w_rows, d_rows, n);
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    matmul_q80_q6_k(out_rows, x, w_rows, d_rows, n);
                }
            }
            QuantKind::F16 | QuantKind::F32 => matmul_q80_dequant(out_rows, x, kind, w_rows, d_rows, n),
        }
}

// ---------------------------------------------------------------------------
// G6: AVX2 kernels for the K-quant matmuls. The integer block dots are
// order-free (exact under any vectorization); the per-block/per-sub-block
// float update chain keeps the exact scalar order — fma for q8_0 (mirrors
// the q40 scheme), plain mul+add for the K-quant terms — so these are
// bit-identical to the scalar reference (pinned by unit tests).
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_i32x8(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let mut lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    _mm_extract_epi32(lo, 0)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matmul_q80_q8_0_impl(out: &mut [f32], x: &[u8], w: &[u8], d: usize, n: usize) {
    use std::arch::x86_64::*;
    debug_assert_eq!(n % 32, 0);
    let n_blocks = n / 32;
    let ones = _mm256_set1_epi16(1);
    for di in 0..d {
        let row = &w[di * n_blocks * 34..][..n_blocks * 34];
        let mut sum = 0.0f32;
        for j in 0..n_blocks {
            let wb = &row[j * 34..][..34];
            let xb = &x[j * 34..][..34];
            let s = f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]))
                * f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
            let w0 = _mm_loadu_si128(wb.as_ptr().add(2) as *const __m128i);
            let w1 = _mm_loadu_si128(wb.as_ptr().add(18) as *const __m128i);
            let x0 = _mm_loadu_si128(xb.as_ptr().add(2) as *const __m128i);
            let x1 = _mm_loadu_si128(xb.as_ptr().add(18) as *const __m128i);
            // i8 x i8 products fit i16 (127*127 = 16129); pairwise sums 32258 < 32768
            let p0 = _mm256_mullo_epi16(_mm256_cvtepi8_epi16(w0), _mm256_cvtepi8_epi16(x0));
            let p1 = _mm256_mullo_epi16(_mm256_cvtepi8_epi16(w1), _mm256_cvtepi8_epi16(x1));
            let acc = sum_i32x8(_mm256_madd_epi16(_mm256_add_epi16(p0, p1), ones));
            sum = (acc as f32).mul_add(s, sum);
        }
        out[di] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matmul_q80_q4_k_impl(out: &mut [f32], x: &[u8], w: &[u8], d: usize, n: usize) {
    use std::arch::x86_64::*;
    debug_assert_eq!(n % 256, 0);
    let row_blocks = n / 256;
    let ones = _mm256_set1_epi16(1);
    let mask0f = _mm_set1_epi8(0x0F);
    for di in 0..d {
        let row = &w[di * row_blocks * 144..][..row_blocks * 144];
        let mut sum = 0.0f32;
        for bi in 0..row_blocks {
            let b = &row[bi * 144..][..144];
            let d_all = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
            let scales = &b[4..16];
            let qs = &b[16..144];
            for sb in 0..8 {
                let (sc, m) = scale_min_k4(sb, scales);
                let ds = d_all * sc as f32;
                let ms = dmin * m as f32;
                let xb = &x[(bi * 8 + sb) * 34..][..34];
                let dx = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
                let lo = sb % 2 == 0;
                let qs_off = 32 * (sb / 2);
                // 32 nibbles: low or high per byte; per-byte >>4 via srli_epi16 + mask
                let nb0 = _mm_loadu_si128(qs.as_ptr().add(qs_off) as *const __m128i);
                let nb1 = _mm_loadu_si128(qs.as_ptr().add(qs_off + 16) as *const __m128i);
                let (n0, n1) = if lo {
                    (_mm_and_si128(nb0, mask0f), _mm_and_si128(nb1, mask0f))
                } else {
                    (
                        _mm_and_si128(_mm_srli_epi16(nb0, 4), mask0f),
                        _mm_and_si128(_mm_srli_epi16(nb1, 4), mask0f),
                    )
                };
                let xq0 = _mm_loadu_si128(xb.as_ptr().add(2) as *const __m128i);
                let xq1 = _mm_loadu_si128(xb.as_ptr().add(18) as *const __m128i);
                // nibbles are unsigned 0..15 — zero-extend; xq sign-extends
                let n0_16 = _mm256_cvtepu8_epi16(n0);
                let n1_16 = _mm256_cvtepu8_epi16(n1);
                let x0_16 = _mm256_cvtepi8_epi16(xq0);
                let x1_16 = _mm256_cvtepi8_epi16(xq1);
                let p0 = _mm256_mullo_epi16(n0_16, x0_16); // <= 15*127 = 1905
                let p1 = _mm256_mullo_epi16(n1_16, x1_16);
                let dot = sum_i32x8(_mm256_madd_epi16(_mm256_add_epi16(p0, p1), ones));
                let sumx = sum_i32x8(_mm256_madd_epi16(_mm256_add_epi16(x0_16, x1_16), ones));
                sum += dx * (ds * dot as f32 - ms * sumx as f32);
            }
        }
        out[di] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matmul_q80_q6_k_impl(out: &mut [f32], x: &[u8], w: &[u8], d: usize, n: usize) {
    use std::arch::x86_64::*;
    debug_assert_eq!(n % 256, 0);
    let row_blocks = n / 256;
    let ones = _mm256_set1_epi16(1);
    let mask0f = _mm_set1_epi8(0x0F);
    let mask3 = _mm_set1_epi8(3);
    let maskf0 = _mm_set1_epi8(-16); // 0xF0
    let sub32 = _mm256_set1_epi16(32);
    for di in 0..d {
        let row = &w[di * row_blocks * 210..][..row_blocks * 210];
        let mut sum = 0.0f32;
        for bi in 0..row_blocks {
            let b = &row[bi * 210..][..210];
            let d_all = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
            let ql = &b[0..128];
            let qh = &b[128..192];
            let sc = &b[192..208];
            for h in 0..8 {
                let grp = h / 4;
                let r = h % 4;
                let xb = &x[(bi * 8 + h) * 34..][..34];
                let dx = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
                // l in [0,16) -> dot_a; l in [16,32) -> dot_b. The ql selector
                // and the 2-bit qh shift are fixed per r (the scalar match arm).
                let (ql_sel_lo, qh_shift, ql_off) = match r {
                    0 => (true, 0, 0),
                    1 => (true, 2, 32),
                    2 => (false, 4, 0),
                    _ => (false, 6, 32),
                };
                let mut dot = [0i32; 2];
                for (half, l_off) in [(0usize, 0usize), (1usize, 16usize)] {
                    let qlv_raw = _mm_loadu_si128(ql.as_ptr().add(grp * 64 + ql_off + l_off) as *const __m128i);
                    let qhv_raw = _mm_loadu_si128(qh.as_ptr().add(grp * 32 + l_off) as *const __m128i);
                    // low nibble, or high nibble (srli_epi16 + mask trick)
                    let qlv = if ql_sel_lo {
                        _mm_and_si128(qlv_raw, mask0f)
                    } else {
                        _mm_and_si128(_mm_srli_epi16(qlv_raw, 4), mask0f)
                    };
                    // 2 bits of qh at bit position qh_shift (0/2/4/6) — SIMD
                    // shift immediates must be constants, hence the match
                    let qh2 = match qh_shift {
                        0 => _mm_and_si128(qhv_raw, mask3),
                        2 => _mm_and_si128(_mm_srli_epi16(qhv_raw, 2), mask3),
                        4 => _mm_and_si128(_mm_srli_epi16(qhv_raw, 4), mask3),
                        _ => _mm_and_si128(_mm_srli_epi16(qhv_raw, 6), mask3),
                    };
                    // q6 = qlv | (qh2 << 4) — per-byte <<4 needs the 0xF0 mask
                    let q6 = _mm_or_si128(qlv, _mm_and_si128(_mm_slli_epi16(qh2, 4), maskf0));
                    let xqv = _mm_loadu_si128(xb.as_ptr().add(2 + l_off) as *const __m128i);
                    let q16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(q6), sub32); // -32..31
                    let x16 = _mm256_cvtepi8_epi16(xqv);
                    let p = _mm256_mullo_epi16(q16, x16); // <= 32*127 = 4064
                    dot[half] = sum_i32x8(_mm256_madd_epi16(p, ones));
                }
                let sa = d_all * sc[grp * 8 + r * 2] as i8 as f32;
                let sb = d_all * sc[grp * 8 + r * 2 + 1] as i8 as f32;
                sum += dx * (sa * dot[0] as f32 + sb * dot[1] as f32);
            }
        }
        out[di] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// LCG byte stream with sane f16 scale fields, so values stay well-conditioned.
    fn synth_weights(kind: QuantKind, d: usize, n: usize, seed: u64) -> Vec<u8> {
        let row = kind.row_bytes(n) as usize;
        let mut w = vec![0u8; d * row];
        let mut s = seed;
        for di in 0..d {
            for b in w[di * row..(di + 1) * row].iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *b = (s >> 33) as u8;
            }
            let (off, extra) = match kind {
                QuantKind::GgufQ4K => (0, 4),
                QuantKind::GgufQ6K => (208, 2),
                _ => (0, 2),
            };
            let row_bytes = &mut w[di * row..(di + 1) * row];
            // f16 0.01 = 0x211F (LE 1F 21); q4_k also patches dmin at bytes 2..4
            row_bytes[off] = 0x1F;
            row_bytes[off + 1] = 0x21;
            if extra == 4 {
                row_bytes[2] = 0x14; // f16 ~0.0021
                row_bytes[3] = 0x21;
            }
        }
        w
    }

    fn synth_q80(n: usize) -> Vec<u8> {
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.25) - 8.0).collect();
        let mut xq = vec![0u8; dllama_quant::q80_bytes(n)];
        dllama_quant::quantize_row_q80(&x, &mut xq);
        xq
    }

    /// G6: the AVX2 K-quant kernels must be bit-identical to the scalar
    /// reference (integer dots are order-free; the float update chains keep
    /// the scalar order).
    #[test]
    fn avx2_k_quant_matmuls_match_scalar_bits() {
        if !avx2::supported() {
            eprintln!("skip: no AVX2 on this host");
            return;
        }
        let (d, n) = (32usize, 256usize);
        let xq = synth_q80(n);
        for (kind, seed) in [
            (QuantKind::GgufQ8_0, 7u64),
            (QuantKind::GgufQ4K, 11),
            (QuantKind::GgufQ6K, 13),
        ] {
            let w = synth_weights(kind, d, n, seed);
            let mut scalar = vec![0.0f32; d];
            let mut avx = vec![0.0f32; d];
            match kind {
                QuantKind::GgufQ8_0 => {
                    matmul_q80_q8_0(&mut scalar, &xq, &w, d, n);
                    unsafe { matmul_q80_q8_0_impl(&mut avx, &xq, &w, d, n) };
                }
                QuantKind::GgufQ4K => {
                    matmul_q80_q4_k(&mut scalar, &xq, &w, d, n);
                    unsafe { matmul_q80_q4_k_impl(&mut avx, &xq, &w, d, n) };
                }
                _ => {
                    matmul_q80_q6_k(&mut scalar, &xq, &w, d, n);
                    unsafe { matmul_q80_q6_k_impl(&mut avx, &xq, &w, d, n) };
                }
            }
            assert!(bits_eq(&scalar, &avx), "{kind:?}: AVX2 != scalar bits");
        }
    }

    /// G6: row threading must not change a single bit (rows independent).
    #[test]
    fn threaded_matmul_matches_serial() {
        let (d, n) = (256usize, 256usize);
        let xq = synth_q80(n);
        for kind in [QuantKind::DllamaQ40, QuantKind::GgufQ4K] {
            let w = synth_weights(kind, d, n, 99);
            let mut serial = vec![0.0f32; d];
            let mut threaded = vec![0.0f32; d];
            crate::set_threads(1);
            matmul_q80(&mut serial, &xq, kind, &w, d, n);
            crate::set_threads(4);
            matmul_q80(&mut threaded, &xq, kind, &w, d, n);
            crate::set_threads(0); // back to auto for other tests
            assert!(bits_eq(&serial, &threaded), "{kind:?}: threads != serial bits");
        }
    }

    #[test]
    fn q4_k_dequant_known_block() {
        // one super-block: d=2.0, dmin=1.0, sub-block scale/min from scales[12]
        // craft scales so sub-block sb has (sc, m) = (sb+1, sb+2)
        let mut scales = [0u8; 12];
        for j in 0..8 {
            // pack 6-bit sc in low bits of bytes 0..3 & high nibble pattern
            // use the simple path: j<4 -> sc = q[j]&63, m = q[j+4]&63
            // j>=4 -> sc = (q[j+4]&0xF) | ((q[j-4]>>6)<<4), etc.
            let (sc, m) = ((j + 1) as u8, (j + 2) as u8);
            if j < 4 {
                scales[j] = sc;
                scales[j + 4] = m;
            } else {
                scales[j + 4] = (sc & 0xF) | ((m & 0xF) << 4);
                // high bits go into byte j-4 / j >> 6 — sc,m <= 10 fit in low parts
            }
        }
        let mut b = [0u8; 144];
        b[0..2].copy_from_slice(&dllama_quant::f32_to_f16(2.0).to_le_bytes());
        b[2..4].copy_from_slice(&dllama_quant::f32_to_f16(1.0).to_le_bytes());
        b[4..16].copy_from_slice(&scales);
        for j in 0..128 {
            b[16 + j] = 0x85; // low nibble 5, high nibble 8
        }
        let mut y = vec![0.0f32; 256];
        dequant_row_q4_k(&b, &mut y, 256);
        // sub-block sb: value = d*sc*(nib) - dmin*m
        // sb=0: sc=1, m=2 -> low nibble 5: 2*5 - 2 = 8
        for l in 0..32 {
            assert_eq!(y[l], 8.0, "sub0 low l={l}");
        }
        // sb=1: sc=2, m=3 -> high nibble 8: 2*2*8 - 3 = 29
        for l in 0..32 {
            assert_eq!(y[32 + l], 29.0, "sub1 high l={l}");
        }
        // sb=2: sc=3, m=4 -> low nibble 5: 2*3*5 - 4 = 26
        for l in 0..32 {
            assert_eq!(y[64 + l], 26.0, "sub2 low l={l}");
        }
        // sb=4 (j>=4 path): sc=5, m=6 -> low nibble 5: 2*5*5 - 6 = 44
        for l in 0..32 {
            assert_eq!(y[128 + l], 44.0, "sub4 low l={l}");
        }
    }

    #[test]
    fn q6_k_dequant_known_block() {
        let mut b = [0u8; 210];
        b[208..210].copy_from_slice(&dllama_quant::f32_to_f16(1.0).to_le_bytes());
        for i in 0..16 {
            b[192 + i] = 40i8 as u8; // scale = 40
        }
        // ql low nibble = 1, high nibble = 2; qh = 0
        for j in 0..128 {
            b[j] = 0x21;
        }
        let mut y = vec![0.0f32; 256];
        dequant_row_q6_k(&b, &mut y, 256);
        // all q1 = (1) - 32 = -31; y = d*sc*q = 1*40*(-31) = -1240
        assert_eq!(y[0], -1240.0);
        assert_eq!(y[31], -1240.0);
        // q2 uses ql[l+32] low -> 1 as well, sc[is+2]=40
        assert_eq!(y[32], -1240.0);
        // q3 uses ql high nibble = 2 -> (2|0<<4)-32 = -30 -> -1200
        assert_eq!(y[64], -1200.0);
        assert_eq!(y[96], -1200.0);
        assert_eq!(y[128], -1240.0); // second group
    }

    #[test]
    fn q6_k_matmul_matches_dequant_dot() {
        // craft one Q6_K row (256 values) + one q80 input block row
        let mut b = [0u8; 210];
        b[208..210].copy_from_slice(&dllama_quant::f32_to_f16(1.0).to_le_bytes());
        for i in 0..16 {
            b[192 + i] = (30 + i as i8) as u8;
        }
        for (j, v) in b.iter_mut().take(192).enumerate() {
            *v = ((j * 7 + 13) % 256) as u8;
        }
        // q80 input: scale d=1, values 1..32 pattern
        let mut xb = [0u8; 34 * 8];
        for blk in 0..8 {
            xb[blk * 34..blk * 34 + 2]
                .copy_from_slice(&dllama_quant::f32_to_f16(1.0).to_le_bytes());
            for k in 0..32 {
                xb[blk * 34 + 2 + k] = ((blk as i32 * 4 + (k % 32) as i32) % 60 - 30) as i8 as u8;
            }
        }
        let mut out_m = vec![0.0f32; 1];
        matmul_q80_q6_k(&mut out_m, &xb, &b, 1, 256);
        // reference: dequant w then dot with dequantized x (f64 comparison)
        let mut wrow = vec![0.0f32; 256];
        dequant_row_q6_k(&b, &mut wrow, 256);
        let mut xq = vec![0.0f32; 256];
        dllama_quant::dequantize_row_q80(&xb, &mut xq, 256);
        let expected: f64 = wrow.iter().zip(xq.iter()).map(|(a, b)| *a as f64 * *b as f64).sum();
        let abs_sum: f64 = wrow.iter().zip(xq.iter()).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
        let diff = (out_m[0] as f64 - expected).abs();
        assert!(
            diff < abs_sum * 1e-6 + 1e-3,
            "matmul={} dot={expected} absdiff={diff} abs_sum={abs_sum}",
            out_m[0]
        );
    }

    #[test]
    fn q4_k_matmul_matches_dequant_dot() {
        let mut b = [0u8; 144];
        b[0..2].copy_from_slice(&dllama_quant::f32_to_f16(2.0).to_le_bytes());
        b[2..4].copy_from_slice(&dllama_quant::f32_to_f16(1.0).to_le_bytes());
        for (j, v) in b.iter_mut().enumerate().skip(4) {
            *v = ((j * 11 + 3) % 256) as u8;
        }
        let mut xb = [0u8; 34 * 8];
        for blk in 0..8 {
            xb[blk * 34..blk * 34 + 2]
                .copy_from_slice(&dllama_quant::f32_to_f16(1.0).to_le_bytes());
            for k in 0..32 {
                xb[blk * 34 + 2 + k] = ((blk as i32 * 3 + k as i32) % 40 - 20) as i8 as u8;
            }
        }
        let mut out_m = vec![0.0f32; 1];
        matmul_q80_q4_k(&mut out_m, &xb, &b, 1, 256);
        let mut wrow = vec![0.0f32; 256];
        dequant_row_q4_k(&b, &mut wrow, 256);
        let mut xq = vec![0.0f32; 256];
        dllama_quant::dequantize_row_q80(&xb, &mut xq, 256);
        // f64 reference to avoid f32 accumulation-order artifacts in the test
        let expected: f64 = wrow.iter().zip(xq.iter()).map(|(a, b)| *a as f64 * *b as f64).sum();
        let abs_sum: f64 = wrow.iter().zip(xq.iter()).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
        let diff = (out_m[0] as f64 - expected).abs();
        assert!(
            diff < abs_sum * 1e-6 + 1e-3,
            "matmul={} dot={expected} absdiff={diff} abs_sum={abs_sum}",
            out_m[0]
        );
    }

    #[test]
    fn q8_0_matmul_matches_dequant_dot() {
        let n = 64;
        let mut w = vec![0u8; n / 32 * 34];
        for blk in 0..n / 32 {
            w[blk * 34..blk * 34 + 2]
                .copy_from_slice(&dllama_quant::f32_to_f16(0.5).to_le_bytes());
            for k in 0..32 {
                w[blk * 34 + 2 + k] = ((blk as i32 * 3 + k as i32) % 50 - 25) as i8 as u8;
            }
        }
        let mut x = vec![0u8; n / 32 * 34];
        for blk in 0..n / 32 {
            x[blk * 34..blk * 34 + 2]
                .copy_from_slice(&dllama_quant::f32_to_f16(1.0).to_le_bytes());
            for k in 0..32 {
                x[blk * 34 + 2 + k] = ((k as i32) % 30 - 15) as i8 as u8;
            }
        }
        let mut out_m = vec![0.0f32; 1];
        matmul_q80_q8_0(&mut out_m, &x, &w, 1, n);
        let mut wrow = vec![0.0f32; n];
        dequant_row_q8_0(&w, &mut wrow, n);
        let mut xq = vec![0.0f32; n];
        dllama_quant::dequantize_row_q80(&x, &mut xq, n);
        let expected: f32 = wrow.iter().zip(xq.iter()).map(|(a, b)| a * b).sum();
        assert!((out_m[0] - expected).abs() < expected.abs().max(1.0) * 1e-4);
    }
}
