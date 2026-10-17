//! Bit-exact ports of the C++ AVX2 kernels (`src/nn/nn-cpu-ops.cpp`) — G1.5.
//!
//! Two execution paths, identical results on AVX2+FMA machines:
//! 1. `#[target_feature(enable = "avx2,fma")]` intrinsic ports (fast),
//! 2. scalar emulations that replicate the *lane structure* with
//!    `f32::mul_add` (bit-exact: fma is correctly rounded in both paths).
//!
//! The q80 x q40 matmul is integer-exact per 32-block (order-independent),
//! then one sequential f32 multiply-add per block — both paths bit-exact.

#![allow(clippy::too_many_arguments)]

use dllama_quant::f16_to_f32;
use std::sync::OnceLock;

/// Direct f32 libm calls. Rust's `f32::exp/cos/sin/powf` on windows-msvc are
/// promoted to f64 (msvcrt has no f32 variants), which breaks bit-parity with
/// the C++ engine's f32 `expf/cosf/sinf/powf`. These externs hit the same
/// runtime implementations the C++ oracle links against.
pub mod libm {
    extern "C" {
        pub fn expf(x: f32) -> f32;
        pub fn cosf(x: f32) -> f32;
        pub fn sinf(x: f32) -> f32;
        pub fn powf(x: f32, y: f32) -> f32;
    }
}

/// AVX2+FMA available (cached).
pub fn supported() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

// ---------------------------------------------------------------------------
// scalar bit-exact emulation helpers (match the AVX2 lane structure)
// ---------------------------------------------------------------------------

/// Emulates `horizontalSum_avx2` (nn-cpu-ops.cpp:59):
/// hi = extractf128(1); res = hi + lo; res += movehl(res); res += movehdup(res).
fn horizontal_sum_emulated(lanes: &[f32; 8]) -> f32 {
    let m = [
        lanes[0] + lanes[4],
        lanes[1] + lanes[5],
        lanes[2] + lanes[6],
        lanes[3] + lanes[7],
    ];
    // movehl: [m2, m3, m2, m3] -> add: [m0+m2, m1+m3, ..]
    let a0 = m[0] + m[2];
    let a1 = m[1] + m[3];
    // movehdup: [a1, a1, ..] -> add_ss: a0 + a1
    a0 + a1
}

/// Emulates `expf_avx2` per element (nn-cpu-ops.cpp:88) — degree-4 2^x
/// polynomial with fused multiply-adds. Bit-exact vs the SIMD lanes.
pub fn expf_avx2_scalar(x: f32) -> f32 {
    let x = x.max(-88.0).min(88.0);
    let log2e = 1.4426950408889634f32;
    let c1 = 0.6931471805599453f32;
    let c2 = 0.2402265069591007f32;
    let c3 = 0.05550410866482158f32;
    let c4 = 0.009618129107628477f32;
    let y = x * log2e;
    let n = y.round_ties_even() as i32; // _mm256_cvtps_epi32: round-to-nearest-even
    let f = y - n as f32;
    let mut p = c4;
    p = p.mul_add(f, c3);
    p = p.mul_add(f, c2);
    p = p.mul_add(f, c1);
    p = p.mul_add(f, 1.0);
    let two_n = f32::from_bits(((n + 127) as u32) << 23);
    p * two_n
}

// ---------------------------------------------------------------------------
// invRms (nn-cpu-ops.cpp:114): fmadd lane accumulation + horizontal sum,
// then 1/sqrt(sum/size + eps)
// ---------------------------------------------------------------------------

pub fn inv_rms(x: &[f32], eps: f32) -> f32 {
    debug_assert!(x.len() % 8 == 0);
    let sum = inv_rms_sum(x);
    let mut sum = sum / x.len() as f32;
    sum += eps;
    1.0 / sum.sqrt()
}

fn inv_rms_sum(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if supported() {
            unsafe { inv_rms_impl(x) }
        } else {
            inv_rms_emulated(x)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        inv_rms_emulated(x)
    }
}

/// Scalar emulation of the AVX2 fmadd lane accumulation (bit-exact).
fn inv_rms_emulated(x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    for chunk in x.chunks_exact(8) {
        for l in 0..8 {
            lanes[l] = chunk[l].mul_add(chunk[l], lanes[l]);
        }
    }
    horizontal_sum_emulated(&lanes)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn inv_rms_impl(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut u = _mm256_setzero_ps();
    let mut j = 0usize;
    while j < x.len() {
        let a = _mm256_loadu_ps(x.as_ptr().add(j));
        u = _mm256_fmadd_ps(a, a, u);
        j += 8;
    }
    horizontal_sum_avx2(u)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(x: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let mut res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

// ---------------------------------------------------------------------------
// rmsNorm multiply phase (nn-cpu-ops.cpp:161): result = (x * invRms) * w.
// Multiplication is commutative in IEEE, so the scalar form is bit-exact
// with both the AVX2 vector path and the C++ scalar tail.
// ---------------------------------------------------------------------------

pub fn rms_norm(out: &mut [f32], x: &[f32], w: &[f32], inv_rms: f32) {
    debug_assert_eq!(x.len(), w.len());
    for i in 0..x.len() {
        out[i] = w[i] * (inv_rms * x[i]);
    }
}

// ---------------------------------------------------------------------------
// dotProduct_F32 (nn-cpu-ops.cpp:722): fmadd lanes + horizontal sum,
// scalar tail `sum += a[i]*b[i]` (no fma) for non-multiples of 8.
// ---------------------------------------------------------------------------

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n8 = a.len() - a.len() % 8;
    let mut sum = dot_product_sum(&a[..n8], &b[..n8]);
    for i in n8..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

fn dot_product_sum(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if supported() {
            unsafe { dot_product_impl(a, b) }
        } else {
            dot_product_emulated(a, b)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        dot_product_emulated(a, b)
    }
}

fn dot_product_emulated(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    for (ca, cb) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for l in 0..8 {
            lanes[l] = ca[l].mul_add(cb[l], lanes[l]);
        }
    }
    horizontal_sum_emulated(&lanes)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_impl(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut u = _mm256_set1_ps(0.0);
    let mut i = 0usize;
    while i < a.len() {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        u = _mm256_fmadd_ps(a0, b0, u);
        i += 8;
    }
    horizontal_sum_avx2(u)
}

// ---------------------------------------------------------------------------
// softmax_F32 (nn-cpu-ops.cpp:595) AVX2 path: exp via expf_avx2, lane-wise
// sum + horizontal reduce, normalize by *multiplying* with 1/sum.
// The scalar tail uses the real expf (libm) — see G1.5 notes in DESIGN.md.
// ---------------------------------------------------------------------------

pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let size = x.len();
    let avx_end = size - size % 8;
    // max — exact and order-independent, any implementation matches
    let mut max_val = x[0];
    for i in 1..size {
        if x[i] > max_val {
            max_val = x[i];
        }
    }
    // exp + lane-wise sums
    let mut lanes = [0.0f32; 8];
    for i in (0..avx_end).step_by(8) {
        for l in 0..8 {
            let v = expf_avx2_scalar(x[i + l] - max_val);
            x[i + l] = v;
            lanes[l] += v;
        }
    }
    let mut sum = horizontal_sum_emulated(&lanes);
    for i in avx_end..size {
        // C++ scalar tail: expf() — call the real f32 libm (not Rust's promoted path)
        let v = unsafe { libm::expf(x[i] - max_val) };
        x[i] = v;
        sum += v;
    }
    if sum == 0.0 {
        sum = 0.000001;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// silu_F32 (nn-cpu-ops.cpp:462): x / (1 + expf_avx2(-x)).
// ---------------------------------------------------------------------------

pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + expf_avx2_scalar(-*v));
    }
}

// ---------------------------------------------------------------------------
// matmul_Q80_Q40_F32 (nn-cpu-ops.cpp:231) — q80 activations x q40 weights.
// Per 32-block: exact integer dot product (order-independent), then one
// sequential f32 fma `sum += block_sum * s` across blocks.
// ---------------------------------------------------------------------------

/// out[d] = x_q80 · W^T, x as q80 blocks (34 B/block), W as q40 rows
/// (18 B per 32 input values), W layout [d rows][n/32 blocks].
#[allow(clippy::needless_range_loop)]
pub fn matmul_q80_q40(out: &mut [f32], x: &[u8], w: &[u8], d: usize, n: usize) {
    debug_assert_eq!(n % 32, 0);
    debug_assert!(out.len() >= d);
    let n_blocks = n / 32;
    #[cfg(target_arch = "x86_64")]
    {
        if supported() {
            unsafe { matmul_q80_q40_impl(out, x, w, d, n_blocks) };
            return;
        }
    }
    matmul_q80_q40_scalar(out, x, w, d, n_blocks);
}

/// Bit-exact scalar emulation (also matches the C++ scalar fallback path).
fn matmul_q80_q40_scalar(out: &mut [f32], x: &[u8], w: &[u8], d: usize, n_blocks: usize) {
    for di in 0..d {
        let mut sum = 0.0f32;
        for j in 0..n_blocks {
            let wb = &w[(di * n_blocks + j) * 18..][..18];
            let xb = &x[j * 34..][..34];
            let s = f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]))
                * f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
            let mut acc: i32 = 0;
            for k in 0..16 {
                let w0 = ((wb[2 + k] & 0x0f) as i32) - 8;
                let w1 = ((wb[2 + k] >> 4) as i32) - 8;
                let i1 = (xb[2 + k] as i8) as i32;
                let i2 = (xb[2 + k + 16] as i8) as i32;
                acc += w0 * i1 + w1 * i2;
            }
            // same fma contraction as the C++ (bit-exact with the impl path)
            sum = (acc as f32).mul_add(s, sum);
        }
        out[di] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matmul_q80_q40_impl(out: &mut [f32], x: &[u8], w: &[u8], d: usize, n_blocks: usize) {
    use std::arch::x86_64::*;
    let mask0f = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let ones = _mm256_set1_epi16(1);
    for di in 0..d {
        let mut sum = 0.0f32;
        for j in 0..n_blocks {
            let base = w.as_ptr().add((di * n_blocks + j) * 18);
            let wb = base.add(2);
            let xb = x.as_ptr().add(j * 34);
            let s = f16_to_f32(u16::from_le_bytes([*base, *base.add(1)]))
                * f16_to_f32(u16::from_le_bytes([*xb, *xb.add(1)]));

            let w_packed = _mm_loadu_si128(wb as *const __m128i);
            let w0_low = _mm_and_si128(w_packed, mask0f);
            let w0 = _mm_sub_epi8(w0_low, sub8);
            let w1_high = _mm_srli_epi16(w_packed, 4);
            let w1_high = _mm_and_si128(w1_high, mask0f);
            let w1 = _mm_sub_epi8(w1_high, sub8);

            let w0_16 = _mm256_cvtepi8_epi16(w0);
            let w1_16 = _mm256_cvtepi8_epi16(w1);
            let i1_8 = _mm_loadu_si128(xb.add(2) as *const __m128i);
            let i2_8 = _mm_loadu_si128(xb.add(2 + 16) as *const __m128i);
            let i1_16 = _mm256_cvtepi8_epi16(i1_8);
            let i2_16 = _mm256_cvtepi8_epi16(i2_8);

            let prod0 = _mm256_mullo_epi16(w0_16, i1_16);
            let prod1 = _mm256_mullo_epi16(w1_16, i2_16);
            let sum_prod = _mm256_add_epi16(prod0, prod1);
            let sum32 = _mm256_madd_epi16(sum_prod, ones);

            let mut sum_low = _mm256_castsi256_si128(sum32);
            let sum_high = _mm256_extracti128_si256(sum32, 1);
            sum_low = _mm_add_epi32(sum_low, sum_high);
            sum_low = _mm_hadd_epi32(sum_low, sum_low);
            sum_low = _mm_hadd_epi32(sum_low, sum_low);
            let block_sum = _mm_extract_epi32(sum_low, 0);

            // GCC contracts `sum += block_sum * s` into one fma (-ffp-contract=fast)
            sum = (block_sum as f32).mul_add(s, sum);
        }
        out[di] = sum;
    }
}
