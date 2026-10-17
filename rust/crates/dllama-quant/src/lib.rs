//! Byte-compatible quantization codecs for Distributed Llama (Rust engine).
//!
//! Layouts mirror `src/nn/nn-quants.hpp` of the C++ reference implementation:
//! - `Q40`: block of 32 f32 -> 2-byte IEEE-f16 scale + 16 bytes packed 4-bit values
//!   (18 bytes per block, 4.5 bits/value)
//! - `Q80`: block of 32 f32 -> 2-byte IEEE-f16 scale + 32 int8 values
//!   (34 bytes per block, 8.5 bits/value)
//!
//! Wire float-type ids (`NnFloatType`): F_32=0, F_16=1, F_Q40=2, F_80=3.

#![allow(dead_code)]

pub const Q40_BLOCK_SIZE: usize = 32;
pub const Q80_BLOCK_SIZE: usize = 32;
/// 2-byte f16 scale + 16 bytes of nibbles.
pub const Q40_BLOCK_BYTES: usize = 18;
/// 2-byte f16 scale + 32 int8 values.
pub const Q80_BLOCK_BYTES: usize = 34;

// ---------------------------------------------------------------------------
// IEEE 754 half-float conversion (ports of nn-quants.cpp)
// ---------------------------------------------------------------------------

/// Exact port of `convertF32ToF16Impl` (round-to-nearest-even style rounding).
pub fn f32_to_f16(x: f32) -> u16 {
    let i = x.to_bits() as i32;
    let s = (i >> 16) & 0x0000_8000;
    let e = (((i >> 23) & 0x0000_00ff) - (127 - 15)) as i32;
    let m = i & 0x007f_ffff;

    if e <= 0 {
        // f32 subnormal or half-underflow: becomes half subnormal or zero.
        if e < -10 {
            return s as u16;
        }
        let mut m = m | 0x0080_0000;
        let t = 14 - e;
        let a = (1 << (t - 1)) - 1;
        let b = (m >> t) & 1;
        m = (m + a + b) >> t;
        return (s | m) as u16;
    }
    if e == 0xff - (127 - 15) {
        // inf / nan
        if m == 0 {
            return (s | 0x7c00) as u16;
        }
        let m16 = m >> 13;
        return (s | 0x7c00 | m16 | (m16 == 0) as i32) as u16;
    }
    let mut m = m + 0x0000_0fff + ((m >> 13) & 1);
    let mut e = e;
    if (m & 0x0080_0000) != 0 {
        // rounding overflowed the mantissa: carry into the exponent
        m = 0;
        e += 1;
    }
    debug_assert!(e <= 30);
    (s | (e << 10) | (m >> 13)) as u16
}

/// f16 -> f32, bit-exact (the C++ engine uses a LUT with identical semantics).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    match exp {
        0 => {
            if mant == 0 {
                f32::from_bits(sign)
            } else {
                // half subnormal: mant * 2^-24, still normal in f32
                let v = (mant as f32) * 5.960_464_5e-8; // 2^-24
                f32::from_bits(sign | v.to_bits())
            }
        }
        31 => f32::from_bits(sign | 0x7f80_0000 | (mant << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13)),
    }
}

// ---------------------------------------------------------------------------
// Q40 (weights) — port of quantizeF32toQ40 / dequantizeQ40toF32
// ---------------------------------------------------------------------------

/// Bytes needed to store `n` values (n must be a multiple of 32).
pub const fn q40_bytes(n: usize) -> usize {
    debug_assert!(n % Q40_BLOCK_SIZE == 0);
    n / Q40_BLOCK_SIZE * Q40_BLOCK_BYTES
}

/// Quantize a row (length multiple of 32) into `q40_bytes(n)` bytes.
///
/// Scale: `d = signed_amax / -8` (signed value with the largest magnitude),
/// `id = 1/d`; each nibble = `(u8)(x*id + 8.5)` clamped to 15.
/// Low nibble holds values 0..15, high nibble values 16..31.
pub fn quantize_row_q40(x: &[f32], out: &mut [u8]) {
    debug_assert_eq!(x.len() % Q40_BLOCK_SIZE, 0);
    assert!(out.len() >= q40_bytes(x.len()));
    let n_blocks = x.len() / Q40_BLOCK_SIZE;
    for i in 0..n_blocks {
        let blk = &x[i * Q40_BLOCK_SIZE..][..Q40_BLOCK_SIZE];
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in blk {
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0f32;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let o = &mut out[i * Q40_BLOCK_BYTES..][..Q40_BLOCK_BYTES];
        o[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..Q40_BLOCK_SIZE / 2 {
            let x0 = blk[j] * id;
            let x1 = blk[j + Q40_BLOCK_SIZE / 2] * id;
            let mut xi0 = (x0 + 8.5f32) as i32 as u8;
            let mut xi1 = (x1 + 8.5f32) as i32 as u8;
            if xi0 > 15 {
                xi0 = 15;
            }
            if xi1 > 15 {
                xi1 = 15;
            }
            o[2 + j] = xi0 | (xi1 << 4);
        }
    }
}

/// Dequantize `n_blocks` Q40 blocks into `n_blocks * 32` f32 values.
pub fn dequantize_row_q40(src: &[u8], out: &mut [f32], n_values: usize) {
    debug_assert_eq!(n_values % Q40_BLOCK_SIZE, 0);
    assert!(src.len() >= q40_bytes(n_values));
    assert!(out.len() >= n_values);
    let n_blocks = n_values / Q40_BLOCK_SIZE;
    for i in 0..n_blocks {
        let b = &src[i * Q40_BLOCK_BYTES..][..Q40_BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        for j in 0..Q40_BLOCK_SIZE / 2 {
            let x0 = ((b[2 + j] & 0x0f) as i32) - 8;
            let x1 = ((b[2 + j] >> 4) as i32) - 8;
            out[i * Q40_BLOCK_SIZE + j] = x0 as f32 * d;
            out[i * Q40_BLOCK_SIZE + j + Q40_BLOCK_SIZE / 2] = x1 as f32 * d;
        }
    }
}

/// Convenience: dequantize a whole row given its value length.
pub fn dequantize_q40_row(src: &[u8], k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; k];
    dequantize_row_q40(src, &mut out, k);
    out
}

// ---------------------------------------------------------------------------
// Q80 (activations / sync buffers) — port of quantizeF32toQ80 / dequantizeQ80toF32
// ---------------------------------------------------------------------------

/// Bytes needed to store `n` values (n must be a multiple of 32).
pub const fn q80_bytes(n: usize) -> usize {
    debug_assert!(n % Q80_BLOCK_SIZE == 0);
    n / Q80_BLOCK_SIZE * Q80_BLOCK_BYTES
}

/// Quantize a row (length multiple of 32) into `q80_bytes(n)` bytes.
///
/// Scale: `d = amax / 127`, `id = 1/d`, value = round-half-away(x * id) in [-127, 127].
///
/// NOTE(open): rounding mode of the scalar C++ path must be verified against
/// `nn-quants.cpp` for byte parity before Phase 3 (see DESIGN.md §11.3).
pub fn quantize_row_q80(x: &[f32], out: &mut [u8]) {
    debug_assert_eq!(x.len() % Q80_BLOCK_SIZE, 0);
    assert!(out.len() >= q80_bytes(x.len()));
    let n_blocks = x.len() / Q80_BLOCK_SIZE;
    for i in 0..n_blocks {
        let blk = &x[i * Q80_BLOCK_SIZE..][..Q80_BLOCK_SIZE];
        let mut amax = 0.0f32;
        for &v in blk {
            if amax < v.abs() {
                amax = v.abs();
            }
        }
        let d = amax / 127.0f32;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let o = &mut out[i * Q80_BLOCK_BYTES..][..Q80_BLOCK_BYTES];
        o[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for (j, &v) in blk.iter().enumerate() {
            // AVX2 path: _mm256_round_ps(_MM_FROUND_TO_NEAREST_INT) — ties to even
            let r = (v * id).round_ties_even();
            o[2 + j] = r.clamp(-127.0, 127.0) as i8 as u8;
        }
    }
}

/// Dequantize `n_values` Q80 values (`n_values % 32 == 0`).
pub fn dequantize_row_q80(src: &[u8], out: &mut [f32], n_values: usize) {
    debug_assert_eq!(n_values % Q80_BLOCK_SIZE, 0);
    assert!(src.len() >= q80_bytes(n_values));
    assert!(out.len() >= n_values);
    let n_blocks = n_values / Q80_BLOCK_SIZE;
    for i in 0..n_blocks {
        let b = &src[i * Q80_BLOCK_BYTES..][..Q80_BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        for j in 0..Q80_BLOCK_SIZE {
            out[i * Q80_BLOCK_SIZE + j] = (b[2 + j] as i8) as f32 * d;
        }
    }
}

pub fn dequantize_q80_row(src: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    dequantize_row_q80(src, &mut out, n);
    out
}

// ---------------------------------------------------------------------------
// Quant kinds — shared by all model-format loaders (dllama .m, GGUF)
// ---------------------------------------------------------------------------

/// Per-tensor quantization format.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuantKind {
    F32,
    F16,
    /// dllama's q40 — byte-identical layout to GGUF Q4_0 (18 B per 32 values)
    DllamaQ40,
    /// GGUF Q4_0 — same layout as dllama q40
    GgufQ4_0,
    /// GGUF Q8_0 — byte-identical layout to dllama q80 (34 B per 32 values)
    GgufQ8_0,
    /// GGUF K-quant: 256-value super-blocks (144 B)
    GgufQ4K,
    /// GGUF K-quant: 256-value super-blocks (210 B)
    GgufQ6K,
}

impl QuantKind {
    /// values per block (row length must be a multiple of this)
    pub const fn block_elems(self) -> usize {
        match self {
            QuantKind::F32 | QuantKind::F16 | QuantKind::DllamaQ40 | QuantKind::GgufQ4_0
            | QuantKind::GgufQ8_0 => 32,
            QuantKind::GgufQ4K | QuantKind::GgufQ6K => 256,
        }
    }
    pub const fn block_bytes(self) -> usize {
        match self {
            QuantKind::F32 => 128,
            QuantKind::F16 => 64,
            QuantKind::DllamaQ40 | QuantKind::GgufQ4_0 => 18,
            QuantKind::GgufQ8_0 => 34,
            QuantKind::GgufQ4K => 144,
            QuantKind::GgufQ6K => 210,
        }
    }
    /// bytes for one row of `k` values (k must be block-aligned)
    pub const fn row_bytes(self, k: usize) -> u64 {
        k as u64 / self.block_elems() as u64 * self.block_bytes() as u64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_known_values() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF); // half max
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        for &v in &[
            0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 3.14159, 1e-4, 1e4, 123.456, 6.1e-5,
        ] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            // f16 has ~11 bits of mantissa: relative tolerance 1/2048
            if v != 0.0 {
                assert!(
                    (back - v).abs() / v.abs() < 1.0 / 1024.0,
                    "v={v} h={h:04x} back={back}"
                );
            }
        }
    }

    #[test]
    fn f16_subnormal_and_nan() {
        // smallest half subnormal = 2^-24
        assert_eq!(f16_to_f32(0x0001), 5.9604645e-8);
        assert!(f16_to_f32(0x7E00).is_nan());
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
    }

    #[test]
    fn q40_roundtrip_within_block_error_bound() {
        let n = 32 * 7;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.37) % 7.0) - 3.5 + (i as f32) * 0.001)
            .collect();
        let mut bytes = vec![0u8; q40_bytes(n)];
        quantize_row_q40(&x, &mut bytes);
        assert_eq!(bytes.len(), n / 32 * 18);
        let mut y = vec![0.0f32; n];
        dequantize_row_q40(&bytes, &mut y, n);
        // The C++ scale rule `d = signed_max / -8` makes the block range
        // ASYMMETRIC: values sharing the sign of `max` reach ±amax exactly
        // (step amax/8 -> error <= amax/16), values of the OPPOSITE sign
        // clamp at 7/8*amax (error <= amax/8). Add f16 scale rounding (~amax/512).
        for (i, (&a, &b)) in x.iter().zip(y.iter()).enumerate() {
            let blk = &x[(i / 32) * 32..][..32];
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                (a - b).abs() <= amax / 8.0 + amax / 16.0 + amax / 512.0 + 1e-6,
                "i={i} a={a} b={b} amax={amax}"
            );
        }
    }

    #[test]
    fn q40_nibble_layout_matches_cpp() {
        // all values within [-8, 7] under scale d = 1.0:
        // the signed max is x[0] = -8 (first strict max) -> d = -8/-8 = 1.0
        let mut x = [0.0f32; 32];
        for (i, v) in x.iter_mut().enumerate() {
            *v = if i < 16 { i as f32 - 8.0 } else { i as f32 - 24.0 };
        }
        let mut b = [0u8; 18];
        quantize_row_q40(&x, &mut b);
        assert_eq!(b[0..2], 0x3C00u16.to_le_bytes()); // f16(1.0)
        // x[0] = -8 -> low nibble of qs[0] is 0; x[16] = -8 -> high nibble is 0
        assert_eq!(b[2] & 0x0f, 0);
        assert_eq!(b[2] >> 4, 0);
        let mut y = [0.0f32; 32];
        dequantize_row_q40(&b, &mut y, 32);
        // integers in range reconstruct exactly under d = 1.0
        for i in 0..32 {
            assert!((y[i] - x[i]).abs() < 1e-6, "i={i} x={} y={}", x[i], y[i]);
        }
        assert!((y[0] - -8.0).abs() < 1e-6);
        assert!((y[5] - -3.0).abs() < 1e-6);
        assert!((y[17] - -7.0).abs() < 1e-6);
    }

    #[test]
    fn q80_roundtrip_error_bound() {
        let n = 32 * 4;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32 * 1.7) % 11.0) - 5.5).collect();
        let mut bytes = vec![0u8; q80_bytes(n)];
        quantize_row_q80(&x, &mut bytes);
        assert_eq!(bytes.len(), n / 32 * 34);
        let mut y = vec![0.0f32; n];
        dequantize_row_q80(&bytes, &mut y, n);
        for (i, (&a, &b)) in x.iter().zip(y.iter()).enumerate() {
            let blk = &x[(i / 32) * 32..][..32];
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                (a - b).abs() <= amax / 127.0 + 1e-6,
                "i={i} a={a} b={b}"
            );
        }
    }

    #[test]
    fn zero_blocks_dequantize_to_zero() {
        // NOTE: matches C++ exactly — a zero block does NOT produce zero bytes:
        // d = 0/-8 = -0.0 -> f16 0x8000, id = 0 -> every nibble = trunc(8.5) = 8.
        // The dequantized values are still exactly zero.
        let x = vec![0.0f32; 64];
        let mut b40 = vec![0u8; q40_bytes(64)];
        quantize_row_q40(&x, &mut b40);
        assert_eq!(u16::from_le_bytes([b40[0], b40[1]]), 0x8000); // -0.0
        // both blocks: scale = -0.0, all nibble bytes = 8|8 = 0x88
        for block in 0..2 {
            let base = block * 18;
            assert_eq!(u16::from_le_bytes([b40[base], b40[base + 1]]), 0x8000);
            assert!(b40[base + 2..base + 18].iter().all(|&b| b == 0x88));
        }
        let y = dequantize_q40_row(&b40, 64);
        assert!(y.iter().all(|&v| v == 0.0));

        let mut b80 = vec![0u8; q80_bytes(64)];
        quantize_row_q80(&x, &mut b80);
        assert!(b80.iter().all(|&b| b == 0));
        let z = dequantize_q80_row(&b80, 64);
        assert!(z.iter().all(|&v| v == 0.0));
    }
}
