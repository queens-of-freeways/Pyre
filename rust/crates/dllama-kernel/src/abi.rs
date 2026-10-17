//! Kernel ABI v1 — the stable C contract between the engine and a kernel
//! backend (G5).
//!
//! Mojo has no Windows toolchain at the time of this gate, so the reference
//! backend is the Rust AVX2 kernel suite (G1.5 bit-exact ports). The same
//! functions are exposed twice:
//!
//! - **built-in**: the engine links `dllama-kernel` and fills the `Kernels`
//!   table with the wrappers below;
//! - **cdylib**: the `dllama-kernel-c` crate compiles them into
//!   `dllama_kernel_c.dll`, resolved at runtime through `libloading` when
//!   `DLLAMA_KERNEL_LIB` is set — the reference "external backend" that
//!   proves the dynamic path end-to-end.
//!
//! A future Mojo (or GPU) backend exports the same symbols from its own
//! library — no engine change. `dllama_kernels_abi_version` gates the table;
//! v2 (batch matmuls for prompt prefill) will bump it.
//!
//! # Contract (v1)
//! - `usize` is C `size_t`; pointers must be valid for the stated lengths
//!   and 4-byte aligned where f32.
//! - Quant kinds travel as `u32` (`kind_to_wire`); an unknown id is a caller
//!   bug (the engine validates kinds before dispatch).
//! - Quantized-input matmuls are single-token (`m = 1`): the engine feeds
//!   one position per forward step (batch variants are ABI v2).
//! - Matmul `d`/`n` follow the dllama convention: `d` = output rows,
//!   `n` = input dim — `out[d] = x · Wᵀ`, W row-major `[d][n]`.
//! - `dllama_attention` requires the k/v cache slices valid for at least
//!   `(pos + 1) * kv_dim` elements.
//! - The engine falls back to the built-in table when a plugin fails to
//!   load or version-check: a bad backend can never brick the binary.

use crate::{avx2, cpu, gguf};
use dllama_quant::{q40_bytes, q80_bytes, QuantKind};
use std::ffi::c_char;

/// ABI version implemented by this crate (and required from plugins).
pub const ABI_VERSION: u32 = 1;

/// Built-in backend name (must be NUL-terminated for the C string return).
pub static BUILTIN_NAME_C: &[u8] = b"rust-avx2 (built-in)\0";

// ---------------------------------------------------------------------------
// QuantKind wire encoding (stable across ABI versions)
// ---------------------------------------------------------------------------

pub fn kind_to_wire(kind: QuantKind) -> u32 {
    match kind {
        QuantKind::F32 => 0,
        QuantKind::F16 => 1,
        QuantKind::DllamaQ40 => 2,
        QuantKind::GgufQ4_0 => 3,
        QuantKind::GgufQ8_0 => 4,
        QuantKind::GgufQ4K => 5,
        QuantKind::GgufQ6K => 6,
    }
}

pub fn kind_from_wire(v: u32) -> Option<QuantKind> {
    match v {
        0 => Some(QuantKind::F32),
        1 => Some(QuantKind::F16),
        2 => Some(QuantKind::DllamaQ40),
        3 => Some(QuantKind::GgufQ4_0),
        4 => Some(QuantKind::GgufQ8_0),
        5 => Some(QuantKind::GgufQ4K),
        6 => Some(QuantKind::GgufQ6K),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Function-pointer types (the table layout)
// ---------------------------------------------------------------------------

pub type AbiVersionFn = unsafe extern "C" fn() -> u32;
pub type NameFn = unsafe extern "C" fn() -> *const c_char;
pub type MatmulF32Fn = unsafe extern "C" fn(*mut f32, *const f32, *const f32, usize, usize, usize);
pub type MatmulQ40Fn = unsafe extern "C" fn(*mut f32, *const f32, *const u8, usize, usize, usize);
pub type MatmulQ80Fn = unsafe extern "C" fn(*mut f32, *const u8, *const u8, u32, usize, usize);
pub type DequantRowFn = unsafe extern "C" fn(*mut f32, *const u8, u32, usize);
pub type RmsnormFn = unsafe extern "C" fn(*mut f32, *const f32, *const f32, usize, f32);
pub type InvRmsFn = unsafe extern "C" fn(*const f32, usize, f32) -> f32;
pub type SoftmaxFn = unsafe extern "C" fn(*mut f32, usize);
pub type ActivatedMulFn = unsafe extern "C" fn(*mut f32, *const f32, *const f32, usize, u32);
pub type DotFn = unsafe extern "C" fn(*const f32, *const f32, usize) -> f32;
pub type ExpfFn = unsafe extern "C" fn(f32) -> f32;
pub type AttentionFn = unsafe extern "C" fn(
    *mut f32,   // k cache
    *mut f32,   // v cache
    usize,      // kv_dim
    *const f32, // q [n_heads * head_dim]
    *const f32, // k [kv_dim]
    *const f32, // v [kv_dim]
    usize,      // pos
    usize,      // n_heads
    usize,      // n_kv_heads
    usize,      // head_dim
    *mut f32,   // out [n_heads * head_dim]
);

// ---------------------------------------------------------------------------
// Reference implementations (Rust AVX2 backend)
//
// These are the built-in table entries AND the semantics every other backend
// must reproduce bit-for-bit (see DESIGN.md §10.2/§10.3 for the parity
// methodology that pinned them down).
// ---------------------------------------------------------------------------

/// # Safety
/// Standard ABI contract (see module docs).
pub unsafe extern "C" fn dllama_kernels_abi_version() -> u32 {
    ABI_VERSION
}

/// # Safety
/// Returns a pointer to a static NUL-terminated UTF-8 string.
pub unsafe extern "C" fn dllama_kernels_name() -> *const c_char {
    BUILTIN_NAME_C.as_ptr() as *const c_char
}

/// # Safety
/// `out` valid for `m * n`; `x` valid for `m * k`; `w` valid for `n * k`.
pub unsafe extern "C" fn dllama_matmul_f32(
    out: *mut f32,
    x: *const f32,
    w: *const f32,
    m: usize,
    n: usize,
    k: usize,
) {
    cpu::matmul_f32(
        std::slice::from_raw_parts_mut(out, m * n),
        std::slice::from_raw_parts(x, m * k),
        std::slice::from_raw_parts(w, n * k),
        m,
        n,
        k,
    );
}

/// # Safety
/// `out` valid for `m * n`; `x` valid for `m * k`; `w` valid for `n * q40_bytes(k)`.
pub unsafe extern "C" fn dllama_matmul_q40_f32(
    out: *mut f32,
    x: *const f32,
    w: *const u8,
    m: usize,
    n: usize,
    k: usize,
) {
    cpu::matmul_q40(
        std::slice::from_raw_parts_mut(out, m * n),
        std::slice::from_raw_parts(x, m * k),
        std::slice::from_raw_parts(w, n * q40_bytes(k)),
        m,
        n,
        k,
    );
}

/// # Safety
/// `out` valid for `d`; `xq80` valid for `q80_bytes(n)`; `w` valid for
/// `kind.row_bytes(n) * d`.
pub unsafe extern "C" fn dllama_matmul_q80(
    out: *mut f32,
    xq80: *const u8,
    w: *const u8,
    kind: u32,
    d: usize,
    n: usize,
) {
    let Some(kind) = kind_from_wire(kind) else {
        panic!("kernel abi: unknown quant kind wire id {kind}");
    };
    let row_bytes = kind.row_bytes(n) as usize;
    gguf::matmul_q80(
        std::slice::from_raw_parts_mut(out, d),
        std::slice::from_raw_parts(xq80, q80_bytes(n)),
        kind,
        std::slice::from_raw_parts(w, d * row_bytes),
        d,
        n,
    );
}

/// # Safety
/// `out` valid for `k` f32; `bytes` valid for `kind.block_bytes() * (k / block_elems)`.
pub unsafe extern "C" fn dllama_dequant_row(out: *mut f32, bytes: *const u8, kind: u32, k: usize) {
    let Some(kind) = kind_from_wire(kind) else {
        panic!("kernel abi: unknown quant kind wire id {kind}");
    };
    gguf::dequant_row(
        kind,
        std::slice::from_raw_parts(bytes, kind.row_bytes(k) as usize),
        std::slice::from_raw_parts_mut(out, k),
        k,
    );
}

/// # Safety
/// `out`, `x`, `w` valid for `dim` f32.
pub unsafe extern "C" fn dllama_rmsnorm(
    out: *mut f32,
    x: *const f32,
    w: *const f32,
    dim: usize,
    eps: f32,
) {
    cpu::rmsnorm(
        std::slice::from_raw_parts_mut(out, dim),
        std::slice::from_raw_parts(x, dim),
        std::slice::from_raw_parts(w, dim),
        eps,
    );
}

/// # Safety
/// `x` valid for `dim` f32.
pub unsafe extern "C" fn dllama_inv_rms(x: *const f32, dim: usize, eps: f32) -> f32 {
    avx2::inv_rms(std::slice::from_raw_parts(x, dim), eps)
}

/// # Safety
/// `x` valid for `n` f32. In-place softmax over one row of width `n`.
pub unsafe extern "C" fn dllama_softmax(x: *mut f32, n: usize) {
    avx2::softmax(std::slice::from_raw_parts_mut(x, n));
}

/// # Safety
/// `out`, `gate`, `up` valid for `n` f32; `gelu` is 0 (silu) or 1.
pub unsafe extern "C" fn dllama_activated_mul(
    out: *mut f32,
    gate: *const f32,
    up: *const f32,
    n: usize,
    gelu: u32,
) {
    cpu::activated_mul(
        std::slice::from_raw_parts_mut(out, n),
        std::slice::from_raw_parts(gate, n),
        std::slice::from_raw_parts(up, n),
        gelu != 0,
    );
}

/// # Safety
/// `x`, `y` valid for `n` f32.
pub unsafe extern "C" fn dllama_dot(x: *const f32, y: *const f32, n: usize) -> f32 {
    avx2::dot_product(std::slice::from_raw_parts(x, n), std::slice::from_raw_parts(y, n))
}

/// The C++ `expf_avx2` polynomial (G1.5 bit-parity) — backends must use this
/// exact approximation wherever the engine applies expf (silu, softmax tails).
pub extern "C" fn dllama_expf(x: f32) -> f32 {
    avx2::expf_avx2_scalar(x)
}

/// # Safety
/// k/v caches valid for `(pos + 1) * kv_dim`; q/out valid for
/// `n_heads * head_dim`; k/v valid for `kv_dim`.
pub unsafe extern "C" fn dllama_attention(
    k_cache: *mut f32,
    v_cache: *mut f32,
    kv_dim: usize,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: *mut f32,
) {
    let used = (pos + 1) * kv_dim;
    cpu::attention(
        std::slice::from_raw_parts_mut(k_cache, used),
        std::slice::from_raw_parts_mut(v_cache, used),
        kv_dim,
        std::slice::from_raw_parts(q, n_heads * head_dim),
        std::slice::from_raw_parts(k, kv_dim),
        std::slice::from_raw_parts(v, kv_dim),
        pos,
        n_heads,
        n_kv_heads,
        head_dim,
        std::slice::from_raw_parts_mut(out, n_heads * head_dim),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_wire_roundtrip() {
        let all = [
            QuantKind::F32,
            QuantKind::F16,
            QuantKind::DllamaQ40,
            QuantKind::GgufQ4_0,
            QuantKind::GgufQ8_0,
            QuantKind::GgufQ4K,
            QuantKind::GgufQ6K,
        ];
        let ids: Vec<u32> = all.iter().map(|k| kind_to_wire(*k)).collect();
        // distinct and dense from 0
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted, (0..all.len() as u32).collect::<Vec<u32>>());
        for (k, &id) in all.iter().zip(ids.iter()) {
            assert_eq!(kind_from_wire(id), Some(*k));
        }
        assert_eq!(kind_from_wire(999), None);
    }
}
