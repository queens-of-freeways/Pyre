//! Reference kernel backend cdylib (G5).
//!
//! Exports the C-ABI v1 symbols defined in `dllama_kernel::abi` as a
//! standalone `dllama_kernel_c.dll` / `libdllama_kernel_c.so`, so the
//! dynamic-backend path can be exercised end-to-end:
//!
//! ```text
//! cargo build -p dllama-kernel-c --release
//! set DLLAMA_KERNEL_LIB=target\release\dllama_kernel_c.dll
//! dllama-cli perplexity ...   # logs: [kernel] backend "rust-avx2 (cdylib)" ...
//! ```
//!
//! The shims forward to the same Rust AVX2 functions the engine uses
//! statically, so results must be bit-identical — this crate doubles as the
//! ABI conformance reference a Mojo backend has to match.
//!
//! Mojo drop-in: `kernels.mojo` compiles to a DLL exporting the same symbol
//! names (`@export abi("C")`), returns the same ABI version, and implements
//! the same numerics (see DESIGN.md §10.9).

use std::ffi::c_char;

#[no_mangle]
pub extern "C" fn dllama_kernels_abi_version() -> u32 {
    dllama_kernel::abi::ABI_VERSION
}

#[no_mangle]
pub extern "C" fn dllama_kernels_name() -> *const c_char {
    static NAME: &[u8] = b"rust-avx2 (cdylib reference)\0";
    NAME.as_ptr() as *const c_char
}

#[no_mangle]
pub unsafe extern "C" fn dllama_matmul_f32(
    out: *mut f32,
    x: *const f32,
    w: *const f32,
    m: usize,
    n: usize,
    k: usize,
) {
    dllama_kernel::abi::dllama_matmul_f32(out, x, w, m, n, k)
}

#[no_mangle]
pub unsafe extern "C" fn dllama_matmul_q40_f32(
    out: *mut f32,
    x: *const f32,
    w: *const u8,
    m: usize,
    n: usize,
    k: usize,
) {
    dllama_kernel::abi::dllama_matmul_q40_f32(out, x, w, m, n, k)
}

#[no_mangle]
pub unsafe extern "C" fn dllama_matmul_q80(
    out: *mut f32,
    xq80: *const u8,
    w: *const u8,
    kind: u32,
    d: usize,
    n: usize,
) {
    dllama_kernel::abi::dllama_matmul_q80(out, xq80, w, kind, d, n)
}

#[no_mangle]
pub unsafe extern "C" fn dllama_dequant_row(out: *mut f32, bytes: *const u8, kind: u32, k: usize) {
    dllama_kernel::abi::dllama_dequant_row(out, bytes, kind, k)
}

#[no_mangle]
pub unsafe extern "C" fn dllama_rmsnorm(
    out: *mut f32,
    x: *const f32,
    w: *const f32,
    dim: usize,
    eps: f32,
) {
    dllama_kernel::abi::dllama_rmsnorm(out, x, w, dim, eps)
}

#[no_mangle]
pub unsafe extern "C" fn dllama_inv_rms(x: *const f32, dim: usize, eps: f32) -> f32 {
    dllama_kernel::abi::dllama_inv_rms(x, dim, eps)
}

#[no_mangle]
pub unsafe extern "C" fn dllama_softmax(x: *mut f32, n: usize) {
    dllama_kernel::abi::dllama_softmax(x, n)
}

#[no_mangle]
pub unsafe extern "C" fn dllama_activated_mul(
    out: *mut f32,
    gate: *const f32,
    up: *const f32,
    n: usize,
    gelu: u32,
) {
    dllama_kernel::abi::dllama_activated_mul(out, gate, up, n, gelu)
}

#[no_mangle]
pub unsafe extern "C" fn dllama_dot(x: *const f32, y: *const f32, n: usize) -> f32 {
    dllama_kernel::abi::dllama_dot(x, y, n)
}

#[no_mangle]
pub extern "C" fn dllama_expf(x: f32) -> f32 {
    dllama_kernel::abi::dllama_expf(x)
}

#[no_mangle]
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
    dllama_kernel::abi::dllama_attention(
        k_cache, v_cache, kv_dim, q, k, v, pos, n_heads, n_kv_heads, head_dim, out,
    )
}
