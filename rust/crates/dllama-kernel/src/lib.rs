//! dllama-kernel: compute kernels + the pluggable backend seam (G5).
//!
//! `Kernels` is a table of C-ABI function pointers (see `abi`). The engine
//! (`dllama-exec`) calls every per-token kernel through it. The table is
//! filled from the built-in Rust AVX2 reference implementation, or — when
//! `DLLAMA_KERNEL_LIB` points at a backend cdylib — loaded at runtime via
//! `libloading` (`dllama-kernel-c` is the reference plugin; a Mojo backend
//! exporting the same symbols drops in without engine changes).
//!
//! Load-time work (rope-cache libm math, norm-weight dequantization, q80
//! codecs) stays host-side, mirroring the C++ layering (nn-core/nn-quants
//! vs nn-cpu-ops).

pub mod abi;
pub mod avx2;
pub mod cpu;
pub mod gguf;

use dllama_quant::QuantKind;

pub mod opencl;

use std::sync::atomic::{AtomicUsize, Ordering};

/// Run `f(first_index, out_slice)` over `count` independent tasks, each
/// writing a disjoint `slice_len`-element slice of `buf`, across threads
/// (G6). Tasks are grouped into ~`threads()` batches to bound spawn count.
/// Used for MoE experts: each expert's compute is independent; the caller
/// merges serially in the original order to stay bit-exact.
pub fn for_task_slices(
    buf: &mut [f32],
    count: usize,
    slice_len: usize,
    f: impl Fn(usize, &mut [f32]) + Sync,
) {
    debug_assert_eq!(buf.len(), count * slice_len);
    let threads = threads();
    if threads <= 1 || count < 2 {
        for i in 0..count {
            f(i, &mut buf[i * slice_len..(i + 1) * slice_len]);
        }
        return;
    }
    let per_group_tasks = (count + threads - 1) / threads;
    let group_bytes = per_group_tasks * slice_len;
    std::thread::scope(|s| {
        let f = &f;
        let mut first = 0;
        for group in buf.chunks_mut(group_bytes) {
            let g_first = first;
            let g_tasks = group.len() / slice_len;
            s.spawn(move || {
                for (gi, task_slice) in group.chunks_mut(slice_len).enumerate() {
                    f(g_first + gi, task_slice);
                }
            });
            first += g_tasks;
        }
    });
}

// ---------------------------------------------------------------------------
// G6: row threading. Every matmul in this crate computes output rows
// independently (serial fma chain per row), so distributing rows across
// threads is bit-exact by construction.
// ---------------------------------------------------------------------------

static THREADS: AtomicUsize = AtomicUsize::new(0);

/// Set the matmul thread count (CLI `--threads`).
pub fn set_threads(n: usize) {
    THREADS.store(n.max(1), Ordering::Relaxed);
}

/// Effective thread count: explicit set → `DLLAMA_RS_THREADS` env → logical cores.
pub fn threads() -> usize {
    let t = THREADS.load(Ordering::Relaxed);
    if t > 0 {
        return t;
    }
    if let Ok(v) = std::env::var("DLLAMA_RS_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Rows-per-thread floor below which a matmul runs serial (spawn overhead
/// would dominate; expert matmuls at d=768 still split 4-way).
const MIN_ROWS_PER_THREAD: usize = 64;

/// Run `f(range_start, range_end)` over `count` items split across threads
/// (G6.3 prefill: weight-stationary batch matmuls partition by output row).
/// Disjoint ranges — bit-exact under any partition.
pub fn for_range(count: usize, f: impl Fn(usize, usize) + Sync) {
    let threads = threads();
    if threads <= 1 || count < 2 * MIN_ROWS_PER_THREAD {
        f(0, count);
        return;
    }
    let n_threads = threads.min(count / MIN_ROWS_PER_THREAD);
    let chunk = (count + n_threads - 1) / n_threads;
    std::thread::scope(|s| {
        let f = &f;
        let mut start = 0;
        while start < count {
            let end = (start + chunk).min(count);
            let st = start;
            s.spawn(move || f(st, end));
            start = end;
        }
    });
}

/// Run `f(seg_start_chunk, segment)` over `buf`'s chunks (length `chunk_len`
/// each), grouped into `threads()` segments. The closure owns its segment for
/// its lifetime — hoist per-thread scratch inside it (G6.4 prefill kernels:
/// column-major batch matmuls, one chunk = one output row's m results).
pub fn for_chunk_segments_mut(buf: &mut [f32], chunk_len: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    debug_assert_eq!(buf.len() % chunk_len, 0);
    let count = buf.len() / chunk_len;
    let threads = threads();
    if threads <= 1 || count < 2 {
        f(0, buf);
        return;
    }
    let per = ((count + threads - 1) / threads).max(1);
    std::thread::scope(|s| {
        let f = &f;
        let mut base = 0usize;
        for seg in buf.chunks_mut(per * chunk_len) {
            let b = base;
            let seg_len = seg.len();
            s.spawn(move || f(b, seg));
            base += seg_len / chunk_len;
        }
    });
}

/// Run `f(row_start, row_end, out_rows)` over `d` output rows in `out`,/// split across threads when worthwhile. `out_rows` is the caller's slice
/// restricted to `[row_start, row_end)` — workers only write their own
/// range, which is what makes every matmul here bit-exact under threading.
pub fn for_rows(out: &mut [f32], d: usize, f: impl Fn(usize, usize, &mut [f32]) + Sync) {
    debug_assert_eq!(out.len(), d);
    let threads = threads();
    if threads <= 1 || d < 2 * MIN_ROWS_PER_THREAD {
        f(0, d, out);
        return;
    }
    let n_threads = threads.min(d / MIN_ROWS_PER_THREAD);
    let chunk = (d + n_threads - 1) / n_threads;
    std::thread::scope(|s| {
        let f = &f;
        let mut start = 0;
        for slice in out.chunks_mut(chunk) {
            let end = start + slice.len();
            let st = start;
            s.spawn(move || f(st, end, slice));
            start = end;
        }
    });
}

/// Per-layer key/value cache (f32 activations; sharding handled by the caller
/// passing only this node's kv slice).
pub struct KvCache {
    /// [pos][kv_dim]
    pub k: Vec<f32>,
    /// [pos][kv_dim]
    pub v: Vec<f32>,
    pub seq_len: usize,
    pub kv_dim: usize,
}

impl KvCache {
    pub fn new(seq_len: usize, kv_dim: usize) -> Self {
        Self {
            k: vec![0.0; seq_len * kv_dim],
            v: vec![0.0; seq_len * kv_dim],
            seq_len,
            kv_dim,
        }
    }
}

/// Default plugin search paths for the platform's Mojo backend (G5.3):
/// `<exe_dir>/<plugin>`, repo layout `<exe_dir>/../../mojo/<plugin>`, and
/// the same two relative to the working directory. Windows currently has no
/// Mojo toolchain, so `dllama_mojo.dll` will simply not exist there — the
/// built-in Rust backend (with the OpenCL accelerant) remains.
fn default_plugin_candidates() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;
    let name = match std::env::consts::OS {
        "windows" => "dllama_mojo.dll",
        "macos" => "libdllama_mojo.dylib",
        _ => "libdllama_mojo.so",
    };
    let mut out: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.join(name));
            if let Some(repo_root) = dir.parent().and_then(|p| p.parent()) {
                out.push(repo_root.join("mojo").join(name));
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        out.push(cwd.join(name));
        out.push(cwd.join("mojo").join(name));
    }
    out
}

/// Resolved kernel backend: a name plus the v1 ABI function table.
///
/// Built with [`Kernels::builtin`] (static Rust reference) or
/// [`Kernels::load_path`] (dynamic library). The safe methods below are the
/// engine-facing surface: they assert slice geometry and forward to the raw
/// pointers, so both paths execute identical code.
pub struct Kernels {
    name: String,
    /// built-in table may use the OpenCL GPU accelerant for the matmul
    /// family (external plugins manage their own devices).
    gpu_accel: bool,
    matmul_f32: abi::MatmulF32Fn,
    matmul_q40: abi::MatmulQ40Fn,
    matmul_q80: abi::MatmulQ80Fn,
    dequant_row: abi::DequantRowFn,
    rmsnorm: abi::RmsnormFn,
    inv_rms: abi::InvRmsFn,
    softmax: abi::SoftmaxFn,
    activated_mul: abi::ActivatedMulFn,
    dot: abi::DotFn,
    expf: abi::ExpfFn,
    attention: abi::AttentionFn,
}

impl Kernels {
    /// Backend name (diagnostics; e.g. "rust-avx2 (built-in)").
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The static Rust reference backend (same functions the cdylib exports).
    pub fn builtin() -> Kernels {
        Kernels {
            name: opencl::device_name()
                .map(|d| format!("rust-avx2 + {d} (built-in)"))
                .unwrap_or_else(|| "rust-avx2 (built-in)".into()),
            gpu_accel: true,
            matmul_f32: abi::dllama_matmul_f32,
            matmul_q40: abi::dllama_matmul_q40_f32,
            matmul_q80: abi::dllama_matmul_q80,
            dequant_row: abi::dllama_dequant_row,
            rmsnorm: abi::dllama_rmsnorm,
            inv_rms: abi::dllama_inv_rms,
            softmax: abi::dllama_softmax,
            activated_mul: abi::dllama_activated_mul,
            dot: abi::dllama_dot,
            expf: abi::dllama_expf,
            attention: abi::dllama_attention,
        }
    }

    /// Resolve the backend: `DLLAMA_KERNEL_LIB` (explicit override) →
    /// OS-detected default plugin (Mojo backend on Linux when present) →
    /// the built-in Rust table (+ optional OpenCL GPU accelerant on Windows/
    /// no-plugin systems). A plugin that fails to load/version-check
    /// downgrades with a warning — the engine always runs.
    pub fn load() -> Kernels {
        match std::env::var("DLLAMA_KERNEL_LIB") {
            Ok(path) if !path.trim().is_empty() => {
                return match Self::load_path(&path) {
                    Ok(k) => k,
                    Err(e) => {
                        eprintln!("[kernel] '{path}': {e} — falling back to the built-in Rust backend");
                        Self::builtin()
                    }
                };
            }
            _ => {}
        }
        // OS-aware default: look for the Mojo backend plugin that ships for
        // this platform (Windows has no Mojo toolchain today, so the probe
        // finds nothing there and the built-in Rust backend stays in charge).
        for cand in default_plugin_candidates() {
            if cand.exists() {
                let cand = cand.display().to_string();
                match Self::load_path(&cand) {
                    Ok(k) => return k,
                    Err(e) => {
                        eprintln!(
                            "[kernel] OS '{}' default plugin '{cand}' failed: {e} — falling back to the built-in Rust backend",
                            std::env::consts::OS
                        );
                        return Self::builtin();
                    }
                }
            }
        }
        Self::builtin()
    }

    /// Load a backend cdylib and version-check it. The library is leaked:
    /// the table holds raw pointers into it for the process lifetime.
    pub fn load_path(path: &str) -> Result<Kernels, String> {
        use libloading::{Library, Symbol};
        // SAFETY: the library is intentionally leaked and never unloaded, so
        // the fn pointers stay valid; every symbol's contract is checked by
        // the version gate + the safe wrappers below.
        unsafe {
            let lib: &'static Library =
                Box::leak(Box::new(Library::new(path).map_err(|e| e.to_string())?));

            macro_rules! sym {
                ($t:ty, $name:literal) => {{
                    let s: Symbol<'static, $t> = lib
                        .get(concat!($name, "\0").as_bytes())
                        .map_err(|e| format!("symbol {}: {e}", $name))?;
                    *s
                }};
            }

            let version: Symbol<'static, abi::AbiVersionFn> = lib
                .get(b"dllama_kernels_abi_version\0")
                .map_err(|e| format!("symbol dllama_kernels_abi_version: {e}"))?;
            let v = version();
            if v != abi::ABI_VERSION {
                return Err(format!(
                    "plugin ABI v{v}, engine expects v{} — rebuild the backend against this engine",
                    abi::ABI_VERSION
                ));
            }

            let name_fn: Symbol<'static, abi::NameFn> = lib
                .get(b"dllama_kernels_name\0")
                .map_err(|e| format!("symbol dllama_kernels_name: {e}"))?;
            let name = {
                let p = name_fn();
                if p.is_null() {
                    return Err("backend returned a null name".into());
                }
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            };

            let k = Kernels {
                name,
                gpu_accel: false,
                matmul_f32: sym!(abi::MatmulF32Fn, "dllama_matmul_f32"),
                matmul_q40: sym!(abi::MatmulQ40Fn, "dllama_matmul_q40_f32"),
                matmul_q80: sym!(abi::MatmulQ80Fn, "dllama_matmul_q80"),
                dequant_row: sym!(abi::DequantRowFn, "dllama_dequant_row"),
                rmsnorm: sym!(abi::RmsnormFn, "dllama_rmsnorm"),
                inv_rms: sym!(abi::InvRmsFn, "dllama_inv_rms"),
                softmax: sym!(abi::SoftmaxFn, "dllama_softmax"),
                activated_mul: sym!(abi::ActivatedMulFn, "dllama_activated_mul"),
                dot: sym!(abi::DotFn, "dllama_dot"),
                expf: sym!(abi::ExpfFn, "dllama_expf"),
                attention: sym!(abi::AttentionFn, "dllama_attention"),
            };
            eprintln!(
                "[kernel] backend \"{}\" loaded from {path} (ABI v{})",
                k.name,
                abi::ABI_VERSION
            );
            Ok(k)
        }
    }

    /// out[m][n] = x[m][k] · Wᵀ, W row-major [n][k].
    pub fn matmul_f32(&self, out: &mut [f32], x: &[f32], w: &[f32], m: usize, n: usize, k: usize) {
        debug_assert_eq!(out.len(), m * n);
        debug_assert_eq!(x.len(), m * k);
        debug_assert_eq!(w.len(), n * k);
        unsafe { (self.matmul_f32)(out.as_mut_ptr(), x.as_ptr(), w.as_ptr(), m, n, k) }
    }

    /// Q40 weights, f32 input (f32-buffer path; m = 1 in the engine today).
    pub fn matmul_q40(&self, out: &mut [f32], x: &[f32], w: &[u8], m: usize, n: usize, k: usize) {
        debug_assert_eq!(out.len(), m * n);
        debug_assert_eq!(x.len(), m * k);
        debug_assert!(w.len() >= n * dllama_quant::q40_bytes(k));
        unsafe { (self.matmul_q40)(out.as_mut_ptr(), x.as_ptr(), w.as_ptr(), m, n, k) }
    }

    /// q80-quantized input × quantized weights of any kind:
    /// out[d] = dequant(xq80) · dequant(W)ᵀ (the engine's main path).
    pub fn matmul_q80(&self, out: &mut [f32], xq80: &[u8], kind: QuantKind, w: &[u8], d: usize, n: usize) {
        debug_assert_eq!(out.len(), d);
        debug_assert_eq!(xq80.len(), dllama_quant::q80_bytes(n));
        debug_assert!(w.len() >= d * kind.row_bytes(n) as usize);
        // built-in backend only: OpenCL GPU accelerant (bit-exact, falls
        // back per call on any failure)
        if self.gpu_accel && opencl::try_matmul_q80(out, xq80, kind, w, d, n) {
            return;
        }
        unsafe {
            (self.matmul_q80)(
                out.as_mut_ptr(),
                xq80.as_ptr(),
                w.as_ptr(),
                abi::kind_to_wire(kind),
                d,
                n,
            )
        }
    }

    /// Dequantize one weight row of `kind` (k values) into `out`.
    pub fn dequant_row(&self, kind: QuantKind, bytes: &[u8], out: &mut [f32], k: usize) {
        debug_assert_eq!(out.len(), k);
        debug_assert!(bytes.len() >= kind.row_bytes(k) as usize);
        unsafe {
            (self.dequant_row)(
                out.as_mut_ptr(),
                bytes.as_ptr(),
                abi::kind_to_wire(kind),
                k,
            )
        }
    }

    /// out = x * w / rms(x) (dim = x.len()).
    pub fn rmsnorm(&self, out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
        let dim = x.len();
        debug_assert_eq!(out.len(), dim);
        debug_assert_eq!(w.len(), dim);
        unsafe { (self.rmsnorm)(out.as_mut_ptr(), x.as_ptr(), w.as_ptr(), dim, eps) }
    }

    /// 1/rms(x) (exact C++ association; see avx2::inv_rms).
    pub fn inv_rms(&self, x: &[f32], eps: f32) -> f32 {
        unsafe { (self.inv_rms)(x.as_ptr(), x.len(), eps) }
    }

    /// In-place softmax over each row of `x` (rows of width `n`).
    pub fn softmax(&self, x: &mut [f32], n: usize) {
        debug_assert_eq!(x.len() % n, 0);
        for row in x.chunks_mut(n) {
            unsafe { (self.softmax)(row.as_mut_ptr(), row.len()) }
        }
    }

    /// out = silu(gate) * up (gelu = 1 selects the gelu variant).
    pub fn activated_mul(&self, out: &mut [f32], gate: &[f32], up: &[f32], gelu: bool) {
        let n = out.len();
        debug_assert_eq!(gate.len(), n);
        debug_assert_eq!(up.len(), n);
        unsafe { (self.activated_mul)(out.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), n, gelu as u32) }
    }

    /// Bit-exact f32 dot (AVX2 reduction tree).
    pub fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        debug_assert_eq!(x.len(), y.len());
        unsafe { (self.dot)(x.as_ptr(), y.as_ptr(), x.len()) }
    }

    /// The C++ expf_avx2 polynomial — silu/softmax parity lives here.
    pub fn expf(&self, x: f32) -> f32 {
        unsafe { (self.expf)(x) }
    }

    /// Fused GQA attention over the caller-owned k/v cache slices.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
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
        let used = (pos + 1) * kv_dim;
        debug_assert!(k_cache.len() >= used);
        debug_assert!(v_cache.len() >= used);
        debug_assert_eq!(k.len(), kv_dim);
        debug_assert_eq!(v.len(), kv_dim);
        debug_assert_eq!(q.len(), n_heads * head_dim);
        debug_assert_eq!(out.len(), n_heads * head_dim);
        unsafe {
            (self.attention)(
                k_cache.as_mut_ptr(),
                v_cache.as_mut_ptr(),
                kv_dim,
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                pos,
                n_heads,
                n_kv_heads,
                head_dim,
                out.as_mut_ptr(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dllama_quant::{q40_bytes, quantize_row_q40, quantize_row_q80};

    /// The built-in table must be bit-identical to the direct kernel calls
    /// (it is the same code through pointers — this pins the plumbing).
    #[test]
    fn builtin_table_matches_direct_kernels() {
        let k = Kernels::builtin();
        assert!(
            k.name().starts_with("rust-avx2"),
            "unexpected backend name: {}",
            k.name()
        );

        fn bits_eq(a: &[f32], b: &[f32]) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        }

        // matmul_q80 × q40: quantize a row pair, run both paths
        let n = 128usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.25) - 8.0).collect();
        let w_f32: Vec<f32> = (0..2 * n).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let row_bytes = q40_bytes(n);
        let mut w_q40 = vec![0u8; 2 * row_bytes];
        quantize_row_q40(&w_f32[..n], &mut w_q40[..row_bytes]);
        quantize_row_q40(&w_f32[n..], &mut w_q40[row_bytes..]);
        let mut xq80 = vec![0u8; dllama_quant::q80_bytes(n)];
        quantize_row_q80(&x, &mut xq80);

        let mut out_table = vec![0.0f32; 2];
        k.matmul_q80(&mut out_table, &xq80, QuantKind::DllamaQ40, &w_q40, 2, n);
        let mut out_direct = vec![0.0f32; 2];
        avx2::matmul_q80_q40(&mut out_direct, &xq80, &w_q40, 2, n);
        assert!(bits_eq(&out_table, &out_direct));

        // rmsnorm / softmax / dot / inv_rms / expf
        let w: Vec<f32> = (0..n).map(|i| 1.0 + (i % 3) as f32 * 0.1).collect();
        let mut o_table = vec![0.0f32; n];
        let mut o_direct = vec![0.0f32; n];
        k.rmsnorm(&mut o_table, &x, &w, 1e-5);
        cpu::rmsnorm(&mut o_direct, &x, &w, 1e-5);
        assert!(bits_eq(&o_table, &o_direct));

        let mut s_table = x.clone();
        let mut s_direct = x.clone();
        k.softmax(&mut s_table, n);
        avx2::softmax(&mut s_direct);
        assert!(bits_eq(&s_table, &s_direct));

        assert_eq!(k.dot(&x, &w).to_bits(), avx2::dot_product(&x, &w).to_bits());
        assert_eq!(k.inv_rms(&x, 1e-5).to_bits(), avx2::inv_rms(&x, 1e-5).to_bits());
        assert_eq!(k.expf(1.234).to_bits(), avx2::expf_avx2_scalar(1.234).to_bits());

        // attention: table vs direct on a tiny cache
        let (n_heads, n_kv, hd) = (2usize, 1usize, 2usize);
        let kv_dim = n_kv * hd;
        let mut kt = KvCache::new(4, kv_dim);
        let mut kd = KvCache::new(4, kv_dim);
        let q = vec![1.0f32, 0.0, 1.0, 0.0];
        let kv = vec![1.0f32, 0.0];
        let vv = vec![5.0f32, -5.0];
        let mut o_table = vec![0.0f32; n_heads * hd];
        let mut o_direct = vec![0.0f32; n_heads * hd];
        k.attention(&mut kt.k, &mut kt.v, kv_dim, &q, &kv, &vv, 0, n_heads, n_kv, hd, &mut o_table);
        cpu::attention(&mut kd.k, &mut kd.v, kv_dim, &q, &kv, &vv, 0, n_heads, n_kv, hd, &mut o_direct);
        assert!(bits_eq(&o_table, &o_direct));
    }

    #[test]
    fn load_path_rejects_missing_and_garbage() {
        assert!(Kernels::load_path("definitely-not-a-backend.dll").is_err());
        // a real file that is not a library / has no symbols
        let tmp = std::env::temp_dir().join("dllama-abi-garbage.bin");
        std::fs::write(&tmp, b"not a dll").unwrap();
        assert!(Kernels::load_path(tmp.to_str().unwrap()).is_err());
    }
}
