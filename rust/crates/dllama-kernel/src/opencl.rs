//! OpenCL GPU accelerant for the Rust kernel backend (G5.3, DESIGN.md §10.12).
//!
//! "Any kind of GPU" the vendor-agnostic way: OpenCL runs on NVIDIA, AMD and
//! Intel (including this repo's dev machine — an Intel HD 620). OpenCL C has
//! a correctly-rounded `fma()` builtin, which wgpu/WGSL lacks — that's why
//! OpenCL and not wgpu: the bit-exactness contract from G1.5 survives.
//!
//! Numerics (bit-exact with the CPU reference by construction):
//! - one GPU work-item per output row, walking the weight row in block order
//!   (the exact scalar-emulation order from avx2.rs),
//! - integer-exact block dots (order-independent),
//! - one sequential `fma()` per block for q40/Q8_0; plain mul+add for the
//!   K-quant terms — `#pragma OPENCL FP_CONTRACT off` keeps the compiler from
//!   contracting those (the same trap Mojo has, see §10.10).
//!
//! Scope: the `matmul_q80` family only (the heavyweight ops); norms,
//! attention and codecs stay on the CPU reference path.
//!
//! Host notes:
//! - hand-rolled FFI via libloading (Windows `OpenCL.dll`, Linux
//!   `libOpenCL.so`) — no new crate dependencies, mirroring the plugin loader,
//! - the weight cache is keyed by (host pointer, bytes): weights are mmap'd
//!   and never move, so device copies are uploaded once and kept for the
//!   process lifetime,
//! - activation (q80) buffers are uploaded per call (KBs), outputs read back
//!   per call,
//! - any failure (no device, allocation too big, build error) degrades the
//!   specific call to the CPU path — a GPU can never break correctness.

use dllama_quant::QuantKind;
use libloading::Library;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CString};
use std::sync::{Mutex, OnceLock};

// ---------------------------------------------------------------------------
// OpenCL types + constants (subset of CL headers)
// ---------------------------------------------------------------------------

type ClPlatformId = *mut c_void;
type ClDeviceId = *mut c_void;
type ClContext = *mut c_void;
type ClCommandQueue = *mut c_void;
type ClMem = *mut c_void;
type ClProgram = *mut c_void;
type ClKernel = *mut c_void;
type ClInt = i32;
type ClUint = u32;
type ClUlong = u64;
type ClDeviceType = ClUlong;
type ClMemFlags = ClUlong;
type ClQueueProperties = isize;

const CL_SUCCESS: ClInt = 0;
const CL_DEVICE_TYPE_GPU: ClDeviceType = 1 << 2;
const CL_MEM_WRITE_ONLY: ClMemFlags = 1 << 1;
const CL_MEM_READ_ONLY: ClMemFlags = 1 << 2;
const CL_PROGRAM_BUILD_LOG: ClUint = 0x1183;

/// Number of kernels compiled from KERNEL_SOURCE (order = KERNEL_NAMES).
const N_KERNELS: usize = 4;
const KERNEL_NAMES: [&str; N_KERNELS] = ["mm_q80_q40", "mm_q80_q8_0", "mm_q80_q4_k", "mm_q80_q6_k"];

// kernel index <-> QuantKind mapping
fn kernel_index(kind: QuantKind) -> Option<usize> {
    match kind {
        QuantKind::DllamaQ40 | QuantKind::GgufQ4_0 => Some(0),
        QuantKind::GgufQ8_0 => Some(1),
        QuantKind::GgufQ4K => Some(2),
        QuantKind::GgufQ6K => Some(3),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// device kernels (OpenCL C). FP_CONTRACT OFF keeps K-quant terms on separate
// mul+add, matching the Rust reference exactly.
// ---------------------------------------------------------------------------

const KERNEL_SOURCE: &str = r#"
#pragma OPENCL FP_CONTRACT off

static float f16f(unsigned int h) {
    unsigned int sign = (h & 0x8000u) << 16;
    unsigned int e = (h >> 10) & 0x1fu;
    unsigned int mant = h & 0x3ffu;
    if (e == 0) {
        if (mant == 0) return as_float(sign);
        float v = (float)mant * 5.9604645e-8f; /* 2^-24 */
        return as_float(sign | as_uint(v));
    }
    if (e == 31u) return as_float(sign | 0x7f800000u | (mant << 13));
    return as_float(sign | ((e + 112u) << 23) | (mant << 13));
}

static int sext(uchar u) { return (int)(char)u; }

__kernel void mm_q80_q40(__global const uchar *xq80, __global const uchar *w,
                         __global float *dst, const int d, const int n) {
    int row = get_global_id(0);
    if (row >= d) return;
    int nb = n >> 5;
    float sum = 0.0f;
    for (int j = 0; j < nb; j++) {
        __global const uchar *wb = w + (ulong)(row * nb + j) * 18ul;
        __global const uchar *xb = xq80 + (ulong)j * 34ul;
        float s = f16f(wb[0] | (wb[1] << 8)) * f16f(xb[0] | (xb[1] << 8));
        int acc = 0;
        for (int k = 0; k < 16; k++) {
            int w0 = (int)(wb[2 + k] & 0x0F) - 8;
            int w1 = (int)(wb[2 + k] >> 4) - 8;
            int i1 = sext(xb[2 + k]);
            int i2 = sext(xb[2 + k + 16]);
            acc += w0 * i1 + w1 * i2;
        }
        sum = fma((float)acc, s, sum);
    }
    dst[row] = sum;
}

__kernel void mm_q80_q8_0(__global const uchar *xq80, __global const uchar *w,
                          __global float *dst, const int d, const int n) {
    int row = get_global_id(0);
    if (row >= d) return;
    int nb = n >> 5;
    float sum = 0.0f;
    for (int j = 0; j < nb; j++) {
        __global const uchar *wb = w + (ulong)(row * nb + j) * 34ul;
        __global const uchar *xb = xq80 + (ulong)j * 34ul;
        float s = f16f(wb[0] | (wb[1] << 8)) * f16f(xb[0] | (xb[1] << 8));
        int acc = 0;
        for (int k = 0; k < 32; k++)
            acc += sext(wb[2 + k]) * sext(xb[2 + k]);
        sum = fma((float)acc, s, sum);
    }
    dst[row] = sum;
}

__kernel void mm_q80_q4_k(__global const uchar *xq80, __global const uchar *w,
                          __global float *dst, const int d, const int n) {
    int row = get_global_id(0);
    if (row >= d) return;
    int rb = n >> 8;
    float sum = 0.0f;
    for (int bi = 0; bi < rb; bi++) {
        __global const uchar *b = w + (ulong)(row * rb + bi) * 144ul;
        float d_all = f16f(b[0] | (b[1] << 8));
        float dmin = f16f(b[2] | (b[3] << 8));
        __global const uchar *scales = b + 4;
        __global const uchar *qs = b + 16;
        for (int sb = 0; sb < 8; sb++) {
            unsigned int sc, m;
            if (sb < 4) { sc = scales[sb] & 63u; m = scales[sb + 4] & 63u; }
            else {
                sc = (scales[sb + 4] & 0x0Fu) | ((unsigned)(scales[sb - 4] >> 6) << 4);
                m = (scales[sb + 4] >> 4) | ((unsigned)(scales[sb] >> 6) << 4);
            }
            float ds = d_all * (float)sc;
            float ms = dmin * (float)m;
            __global const uchar *xb = xq80 + (ulong)(bi * 8 + sb) * 34ul;
            float dx = f16f(xb[0] | (xb[1] << 8));
            int lo = (sb % 2) == 0;
            int qs_off = 32 * (sb / 2);
            int dot = 0, sumx = 0;
            for (int k = 0; k < 32; k++) {
                int nib = lo ? (qs[qs_off + k] & 0x0F) : (qs[qs_off + k] >> 4);
                int xq = sext(xb[2 + k]);
                dot += nib * xq;
                sumx += xq;
            }
            volatile float va = ds * (float)dot;
            volatile float vb = ms * (float)sumx;
            float inner4 = va - vb;
            volatile float vp4 = dx * inner4;
            sum += vp4;
        }
    }
    dst[row] = sum;
}

__kernel void mm_q80_q6_k(__global const uchar *xq80, __global const uchar *w,
                          __global float *dst, const int d, const int n) {
    int row = get_global_id(0);
    if (row >= d) return;
    int rb = n >> 8;
    float sum = 0.0f;
    for (int bi = 0; bi < rb; bi++) {
        __global const uchar *b = w + (ulong)(row * rb + bi) * 210ul;
        float d_all = f16f(b[208] | (b[209] << 8));
        __global const uchar *ql = b;
        __global const uchar *qh = b + 128;
        __global const uchar *sc = b + 192;
        for (int h = 0; h < 8; h++) {
            int grp = h / 4, r = h % 4;
            __global const uchar *xb = xq80 + (ulong)(bi * 8 + h) * 34ul;
            float dx = f16f(xb[0] | (xb[1] << 8));
            int dot_a = 0, dot_b = 0;
            for (int l = 0; l < 32; l++) {
                int q;
                if (r == 0)
                    q = ((ql[grp*64 + l] & 0x0F) | (((int)qh[grp*32 + l] & 3) << 4)) - 32;
                else if (r == 1)
                    q = ((ql[grp*64 + l + 32] & 0x0F) | (((int)(qh[grp*32 + l] >> 2) & 3) << 4)) - 32;
                else if (r == 2)
                    q = ((ql[grp*64 + l] >> 4) | (((int)(qh[grp*32 + l] >> 4) & 3) << 4)) - 32;
                else
                    q = ((ql[grp*64 + l + 32] >> 4) | (((int)(qh[grp*32 + l] >> 6) & 3) << 4)) - 32;
                int xq = sext(xb[2 + l]);
                if (l < 16) dot_a += q * xq; else dot_b += q * xq;
            }
            float sa = d_all * (float)sext(sc[grp*8 + r*2]);
            float sbv = d_all * (float)sext(sc[grp*8 + r*2 + 1]);
            volatile float v1 = sa * (float)dot_a;
            volatile float v2 = sbv * (float)dot_b;
            float inner6 = v1 + v2;
            volatile float vp6 = dx * inner6;
            sum += vp6;
        }
    }
    dst[row] = sum;
}
"#;

// ---------------------------------------------------------------------------
// loaded API
// ---------------------------------------------------------------------------

macro_rules! cl_fn {
    ($lib:expr, $name:literal, $ty:ty) => {{
        match $lib.get::<$ty>(concat!($name, "\0").as_bytes()) {
            Ok(s) => *s,
            Err(e) => {
                warn_once(&format!("OpenCL symbol {}: {e}", $name));
                return None;
            }
        }
    }};
}

struct Gpu {
    _lib: &'static Library,
    ctx: ClContext,
    queue: ClCommandQueue,
    kernels: [ClKernel; N_KERNELS],
    device_name: String,
    // raw fns (kept for readability)
    set_arg: unsafe extern "C" fn(ClKernel, ClUint, usize, *const c_void) -> ClInt,
    enqueue_nd: unsafe extern "C" fn(
        ClCommandQueue,
        ClKernel,
        ClUint,
        *const usize,
        *const usize,
        *const usize,
        ClUint,
        *mut c_void,
        *mut c_void,
    ) -> ClInt,
    write_buf: unsafe extern "C" fn(ClCommandQueue, ClMem, ClUint, usize, usize, *const c_void, ClUint, *mut c_void, *mut c_void) -> ClInt,
    read_buf: unsafe extern "C" fn(ClCommandQueue, ClMem, ClUint, usize, usize, *mut c_void, ClUint, *mut c_void, *mut c_void) -> ClInt,
    create_buf: unsafe extern "C" fn(ClContext, ClMemFlags, usize, *mut c_void, *mut ClInt) -> ClMem,
    finish: unsafe extern "C" fn(ClCommandQueue) -> ClInt,
    release_mem: unsafe extern "C" fn(ClMem) -> ClInt,
}

unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
static WEIGHT_BUFS: OnceLock<Mutex<WeightMap>> = OnceLock::new();
static WARNED_FALLBACK: OnceLock<bool> = OnceLock::new();

/// cl_mem handle wrapper (raw pointers aren't Send; the buffer is owned by
/// the process-lifetime OpenCL context and never touched concurrently).
struct ClMemHandle(ClMem);
unsafe impl Send for ClMemHandle {}
type WeightMap = HashMap<(usize, u64), ClMemHandle>;

fn warn_once(msg: &str) {
    if WARNED_FALLBACK.set(true).is_ok() {
        eprintln!("[kernel-gpu] {msg}");
    }
}

/// Environment gate: `DLLAMA_RUST_GPU=0` (or `false`) disables; anything else
/// (including unset) uses the GPU when one is found.
fn env_enabled() -> bool {
    match std::env::var("DLLAMA_RUST_GPU") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "off"),
        Err(_) => true,
    }
}

fn init_gpu() -> Option<Gpu> {
    if !env_enabled() {
        eprintln!("[kernel-gpu] disabled by DLLAMA_RUST_GPU");
        return None;
    }
    let names: &[&str] = if cfg!(windows) {
        &["OpenCL.dll"]
    } else {
        &["libOpenCL.so.1", "libOpenCL.so"]
    };
    let mut last_err = String::from("no OpenCL loader found");
    let mut lib: Option<&'static Library> = None;
    for n in names {
        // SAFETY: the library is intentionally leaked (process lifetime);
        // all raw CL handles derived from it live in GPU (also leaked).
        match unsafe { Library::new(n) } {
            Ok(l) => {
                lib = Some(Box::leak(Box::new(l)));
                break;
            }
            Err(e) => last_err = format!("{n}: {e}"),
        }
    }
    let lib = match lib {
        Some(l) => l,
        None => {
            warn_once(&format!("OpenCL unavailable: {last_err}"));
            return None;
        }
    };
    // SAFETY: the library is leaked; all pointers live for the process.
    unsafe {
        let get_platform_ids = cl_fn!(
            lib,
            "clGetPlatformIDs",
            unsafe extern "C" fn(ClUint, *mut ClPlatformId, *mut ClUint) -> ClInt
        );
        let get_device_ids = cl_fn!(
            lib,
            "clGetDeviceIDs",
            unsafe extern "C" fn(ClPlatformId, ClDeviceType, ClUint, *mut ClDeviceId, *mut ClUint) -> ClInt
        );
        let create_context = cl_fn!(
            lib,
            "clCreateContext",
            unsafe extern "C" fn(*const isize, ClUint, *const ClDeviceId, *const c_void, *mut c_void, *mut ClInt) -> ClContext
        );

        // platform + first GPU device
        let mut n_platforms: ClUint = 0;
        if get_platform_ids(0, std::ptr::null_mut(), &mut n_platforms) != CL_SUCCESS || n_platforms == 0 {
            warn_once("no OpenCL platforms");
            return None;
        }
        let mut platforms = vec![std::ptr::null_mut(); n_platforms as usize];
        if get_platform_ids(n_platforms, platforms.as_mut_ptr(), std::ptr::null_mut()) != CL_SUCCESS {
            warn_once("clGetPlatformIDs failed");
            return None;
        }
        let mut device: ClDeviceId = std::ptr::null_mut();
        for p in platforms {
            let mut n_dev: ClUint = 0;
            if get_device_ids(p, CL_DEVICE_TYPE_GPU, 0, std::ptr::null_mut(), &mut n_dev) == CL_SUCCESS
                && n_dev > 0
            {
                if get_device_ids(p, CL_DEVICE_TYPE_GPU, 1, &mut device, std::ptr::null_mut()) == CL_SUCCESS {
                    break;
                }
            }
            device = std::ptr::null_mut();
        }
        if device.is_null() {
            warn_once("no OpenCL GPU device");
            return None;
        }

        // device name (best effort)
        let get_dev_info = cl_fn!(
            lib,
            "clGetDeviceInfo",
            unsafe extern "C" fn(ClDeviceId, ClUint, usize, *mut c_void, *mut usize) -> ClInt
        );
        let mut device_name = String::from("opencl-gpu");
        let mut buf = [0u8; 256];
        let mut got: usize = 0;
        // CL_DEVICE_NAME = 0x02B6
        if get_dev_info(device, 0x02B6, buf.len(), buf.as_mut_ptr() as *mut c_void, &mut got) == CL_SUCCESS
            && got > 0
        {
            if let Some(end) = buf[..got].iter().position(|&c| c == 0) {
                device_name = String::from_utf8_lossy(&buf[..end]).into_owned();
            }
        }

        // context + queue
        let mut err: ClInt = -1;
        let ctx = create_context(std::ptr::null(), 1, &device, std::ptr::null(), std::ptr::null_mut(), &mut err);
        if err != CL_SUCCESS {
            warn_once(&format!("clCreateContext failed ({err})"));
            return None;
        }
        // queue: prefer the OpenCL 2.0 entry point, fall back to the 1.x one
        let queue = if lib.get::<*mut c_void>(b"clCreateCommandQueueWithProperties\0").is_ok() {
            let f = cl_fn!(
                lib,
                "clCreateCommandQueueWithProperties",
                unsafe extern "C" fn(ClContext, ClDeviceId, *const ClQueueProperties, *mut ClInt) -> ClCommandQueue
            );
            f(ctx, device, std::ptr::null(), &mut err)
        } else {
            let f = cl_fn!(
                lib,
                "clCreateCommandQueue",
                unsafe extern "C" fn(ClContext, ClDeviceId, ClUlong, *mut ClInt) -> ClCommandQueue
            );
            f(ctx, device, 0, &mut err)
        };
        if err != CL_SUCCESS {
            warn_once(&format!("queue creation failed ({err})"));
            return None;
        }

        // program build
        let create_prog_src = cl_fn!(
            lib,
            "clCreateProgramWithSource",
            unsafe extern "C" fn(ClContext, ClUint, *const *const c_char, *const usize, *mut ClInt) -> ClProgram
        );
        let build_prog = cl_fn!(
            lib,
            "clBuildProgram",
            unsafe extern "C" fn(ClProgram, ClUint, *const ClDeviceId, *const c_char, *const c_void, *mut c_void) -> ClInt
        );
        let get_build_info = cl_fn!(
            lib,
            "clGetProgramBuildInfo",
            unsafe extern "C" fn(ClProgram, ClDeviceId, ClUint, usize, *mut c_void, *mut usize) -> ClInt
        );
        let create_kernel = cl_fn!(
            lib,
            "clCreateKernel",
            unsafe extern "C" fn(ClProgram, *const c_char, *mut ClInt) -> ClKernel
        );

        let src = CString::new(KERNEL_SOURCE).unwrap();
        let src_ptr: *const c_char = src.as_ptr();
        let mut err: ClInt = -1;
        let program = create_prog_src(ctx, 1, &src_ptr, std::ptr::null(), &mut err);
        if err != CL_SUCCESS {
            warn_once(&format!("clCreateProgramWithSource failed ({err})"));
            return None;
        }
        if build_prog(program, 1, &device, std::ptr::null(), std::ptr::null(), std::ptr::null_mut()) != CL_SUCCESS {
            let mut got: usize = 0;
            get_build_info(program, device, CL_PROGRAM_BUILD_LOG, 0, std::ptr::null_mut(), &mut got);
            let mut log = vec![0u8; got + 1];
            get_build_info(program, device, CL_PROGRAM_BUILD_LOG, log.len(), log.as_mut_ptr() as *mut c_void, std::ptr::null_mut());
            let s = String::from_utf8_lossy(&log);
            warn_once(&format!("OpenCL kernel build failed: {}", s.trim()));
            return None;
        }
        let mut kernels = [std::ptr::null_mut(); N_KERNELS];
        for (i, name) in KERNEL_NAMES.iter().enumerate() {
            let cname = CString::new(*name).unwrap();
            let mut kerr: ClInt = -1;
            kernels[i] = create_kernel(program, cname.as_ptr(), &mut kerr);
            if kerr != CL_SUCCESS {
                warn_once(&format!("clCreateKernel({name}) failed ({kerr})"));
                return None;
            }
        }

        eprintln!("[kernel-gpu] OpenCL device \"{device_name}\" — matmuls will use it (DLLAMA_RUST_GPU=0 to disable)");
        Some(Gpu {
            _lib: lib,
            ctx,
            queue,
            kernels,
            device_name,
            set_arg: cl_fn!(
                lib,
                "clSetKernelArg",
                unsafe extern "C" fn(ClKernel, ClUint, usize, *const c_void) -> ClInt
            ),
            enqueue_nd: cl_fn!(
                lib,
                "clEnqueueNDRangeKernel",
                unsafe extern "C" fn(ClCommandQueue, ClKernel, ClUint, *const usize, *const usize, *const usize, ClUint, *mut c_void, *mut c_void) -> ClInt
            ),
            write_buf: cl_fn!(
                lib,
                "clEnqueueWriteBuffer",
                unsafe extern "C" fn(ClCommandQueue, ClMem, ClUint, usize, usize, *const c_void, ClUint, *mut c_void, *mut c_void) -> ClInt
            ),
            read_buf: cl_fn!(
                lib,
                "clEnqueueReadBuffer",
                unsafe extern "C" fn(ClCommandQueue, ClMem, ClUint, usize, usize, *mut c_void, ClUint, *mut c_void, *mut c_void) -> ClInt
            ),
            create_buf: cl_fn!(
                lib,
                "clCreateBuffer",
                unsafe extern "C" fn(ClContext, ClMemFlags, usize, *mut c_void, *mut ClInt) -> ClMem
            ),
            finish: cl_fn!(lib, "clFinish", unsafe extern "C" fn(ClCommandQueue) -> ClInt),
            release_mem: cl_fn!(lib, "clReleaseMemObject", unsafe extern "C" fn(ClMem) -> ClInt),
        })
    }
}

/// True if the GPU accelerant is active (initializes on first use).
pub fn available() -> bool {
    GPU.get_or_init(init_gpu).is_some()
}

/// The resolved OpenCL device name, if active.
pub fn device_name() -> Option<&'static str> {
    GPU.get_or_init(init_gpu).as_ref().map(|g| g.device_name.as_str())
}

fn weight_buffer(gpu: &Gpu, w: &[u8]) -> Option<ClMem> {
    let key = (w.as_ptr() as usize, w.len() as u64);
    let map = WEIGHT_BUFS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap();
    if let Some(&ClMemHandle(m)) = guard.get(&key) {
        return Some(m);
    }
    unsafe {
        let mut err: ClInt = -1;
        let m = (gpu.create_buf)(gpu.ctx, CL_MEM_READ_ONLY, w.len(), std::ptr::null_mut(), &mut err);
        if err != CL_SUCCESS {
            warn_once(&format!(
                "GPU buffer alloc of {} MB failed ({err}); affected tensors stay on CPU",
                w.len() / 1_048_576
            ));
            return None;
        }
        if (gpu.write_buf)(gpu.queue, m, 1, 0, w.len(), w.as_ptr() as *const c_void, 0, std::ptr::null_mut(), std::ptr::null_mut()) != CL_SUCCESS {
            warn_once("GPU weight upload failed; affected tensors stay on CPU");
            (gpu.release_mem)(m);
            return None;
        }
        guard.insert(key, ClMemHandle(m));
        Some(m)
    }
}

/// Try to run out[d] = xq80 · W(kind)ᵀ on the GPU.
/// Returns false (caller falls back to CPU) on any unsupported kind or error.
pub fn try_matmul_q80(out: &mut [f32], xq80: &[u8], kind: QuantKind, w: &[u8], d: usize, n: usize) -> bool {
    let Some(idx) = kernel_index(kind) else { return false };
    let Some(gpu) = GPU.get_or_init(init_gpu).as_ref() else { return false };
    if out.len() < d || n % 32 != 0 {
        return false;
    }
    let Some(wbuf) = weight_buffer(gpu, w) else { return false };
    unsafe {
        let mut err: ClInt = -1;
        let xbuf = (gpu.create_buf)(gpu.ctx, CL_MEM_READ_ONLY, xq80.len(), std::ptr::null_mut(), &mut err);
        if err != CL_SUCCESS {
            return false;
        }
        let obuf = (gpu.create_buf)(gpu.ctx, CL_MEM_WRITE_ONLY, d * 4, std::ptr::null_mut(), &mut err);
        if err != CL_SUCCESS {
            (gpu.release_mem)(xbuf);
            return false;
        }
        let ok = (gpu.write_buf)(
            gpu.queue,
            xbuf,
            1,
            0,
            xq80.len(),
            xq80.as_ptr() as *const c_void,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ) == CL_SUCCESS
            && (gpu.set_arg)(gpu.kernels[idx], 0, size_of::<ClMem>(), &xbuf as *const _ as *const c_void) == CL_SUCCESS
            && (gpu.set_arg)(gpu.kernels[idx], 1, size_of::<ClMem>(), &wbuf as *const _ as *const c_void) == CL_SUCCESS
            && (gpu.set_arg)(gpu.kernels[idx], 2, size_of::<ClMem>(), &obuf as *const _ as *const c_void) == CL_SUCCESS
            && {
                let dv = d as i32;
                let nv = n as i32;
                (gpu.set_arg)(gpu.kernels[idx], 3, 4, &dv as *const i32 as *const c_void) == CL_SUCCESS
                    && (gpu.set_arg)(gpu.kernels[idx], 4, 4, &nv as *const i32 as *const c_void) == CL_SUCCESS
            };
        let global = d;
        let ok = ok
            && (gpu.enqueue_nd)(gpu.queue, gpu.kernels[idx], 1, std::ptr::null(), &global, std::ptr::null(), 0, std::ptr::null_mut(), std::ptr::null_mut()) == CL_SUCCESS
            && (gpu.finish)(gpu.queue) == CL_SUCCESS
            && (gpu.read_buf)(gpu.queue, obuf, 1, 0, d * 4, out.as_mut_ptr() as *mut c_void, 0, std::ptr::null_mut(), std::ptr::null_mut()) == CL_SUCCESS;
        (gpu.release_mem)(obuf);
        (gpu.release_mem)(xbuf);
        ok
    }
}

/// CLI helper: report OpenCL availability.
pub fn probe_report() -> String {
    if env_enabled() {
        match device_name() {
            Some(name) => format!("available: {name}"),
            None => "not available (no OpenCL GPU, driver, or disabled)".into(),
        }
    } else {
        "disabled by DLLAMA_RUST_GPU".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dllama_quant::{quantize_row_q40, quantize_row_q80};

    /// GPU vs CPU bit comparison with first-diff reporting, per quant kind.
    #[test]
    fn gpu_matches_cpu_bits() {
        if !available() {
            eprintln!("[test] no OpenCL GPU — skipping bit comparison");
            return;
        }
        let n = 256usize;
        let d = 8usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.25) - 8.0).collect();
        let mut xq80 = vec![0u8; dllama_quant::q80_bytes(n)];
        quantize_row_q80(&x, &mut xq80);

        // build one weight blob per kind: q40 | q8_0 | q4_k | q6_k, with sane
        // f16 scales (0.01 -> 0x211F) so the data stays well-conditioned
        let make_w = |kind: QuantKind, seed: u64| -> Vec<u8> {
            let row = kind.row_bytes(n) as usize;
            let mut w = Vec::with_capacity(d * row);
            let mut s = seed;
            for _ in 0..d {
                let mut bytes = vec![0u8; row];
                for b in bytes.iter_mut() {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    *b = (s >> 33) as u8;
                }
                // overwrite f16 scale fields with 0.01 (LE bytes 1F 21)
                match kind {
                    QuantKind::GgufQ4K => {
                        bytes[0] = 0x1F; bytes[1] = 0x21;   // d
                        bytes[2] = 0x14; bytes[3] = 0x21;   // dmin (0.002-ish)
                    }
                    QuantKind::GgufQ6K => {
                        bytes[208] = 0x1F; bytes[209] = 0x21;
                    }
                    _ => {
                        bytes[0] = 0x1F; bytes[1] = 0x21;
                    }
                }
                w.extend_from_slice(&bytes);
            }
            w
        };

        for kind in [QuantKind::GgufQ6K, QuantKind::GgufQ4K, QuantKind::DllamaQ40, QuantKind::GgufQ8_0] {
            let w = make_w(kind, 42);
            let mut gpu_out = vec![0.0f32; d];
            let mut cpu_out = vec![0.0f32; d];
            let handled = try_matmul_q80(&mut gpu_out, &xq80, kind, &w, d, n);
            crate::gguf::matmul_q80(&mut cpu_out, &xq80, kind, &w, d, n);
            // the weight cache is keyed by (ptr, len) — valid for the engine
            // (mmap'd weights never move), so keep the test blobs alive to
            // avoid allocator address reuse hitting a stale cache entry.
            std::mem::forget(w);
            assert!(handled, "GPU did not handle {kind:?}");
            for i in 0..d {
                let g = gpu_out[i].to_bits();
                let c = cpu_out[i].to_bits();
                assert_eq!(
                    g, c,
                    "{kind:?} row {i}: gpu={:08x} ({}) cpu={:08x} ({})",
                    g, gpu_out[i], c, cpu_out[i]
                );
            }
            eprintln!("[test] {kind:?}: bit-exact GPU vs CPU");
        }
    }
}
