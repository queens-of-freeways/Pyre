# ===----------------------------------------------------------------------=== #
# dllama GPU kernels — Mojo device code for the matmul family (G5.2, §10.11)
#
# Design: bit-exact with the CPU reference. One GPU thread per output row,
# each thread walks its weight row in block order doing the integer-exact
# block dot + ONE SEQUENTIAL fma per block — the exact G1.5 numerics chain
# (integer dots are order-independent; the fma chain is serial by
# construction), so a GPU result is bit-identical to the CPU backend.
#
# Portable device code: `max.gpu` intrinsics compile to PTX (NVIDIA),
# AMDGCN (AMD) or Metal — the "any Mojo-supported GPU" story.
#
# Status (2026-10-04): compile-checked host+device code. Runtime validation
# needs NVIDIA/AMD hardware (this dev box has an Intel iGPU, which Mojo does
# not target). Execution is opt-in via DLLAMA_MOJO_GPU=1 and gated on
# DeviceContext.number_of_devices() > 0; otherwise the caller uses the CPU
# backend (already validated bit-exact).
#
# TODO(next, needs hardware): weight residency — per-call upload is
# correctness-first only. Planned state bootstrap (Mojo 1.1 has no mutable
# globals): on first call, open("/tmp/.dllama-mojo-<pid>.bin", O_CREAT|O_RDWR)
# via external_call["open"], ftruncate to one page, mmap; the mapping holds
# {state_magic, DeviceContext handle, entry[4096] of (host_ptr, device_ptr,
# bytes)}; weight uploads cached by host pointer (weights are mmap-stable).
# ~20 µs/call overhead, ABI v1 untouched.
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.math import fma
from std.ffi import external_call


def f16_to_f32(h: UInt16) -> Float32:
    var sign = (UInt32(h & 0x8000)) << 16
    var e = UInt32((h >> 10) & 0x1F)
    var mant = UInt32(h & 0x03FF)
    if e == 0:
        if mant == 0:
            return bitcast[DType.float32, 1](sign)
        var v = Float32(mant) * 5.9604645e-8
        return bitcast[DType.float32, 1](sign | bitcast[DType.uint32, 1](v))
    elif e == 31:
        return bitcast[DType.float32, 1](sign | 0x7F800000 | (mant << 13))
    return bitcast[DType.float32, 1](sign | ((e + 112) << 23) | (mant << 13))


def load_u16_le(b: Pointer[Scalar[DType.uint8], _],
                off: Int) -> UInt16:
    return UInt16(b[unsafe_offset=off]) | (UInt16(b[unsafe_offset=off + 1]) << 8)


def sign_extend_u8(u: UInt8) -> Int32:
    return (Int32(u ^ 0x80)) - 128


# ---------------------------------------------------------------------------
# GPU availability gate (non-raising)
# ---------------------------------------------------------------------------

def gpu_requested_and_available() -> Bool:
    if DeviceContext.number_of_devices() == 0:
        return False
    comptime ENV_KEY = "DLLAMA_MOJO_GPU"
    var requested = False
    # getenv via libc: const char* getenv(const char*)
    comptime NULL_PTR = Pointer[Scalar[DType.int8], ImmUntrackedOrigin](
        unsafe_from_address=0)
    var present = external_call["getenv", Pointer[Scalar[DType.int8],
        ImmUntrackedOrigin]](ENV_KEY.ptr())
    if present != NULL_PTR:
        requested = True
    return requested


# ---------------------------------------------------------------------------
# host dispatchers: upload -> launch -> sync -> download (correctness-first:
# per-call upload; see TODO for the weight-residency cache)
# ---------------------------------------------------------------------------

def gpu_matmul_q80_q40(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
                       xq80: Pointer[Scalar[DType.uint8], MutUntrackedOrigin],
                       w: Pointer[Scalar[DType.uint8], MutUntrackedOrigin],
                       d: Int, n: Int) raises:
    var ctx = DeviceContext()
    var x_dev = ctx.create_buffer_sync[DType.uint8]((n // 32) * 34)
    x_dev.enqueue_copy_from(xq80)
    var w_dev = ctx.create_buffer_sync[DType.uint8](d * (n // 32) * 18)
    w_dev.enqueue_copy_from(w)
    var out_dev = ctx.create_buffer_sync[DType.float32](d)

    @__parameter
    def mm_kernel(dp: Int32, np: Int32, n_blocks: Int32):
        var row = Int(block_dim.x * block_idx.x + thread_idx.x)
        if row < Int(dp):
            var n_blocks_i = Int(n_blocks)
            var sum = Float32(0)
            for j in range(n_blocks_i):
                var wb = w_dev.unsafe_ptr().unsafe_offset((row * n_blocks_i + j) * 18)
                var xb = x_dev.unsafe_ptr().unsafe_offset(j * 34)
                var s = f16_to_f32(load_u16_le(wb, 0)) * f16_to_f32(load_u16_le(xb, 0))
                var acc = Int32(0)
                for k in range(16):
                    var w0 = (Int32(wb[unsafe_offset=2 + k] & 0x0F)) - 8
                    var w1 = (Int32(wb[unsafe_offset=2 + k] >> 4)) - 8
                    var i1 = sign_extend_u8(xb[unsafe_offset=2 + k])
                    var i2 = sign_extend_u8(xb[unsafe_offset=2 + k + 16])
                    acc += w0 * i1 + w1 * i2
                # sequential fma per block — bit-exact with the CPU reference
                sum = fma(Float32(acc), s, sum)
            out_dev.unsafe_ptr()[unsafe_offset=row] = sum

    comptime BLOCK = 128
    var grid = (d + BLOCK - 1) // BLOCK
    ctx.enqueue_function[mm_kernel](
        Int32(d), Int32(n), Int32(n // 32), grid_dim=grid, block_dim=BLOCK
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst, out_dev)


def gpu_matmul_q80_q8_0(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
                        xq80: Pointer[Scalar[DType.uint8], MutUntrackedOrigin],
                        w: Pointer[Scalar[DType.uint8], MutUntrackedOrigin],
                        d: Int, n: Int) raises:
    var ctx = DeviceContext()
    var x_dev = ctx.create_buffer_sync[DType.uint8]((n // 32) * 34)
    x_dev.enqueue_copy_from(xq80)
    var w_dev = ctx.create_buffer_sync[DType.uint8](d * (n // 32) * 34)
    w_dev.enqueue_copy_from(w)
    var out_dev = ctx.create_buffer_sync[DType.float32](d)

    @__parameter
    def mm_kernel(dp: Int32, np: Int32, n_blocks: Int32):
        var row = Int(block_dim.x * block_idx.x + thread_idx.x)
        if row < Int(dp):
            var n_blocks_i = Int(n_blocks)
            var sum = Float32(0)
            for j in range(n_blocks_i):
                var wb = w_dev.unsafe_ptr().unsafe_offset((row * n_blocks_i + j) * 34)
                var xb = x_dev.unsafe_ptr().unsafe_offset(j * 34)
                var s = f16_to_f32(load_u16_le(wb, 0)) * f16_to_f32(load_u16_le(xb, 0))
                var acc = Int32(0)
                for k in range(32):
                    acc += sign_extend_u8(wb[unsafe_offset=2 + k]) * sign_extend_u8(
                        xb[unsafe_offset=2 + k])
                sum = fma(Float32(acc), s, sum)
            out_dev.unsafe_ptr()[unsafe_offset=row] = sum

    comptime BLOCK = 128
    var grid = (d + BLOCK - 1) // BLOCK
    ctx.enqueue_function[mm_kernel](
        Int32(d), Int32(n), Int32(n // 32), grid_dim=grid, block_dim=BLOCK
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst, out_dev)


def gpu_matmul_q80_q4_k(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
                        xq80: Pointer[Scalar[DType.uint8], MutUntrackedOrigin],
                        w: Pointer[Scalar[DType.uint8], MutUntrackedOrigin],
                        d: Int, n: Int) raises:
    var ctx = DeviceContext()
    var x_dev = ctx.create_buffer_sync[DType.uint8]((n // 32) * 34)
    x_dev.enqueue_copy_from(xq80)
    var w_dev = ctx.create_buffer_sync[DType.uint8](d * (n // 256) * 144)
    w_dev.enqueue_copy_from(w)
    var out_dev = ctx.create_buffer_sync[DType.float32](d)

    @__parameter
    def mm_kernel(dp: Int32, np: Int32, row_blocks: Int32):
        var row = Int(block_dim.x * block_idx.x + thread_idx.x)
        if row < Int(dp):
            var rb = Int(row_blocks)
            var sum = Float32(0)
            for bi in range(rb):
                var b = w_dev.unsafe_ptr().unsafe_offset((row * rb + bi) * 144)
                var d_all = f16_to_f32(load_u16_le(b, 0))
                var dmin = f16_to_f32(load_u16_le(b, 2))
                var scales = b.unsafe_offset(4)
                var qs = b.unsafe_offset(16)
                for sb in range(8):
                    var sc: UInt8
                    var m: UInt8
                    if sb < 4:
                        sc = scales[unsafe_offset=sb] & 63
                        m = scales[unsafe_offset=sb + 4] & 63
                    else:
                        sc = (scales[unsafe_offset=sb + 4] & 0x0F) | (
                            (scales[unsafe_offset=sb - 4] >> 6) << 4)
                        m = (scales[unsafe_offset=sb + 4] >> 4) | (
                            (scales[unsafe_offset=sb] >> 6) << 4)
                    var ds = d_all * Float32(sc)
                    var ms = dmin * Float32(m)
                    var xb = x_dev.unsafe_ptr().unsafe_offset((bi * 8 + sb) * 34)
                    var dx = f16_to_f32(load_u16_le(xb, 0))
                    var lo = (sb % 2) == 0
                    var qs_off = 32 * (sb // 2)
                    var dot = Int32(0)
                    var sumx = Int32(0)
                    for k in range(32):
                        var nib = Int32(qs[unsafe_offset=qs_off + k] & 0x0F)
                        if not lo:
                            nib = Int32(qs[unsafe_offset=qs_off + k] >> 4)
                        var xq = sign_extend_u8(xb[unsafe_offset=2 + k])
                        dot += nib * xq
                        sumx += xq
                    # plain mul+add (no contraction) — matches the Rust reference
                    sum = sum + dx * (ds * Float32(dot) - ms * Float32(sumx))
            out_dev.unsafe_ptr()[unsafe_offset=row] = sum

    comptime BLOCK = 128
    var grid = (d + BLOCK - 1) // BLOCK
    ctx.enqueue_function[mm_kernel](
        Int32(d), Int32(n), Int32(n // 256), grid_dim=grid, block_dim=BLOCK
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst, out_dev)


def gpu_matmul_q80_q6_k(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
                        xq80: Pointer[Scalar[DType.uint8], MutUntrackedOrigin],
                        w: Pointer[Scalar[DType.uint8], MutUntrackedOrigin],
                        d: Int, n: Int) raises:
    var ctx = DeviceContext()
    var x_dev = ctx.create_buffer_sync[DType.uint8]((n // 32) * 34)
    x_dev.enqueue_copy_from(xq80)
    var w_dev = ctx.create_buffer_sync[DType.uint8](d * (n // 256) * 210)
    w_dev.enqueue_copy_from(w)
    var out_dev = ctx.create_buffer_sync[DType.float32](d)

    @__parameter
    def mm_kernel(dp: Int32, np: Int32, row_blocks: Int32):
        var row = Int(block_dim.x * block_idx.x + thread_idx.x)
        if row < Int(dp):
            var rb = Int(row_blocks)
            var sum = Float32(0)
            for bi in range(rb):
                var b = w_dev.unsafe_ptr().unsafe_offset((row * rb + bi) * 210)
                var d_all = f16_to_f32(load_u16_le(b, 208))
                var ql = b.unsafe_offset(0)
                var qh = b.unsafe_offset(128)
                var sc = b.unsafe_offset(192)
                for h in range(8):
                    var grp = h // 4
                    var r = h % 4
                    var xb = x_dev.unsafe_ptr().unsafe_offset((bi * 8 + h) * 34)
                    var dx = f16_to_f32(load_u16_le(xb, 0))
                    var dot_a = Int32(0)
                    var dot_b = Int32(0)
                    for l in range(32):
                        var q: Int32
                        if r == 0:
                            q = (Int32(ql[unsafe_offset=grp * 64 + l] & 0x0F)
                                 | ((Int32(qh[unsafe_offset=grp * 32 + l]) & 3) << 4)) - 32
                        elif r == 1:
                            q = (Int32(ql[unsafe_offset=grp * 64 + l + 32] & 0x0F)
                                 | ((Int32(qh[unsafe_offset=grp * 32 + l] >> 2) & 3) << 4)) - 32
                        elif r == 2:
                            q = (Int32(ql[unsafe_offset=grp * 64 + l] >> 4)
                                 | ((Int32(qh[unsafe_offset=grp * 32 + l] >> 4) & 3) << 4)) - 32
                        else:
                            q = (Int32(ql[unsafe_offset=grp * 64 + l + 32] >> 4)
                                 | ((Int32(qh[unsafe_offset=grp * 32 + l] >> 6) & 3) << 4)) - 32
                        var xq = sign_extend_u8(xb[unsafe_offset=2 + l])
                        if l < 16:
                            dot_a += q * xq
                        else:
                            dot_b += q * xq
                    var sa = d_all * Float32(sign_extend_u8(sc[unsafe_offset=grp * 8 + r * 2]))
                    var sb2 = d_all * Float32(sign_extend_u8(sc[unsafe_offset=grp * 8 + r * 2 + 1]))
                    sum = sum + dx * (sa * Float32(dot_a) + sb2 * Float32(dot_b))
            out_dev.unsafe_ptr()[unsafe_offset=row] = sum

    comptime BLOCK = 128
    var grid = (d + BLOCK - 1) // BLOCK
    ctx.enqueue_function[mm_kernel](
        Int32(d), Int32(n), Int32(n // 256), grid_dim=grid, block_dim=BLOCK
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst, out_dev)
