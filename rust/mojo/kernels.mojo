# ===----------------------------------------------------------------------=== #
# dllama kernel backend — Mojo 1.1 (C-ABI v1, DESIGN.md §10.10)
#
# Bit-exact port of the Rust AVX2 reference kernels (dllama-kernel):
#   - integer-exact q40/Q8_0/Q4_K/Q6_K block dots + one sequential fma per
#     block (matches matmul_q80_q40_scalar / gguf.rs semantics),
#   - 8-lane fma accumulation + the horizontalSum_avx2 reduction tree
#     (extractf128 + movehl + movehdup) emulated lane-by-lane,
#   - the C++ expf_avx2 degree-4 2^x polynomial with fma Horner steps,
#   - glibc expf for the softmax scalar tail (external_call — matches the
#     Linux Rust engine's libm::expf bit-for-bit).
#
# Build (WSL/Linux):
#   mojo build --emit shared-lib kernels.mojo -o libdllama_mojo.so
# Engine:
#   DLLAMA_KERNEL_LIB=./libdllama_mojo.so dllama-rs perplexity ...
# ===----------------------------------------------------------------------=== #

from std.math import fma, sqrt, round
from std.memory import bitcast
from std.memory.alloc import alloc, dealloc, Layout
from std.runtime import initialize_runtime
from std.ffi import external_call

comptime ABI_VERSION = UInt32(1)



# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def f16_to_f32(h: UInt16) -> Float32:
    var sign = (UInt32(h & 0x8000)) << 16
    var e = UInt32((h >> 10) & 0x1F)
    var mant = UInt32(h & 0x03FF)
    if e == 0:
        if mant == 0:
            return bitcast[DType.float32, 1](sign)
        var v = Float32(mant) * 5.9604645e-8  # 2^-24
        return bitcast[DType.float32, 1](sign | bitcast[DType.uint32, 1](v))
    elif e == 31:
        return bitcast[DType.float32, 1](sign | 0x7F800000 | (mant << 13))
    return bitcast[DType.float32, 1](sign | ((e + 112) << 23) | (mant << 13))


def load_u16_le(b: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], off: Int) -> UInt16:
    return UInt16(b[unsafe_offset=off]) | (UInt16(b[unsafe_offset=off + 1]) << 8)


def sign_extend_u8(u: UInt8) -> Int32:
    # (u ^ 0x80) - 128 == signed reinterpretation of the byte
    return (Int32(u ^ 0x80)) - 128


# ---------------------------------------------------------------------------
# C++ expf_avx2 polynomial (nn-cpu-ops.cpp:88) — degree-4 2^x, fma Horner
# ---------------------------------------------------------------------------

def expf_avx2_scalar(x_in: Float32) -> Float32:
    var x = x_in
    if x < -88.0:
        x = -88.0
    if x > 88.0:
        x = 88.0
    var y = x * 1.4426950408889634
    var n = Int32(round(y))  # round-to-nearest-even (matches cvtps_epi32)
    var f = y - Float32(n)
    var p = Float32(0.009618129107628477)
    p = fma(p, f, Float32(0.05550410866482158))
    p = fma(p, f, Float32(0.2402265069591007))
    p = fma(p, f, Float32(0.6931471805599453))
    p = fma(p, f, Float32(1.0))
    var two_n = bitcast[DType.float32, 1]((UInt32(n + 127)) << 23)
    return p * two_n


# ---------------------------------------------------------------------------
# horizontalSum_avx2 tree (nn-cpu-ops.cpp:59):
#   hi = extractf128(1); res = hi + lo; res += movehl(res); res += movehdup(res)
# ---------------------------------------------------------------------------

def horizontal_sum(l0: Float32, l1: Float32, l2: Float32, l3: Float32,
                   l4: Float32, l5: Float32, l6: Float32, l7: Float32) -> Float32:
    var m0 = l0 + l4
    var m1 = l1 + l5
    var m2 = l2 + l6
    var m3 = l3 + l7
    var a0 = m0 + m2
    var a1 = m1 + m3
    return a0 + a1


# ---------------------------------------------------------------------------
# invRms (nn-cpu-ops.cpp:114): fma lanes + tree, then 1/sqrt(sum/size + eps)
# ---------------------------------------------------------------------------

def inv_rms_impl(x: Pointer[Scalar[DType.float32], MutUntrackedOrigin], n: Int, eps: Float32) -> Float32:
    var l0 = Float32(0)
    var l1 = Float32(0)
    var l2 = Float32(0)
    var l3 = Float32(0)
    var l4 = Float32(0)
    var l5 = Float32(0)
    var l6 = Float32(0)
    var l7 = Float32(0)
    var i = 0
    while i + 8 <= n:
        l0 = fma(x[unsafe_offset=i + 0], x[unsafe_offset=i + 0], l0)
        l1 = fma(x[unsafe_offset=i + 1], x[unsafe_offset=i + 1], l1)
        l2 = fma(x[unsafe_offset=i + 2], x[unsafe_offset=i + 2], l2)
        l3 = fma(x[unsafe_offset=i + 3], x[unsafe_offset=i + 3], l3)
        l4 = fma(x[unsafe_offset=i + 4], x[unsafe_offset=i + 4], l4)
        l5 = fma(x[unsafe_offset=i + 5], x[unsafe_offset=i + 5], l5)
        l6 = fma(x[unsafe_offset=i + 6], x[unsafe_offset=i + 6], l6)
        l7 = fma(x[unsafe_offset=i + 7], x[unsafe_offset=i + 7], l7)
        i += 8
    var sum = horizontal_sum(l0, l1, l2, l3, l4, l5, l6, l7)
    sum = sum / Float32(n)
    sum = sum + eps
    return 1.0 / sqrt(sum)


# ---------------------------------------------------------------------------
# dotProduct_F32 (nn-cpu-ops.cpp:722): fma lanes + tree; scalar tail no-fma
# ---------------------------------------------------------------------------

def dot_product(a: Pointer[Scalar[DType.float32], MutUntrackedOrigin], b: Pointer[Scalar[DType.float32], MutUntrackedOrigin], n: Int) -> Float32:
    var l0 = Float32(0)
    var l1 = Float32(0)
    var l2 = Float32(0)
    var l3 = Float32(0)
    var l4 = Float32(0)
    var l5 = Float32(0)
    var l6 = Float32(0)
    var l7 = Float32(0)
    var n8 = n - n % 8
    var i = 0
    while i < n8:
        l0 = fma(a[unsafe_offset=i + 0], b[unsafe_offset=i + 0], l0)
        l1 = fma(a[unsafe_offset=i + 1], b[unsafe_offset=i + 1], l1)
        l2 = fma(a[unsafe_offset=i + 2], b[unsafe_offset=i + 2], l2)
        l3 = fma(a[unsafe_offset=i + 3], b[unsafe_offset=i + 3], l3)
        l4 = fma(a[unsafe_offset=i + 4], b[unsafe_offset=i + 4], l4)
        l5 = fma(a[unsafe_offset=i + 5], b[unsafe_offset=i + 5], l5)
        l6 = fma(a[unsafe_offset=i + 6], b[unsafe_offset=i + 6], l6)
        l7 = fma(a[unsafe_offset=i + 7], b[unsafe_offset=i + 7], l7)
        i += 8
    var sum = horizontal_sum(l0, l1, l2, l3, l4, l5, l6, l7)
    while i < n:
        sum = sum + a[unsafe_offset=i] * b[unsafe_offset=i]
        i += 1
    return sum


# ---------------------------------------------------------------------------
# softmax_F32 (nn-cpu-ops.cpp:595): max, poly exp + lane sums, glibc tail
# ---------------------------------------------------------------------------

def softmax_impl(x: Pointer[Scalar[DType.float32], MutUntrackedOrigin], size: Int):
    if size == 0:
        return
    var max_val = x[unsafe_offset=0]
    for i in range(1, size):
        if x[unsafe_offset=i] > max_val:
            max_val = x[unsafe_offset=i]
    var avx_end = size - size % 8
    var l0 = Float32(0)
    var l1 = Float32(0)
    var l2 = Float32(0)
    var l3 = Float32(0)
    var l4 = Float32(0)
    var l5 = Float32(0)
    var l6 = Float32(0)
    var l7 = Float32(0)
    var i = 0
    while i < avx_end:
        var v0 = expf_avx2_scalar(x[unsafe_offset=i + 0] - max_val)
        var v1 = expf_avx2_scalar(x[unsafe_offset=i + 1] - max_val)
        var v2 = expf_avx2_scalar(x[unsafe_offset=i + 2] - max_val)
        var v3 = expf_avx2_scalar(x[unsafe_offset=i + 3] - max_val)
        var v4 = expf_avx2_scalar(x[unsafe_offset=i + 4] - max_val)
        var v5 = expf_avx2_scalar(x[unsafe_offset=i + 5] - max_val)
        var v6 = expf_avx2_scalar(x[unsafe_offset=i + 6] - max_val)
        var v7 = expf_avx2_scalar(x[unsafe_offset=i + 7] - max_val)
        x[unsafe_offset=i + 0] = v0
        x[unsafe_offset=i + 1] = v1
        x[unsafe_offset=i + 2] = v2
        x[unsafe_offset=i + 3] = v3
        x[unsafe_offset=i + 4] = v4
        x[unsafe_offset=i + 5] = v5
        x[unsafe_offset=i + 6] = v6
        x[unsafe_offset=i + 7] = v7
        l0 = l0 + v0
        l1 = l1 + v1
        l2 = l2 + v2
        l3 = l3 + v3
        l4 = l4 + v4
        l5 = l5 + v5
        l6 = l6 + v6
        l7 = l7 + v7
        i += 8
    var sum = horizontal_sum(l0, l1, l2, l3, l4, l5, l6, l7)
    while i < size:
        # C++ scalar tail: real libm expf (glibc on Linux — matches the
        # Rust engine's extern "C" expf)
        var v = external_call["expf", Float32](x[unsafe_offset=i] - max_val)
        x[unsafe_offset=i] = v
        sum = sum + v
        i += 1
    if sum == 0.0:
        sum = 0.000001
    var inv_sum = 1.0 / sum
    for i in range(size):
        x[unsafe_offset=i] = x[unsafe_offset=i] * inv_sum


# ---------------------------------------------------------------------------
# quantized matmuls: integer-exact block dots + sequential fma (G1.5 scheme)
# ---------------------------------------------------------------------------

def matmul_q80_q40_impl(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], x: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], w: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], d: Int, n: Int):
    var n_blocks = n // 32
    for di in range(d):
        var sum = Float32(0)
        for j in range(n_blocks):
            var wb = w.unsafe_offset((di * n_blocks + j) * 18)
            var xb = x.unsafe_offset(j * 34)
            var s = f16_to_f32(load_u16_le(wb, 0)) * f16_to_f32(load_u16_le(xb, 0))
            var acc = Int32(0)
            for k in range(16):
                var w0 = (Int32(wb[unsafe_offset=2 + k] & 0x0F)) - 8
                var w1 = (Int32(wb[unsafe_offset=2 + k] >> 4)) - 8
                var i1 = sign_extend_u8(xb[unsafe_offset=2 + k])
                var i2 = sign_extend_u8(xb[unsafe_offset=2 + k + 16])
                acc += w0 * i1 + w1 * i2
            sum = fma(Float32(acc), s, sum)
        dst[unsafe_offset=di] = sum


def matmul_q80_q8_0_impl(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], x: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], w: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], d: Int, n: Int):
    var n_blocks = n // 32
    for di in range(d):
        var row = w.unsafe_offset(di * n_blocks * 34)
        var sum = Float32(0)
        for j in range(n_blocks):
            var wb = row.unsafe_offset(j * 34)
            var xb = x.unsafe_offset(j * 34)
            var s = f16_to_f32(load_u16_le(wb, 0)) * f16_to_f32(load_u16_le(xb, 0))
            var acc = Int32(0)
            for k in range(32):
                acc += sign_extend_u8(wb[unsafe_offset=2 + k]) * sign_extend_u8(
                    xb[unsafe_offset=2 + k])
            sum = fma(Float32(acc), s, sum)
        dst[unsafe_offset=di] = sum


def scale_min_k4(j: Int, q: Pointer[Scalar[DType.uint8], MutUntrackedOrigin]) -> Tuple[UInt8, UInt8]:
    if j < 4:
        return (q[unsafe_offset=j] & 63, q[unsafe_offset=j + 4] & 63)
    var d = (q[unsafe_offset=j + 4] & 0x0F) | ((q[unsafe_offset=j - 4] >> 6) << 4)
    var m = (q[unsafe_offset=j + 4] >> 4) | ((q[unsafe_offset=j] >> 6) << 4)
    return (d, m)


def matmul_q80_q4_k_impl(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], x: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], w: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], d: Int, n: Int):
    var row_blocks = n // 256
    for di in range(d):
        var row = w.unsafe_offset(di * row_blocks * 144)
        var sum = Float32(0)
        for bi in range(row_blocks):
            var b = row.unsafe_offset(bi * 144)
            var d_all = f16_to_f32(load_u16_le(b, 0))
            var dmin = f16_to_f32(load_u16_le(b, 2))
            var scales = b.unsafe_offset(4)
            var qs = b.unsafe_offset(16)
            for sb in range(8):
                var pair = scale_min_k4(sb, scales)
                var ds = d_all * Float32(pair[0])
                var ms = dmin * Float32(pair[1])
                var xb = x.unsafe_offset((bi * 8 + sb) * 34)
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
                sum = sum + dx * (ds * Float32(dot) - ms * Float32(sumx))
        dst[unsafe_offset=di] = sum


def matmul_q80_q6_k_impl(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], x: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], w: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], d: Int, n: Int):
    var row_blocks = n // 256
    for di in range(d):
        var row = w.unsafe_offset(di * row_blocks * 210)
        var sum = Float32(0)
        for bi in range(row_blocks):
            var b = row.unsafe_offset(bi * 210)
            var d_all = f16_to_f32(load_u16_le(b, 208))
            var ql = b.unsafe_offset(0)
            var qh = b.unsafe_offset(128)
            var sc = b.unsafe_offset(192)
            for h in range(8):
                var grp = h // 4
                var r = h % 4
                var xb = x.unsafe_offset((bi * 8 + h) * 34)
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
        dst[unsafe_offset=di] = sum


def dequant_row_q80_bytes(x: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], n: Int):
    var n_blocks = n // 32
    for j in range(n_blocks):
        var xb = x.unsafe_offset(j * 34)
        var d = f16_to_f32(load_u16_le(xb, 0))
        for k in range(32):
            dst[unsafe_offset=j * 32 + k] = Float32(sign_extend_u8(
                xb[unsafe_offset=2 + k])) * d


# ---------------------------------------------------------------------------
# row dequantization (F32/F16/Q40/Q4_0/Q8_0/Q4_K/Q6_K)
# ---------------------------------------------------------------------------

def dequant_q40(b: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], n_blocks: Int):
    for i in range(n_blocks):
        var blk = b.unsafe_offset(i * 18)
        var d = f16_to_f32(load_u16_le(blk, 0))
        for j in range(16):
            var x0 = (Int32(blk[unsafe_offset=2 + j] & 0x0F)) - 8
            var x1 = (Int32(blk[unsafe_offset=2 + j] >> 4)) - 8
            dst[unsafe_offset=i * 32 + j] = Float32(x0) * d
            dst[unsafe_offset=i * 32 + j + 16] = Float32(x1) * d


def dequant_q8_0(b: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], n_blocks: Int):
    for i in range(n_blocks):
        var blk = b.unsafe_offset(i * 34)
        var d = f16_to_f32(load_u16_le(blk, 0))
        for j in range(32):
            dst[unsafe_offset=i * 32 + j] = Float32(sign_extend_u8(
                blk[unsafe_offset=2 + j])) * d


def dequant_q4_k(b: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], nb: Int):
    for i in range(nb):
        var blk = b.unsafe_offset(i * 144)
        var d = f16_to_f32(load_u16_le(blk, 0))
        var dmin = f16_to_f32(load_u16_le(blk, 2))
        var scales = blk.unsafe_offset(4)
        var qs = blk.unsafe_offset(16)
        var y = dst.unsafe_offset(i * 256)
        var y_off = 0
        var q_off = 0
        var is_idx = 0
        for _it in range(4):
            var p1 = scale_min_k4(is_idx, scales)
            var p2 = scale_min_k4(is_idx + 1, scales)
            var d1 = d * Float32(p1[0])
            var m1 = dmin * Float32(p1[1])
            var d2 = d * Float32(p2[0])
            var m2 = dmin * Float32(p2[1])
            for l in range(32):
                y[unsafe_offset=y_off + l] = d1 * Float32(
                    qs[unsafe_offset=q_off + l] & 0x0F) - m1
            for l in range(32):
                y[unsafe_offset=y_off + 32 + l] = d2 * Float32(
                    qs[unsafe_offset=q_off + l] >> 4) - m2
            y_off += 64
            q_off += 32
            is_idx += 2


def dequant_q6_k(b: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], nb: Int):
    for i in range(nb):
        var blk = b.unsafe_offset(i * 210)
        var d = f16_to_f32(load_u16_le(blk, 208))
        var ql = blk.unsafe_offset(0)
        var qh = blk.unsafe_offset(128)
        var sc = blk.unsafe_offset(192)
        var y = dst.unsafe_offset(i * 256)
        var y_off = 0
        var ql_off = 0
        var qh_off = 0
        var sc_off = 0
        for _n in range(2):
            for l in range(32):
                var idx = l // 16
                var q1 = (Int32(ql[unsafe_offset=ql_off + l] & 0x0F)
                          | ((Int32(qh[unsafe_offset=qh_off + l]) & 3) << 4)) - 32
                var q2 = (Int32(ql[unsafe_offset=ql_off + l + 32] & 0x0F)
                          | ((Int32(qh[unsafe_offset=qh_off + l] >> 2) & 3) << 4)) - 32
                var q3 = (Int32(ql[unsafe_offset=ql_off + l] >> 4)
                          | ((Int32(qh[unsafe_offset=qh_off + l] >> 4) & 3) << 4)) - 32
                var q4 = (Int32(ql[unsafe_offset=ql_off + l + 32] >> 4)
                          | ((Int32(qh[unsafe_offset=qh_off + l] >> 6) & 3) << 4)) - 32
                y[unsafe_offset=y_off + l] = d * Float32(sign_extend_u8(
                    sc[unsafe_offset=sc_off + idx])) * Float32(q1)
                y[unsafe_offset=y_off + l + 32] = d * Float32(sign_extend_u8(
                    sc[unsafe_offset=sc_off + idx + 2])) * Float32(q2)
                y[unsafe_offset=y_off + l + 64] = d * Float32(sign_extend_u8(
                    sc[unsafe_offset=sc_off + idx + 4])) * Float32(q3)
                y[unsafe_offset=y_off + l + 96] = d * Float32(sign_extend_u8(
                    sc[unsafe_offset=sc_off + idx + 6])) * Float32(q4)
            y_off += 128
            ql_off += 64
            qh_off += 32
            sc_off += 8


# ---------------------------------------------------------------------------
# cpu.rs `dot`: 4 accumulators, plain mul+add (no fma), pairwise combine
# ---------------------------------------------------------------------------

def dot4(x: Pointer[Scalar[DType.float32], MutUntrackedOrigin], w: Pointer[Scalar[DType.float32], MutUntrackedOrigin], k: Int) -> Float32:
    var a0 = Float32(0)
    var a1 = Float32(0)
    var a2 = Float32(0)
    var a3 = Float32(0)
    var i = 0
    while i + 4 <= k:
        a0 = a0 + x[unsafe_offset=i] * w[unsafe_offset=i]
        a1 = a1 + x[unsafe_offset=i + 1] * w[unsafe_offset=i + 1]
        a2 = a2 + x[unsafe_offset=i + 2] * w[unsafe_offset=i + 2]
        a3 = a3 + x[unsafe_offset=i + 3] * w[unsafe_offset=i + 3]
        i += 4
    var tail = Float32(0)
    while i < k:
        tail = tail + x[unsafe_offset=i] * w[unsafe_offset=i]
        i += 1
    return (a0 + a1) + (a2 + a3) + tail


# ---------------------------------------------------------------------------
# exported ABI v1 surface
# ---------------------------------------------------------------------------

@export
def dllama_kernels_abi_version() abi("C") -> UInt32:
    initialize_runtime()
    return ABI_VERSION


@export
def dllama_kernels_name() abi("C") -> Pointer[Scalar[DType.int8], ImmStaticOrigin]:
    initialize_runtime()
    # StringLiteral is_idx NUL-terminated by contract (stdlib docs)
    return "mojo-1.1 (wsl)".as_c_string_span().ptr()


@export
def dllama_expf(x: Float32) abi("C") -> Float32:
    initialize_runtime()
    return expf_avx2_scalar(x)


@export
def dllama_dot(x: Pointer[Scalar[DType.float32], MutUntrackedOrigin], y: Pointer[Scalar[DType.float32], MutUntrackedOrigin], n: Int) abi("C") -> Float32:
    initialize_runtime()
    return dot_product(x, y, n)


@export
def dllama_inv_rms(x: Pointer[Scalar[DType.float32], MutUntrackedOrigin], n: Int, eps: Float32) abi("C") -> Float32:
    initialize_runtime()
    return inv_rms_impl(x, n, eps)


@export
def dllama_rmsnorm(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], x: Pointer[Scalar[DType.float32], MutUntrackedOrigin], w: Pointer[Scalar[DType.float32], MutUntrackedOrigin], dim: Int,
                   eps: Float32) abi("C"):
    initialize_runtime()
    var inv = inv_rms_impl(x, dim, eps)
    for i in range(dim):
        dst[unsafe_offset=i] = w[unsafe_offset=i] * (inv * x[unsafe_offset=i])


@export
def dllama_softmax(x: Pointer[Scalar[DType.float32], MutUntrackedOrigin], n: Int) abi("C"):
    initialize_runtime()
    softmax_impl(x, n)


@export
def dllama_activated_mul(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], gate: Pointer[Scalar[DType.float32], MutUntrackedOrigin], up: Pointer[Scalar[DType.float32], MutUntrackedOrigin], n: Int,
                         gelu: UInt32) abi("C"):
    initialize_runtime()
    if gelu != 0:
        # gelu (cpu.rs): g = 0.5*v*expf(1 + sqrt(2/pi)*(v + 0.044715*v^3))
        var c = sqrt(Float32(2.0) / Float32(3.14159265358979))
        for i in range(n):
            var v = gate[unsafe_offset=i]
            var arg = 1.0 + c * (v + 0.044715 * v * v * v)
            var g = 0.5 * v * external_call["expf", Float32](arg)
            dst[unsafe_offset=i] = g * up[unsafe_offset=i]
    else:
        for i in range(n):
            var g = gate[unsafe_offset=i] / (1.0 + expf_avx2_scalar(
                -gate[unsafe_offset=i]))
            dst[unsafe_offset=i] = g * up[unsafe_offset=i]


@export
def dllama_dequant_row(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], bytes: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], kind: UInt32, k: Int) abi("C"):
    initialize_runtime()
    if kind == 0:  # F32
        var src = bytes.unsafe_bitcast[Scalar[DType.float32]]()
        for i in range(k):
            dst[unsafe_offset=i] = src[unsafe_offset=i]
    elif kind == 1:  # F16
        for i in range(k):
            dst[unsafe_offset=i] = f16_to_f32(load_u16_le(bytes, 2 * i))
    elif kind == 2 or kind == 3:  # DllamaQ40 / GgufQ4_0
        dequant_q40(bytes, dst, k // 32)
    elif kind == 4:  # GgufQ8_0
        dequant_q8_0(bytes, dst, k // 32)
    elif kind == 5:  # GgufQ4K
        dequant_q4_k(bytes, dst, k // 256)
    elif kind == 6:  # GgufQ6K
        dequant_q6_k(bytes, dst, k // 256)


@export
def dllama_matmul_q80(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], xq80: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], w: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], kind: UInt32,
                      d: Int, n: Int) abi("C"):
    initialize_runtime()
    if kind == 2 or kind == 3:  # DllamaQ40 / GgufQ4_0
        matmul_q80_q40_impl(dst, xq80, w, d, n)
    elif kind == 4:  # GgufQ8_0
        matmul_q80_q8_0_impl(dst, xq80, w, d, n)
    elif kind == 5:  # GgufQ4K
        matmul_q80_q4_k_impl(dst, xq80, w, d, n)
    elif kind == 6:  # GgufQ6K
        matmul_q80_q6_k_impl(dst, xq80, w, d, n)
    else:  # F16 / F32 weights: dequant both sides + f32 dot
        var xq_a = alloc(Layout[Float32](count=n))
        var xq = xq_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        dequant_row_q80_bytes(xq80, xq, n)
        var wrow_a = alloc(Layout[Float32](count=n))
        var wrow = wrow_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        var row_bytes = n * 4
        if kind == 1:
            row_bytes = n * 2
        for di in range(d):
            if kind == 0:
                var src = w.unsafe_offset(di * row_bytes).unsafe_bitcast[
                    Scalar[DType.float32]]()
                for i in range(n):
                    wrow[unsafe_offset=i] = src[unsafe_offset=i]
            else:
                for i in range(n):
                    wrow[unsafe_offset=i] = f16_to_f32(load_u16_le(
                        w.unsafe_offset(di * row_bytes), 2 * i))
            dst[unsafe_offset=di] = dot_product(xq, wrow, n)
        dealloc(wrow_a^)
        dealloc(xq_a^)


@export
def dllama_matmul_f32(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], x: Pointer[Scalar[DType.float32], MutUntrackedOrigin], w: Pointer[Scalar[DType.float32], MutUntrackedOrigin], m: Int, n: Int,
                      k: Int) abi("C"):
    initialize_runtime()
    for mi in range(m):
        var xr = x.unsafe_offset(mi * k)
        var orow = dst.unsafe_offset(mi * n)
        for ni in range(n):
            orow[unsafe_offset=ni] = dot4(xr, w.unsafe_offset(ni * k), k)


@export
def dllama_matmul_q40_f32(dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin], x: Pointer[Scalar[DType.float32], MutUntrackedOrigin], w: Pointer[Scalar[DType.uint8], MutUntrackedOrigin], m: Int, n: Int,
                          k: Int) abi("C"):
    initialize_runtime()
    var row_bytes = (k // 32) * 18
    var row_a = alloc(Layout[Float32](count=k))
    var row = row_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    for mi in range(m):
        var xr = x.unsafe_offset(mi * k)
        var orow = dst.unsafe_offset(mi * n)
        for ni in range(n):
            dequant_q40(w.unsafe_offset(ni * row_bytes), row, k // 32)
            orow[unsafe_offset=ni] = dot4(xr, row, k)
    dealloc(row_a^)


@export
def dllama_attention(k_cache: Pointer[Scalar[DType.float32], MutUntrackedOrigin], v_cache: Pointer[Scalar[DType.float32], MutUntrackedOrigin], kv_dim: Int, q: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
                     k: Pointer[Scalar[DType.float32], MutUntrackedOrigin], v: Pointer[Scalar[DType.float32], MutUntrackedOrigin], pos: Int, n_heads: Int,
                     n_kv_heads: Int, head_dim: Int, dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin]) abi("C"):
    initialize_runtime()
    # append k/v at pos (engine-owned cache slices)
    for i in range(kv_dim):
        k_cache[unsafe_offset=pos * kv_dim + i] = k[unsafe_offset=i]
        v_cache[unsafe_offset=pos * kv_dim + i] = v[unsafe_offset=i]
    var q_per_kv = n_heads // n_kv_heads
    var head_dim_root = sqrt(Float32(head_dim))
    var scores_a = alloc(Layout[Float32](count=pos + 1))
    var scores = scores_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    for h in range(n_heads):
        var kv_h = h // q_per_kv
        var qh = q.unsafe_offset(h * head_dim)
        for t in range(pos + 1):
            var kh = k_cache.unsafe_offset(t * kv_dim + kv_h * head_dim)
            scores[unsafe_offset=t] = dot_product(qh, kh, head_dim) / head_dim_root
        softmax_impl(scores, pos + 1)
        var oh = dst.unsafe_offset(h * head_dim)
        for i in range(head_dim):
            oh[unsafe_offset=i] = 0.0
        for t in range(pos + 1):
            var vh = v_cache.unsafe_offset(t * kv_dim + kv_h * head_dim)
            var s = scores[unsafe_offset=t]
            for i in range(head_dim):
                oh[unsafe_offset=i] = fma(s, vh[unsafe_offset=i], oh[unsafe_offset=i])
    dealloc(scores_a^)
