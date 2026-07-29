from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.math import sqrt, exp, tanh, cos, sin

# ── helpers ──────────────────────────────────────────────────────────────────

def _ptr_f32(arr: PythonObject) raises -> UnsafePointer[Float32, MutUntrackedOrigin]:
    var data = arr.__array_interface__["data"]
    var addr = Int(py=data[0])
    return UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)

def _dim(arr: PythonObject, i: Int) raises -> Int:
    return Int(py=arr.shape[i])

# ── rms_norm ─────────────────────────────────────────────────────────────────

def rms_norm(x: PythonObject, weight: PythonObject, result: PythonObject) raises -> PythonObject:
    var xp = _ptr_f32(x)
    var wp = _ptr_f32(weight)
    var rp = _ptr_f32(result)
    var B = _dim(x, 0)
    var S = _dim(x, 1)
    var D = _dim(x, 2)
    for b in range(B):
        for s in range(S):
            var offset = (b * S + s) * D
            var sq: Float64 = 0.0
            for d in range(D):
                var v = Float64(xp[offset + d])
                sq += v * v
            var rstd = Float64(1.0) / sqrt(sq / Float64(D) + Float64(1e-6))
            for d in range(D):
                rp[offset + d] = Float32(Float64(xp[offset + d]) * rstd * Float64(wp[d]))
    return result

# ── softmax (last axis) ──────────────────────────────────────────────────────

def softmax(x: PythonObject, result: PythonObject) raises -> PythonObject:
    var xp = _ptr_f32(x)
    var rp = _ptr_f32(result)
    var total = Int(py=x.size)
    var last_dim = Int(py=x.shape[Int(py=len(x.shape)) - 1])
    var n_rows = total // last_dim
    for row in range(n_rows):
        var base = row * last_dim
        var mx: Float32 = -1e10
        for j in range(last_dim):
            var v = xp[base + j]
            if v > mx:
                mx = v
        var sum_exp: Float32 = 0.0
        for j in range(last_dim):
            sum_exp += exp(xp[base + j] - mx)
        var inv_sum = Float32(1.0) / sum_exp
        for j in range(last_dim):
            rp[base + j] = exp(xp[base + j] - mx) * inv_sum
    return result

# ── RoPE ─────────────────────────────────────────────────────────────────────

def apply_rope(
    x: PythonObject, result: PythonObject,
    rope_theta: PythonObject, start_pos: PythonObject,
    rope_fraction: PythonObject,
) raises -> PythonObject:
    var xp = _ptr_f32(x)
    var rp = _ptr_f32(result)
    var B = _dim(x, 0)
    var H = _dim(x, 1)
    var S = _dim(x, 2)
    var D = _dim(x, 3)
    var theta = Float64(py=rope_theta)
    var spos = Int(py=start_pos)
    var frac = Float64(py=rope_fraction)
    var dims = Int(Float64(D) * frac)
    if dims < 2:
        for i in range(B * H * S * D):
            rp[i] = xp[i]
        return result
    var half = dims // 2
    for b in range(B):
        for h in range(H):
            for pos in range(S):
                var base = ((b * H + h) * S + pos) * D
                for i in range(half):
                    var freq = Float64(1.0) / (Float64(theta) ** (Float64(2 * i) / Float64(dims)))
                    var c = Float32(cos(Float64(spos + pos) * freq))
                    var si = Float32(sin(Float64(spos + pos) * freq))
                    var x1 = xp[base + i]
                    var x2 = xp[base + half + i]
                    rp[base + i] = x1 * c - x2 * si
                    rp[base + half + i] = x1 * si + x2 * c
                for i in range(dims, D):
                    rp[base + i] = xp[base + i]
    return result

# ── V_norm ───────────────────────────────────────────────────────────────────

def apply_v_norm(v: PythonObject, result: PythonObject) raises -> PythonObject:
    var vp = _ptr_f32(v)
    var rp = _ptr_f32(result)
    var B = _dim(v, 0)
    var H = _dim(v, 1)
    var S = _dim(v, 2)
    var D = _dim(v, 3)
    var total = B * H * S
    for i in range(total):
        var offset = i * D
        var sq: Float64 = 0.0
        for d in range(D):
            var vv = Float64(vp[offset + d])
            sq += vv * vv
        var rstd = Float64(1.0) / sqrt(sq / Float64(D) + Float64(1e-6))
        for d in range(D):
            rp[offset + d] = Float32(Float64(vp[offset + d]) * rstd)
    return result

# ── SiLU multiply (gate * sigmoid(gate) * up) ───────────────────────────────

def silu_mul(gate: PythonObject, up: PythonObject, result: PythonObject) raises -> PythonObject:
    var gp = _ptr_f32(gate)
    var up_ = _ptr_f32(up)
    var rp = _ptr_f32(result)
    var n = Int(py=gate.size)
    for i in range(n):
        var g = gp[i]
        rp[i] = g / (Float32(1.0) + exp(-g)) * up_[i]
    return result

# ── matmul_f32 (naive, for decode path single-token) ────────────────────────

def matmul_f32(a: PythonObject, b: PythonObject, result: PythonObject) raises -> PythonObject:
    var ap = _ptr_f32(a)
    var bp = _ptr_f32(b)
    var rp = _ptr_f32(result)
    var M = _dim(a, 0)
    var K = _dim(a, 1)
    var N = _dim(b, 1)
    for i in range(M):
        for j in range(N):
            var acc: Float32 = 0.0
            for k in range(K):
                acc += ap[i * K + k] * bp[k * N + j]
            rp[i * N + j] = acc
    return result

# ── batched matmul (3D: [B, M, K] @ [B, K, N] → [B, M, N]) ──────────────

def bmm_f32(a: PythonObject, b: PythonObject, result: PythonObject) raises -> PythonObject:
    var ap = _ptr_f32(a)
    var bp = _ptr_f32(b)
    var rp = _ptr_f32(result)
    var B = _dim(a, 0)
    var M = _dim(a, 1)
    var K = _dim(a, 2)
    var N = _dim(b, 2)
    for batch in range(B):
        var a_off = batch * M * K
        var b_off = batch * K * N
        var o_off = batch * M * N
        for i in range(M):
            for j in range(N):
                var acc: Float32 = 0.0
                for k in range(K):
                    acc += ap[a_off + i * K + k] * bp[b_off + k * N + j]
                rp[o_off + i * N + j] = acc
    return result

# ── Q8_0 dequantize ─────────────────────────────────────────────────────────

def q80_dequantize(
    qdata: PythonObject, scales: PythonObject, result: PythonObject,
) raises -> PythonObject:
    var qp = UnsafePointer[Int8, MutUntrackedOrigin](
        unsafe_from_address=Int(py=qdata.__array_interface__["data"][0])
    )
    var sp = _ptr_f32(scales)
    var rp = _ptr_f32(result)
    alias BS: Int = 32
    var n = Int(py=result.size)
    for i in range(n):
        var bi = i // BS
        rp[i] = Float32(qp[i]) * sp[bi]
    return result

# ── GELU ─────────────────────────────────────────────────────────────────────

def gelu(x: PythonObject, result: PythonObject) raises -> PythonObject:
    var xp = _ptr_f32(x)
    var rp = _ptr_f32(result)
    var n = Int(py=x.size)
    var sqrt_2pi = Float32(sqrt(Float64(2.0) / Float64(3.141592653589793)))
    for i in range(n):
        var v = xp[i]
        var v3 = v * v * v
        rp[i] = Float32(0.5) * v * (Float32(1.0) + tanh(sqrt_2pi * (v + Float32(0.044715) * v3)))
    return result

# ── Broadcast add (a + b where b is [D] and a is [B, S, D]) ────────────────

def broadcast_add(a: PythonObject, b: PythonObject, result: PythonObject) raises -> PythonObject:
    var ap = _ptr_f32(a)
    var bp = _ptr_f32(b)
    var rp = _ptr_f32(result)
    var n = Int(py=a.size)
    var last_dim = Int(py=b.size)
    for i in range(n):
        rp[i] = ap[i] + bp[i % last_dim]
    return result

# ── Linear attention: depthwise conv1d + SiLU ─────────────────────────────
# qkv_pad: [B, S+dc-1, C] padded input (dc = d_conv)
# conv_w:  [C, 1, dc]     depthwise conv weights
# out:     [B, S, C]      output (written in-place, then SiLU applied)

def linear_attn_conv1d(
    qkv_pad: PythonObject, conv_w: PythonObject, result_: PythonObject,
) raises -> PythonObject:
    var qp = _ptr_f32(qkv_pad)
    var wp = _ptr_f32(conv_w)
    var rp = _ptr_f32(result_)
    var B = _dim(qkv_pad, 0)
    var S_pad = _dim(qkv_pad, 1)
    var C = _dim(qkv_pad, 2)
    var dc = _dim(conv_w, 2)
    var S = S_pad - dc + 1  # output seq len

    for b in range(B):
        for c in range(C):
            for t in range(S):
                var acc: Float32 = 0.0
                var base_pad = b * S_pad * C + t * C + c
                for k in range(dc):
                    var pad_idx = base_pad + k * C
                    var w_idx = c * dc + (dc - 1 - k)
                    acc += qp[pad_idx] * wp[w_idx]
                rp[b * S * C + t * C + c] = acc

    # Apply SiLU: result_ /= (1 + exp(-result_))
    var total = B * S * C
    for i in range(total):
        var v = rp[i]
        rp[i] = v / (Float32(1.0) + exp(-v))
    return result_

# ── Linear attention: SSM scan (Mamba2-style recurrent step) ──────────────
# x:  [B, S, D]    SSM input (x_ssm)
# Bp: [B, S, N]    input projection (B_proj)
# Cp: [B, S, N]    output projection (C_proj)
# A:  [N]          discretized A (A_bar)
# h:  [B, D, N]    state — read/write, modified in-place
# y:  [B, S, D]    output (written)

def linear_attn_ssm_scan(
    x: PythonObject, Bp: PythonObject, Cp: PythonObject,
    A: PythonObject, h: PythonObject, y: PythonObject,
) raises -> PythonObject:
    var xp = _ptr_f32(x)
    var bp = _ptr_f32(Bp)
    var cp = _ptr_f32(Cp)
    var ap = _ptr_f32(A)
    var hp = _ptr_f32(h)
    var yp = _ptr_f32(y)
    var B = _dim(x, 0)
    var S = _dim(x, 1)
    var D = _dim(x, 2)
    var N = _dim(Bp, 2)

    for t in range(S):
        # h[b,i,d] = h[b,i,d] * A[d] + B[b,t,d] * x[b,t,i]
        for b in range(B):
            for i in range(D):
                for d in range(N):
                    var h_idx = b * D * N + i * N + d
                    var a_val = ap[d]
                    var b_val = bp[b * S * N + t * N + d]
                    var x_val = xp[b * S * D + t * D + i]
                    hp[h_idx] = hp[h_idx] * a_val + b_val * x_val

        # y[b,t,i] = sum_d h[b,i,d] * C[b,t,d]
        for b in range(B):
            for i in range(D):
                var acc: Float32 = 0.0
                for d in range(N):
                    var h_idx = b * D * N + i * N + d
                    var c_val = cp[b * S * N + t * N + d]
                    acc += hp[h_idx] * c_val
                yp[b * S * D + t * D + i] = acc

    return y

# ── Module init ──────────────────────────────────────────────────────────────

@export
def PyInit_pyre_kernels() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("pyre_kernels")
        m.def_function[rms_norm]("rms_norm")
        m.def_function[softmax]("softmax")
        m.def_function[apply_rope]("apply_rope")
        m.def_function[apply_v_norm]("apply_v_norm")
        m.def_function[silu_mul]("silu_mul")
        m.def_function[matmul_f32]("matmul_f32")
        m.def_function[bmm_f32]("bmm_f32")
        m.def_function[q80_dequantize]("q80_dequantize")
        m.def_function[gelu]("gelu")
        m.def_function[broadcast_add]("broadcast_add")
        m.def_function[linear_attn_conv1d]("linear_attn_conv1d")
        m.def_function[linear_attn_ssm_scan]("linear_attn_ssm_scan")
        return m.finalize()
    except e:
        abort(String("failed to build pyre_kernels: ", e))
