#!/usr/bin/env python3
# Bit-level fuzz: Rust reference cdylib vs Mojo backend, per quant kind.
import ctypes, struct, random

f32 = ctypes.c_float
rust = ctypes.CDLL("/home/sid/dllama/rust/target/release/libdllama_kernel_c.so")
mojo = ctypes.CDLL("/mnt/c/Users/kodur/distributed-llama/rust/mojo/libdllama_mojo.so")
for lib in (rust, mojo):
    lib.dllama_matmul_q80.argtypes = [ctypes.POINTER(f32), ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32, ctypes.c_size_t, ctypes.c_size_t]
    lib.dllama_matmul_q80.restype = None

KINDS = {3: "Q4_0", 4: "Q8_0", 5: "Q4_K", 6: "Q6_K", 1: "F16", 0: "F32"}

def bits(buf):
    return struct.pack(f"<{len(buf)}f", *buf)

random.seed(42)
for kind, name in KINDS.items():
    n = 1024
    d = 64
    # q80 input: n/32 blocks x 34 bytes
    x = bytes(random.randrange(256) for _ in range((n // 32) * 34))
    # weight bytes: random (both sides decode identically; scales may be garbage
    # but deterministic)
    if name == "Q4_K": rb = (n // 256) * 144
    elif name == "Q6_K": rb = (n // 256) * 210
    elif name == "F16": rb = n * 2
    elif name == "F32": rb = n * 4
    else: rb = (n // 32) * (34 if name == "Q8_0" else 18)
    w = bytes(random.randrange(256) for _ in range(rb * d))
    out_r = (f32 * d)()
    out_m = (f32 * d)()
    rust.dllama_matmul_q80(out_r, (ctypes.c_uint8 * len(x)).from_buffer_copy(x),
                           (ctypes.c_uint8 * len(w)).from_buffer_copy(w), kind, d, n)
    mojo.dllama_matmul_q80(out_m, (ctypes.c_uint8 * len(x)).from_buffer_copy(x),
                           (ctypes.c_uint8 * len(w)).from_buffer_copy(w), kind, d, n)
    same = bits(out_r) == bits(out_m)
    ndiff = sum(1 for a, b in zip(out_r, out_m)
                if struct.pack("<f", a) != struct.pack("<f", b))
    print(f"{name:5s}: {'BIT-EXACT' if same else f'DIFF in {ndiff}/{d} rows'}")
    if not same:
        for i, (a, b) in enumerate(zip(out_r, out_m)):
            if struct.pack("<f", a) != struct.pack("<f", b):
                print(f"   row {i}: rust={a!r} mojo={b!r}")
                break
