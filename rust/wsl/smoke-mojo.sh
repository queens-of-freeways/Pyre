#!/bin/bash
set -e
cd /mnt/c/Users/kodur/distributed-llama/rust/mojo
ls -la libdllama_mojo.so
nm -D libdllama_mojo.so | grep " T dllama" | sort
echo "=== ctypes smoke test:"
python3 - <<'EOF'
import ctypes, math

lib = ctypes.CDLL("./libdllama_mojo.so")

lib.dllama_kernels_abi_version.restype = ctypes.c_uint32
assert lib.dllama_kernels_abi_version() == 1, "ABI version mismatch"

lib.dllama_kernels_name.restype = ctypes.c_char_p
name = lib.dllama_kernels_name()
print("backend name:", name)
assert b"mojo" in name, name

f32 = ctypes.c_float
lib.dllama_expf.argtypes = [f32]
lib.dllama_expf.restype = f32
# expf(0) == 1.0 exactly; expf(1) ~ e; ties-even affects y=log2e*x rounding
e0 = lib.dllama_expf(f32(0.0))
e1 = lib.dllama_expf(f32(1.0))
print(f"expf(0)={e0!r} expf(1)={e1!r}")
assert e0 == 1.0, "expf(0) must be exactly 1.0"
# the C++ degree-4 polynomial is ~4e-5 accurate (2.7182374 by hand) — that
# IS the engine's expf; bit-parity is proven by the engine trace-diff later
assert abs(e1 - 2.7182374) < 2e-5, e1

# bit-exactness is arbitrated by the engine trace-diff (the f64 python replica
# here cannot reproduce f32 rounding) — just check monotone sanity:
for probe in [-3.7, -0.5, 0.25, 1.234, 5.5, 20.0, 87.0]:
    got = lib.dllama_expf(f32(probe))
    want = math.exp(probe)
    rel = abs(got - want) / max(1e-30, abs(want))
    assert rel < 5e-5, f"expf({probe}) rel err {rel}"
print("expf polynomial sanity OK (max rel err within poly accuracy)")

# dot product vs naive (tolerance only — tree order differs from naive sum)
lib.dllama_dot.argtypes = [ctypes.POINTER(f32), ctypes.POINTER(f32), ctypes.c_size_t]
lib.dllama_dot.restype = f32
xs = [float(i % 7) - 3.0 for i in range(64)]
ys = [float((i * 3) % 11) - 5.0 for i in range(64)]
xa = (f32 * 64)(*xs); ya = (f32 * 64)(*ys)
got = lib.dllama_dot(xa, ya, 64)
want = sum(a * b for a, b in zip(xs, ys))
print("dot:", got, "naive:", want)
assert abs(got - want) < 1e-4 * max(1, abs(want))

# softmax
lib.dllama_softmax.argtypes = [ctypes.POINTER(f32), ctypes.c_size_t]
sa = (f32 * 5)(3.0, 1.0, 4.0, 1.0, 5.0)
lib.dllama_softmax(sa, 5)
s = list(sa)
print("softmax:", s)
assert abs(sum(s) - 1.0) < 1e-6 and abs(s[4] - max(s)) < 1e-9

print("ALL SMOKE TESTS PASSED")
EOF
