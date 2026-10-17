#!/usr/bin/env python3
# dump tensor names + quant kinds from a GGUF file
import struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else "../models/Qwen_Qwen3-0.6B-Q4_K_M.gguf"
data = open(path, "rb").read(64 * 1024 * 1024)  # header slice is enough

GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
              8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
              13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}

off = 0
magic = data[0:4]
assert magic == b"GGUF", magic
off = 4
(ver,) = struct.unpack_from("<I", data, off); off += 4
(ti_n,) = struct.unpack_from("<Q", data, off); off += 8
(kv_n,) = struct.unpack_from("<Q", data, off); off += 8

def read_str(off):
    (n,) = struct.unpack_from("<Q", data, off); off += 8
    s = data[off:off+n]; off += n
    return s.decode(), off

for _ in range(kv_n):
    _, off = read_str(off)
    (t,) = struct.unpack_from("<I", data, off); off += 4
    if t == 0: off += 1      # u8
    elif t == 1: off += 1    # int8
    elif t == 2: off += 2    # uint16
    elif t == 3: off += 2    # int16
    elif t == 4: off += 4    # uint32
    elif t == 5: off += 4    # int32
    elif t == 6: off += 4    # float32
    elif t == 7: off += 1    # bool
    elif t == 8: _, off = read_str(off)  # string
    elif t == 9:             # array
        (et,) = struct.unpack_from("<I", data, off); off += 4
        (cnt,) = struct.unpack_from("<Q", data, off); off += 8
        if et == 8:           # array of strings
            for _ in range(cnt):
                _, off = read_str(off)
        else:
            esz = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}.get(et, 4)
            off += cnt * esz
    elif t == 10: off += 8    # uint64
    elif t == 11: off += 8    # int64
    elif t == 12: off += 8    # float64
    else:
        raise SystemExit(f"unknown kv type {t} at {off}")

print(f"GGUF v{ver}, {ti_n} tensors")
seen = {}
for _ in range(ti_n):
    name, off = read_str(off)
    (dim_n,) = struct.unpack_from("<I", data, off); off += 4
    dims = []
    for _ in range(dim_n):
        (d,) = struct.unpack_from("<Q", data, off); off += 8
        dims.append(d)
    (ty,) = struct.unpack_from("<I", data, off); off += 4
    (pos,) = struct.unpack_from("<Q", data, off); off += 8
    kind = GGML_TYPES.get(ty, f"ty{ty}")
    seen.setdefault(kind, 0)
    seen[kind] += 1
    if name.startswith(("blk.0.", "token_embd", "output")) and dim_n:
        print(f"  {name:28s} {kind:6s} dims={dims}")
print("kind histogram:", seen)
