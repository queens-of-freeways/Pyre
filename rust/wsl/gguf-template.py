#!/usr/bin/env python3
import struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else "../models/Qwen_Qwen3-0.6B-Q4_K_M.gguf"
data = open(path, "rb").read(8 * 1024 * 1024)

off = 4
(ver,) = struct.unpack_from("<I", data, off); off += 4
(ti_n,) = struct.unpack_from("<Q", data, off); off += 8
(kv_n,) = struct.unpack_from("<Q", data, off); off += 8

def read_str(off):
    (n,) = struct.unpack_from("<Q", data, off); off += 8
    s = data[off:off+n]; off += n
    return s.decode(), off

VT_SIZES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8, 13:2}

for _ in range(kv_n):
    name, off = read_str(off)
    (t,) = struct.unpack_from("<I", data, off); off += 4
    if t == 8:  # string
        val, off = read_str(off)
        if name == "tokenizer.chat_template":
            print("=== tokenizer.chat_template:")
            for i, line in enumerate(val.split("\n"), 1):
                print(f"{i:3}: {line}")
            sys.exit(0)
    elif t == 9:  # array
        (et,) = struct.unpack_from("<I", data, off); off += 4
        (cnt,) = struct.unpack_from("<Q", data, off); off += 8
        if et == 8:
            for _ in range(cnt):
                _, off = read_str(off)
        else:
            off += cnt * VT_SIZES.get(et, 4)
    else:
        off += VT_SIZES.get(t, 4)
print("no chat_template found in metadata")
