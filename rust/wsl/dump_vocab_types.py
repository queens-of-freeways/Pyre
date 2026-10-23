"""Dump tokenizer.ggml.types==6 (BYTE) token pieces from a GGUF fixture."""
import struct
import sys

path = sys.argv[1]
data = open(path, "rb").read()
assert data[:4] == b"GGUF"
ver, tc, kvc = struct.unpack_from("<IQQ", data, 4)
VT = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8, 13: 2}
off = 24


def rd(n):
    global off
    b = data[off : off + n]
    off += n
    return b


kvs = {}
for _ in range(kvc):
    n = struct.unpack("<Q", rd(8))[0]
    key = rd(n).decode()
    t = struct.unpack("<I", rd(4))[0]
    if t == 8:
        ln = struct.unpack("<Q", rd(8))[0]
        val = rd(ln)
    elif t == 9:
        et = struct.unpack("<I", rd(4))[0]
        cnt = struct.unpack("<Q", rd(8))[0]
        vals = []
        for _ in range(cnt):
            if et == 8:
                ln = struct.unpack("<Q", rd(8))[0]
                vals.append(rd(ln))
            else:
                fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}[et]
                vals.append(struct.unpack("<" + fmt, rd(VT[et]))[0])
        val = vals
    else:
        val = struct.unpack("<" + {1: "B", 2: "H", 4: "I", 8: "Q"}[VT[t]], rd(VT[t]))[0]
    kvs[key] = val

toks = kvs.get("tokenizer.ggml.tokens", [])
types = kvs.get("tokenizer.ggml.types", [])
print("vocab:", len(toks), "types:", len(types))
byte_ids = [i for i, t in enumerate(types) if t == 6]
print("BYTE-typed count:", len(byte_ids))
for i in byte_ids[:6]:
    print(f"  id {i}: {toks[i]!r}")
ctrl = [i for i, t in enumerate(types) if t == 3]
print("CONTROL-typed count:", len(ctrl))
for i in ctrl[:4]:
    print(f"  id {i}: {toks[i]!r}")
unk = [i for i, t in enumerate(types) if t == 2]
print("UNKNOWN-typed:", [(i, toks[i]) for i in unk[:4]])
