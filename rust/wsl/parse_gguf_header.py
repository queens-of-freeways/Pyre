"""Fetch the header of the Ornith-1.5-9B Q4_K_M GGUF and dump ALL metadata
key/type pairs as they parse (with drift context), then the first tensor
names if reachable. Evidence for architecture-support analysis.
"""
import struct
import urllib.request

URL = 'https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/resolve/main/Ornith-1.5-9B-Q4_K_M.gguf'
RANGE = 12 * 1024 * 1024

req = urllib.request.Request(URL, headers={
    'Range': f'bytes=0-{RANGE - 1}',
    'User-Agent': 'Mozilla/5.0',
})
data = urllib.request.urlopen(req, timeout=120).read()
print(f'fetched {len(data) / 1e6:.1f} MB of the file header')

off = 0


def u32():
    global off
    v = struct.unpack_from('<I', data, off)[0]
    off += 4
    return v


def u64():
    global off
    v = struct.unpack_from('<Q', data, off)[0]
    off += 8
    return v


def rd(n):
    global off
    b = data[off:off + n]
    off += n
    return b


assert data[:4] == b'GGUF', 'not a GGUF file'
off = 4
ver = u32()
tc = u64()
kvc = u64()
print(f'GGUF v{ver}, tensors={tc}, kv_pairs={kvc}')

VT = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8, 13: 2}


def read_val(t, depth=0):
    if depth > 2:
        return '<nested>'
    if t == 8:
        n = u32()
        if n > 1_000_000:
            raise ValueError(f'string length too big: {n}')
        return rd(n).decode('utf-8', errors='replace')
    if t == 9:
        et = u32()
        cnt = u64()
        vals = []
        for _ in range(min(cnt, 3)):
            vals.append(read_val(et, depth + 1))
        if cnt > 3:
            vals.append(f'...({cnt} elems)')
        return vals
    n = VT.get(t)
    if n is None:
        raise ValueError(f'unknown value type {t}')
    raw = rd(n)
    if t == 6:
        return struct.unpack('<f', raw)[0]
    if t == 12:
        return struct.unpack('<d', raw)[0]
    if t == 7:
        return bool(raw[0])
    fmt = {1: '<B', 2: '<H', 4: '<I', 8: '<Q'}[n]
    return struct.unpack(fmt, raw)[0]


try:
    for i in range(kvc):
        knlen = u32()
        if knlen > 10_000:
            raise ValueError(f'key length implausible: {knlen} (parser drift)')
        key = rd(knlen).decode('utf-8', errors='replace')
        t = u32()
        try:
            val = read_val(t)
            print(f'[{i:2d}] {key} (type {t}) = {repr(val)[:200]}')
        except ValueError as ex:
            print(f'[{i:2d}] {key} (type {t}) -> VALUE PARSE ERROR: {ex}')
            break
except (IndexError, ValueError, struct.error) as ex:
    print(f'(parse stopped: {ex})')

print('\n-- first tensor names --')
names = []
try:
    for _ in range(min(tc, 80)):
        nlen = u32()
        name = rd(nlen).decode('utf-8', errors='replace')
        n_dims = u32()
        for _ in range(n_dims):
            u64()
        u32()
        u64()
        names.append(name)
except (IndexError, ValueError, struct.error):
    pass
for n in names:
    print(' ', n)
if not names:
    print('  (tensor directory beyond fetched range)')
