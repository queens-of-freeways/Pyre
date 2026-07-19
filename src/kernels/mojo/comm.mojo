# Mock communication kernels for distributed-llama.
# BFloat16 is stored as UInt16; q80 is stored as UInt8.

# q80: 1 sign bit + 7 exponent bits (truncated from bfloat16)
# bfloat16: 1 sign + 8 exponent + 7 mantissa

def compress_q80(
    input: UnsafePointer[UInt16, MutUntrackedOrigin],
    length: Int,
) -> UnsafePointer[UInt8, MutUntrackedOrigin]:
    var out = alloc[UInt8](length)
    for i in range(length):
        var v = input[i]
        var sign = UInt8((v >> 8) & 0x80)
        var exp = UInt8(v >> 7) & 0x7F
        out[i] = sign | exp
    return out

def decompress_q80(
    input: UnsafePointer[UInt8, MutUntrackedOrigin],
    length: Int,
) -> UnsafePointer[UInt16, MutUntrackedOrigin]:
    var out = alloc[UInt16](length)
    for i in range(length):
        var q = input[i]
        var sign = UInt16(q & 0x80) << 8
        var exp = UInt16(q & 0x7F) << 7
        out[i] = sign | exp
    return out

def ring_all_reduce_accumulate(
    local: UnsafePointer[UInt16, MutUntrackedOrigin],
    incoming: UnsafePointer[UInt16, MutUntrackedOrigin],
    length: Int,
):
    for i in range(length):
        local[i] += incoming[i]

def all_to_all_dispatch(
    chunk: UnsafePointer[UInt16, MutUntrackedOrigin],
    length: Int,
    target_node_idx: Int,
) -> UnsafePointer[UInt16, MutUntrackedOrigin]:
    var out = alloc[UInt16](length)
    for i in range(length):
        out[i] = chunk[i]
    return out
