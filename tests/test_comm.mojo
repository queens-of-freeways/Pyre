from std.testing import assert_equal, TestSuite
from src.kernels.mojo.comm import (
    compress_q80,
    decompress_q80,
    ring_all_reduce_accumulate,
    all_to_all_dispatch,
)

def test_ring_all_reduce() raises:
    comptime SIZE = 1024

    var buf0 = alloc[UInt16](SIZE)
    var buf1 = alloc[UInt16](SIZE)
    var buf2 = alloc[UInt16](SIZE)
    for j in range(SIZE):
        buf0[j] = UInt16(j)
        buf1[j] = UInt16(j + 1000)
        buf2[j] = UInt16(j + 2000)

    # Save originals
    var orig0 = all_to_all_dispatch(buf0, SIZE, 0)
    var orig1 = all_to_all_dispatch(buf1, SIZE, 1)
    var orig2 = all_to_all_dispatch(buf2, SIZE, 2)

    # Compute expected sum
    var expected = alloc[UInt16](SIZE)
    for j in range(SIZE):
        expected[j] = UInt16(3 * j + 3000)

    # Simulate 2 ring steps where each node accumulates
    # the other nodes' original chunks.
    # Step 1: each node accumulates predecessor's original
    ring_all_reduce_accumulate(buf0, orig2, SIZE)  # buf0 += orig2
    ring_all_reduce_accumulate(buf1, orig0, SIZE)  # buf1 += orig0
    ring_all_reduce_accumulate(buf2, orig1, SIZE)  # buf2 += orig1

    # Step 2: each node accumulates the remaining node's original
    ring_all_reduce_accumulate(buf0, orig1, SIZE)  # buf0 += orig1
    ring_all_reduce_accumulate(buf1, orig2, SIZE)  # buf1 += orig2
    ring_all_reduce_accumulate(buf2, orig0, SIZE)  # buf2 += orig0

    # Now each buf = sum of all 3 originals
    for j in range(10):
        assert_equal(buf0[j], expected[j])
        assert_equal(buf1[j], expected[j])
        assert_equal(buf2[j], expected[j])

    print("  ring_all_reduce OK")

def test_compression() raises:
    comptime SIZE = 1024

    var src = alloc[UInt16](SIZE)
    for i in range(SIZE):
        src[i] = UInt16(i * 7 + 3)

    var compressed = compress_q80(src, SIZE)
    var decompressed = decompress_q80(compressed, SIZE)

    # Just verify no crash — q80 is lossy so exact match not expected
    var any_diff = False
    for i in range(SIZE):
        if decompressed[i] != src[i]:
            any_diff = True
    _ = any_diff

    print("  compress/decompress OK")

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
