"""Phase 2 test: non-uniform FFN shard graph builds, compiles, and runs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from max.graph import DeviceRef

from ffn.builder import ShardSpec, build_ffn_graph, compile_and_run


def test_ffn_graph_3_node_partition():
    """Simulate a 3-node non-uniform partition of a 3072-wide FFN.

    Node 0: dims 0    - 1024
    Node 1: dims 1024 - 2048
    Node 2: dims 2048 - 3072
    """
    hidden_dim = 768
    ffn_total = 3072
    seq_len = 4

    shards = [
        ShardSpec(0, 1024),
        ShardSpec(1024, 2048),
        ShardSpec(2048, 3072),
    ]

    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, seq_len, hidden_dim)).astype(np.float32)

    # Full weights (as if from a checkpoint)
    ffn_up_full = rng.standard_normal((hidden_dim, ffn_total)).astype(np.float32)
    ffn_down_full = rng.standard_normal((ffn_total, hidden_dim)).astype(np.float32)

    partials = []
    for shard in shards:
        width = shard.ffn_width()
        assert width == 1024, f"expected uniform 1024-wide shards, got {width}"

        # Slice the weights for this node's non-uniform shard
        up_slice = ffn_up_full[:, shard.ffn_dim_start : shard.ffn_dim_end]
        down_slice = ffn_down_full[shard.ffn_dim_start : shard.ffn_dim_end, :]

        # Build & run on CPU
        out = compile_and_run(
            shard,
            hidden_dim,
            x,
            up_slice,
            down_slice,
            device=DeviceRef.cpu(),
        )
        assert out.shape == (1, seq_len, hidden_dim), (
            f"Node {shard.ffn_dim_start//1024}: bad output shape {out.shape}"
        )
        partials.append(out)

    # Sum partials and compare against the reference full-FFN computation.
    combined = sum(partials)

    # Reference: x @ ffn_up_full -> silu -> @ ffn_down_full
    h_ref = x @ ffn_up_full            # [1, seq, ffn_total]
    h_ref = h_ref / (1.0 + np.exp(-h_ref))  # silu
    ref = h_ref @ ffn_down_full        # [1, seq, hidden]

    np.testing.assert_allclose(combined, ref, rtol=1e-4, atol=1e-4)
    print("Phase 2 FFN builder test passed.")


if __name__ == "__main__":
    test_ffn_graph_3_node_partition()
