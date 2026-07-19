from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warnings

warnings.filterwarnings("ignore", "overflow encountered in exp")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ffn.builder import ShardSpec, compile_and_run


def test_ffn_graph_3_node_partition():
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
    ffn_up_full = rng.standard_normal((hidden_dim, ffn_total)).astype(np.float32)
    ffn_down_full = rng.standard_normal((ffn_total, hidden_dim)).astype(np.float32)

    partials = []
    for shard in shards:
        width = shard.ffn_width()
        assert width == 1024, f"expected 1024-wide shards, got {width}"

        up_slice = ffn_up_full[:, shard.ffn_dim_start : shard.ffn_dim_end]
        down_slice = ffn_down_full[shard.ffn_dim_start : shard.ffn_dim_end, :]

        out = compile_and_run(shard, hidden_dim, x, up_slice, down_slice)
        assert out.shape == (1, seq_len, hidden_dim), (
            f"bad output shape {out.shape}"
        )
        partials.append(out)

    combined = sum(partials)
    h_ref = x @ ffn_up_full
    h_ref = h_ref / (1.0 + np.exp(-h_ref))
    ref = h_ref @ ffn_down_full

    np.testing.assert_allclose(combined, ref, rtol=1e-3, atol=1e-3)
    print("Phase 2 FFN builder test passed.")


if __name__ == "__main__":
    test_ffn_graph_3_node_partition()
