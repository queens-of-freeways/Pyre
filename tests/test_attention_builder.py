from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from max.driver import CPU
from max.engine import InferenceSession
from max.graph import DeviceRef

from attention.builder import ShardSpec, build_ulysses_attention_graph


def test_ulysses_attention_graph_shapes():
    hidden_dim = 4096
    n_heads = 32
    n_kv_heads = 8
    head_dim = hidden_dim // n_heads  # 128
    
    # 4-node setup
    total_seq_len = 1024
    local_seq_len = total_seq_len // 4  # 256
    
    # Node 0 gets 8 Q heads (8 * 128 = 1024 width), and all 8 KV heads
    shard = ShardSpec(
        ffn_dim_start=0,
        ffn_dim_end=1024,
        seq_start=0,
        seq_end=local_seq_len
    )
    
    device = DeviceRef.CPU()
    graph = build_ulysses_attention_graph(
        shard, hidden_dim, n_heads, n_kv_heads, head_dim, device
    )
    
    session = InferenceSession(devices=[CPU()])
    model = session.load(graph)
    
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, local_seq_len, hidden_dim)).astype(np.float32)
    wq_slice = rng.standard_normal((hidden_dim, 8 * head_dim)).astype(np.float32)
    wk_full = rng.standard_normal((hidden_dim, n_kv_heads * head_dim)).astype(np.float32)
    wv_full = rng.standard_normal((hidden_dim, n_kv_heads * head_dim)).astype(np.float32)
    
    q_out, k_out, v_out = model.execute(x, wq_slice, wk_full, wv_full)
    
    q_np = q_out.to_numpy()
    k_np = k_out.to_numpy()
    v_np = v_out.to_numpy()
    
    assert q_np.shape == (1, local_seq_len, 8, head_dim), f"Q shape mismatch: {q_np.shape}"
    assert k_np.shape == (1, local_seq_len, n_kv_heads, head_dim), f"K shape mismatch: {k_np.shape}"
    assert v_np.shape == (1, local_seq_len, n_kv_heads, head_dim), f"V shape mismatch: {v_np.shape}"
    
    print("Phase 4 Ulysses attention builder test passed.")


if __name__ == "__main__":
    test_ulysses_attention_graph_shapes()
