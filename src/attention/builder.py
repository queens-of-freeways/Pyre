from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


@dataclass(frozen=True)
class ShardSpec:
    ffn_dim_start: int
    ffn_dim_end: int
    seq_start: int
    seq_end: int

    def ffn_width(self) -> int:
        return self.ffn_dim_end - self.ffn_dim_start

    def local_seq_len(self) -> int:
        return self.seq_end - self.seq_start


def build_ulysses_attention_graph(
    shard: ShardSpec,
    hidden_dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    device: DeviceRef,
) -> Graph:
    local_seq_len = shard.local_seq_len()
    n_q_heads_local = shard.ffn_width() // head_dim

    x_type = TensorType(DType.float32, [1, local_seq_len, hidden_dim], device=device)
    wq_slice_type = TensorType(DType.float32, [hidden_dim, n_q_heads_local * head_dim], device=device)
    wk_full_type = TensorType(DType.float32, [hidden_dim, n_kv_heads * head_dim], device=device)
    wv_full_type = TensorType(DType.float32, [hidden_dim, n_kv_heads * head_dim], device=device)

    with Graph(
        "ulysses_attention_shard",
        input_types=[x_type, wq_slice_type, wk_full_type, wv_full_type],
    ) as g:
        x, wq_slice, wk_full, wv_full = g.inputs
        
        q = ops.matmul(x, wq_slice)
        k = ops.matmul(x, wk_full)
        v = ops.matmul(x, wv_full)
        
        q = ops.reshape(q, [1, local_seq_len, n_q_heads_local, head_dim])
        k = ops.reshape(k, [1, local_seq_len, n_kv_heads, head_dim])
        v = ops.reshape(v, [1, local_seq_len, n_kv_heads, head_dim])
        
        g.output(q, k, v)

    return g
