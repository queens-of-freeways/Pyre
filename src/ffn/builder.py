from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.driver import CPU


@dataclass(frozen=True)
class ShardSpec:
    ffn_dim_start: int
    ffn_dim_end: int

    def ffn_width(self) -> int:
        return self.ffn_dim_end - self.ffn_dim_start


def build_ffn_graph(
    shard: ShardSpec,
    hidden_dim: int,
    device: DeviceRef,
    *,
    seq_len: int = 1,
    name: str = "ffn_shard",
) -> Graph:
    width = shard.ffn_width()
    if width <= 0:
        raise ValueError(f"ShardSpec width must be positive, got {width}")

    x_type = TensorType(DType.float32, [1, seq_len, hidden_dim], device=device)
    ffn_up_slice_type = TensorType(DType.float32, [hidden_dim, width], device=device)
    ffn_down_slice_type = TensorType(DType.float32, [width, hidden_dim], device=device)

    with Graph(name, input_types=[x_type, ffn_up_slice_type, ffn_down_slice_type]) as g:
        x, ffn_up_slice, ffn_down_slice = g.inputs
        h = ops.matmul(x, ffn_up_slice)
        h = ops.silu(h)
        partial = ops.matmul(h, ffn_down_slice)
        g.output(partial)

    return g


def compile_and_run(
    shard: ShardSpec,
    hidden_dim: int,
    x,
    ffn_up_slice,
    ffn_down_slice,
    *,
    seq_len: int | None = None,
    device: DeviceRef | None = None,
):
    import numpy as np
    from max.driver import CPU

    if seq_len is None:
        seq_len = x.shape[1]
    if device is None:
        device = DeviceRef.CPU()

    graph = build_ffn_graph(shard, hidden_dim, device, seq_len=seq_len)

    session = InferenceSession(devices=[CPU()])
    model = session.load(graph)
    (out,) = model.execute(
        np.ascontiguousarray(x),
        np.ascontiguousarray(ffn_up_slice),
        np.ascontiguousarray(ffn_down_slice),
    )
    return out.to_numpy()
