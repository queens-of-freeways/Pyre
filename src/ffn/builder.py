"""Phase 2: Python max.graph FFN builder with non-uniform ShardSpec.

Builds a per-node FFN sub-graph that computes a PARTIAL result for a
non-uniform slice of the FFN intermediate dimension.  The caller is
responsible for combining partial results across nodes (e.g. via the
Mojo ring-all-reduce kernel from Phase 3).
"""
from __future__ import annotations

from dataclasses import dataclass

from max.driver import Accelerator, CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


# ---------------------------------------------------------------------------
# ShardSpec - Python mirror of the Mojo struct in src/partitioner/solver.mojo
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShardSpec:
    """Non-uniform shard of the FFN intermediate dimension.

    Attributes:
        ffn_dim_start: Inclusive start index into the FFN intermediate dim.
        ffn_dim_end:   Exclusive end index into the FFU intermediate dim.
    """
    ffn_dim_start: int
    ffn_dim_end: int

    def ffn_width(self) -> int:
        return self.ffn_dim_end - self.ffn_dim_start


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_ffn_graph(
    shard: ShardSpec,
    hidden_dim: int,
    device: DeviceRef,
    *,
    seq_len: int = 1,
    name: str = "ffn_shard",
) -> Graph:
    """Build a MAX Graph for a single node's non-uniform FFN shard.

    The graph accepts:
        - input  x            : [1, seq_len, hidden_dim]
        - weight ffn_up_slice : [hidden_dim, shard.ffn_width()]   (column-parallel slice)
        - weight ffn_down_slice: [shard.ffn_width(), hidden_dim]  (row-parallel slice)

    Operations:
        x @ ffn_up_slice  -> silu  -> @ ffn_down_slice  -> partial output

    No all-reduce is performed inside the graph; the returned tensor is the
    partial contribution of this node and must be summed across nodes by the
    orchestrator.
    """
    width = shard.ffn_width()
    if width <= 0:
        raise ValueError(f"ShardSpec width must be positive, got {width}")

    # --- Tensor types -------------------------------------------------------
    x_type = TensorType(
        DType.float32,
        [1, seq_len, hidden_dim],
        device=device,
    )
    ffn_up_slice_type = TensorType(
        DType.float32,
        [hidden_dim, width],
        device=device,
    )
    ffn_down_slice_type = TensorType(
        DType.float32,
        [width, hidden_dim],
        device=device,
    )

    # --- Graph construction -------------------------------------------------
    with Graph(
        name,
        input_types=[x_type, ffn_up_slice_type, ffn_down_slice_type],
    ) as g:
        x, ffn_up_slice, ffn_down_slice = g.inputs

        # Column-parallel projection: [1, seq, hidden] @ [hidden, width] -> [1, seq, width]
        h = ops.matmul(x, ffn_up_slice)

        # Activation
        h = ops.silu(h)

        # Row-parallel projection: [1, seq, width] @ [width, hidden] -> [1, seq, hidden]
        partial = ops.matmul(h, ffn_down_slice)

        g.output(partial)

    return g


# ---------------------------------------------------------------------------
# Convenience helper for tests / orchestrator
# ---------------------------------------------------------------------------
def compile_and_run(
    shard: ShardSpec,
    hidden_dim: int,
    x,
    ffn_up_slice,
    ffn_down_slice,
    *,
    device: DeviceRef | None = None,
):
    """Compile the FFN shard graph and execute it on the given device.

    Returns the partial output tensor as a numpy array.
    """
    import numpy as np

    if device is None:
        device = DeviceRef.cpu()

    graph = build_ffn_graph(shard, hidden_dim, device)

    # Choose a device driver for the session
    if device == DeviceRef.cpu():
        dev = CPU()
    else:
        dev = Accelerator()

    session = InferenceSession(devices=[dev])
    model = session.load(graph)

    from max.driver import Tensor

    x_t = Tensor.from_numpy(x).to(dev)
    up_t = Tensor.from_numpy(ffn_up_slice).to(dev)
    down_t = Tensor.from_numpy(ffn_down_slice).to(dev)

    (out,) = model.execute(x_t, up_t, down_t)
    return out.to_numpy()
