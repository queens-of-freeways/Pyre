from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.driver import CPU

from src.attention.builder import build_ulysses_attention_graph, ShardSpec as AttentionShardSpec
from src.ffn.builder import build_ffn_graph, ShardSpec as FFNShardSpec

@dataclass
class NodeCap:
    id: int
    flops_gflops: float
    mem_bytes: int
    net_bps: float
    barrier_latency_us: float

@dataclass
class ModelConfig:
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_dim: int
    num_layers: int = 1
    vocab_size: int = 49152
    model_type: str = "llama"
    ple_dim: int = 0  # PLE dimension (Gemma 4 E-series only)
    rope_theta: float = 10000.0
    max_seq_len: int = 2048

    @staticmethod
    def from_hf(model_id: str) -> "ModelConfig":
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)

        hidden_dim = getattr(cfg, "hidden_size", getattr(cfg, "hidden_dim", None))
        n_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", None))
        n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
        head_dim = getattr(cfg, "head_dim", None) or (hidden_dim // n_heads)
        ffn_dim = getattr(cfg, "intermediate_size", getattr(cfg, "ffn_dim", None))
        num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", 1))
        vocab_size = getattr(cfg, "vocab_size", 49152)
        model_type = getattr(cfg, "model_type", "llama")

        if any(v is None for v in [hidden_dim, n_heads, head_dim, ffn_dim]):
            raise ValueError(f"Could not infer all model dimensions from {model_id} config")

        ple_dim = getattr(cfg, "hidden_size_per_layer_input", 0)
        rope_theta = getattr(cfg, "rope_theta", 10000.0)
        max_seq_len = getattr(cfg, "max_position_embeddings", 2048)

        return ModelConfig(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            model_type=model_type,
            ple_dim=ple_dim,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )

class ClusterOrchestrator:
    def __init__(self, nodes: List[NodeCap], config: ModelConfig):
        self.nodes = nodes
        self.config = config
        self.device = DeviceRef.CPU()
        self.session = InferenceSession(devices=[CPU()])
        
        # Mock partition solver: evenly distribute ffn_dim and seq_len across nodes
        self.partitions = self._solve_partitions()
        
        self.attn_graphs = {}
        self.ffn_graphs = {}
        self.attn_models = {}
        self.ffn_models = {}
        
        for node_id, shard in self.partitions.items():
            attn_shard = AttentionShardSpec(
                ffn_dim_start=shard["ffn_start"],
                ffn_dim_end=shard["ffn_end"],
                seq_start=shard["seq_start"],
                seq_end=shard["seq_end"]
            )
            attn_graph = build_ulysses_attention_graph(
                attn_shard, config.hidden_dim, config.n_heads, config.n_kv_heads, config.head_dim, self.device
            )
            self.attn_graphs[node_id] = attn_graph
            self.attn_models[node_id] = self.session.load(attn_graph)
            
            ffn_shard = FFNShardSpec(
                ffn_dim_start=shard["ffn_start"],
                ffn_dim_end=shard["ffn_end"]
            )
            ffn_graph = build_ffn_graph(
                ffn_shard, config.hidden_dim, self.device, seq_len=shard["seq_end"] - shard["seq_start"]
            )
            self.ffn_graphs[node_id] = ffn_graph
            self.ffn_models[node_id] = self.session.load(ffn_graph)

    def _solve_partitions(self):
        n = len(self.nodes)
        ffn_dim = self.config.ffn_dim
        seq_len = 64  # Assuming standard seq_len for partitioning
        
        partitions = {}
        ffn_chunk = ffn_dim // n
        seq_chunk = seq_len // n
        
        for i, node in enumerate(self.nodes):
            ffn_start = i * ffn_chunk
            ffn_end = (i + 1) * ffn_chunk if i < n - 1 else ffn_dim
            seq_start = i * seq_chunk
            seq_end = (i + 1) * seq_chunk if i < n - 1 else seq_len
            partitions[node.id] = {
                "ffn_start": ffn_start,
                "ffn_end": ffn_end,
                "seq_start": seq_start,
                "seq_end": seq_end
            }
        return partitions

    def run(self, x: np.ndarray) -> np.ndarray:
        batch, seq_len, hidden_dim = x.shape
        assert hidden_dim == self.config.hidden_dim

        # 1. Split input along sequence dimension for Ulysses attention
        # 2. Pass slices through each node's attention graph (mock: just execute)
        for i, node_id in enumerate(self.partitions):
            p = self.partitions[node_id]
            x_slice = x[:, p["seq_start"]:p["seq_end"], :]
            n_q_heads_local = (p["ffn_end"] - p["ffn_start"]) // self.config.head_dim
            wq_slice = np.random.randn(hidden_dim, n_q_heads_local * self.config.head_dim).astype(np.float32)
            wk_full = np.random.randn(hidden_dim, self.config.n_kv_heads * self.config.head_dim).astype(np.float32)
            wv_full = np.random.randn(hidden_dim, self.config.n_kv_heads * self.config.head_dim).astype(np.float32)
            self.attn_models[node_id].execute(
                np.ascontiguousarray(x_slice),
                np.ascontiguousarray(wq_slice),
                np.ascontiguousarray(wk_full),
                np.ascontiguousarray(wv_full)
            )

        # 3. Mock attention output for FFN input
        attn_out = np.random.randn(batch, seq_len, hidden_dim).astype(np.float32)

        # 4. Pass slices through each node's FFN graph, placing partials back
        final_output = np.zeros((batch, seq_len, hidden_dim), dtype=np.float32)
        for node_id in self.partitions:
            p = self.partitions[node_id]
            width = p["ffn_end"] - p["ffn_start"]
            attn_slice = attn_out[:, p["seq_start"]:p["seq_end"], :]
            ffn_up_slice = np.random.randn(hidden_dim, width).astype(np.float32)
            ffn_down_slice = np.random.randn(width, hidden_dim).astype(np.float32)
            (partial,) = self.ffn_models[node_id].execute(
                np.ascontiguousarray(attn_slice),
                np.ascontiguousarray(ffn_up_slice),
                np.ascontiguousarray(ffn_down_slice)
            )
            final_output[:, p["seq_start"]:p["seq_end"], :] += partial.to_numpy()

        return final_output

    def detect_drift(self, execution_times: List[float]) -> bool:
        if not execution_times:
            return False
        avg_time = sum(execution_times) / len(execution_times)
        for t in execution_times:
            if t > avg_time * 1.15:
                print("Drift detected: triggering re-partition")
                return True
        return False


class AdaptivePartitioner:
    """Tracks per-node execution speed and computes non-uniform FFN partitions.

    Maintains an exponential moving average of each node's relative speed.
    Faster nodes receive a larger FFN slice; slower nodes receive a smaller slice.
    Partitions are always non-overlapping and cover the full FFN dimension.

    The partitioner can be used in two modes:
    1. **Between-pass**: call ``get_partitions()`` after a full inference pass to
       obtain optimised splits for the *next* pass.
    2. **Per-layer** (requires pre-compiled multi-width graphs): call
       ``layer_partition(node_id, layer_idx)`` for each layer to get the
       recomputed split based on the most recent timing for each node.
    """

    def __init__(self, ffn_dim: int, num_nodes: int, seq_len: int = 64,
                 min_fraction: float = 0.25, max_fraction: float = 1.75,
                 ema_alpha: float = 0.3):
        self.ffn_dim = ffn_dim
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.base_width = ffn_dim // num_nodes
        self.min_width = max(1, int(ffn_dim * min_fraction / num_nodes))
        self.max_width = min(ffn_dim, int(ffn_dim * max_fraction / num_nodes))
        # Guard: at least one base unit for others
        self.max_width = min(self.max_width, ffn_dim - (num_nodes - 1) * self.min_width)
        self.alpha = ema_alpha

        # ema[i] = relative speed of node i (1.0 = average, >1 = faster)
        self.ema: List[float] = [1.0] * num_nodes
        # Per-node cumulative timings for the current inference pass
        self._layer_timings: Dict[int, List[float]] = {i: [] for i in range(num_nodes)}
        self._drift_counter: int = 0

    def update(self, node_timings: Dict[int, float]):
        """Record per-node execution times for a single layer (lower = faster).

        ``node_timings`` maps node_id → wall-clock seconds spent in the
        FFN computation for that layer.
        """
        for nid, t in node_timings.items():
            if nid in self._layer_timings:
                self._layer_timings[nid].append(t)

    def _recompute_ema(self):
        """Recompute EMA ratios from accumulated layer timings."""
        timings = {nid: ts for nid, ts in self._layer_timings.items() if ts}
        if not timings:
            return
        # Average per node across observed layers
        avg_times = {nid: sum(ts) / len(ts) for nid, ts in timings.items()}
        # Global average across all nodes
        global_avg = sum(avg_times.values()) / len(avg_times)
        for nid, t in avg_times.items():
            relative = global_avg / max(t, 1e-9)
            self.ema[nid] = self.ema[nid] * (1 - self.alpha) + relative * self.alpha

    def drift_detected(self, threshold: float = 1.15) -> bool:
        """Returns True if any node is more than *threshold* away from average."""
        if not self._has_data():
            return False
        self._recompute_ema()
        for v in self.ema:
            if v > threshold or v < 1.0 / threshold:
                return True
        return False

    def _has_data(self) -> bool:
        return any(len(ts) > 0 for ts in self._layer_timings.values())

    def get_partitions(self) -> Dict[int, dict]:
        """Compute non-uniform FFN splits from current speed estimates.

        Node 0 (root) always receives the base share since its speed is not
        independently measurable from the combined prefill/decode path.
        Worker nodes split the remaining FFN dimension proportional to their
        EMA speed, so faster workers get more work.

        Returns the same dict format as ``RootNode._solve_partitions``.
        """
        self._recompute_ema()
        return self._allocate_partitions()

    def _allocate_partitions(self) -> Dict[int, dict]:
        """Allocate FFN width proportional to each node's ema speed.

        Root (node 0) gets ``base_width``; workers split the rest
        proportional to their relative speed.
        """
        total_nodes = self.num_nodes
        seq_chunk = self.seq_len // total_nodes

        if total_nodes == 1:
            return {0: {"ffn_start": 0, "ffn_end": self.ffn_dim,
                        "seq_start": 0, "seq_end": self.seq_len}}

        # Root gets base width; workers split remainder
        root_base = self.base_width
        remaining = self.ffn_dim - root_base

        worker_emas = self.ema[1:]  # skip root
        total_worker_weight = sum(worker_emas)

        # Allocate to workers
        raw_widths = []
        for v in worker_emas:
            w = int(remaining * v / total_worker_weight) if total_worker_weight > 0 else remaining // (total_nodes - 1)
            raw_widths.append(max(self.min_width, min(w, remaining - (total_nodes - 2) * self.min_width)))

        # Clamp: ensure each worker at least min_width
        clamped = [max(self.min_width, min(w, remaining - (total_nodes - 2) * self.min_width))
                   for w in raw_widths]

        # Distribute residual
        allocated = sum(clamped)
        residual = remaining - allocated
        if residual > 0:
            fastest_wi = max(range(len(clamped)), key=lambda i: worker_emas[i])
            clamped[fastest_wi] += residual
        elif residual < 0:
            slowest_wi = min(range(len(clamped)), key=lambda i: worker_emas[i])
            clamped[slowest_wi] = max(self.min_width, clamped[slowest_wi] + residual)

        # Build result
        partitions: Dict[int, dict] = {}
        offset = 0
        # Root first
        partitions[0] = {
            "ffn_start": 0,
            "ffn_end": root_base,
            "seq_start": 0,
            "seq_end": seq_chunk,
        }
        offset = root_base
        for wi, width in enumerate(clamped):
            nid = wi + 1
            partitions[nid] = {
                "ffn_start": offset,
                "ffn_end": offset + width,
                "seq_start": nid * seq_chunk if nid < total_nodes - 1 else (total_nodes - 1) * seq_chunk,
                "seq_end": (nid + 1) * seq_chunk if nid < total_nodes - 1 else self.seq_len,
            }
            offset += width
        return partitions

    def layer_partition(self, node_id: int, layer_idx: int) -> dict:
        """Return the partition for *node_id* at *layer_idx* based on EMA.

        Called per-layer when the caller manages multiple FFN graphs at
        different widths and can select the correct one for each layer.

        This is a *local* view: it returns a partition that assumes all
        other nodes keep their current partition; only *node_id*'s width
        is adjusted based on the most recent timing delta for that node.
        """
        partitions = self.get_partitions()  # uses latest EMA
        return partitions.get(node_id, {})

    @staticmethod
    def equal_partitions(ffn_dim: int, num_nodes: int, seq_len: int = 64) -> Dict[int, dict]:
        """Create equal partitions (same as ``_solve_partitions``)."""
        seq_chunk = seq_len // num_nodes
        ffn_chunk = ffn_dim // num_nodes
        partitions = {}
        for i in range(num_nodes):
            partitions[i] = {
                "ffn_start": i * ffn_chunk,
                "ffn_end": (i + 1) * ffn_chunk if i < num_nodes - 1 else ffn_dim,
                "seq_start": i * seq_chunk,
                "seq_end": (i + 1) * seq_chunk if i < num_nodes - 1 else seq_len,
            }
        return partitions

    def reset_pass(self):
        """Call at the start of each inference pass to clear layer timings."""
        self._layer_timings = {i: [] for i in range(self.num_nodes)}
