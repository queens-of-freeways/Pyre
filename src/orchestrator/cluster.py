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
