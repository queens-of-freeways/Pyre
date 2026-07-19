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
        x_slices = []
        for node_id in self.partitions:
            p = self.partitions[node_id]
            x_slice = x[:, p["seq_start"]:p["seq_end"], :]
            x_slices.append(x_slice)
        
        # 2. Pass slices through each node's attention graph
        # Mocking weights for attention
        q_outputs, k_outputs, v_outputs = [], [], []
        for i, node_id in enumerate(self.partitions):
            p = self.partitions[node_id]
            local_seq = p["seq_end"] - p["seq_start"]
            n_q_heads_local = (p["ffn_end"] - p["ffn_start"]) // self.config.head_dim
            
            x_slice = x_slices[i]
            wq_slice = np.random.randn(hidden_dim, n_q_heads_local * self.config.head_dim).astype(np.float32)
            wk_full = np.random.randn(hidden_dim, self.config.n_kv_heads * self.config.head_dim).astype(np.float32)
            wv_full = np.random.randn(hidden_dim, self.config.n_kv_heads * self.config.head_dim).astype(np.float32)
            
            (q, k, v) = self.attn_models[node_id].execute(
                np.ascontiguousarray(x_slice),
                np.ascontiguousarray(wq_slice),
                np.ascontiguousarray(wk_full),
                np.ascontiguousarray(wv_full)
            )
            q_outputs.append(q.to_numpy())
            k_outputs.append(k.to_numpy())
            v_outputs.append(v.to_numpy())
        
        # 3. Simulate all-to-all by concatenating Q, K, V tensors back together
        q_full = np.concatenate(q_outputs, axis=2)  # Concat along head dimension
        k_full = np.concatenate(k_outputs, axis=1)  # Concat along seq dimension
        v_full = np.concatenate(v_outputs, axis=1)
        
        # Mock attention output (Q * K^T * V)
        # For simplicity, just use q_full reshaped back to hidden_dim for FFN input
        attn_out = np.random.randn(batch, seq_len, hidden_dim).astype(np.float32)
        
        # 4. Pass attention output through each node's FFN graph
        partial_ffn_outputs = []
        for i, node_id in enumerate(self.partitions):
            p = self.partitions[node_id]
            width = p["ffn_end"] - p["ffn_start"]
            local_seq = p["seq_end"] - p["seq_start"]
            
            # FFN expects local sequence slice
            attn_slice = attn_out[:, p["seq_start"]:p["seq_end"], :]
            ffn_up_slice = np.random.randn(hidden_dim, width).astype(np.float32)
            ffn_down_slice = np.random.randn(width, hidden_dim).astype(np.float32)
            
            (partial,) = self.ffn_models[node_id].execute(
                np.ascontiguousarray(attn_slice),
                np.ascontiguousarray(ffn_up_slice),
                np.ascontiguousarray(ffn_down_slice)
            )
            partial_ffn_outputs.append(partial.to_numpy())
        
        # 5. Simulate ring all-reduce by summing all partial FFN outputs
        final_output = np.zeros_like(partial_ffn_outputs[0])
        for p_out in partial_ffn_outputs:
            final_output += p_out
            
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
