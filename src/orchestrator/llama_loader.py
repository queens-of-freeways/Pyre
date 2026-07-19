from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


SMOLM_135M_CONFIG = {
    "hidden_dim": 576,
    "n_heads": 9,
    "n_kv_heads": 3,
    "head_dim": 64,
    "ffn_dim": 1536,
    "vocab_size": 49152,
    "num_layers": 8,
}


def get_smollm_config() -> dict:
    return dict(SMOLM_135M_CONFIG)


def create_synthetic_weights(config: dict) -> dict:
    hidden_dim = config["hidden_dim"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = config["head_dim"]
    ffn_dim = config["ffn_dim"]
    vocab_size = config["vocab_size"]

    total_nodes = 3
    ffn_width_per_node = ffn_dim // total_nodes
    n_q_per_node = ffn_width_per_node // head_dim
    total_q_heads = n_q_per_node * total_nodes

    return {
        "q_weight": np.random.randn(hidden_dim, total_q_heads * head_dim).astype(np.float32),
        "k_weight": np.random.randn(hidden_dim, n_kv_heads * head_dim).astype(np.float32),
        "v_weight": np.random.randn(hidden_dim, n_kv_heads * head_dim).astype(np.float32),
        "ffn_gate": np.random.randn(hidden_dim, ffn_dim).astype(np.float32),
        "ffn_up": np.random.randn(hidden_dim, ffn_dim).astype(np.float32),
        "ffn_down": np.random.randn(ffn_dim, hidden_dim).astype(np.float32),
        "lm_head": np.random.randn(vocab_size, hidden_dim).astype(np.float32),
        "embedding": np.random.randn(vocab_size, hidden_dim).astype(np.float32),
    }


def slice_ffn_weights(
    full_weights: dict,
    ffn_start: int,
    ffn_end: int,
) -> Dict[str, np.ndarray]:
    width = ffn_end - ffn_start
    return {
        "ffn_gate": full_weights["ffn_gate"][:, ffn_start:ffn_end].copy(),
        "ffn_up": full_weights["ffn_up"][:, ffn_start:ffn_end].copy(),
        "ffn_down": full_weights["ffn_down"][ffn_start:ffn_end, :].copy(),
    }


def slice_attention_weights(
    full_weights: dict,
    head_start: int,
    head_end: int,
    n_kv_heads: int,
    head_dim: int,
) -> Dict[str, np.ndarray]:
    q_dim_start = head_start * head_dim
    q_dim_end = head_end * head_dim
    kv_dim = n_kv_heads * head_dim
    return {
        "q_slice": full_weights["q_weight"][:, q_dim_start:q_dim_end].copy(),
        "k_full": full_weights["k_weight"][:, :kv_dim].copy(),
        "v_full": full_weights["v_weight"][:, :kv_dim].copy(),
    }


def slice_weights_for_node(
    node_shard: dict,
    full_weights: dict,
    total_nodes: int,
    config: dict,
) -> dict:
    head_dim = config["head_dim"]
    n_kv_heads = config["n_kv_heads"]
    hidden_dim = config["hidden_dim"]
    ffn_width = node_shard["ffn_end"] - node_shard["ffn_start"]
    n_q_local = ffn_width // head_dim

    q_dim_start = 0
    q_dim_end = n_q_local * head_dim
    kv_dim = n_kv_heads * head_dim

    attn = {
        "q_slice": full_weights["q_weight"][:, q_dim_start:q_dim_end].copy(),
        "k_full": full_weights["k_weight"][:, :kv_dim].copy(),
        "v_full": full_weights["v_weight"][:, :kv_dim].copy(),
    }
    ffn = slice_ffn_weights(full_weights, node_shard["ffn_start"], node_shard["ffn_end"])

    return {
        "attn": attn,
        "ffn": ffn,
    }


def validate_weight_shapes(
    node_weights: Dict[int, dict],
    partitions: Dict[int, dict],
    config: dict,
) -> bool:
    hidden_dim = config["hidden_dim"]
    head_dim = config["head_dim"]
    n_kv_heads = config["n_kv_heads"]

    for node_id in partitions:
        p = partitions[node_id]
        width = p["ffn_end"] - p["ffn_start"]
        n_q_local = width // head_dim
        n_kv_local = n_kv_heads

        weights = node_weights[node_id]
        w = weights["ffn"]
        assert w["ffn_gate"].shape == (hidden_dim, width), (
            f"Node {node_id} ffn_gate shape {w['ffn_gate'].shape} != ({hidden_dim}, {width})"
        )
        assert w["ffn_up"].shape == (hidden_dim, width), (
            f"Node {node_id} ffn_up shape {w['ffn_up'].shape} != ({hidden_dim}, {width})"
        )
        assert w["ffn_down"].shape == (width, hidden_dim), (
            f"Node {node_id} ffn_down shape {w['ffn_down'].shape} != ({width}, {hidden_dim})"
        )

        a = weights["attn"]
        assert a["q_slice"].shape == (hidden_dim, n_q_local * head_dim), (
            f"Node {node_id} q_slice shape {a['q_slice'].shape} != ({hidden_dim}, {n_q_local * head_dim})"
        )
        assert a["k_full"].shape == (hidden_dim, n_kv_local * head_dim), (
            f"Node {node_id} k_full shape {a['k_full'].shape} != ({hidden_dim}, {n_kv_local * head_dim})"
        )
        assert a["v_full"].shape == (hidden_dim, n_kv_local * head_dim), (
            f"Node {node_id} v_full shape {a['v_full'].shape} != ({hidden_dim}, {n_kv_local * head_dim})"
        )

    return True
