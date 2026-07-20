"""Graph reuse: compile fewer MAX graphs by padding smaller head_dim weights."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def select_graph_dims(needed: Dict[Tuple, int]) -> Tuple[Tuple, int]:
    """Given {layer_type_key: example_layer_idx}, pick the dims that minimize graphs.
    
    Strategy: compile one graph for the most common head_dim, pad others.
    Returns the selected key for the shared graph.
    """
    if not needed:
        return None, None
    # Count occurrences of each head_dim
    dim_counts = {}
    for key in needed:
        hd = key[0]
        dim_counts[hd] = dim_counts.get(hd, 0) + 1
    # Pick the most common head_dim
    best_hd = max(dim_counts, key=dim_counts.get)
    best_key = next(k for k in needed if k[0] == best_hd)
    return best_key, best_hd


def pad_attn_weight(w: np.ndarray, target_heads: int, target_hd: int) -> np.ndarray:
    """Pad attention weight to target number of heads.
    
    w: [hidden_dim, current_heads * current_hd]
    Returns: [hidden_dim, target_heads * target_hd]
    """
    if w.shape[1] >= target_heads * target_hd:
        return w[:, :target_heads * target_hd].copy()
    padded = np.zeros((w.shape[0], target_heads * target_hd), dtype=w.dtype)
    padded[:, :w.shape[1]] = w
    return padded


def pad_ffn_weights_for_node(p: dict, config) -> dict:
    """Create FFN weights padded to a uniform width for graph reuse."""
    from src.orchestrator.llama_loader import LayerProperties
    width = p["ffn_end"] - p["ffn_start"]
    # Pad to next multiple of head_dim * n_heads = config.head_dim * config.n_heads
    target = config.n_heads * config.head_dim
    if width < target:
        return {
            "gate": np.random.randn(config.hidden_dim, target).astype(np.float32),
            "up": np.random.randn(config.hidden_dim, target).astype(np.float32),
            "down": np.random.randn(target, config.hidden_dim).astype(np.float32),
        }
    return None
