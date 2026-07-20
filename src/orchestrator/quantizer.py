"""Q8_0 block-wise quantization for weight transfer compression."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def quantize_q80(arr: np.ndarray, block_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize float32 array to Q8_0 (block int8 + fp32 scale).
    
    Returns (qdata, scales) where qdata is int8 and scales is float32.
    Compression: 4x for float32 input.
    """
    orig_shape = arr.shape
    flat = arr.ravel().astype(np.float32)
    n = flat.shape[0]
    n_blocks = (n + block_size - 1) // block_size
    padded = np.zeros(n_blocks * block_size, dtype=np.float32)
    padded[:n] = flat

    blocks = padded.reshape(n_blocks, block_size)
    absmax = np.max(np.abs(blocks), axis=1, keepdims=True)
    absmax = np.where(absmax == 0, 1.0, absmax)
    scales = absmax / 127.0
    qdata = np.clip(np.round(blocks / scales), -128, 127).astype(np.int8)

    scales = scales.ravel()
    return qdata, scales, orig_shape


def dequantize_q80(qdata: np.ndarray, scales: np.ndarray,
                   orig_shape: Tuple[int, ...]) -> np.ndarray:
    """Dequantize Q8_0 back to float32."""
    flat = qdata.ravel().astype(np.float32)
    n_blocks = scales.shape[0]
    block_size_used = flat.shape[0] // n_blocks
    blocks = flat.reshape(n_blocks, block_size_used)
    result = (blocks * scales.reshape(n_blocks, 1)).ravel()
    total = 1
    for d in orig_shape:
        total *= d
    return result[:total].reshape(orig_shape)


def quantize_weights_dict(weights: dict) -> dict:
    """Recursively quantize all numpy arrays in a weight dict."""
    qd = {}
    for k, v in weights.items():
        if isinstance(v, dict):
            qd[k] = quantize_weights_dict(v)
        elif isinstance(v, np.ndarray):
            q, s, sh = quantize_q80(v)
            qd[k] = ("q8", q, s, sh)
        else:
            qd[k] = v
    return qd


def dequantize_weights_dict(qd: dict) -> dict:
    """Recursively dequantize all Q8_0 entries in a dict."""
    result = {}
    for k, v in qd.items():
        if isinstance(v, dict):
            result[k] = dequantize_weights_dict(v)
        elif isinstance(v, tuple) and v[0] == "q8":
            _, q, s, sh = v
            result[k] = dequantize_q80(q, s, sh)
        else:
            result[k] = v
    return result
