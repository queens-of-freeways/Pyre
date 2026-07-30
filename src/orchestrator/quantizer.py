"""Q8_0 block-wise quantization for weight transfer compression."""
from __future__ import annotations

from typing import Tuple

import numpy as np

MIN_QUANTIZE_ELEMS = 4096  # skip tiny arrays (norms, biases, conv1d, etc.)


def quantize_q80(arr: np.ndarray, block_size: int = 32) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """Quantize float32 array to Q8_0 (block int8 + fp32 scale)."""
    orig_shape = arr.shape
    flat = arr.ravel().astype(np.float32)
    n = flat.shape[0]
    n_blocks = (n + block_size - 1) // block_size
    padded = np.zeros(n_blocks * block_size, dtype=np.float32)
    padded[:n] = flat

    blocks = padded.reshape(n_blocks, block_size)
    absmax = np.max(np.abs(blocks), axis=1, keepdims=True)
    absmax = np.where(absmax == 0, 1.0, absmax)
    scales = (absmax / 127.0).ravel()
    qdata = np.clip(np.round(blocks / (scales.reshape(n_blocks, 1))), -128, 127).astype(np.int8)

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
    return np.ascontiguousarray(result[:total].reshape(orig_shape))


def quantize_to_f16(arr: np.ndarray) -> tuple:
    """Store float32 as float16 (2x compression, negligible precision loss)."""
    return ("f16", arr.astype(np.float16), arr.shape)


def dequantize_from_f16(data: tuple) -> np.ndarray:
    """Restore float16 back to float32."""
    _, arr, shape = data
    return np.ascontiguousarray(arr.astype(np.float32).reshape(shape))


def quantize_activation(arr: np.ndarray) -> tuple:
    """Q8_0 quantization for network transfer of activations.
    
    Returns a pickle-friendly tuple (qdata, scales, orig_shape).
    ~4x compression vs float32.
    """
    qdata, scales, orig_shape = quantize_q80(arr)
    return (qdata, scales, orig_shape)


def dequantize_activation(data: tuple) -> np.ndarray:
    """Restore a Q8_0 activation tuple back to float32."""
    qdata, scales, orig_shape = data
    return dequantize_q80(qdata, scales, orig_shape)


def quantize_weights_dict(weights: dict, mode: str = "q8") -> dict:
    """Recursively quantize all numpy arrays in a weight dict.

    mode: "q8" for 4x compression (block int8), "f16" for 2x (float16).
    """
    qd = {}
    for k, v in weights.items():
        if isinstance(v, dict):
            qd[k] = quantize_weights_dict(v, mode=mode)
        elif isinstance(v, np.ndarray) and v.size >= MIN_QUANTIZE_ELEMS:
            if mode == "f16":
                qd[k] = quantize_to_f16(v)
            else:
                q, s, sh = quantize_q80(v)
                qd[k] = ("q8", q, s, sh)
        else:
            qd[k] = v
    return qd


def dequantize_weights_dict(qd: dict, subset: str = "") -> dict:
    """Recursively dequantize all quantized entries in a dict.

    If *subset* is set to a top-level key (e.g. ``"ffn"``), only that
    subtree is dequantized; all other top-level keys are returned as-is.
    """
    result = {}
    for k, v in qd.items():
        if subset and k != subset:
            result[k] = v
            continue
        if isinstance(v, dict):
            result[k] = dequantize_weights_dict(v)
        elif isinstance(v, tuple):
            tag = v[0]
            if tag == "q8":
                _, q, s, sh = v
                result[k] = dequantize_q80(q, s, sh)
            elif tag == "f16":
                result[k] = dequantize_from_f16(v)
            else:
                result[k] = v
        else:
            result[k] = v
    return result
