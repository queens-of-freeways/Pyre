from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class LayerProperties:
    head_dim: int
    has_v_proj: bool = True
    rope_fraction: float = 1.0
    use_v_norm: bool = False
    attention_type: str = "standard"  # "standard", "gemma4_global", "gemma4_sliding"
    kv_source_layer: Optional[int] = None  # if set, this layer shares KV from this source

    @staticmethod
    def standard(hd: int) -> "LayerProperties":
        return LayerProperties(head_dim=hd)

    @staticmethod
    def gemma4_global(hd: int) -> "LayerProperties":
        return LayerProperties(head_dim=hd, has_v_proj=False, rope_fraction=0.25,
                              use_v_norm=True, attention_type="gemma4_global")

    @staticmethod
    def gemma4_sliding(hd: int) -> "LayerProperties":
        return LayerProperties(head_dim=hd, attention_type="gemma4_sliding")


@dataclass
class LayerWeightSet:
    q: torch.Tensor          # [hidden_dim, n_heads * head_dim] bfloat16
    k: torch.Tensor          # [hidden_dim, n_kv_heads * head_dim] bfloat16
    ffn_gate: torch.Tensor   # [hidden_dim, ffn_dim] bfloat16
    ffn_up: torch.Tensor     # [hidden_dim, ffn_dim] bfloat16
    ffn_down: torch.Tensor   # [ffn_dim, hidden_dim] bfloat16
    v: Optional[torch.Tensor] = None  # [hidden_dim, n_kv_heads * head_dim] bfloat16
    o: Optional[torch.Tensor] = None  # [n_heads * head_dim, hidden_dim] bfloat16
    q_bias: Optional[np.ndarray] = None          # [n_heads * head_dim] float32
    k_bias: Optional[np.ndarray] = None          # [n_kv_heads * head_dim] float32
    v_bias: Optional[np.ndarray] = None          # [n_kv_heads * head_dim] float32
    input_layernorm: Optional[np.ndarray] = None      # [hidden_dim] float32
    post_attention_layernorm: Optional[np.ndarray] = None  # [hidden_dim] float32
    has_v_proj: bool = True
    props: Optional[LayerProperties] = None
    # PLE (Per-Layer Embeddings) per-layer weights (stored as float32, tiny)
    ple_gate: Optional[np.ndarray] = None      # [hidden_dim, ple_dim] float32
    ple_proj: Optional[np.ndarray] = None      # [ple_dim, hidden_dim] float32
    ple_post_norm: Optional[np.ndarray] = None # [hidden_dim] float32


@dataclass
class FullWeights:
    layer_weights: Dict[int, LayerWeightSet]
    layer_props: Dict[int, LayerProperties]
    embedding: np.ndarray       # [vocab_size, hidden_dim]
    lm_head: np.ndarray         # [vocab_size, hidden_dim]
    num_layers: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_dim: int
    vocab_size: int
    final_norm: Optional[np.ndarray] = None  # [hidden_dim]
    model_type: str = "llama"
    # PLE (Per-Layer Embeddings) — Gemma 4 E-series
    ple_embedding: Optional[np.ndarray] = None   # [ple_vocab_size, num_layers * ple_dim]
    ple_projection: Optional[np.ndarray] = None  # [hidden_dim, num_layers * ple_dim]
    ple_projection_norm: Optional[np.ndarray] = None  # [ple_dim]
    ple_dim: int = 0
    ple_vocab_size: int = 0


def _torch_to_np(t: torch.Tensor, transpose: bool = False) -> np.ndarray:
    """Convert torch tensor to float32 numpy array."""
    t = t.to(torch.float32)
    arr = t.cpu().numpy()
    if transpose:
        arr = arr.T
    return np.ascontiguousarray(arr)

def _torch_to_b16(t: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    """Keep tensor as bfloat16 for memory-efficient storage."""
    t = t.detach().to(torch.bfloat16)
    if transpose:
        t = t.T
    return t.contiguous()

def _ensure_f32(arr: np.ndarray) -> np.ndarray:
    """Ensure array is float32. No-op if already float32."""
    if arr.dtype != np.float32:
        return arr.astype(np.float32, copy=False)
    return arr


def _infer_weight_keys(state: Dict[str, torch.Tensor], model_type: str = "llama",
                        cfg: Any = None) -> dict:
    """Scan state dict keys and infer naming conventions."""
    keys = list(state.keys())

    layer_keys = [k for k in keys if re.match(r"model\.layers\.\d+\.", k)]
    entry = {}
    for k in layer_keys:
        m = re.match(r"model\.layers\.(\d+)\.(.+)", k)
        if m:
            entry.setdefault(int(m.group(1)), []).append(m.group(2))

    if not entry:
        raise ValueError("No model.layers.N.* keys found in state dict")

    first_idx = min(entry.keys())
    first_keys = entry[first_idx]

    def has(pat, keys=first_keys):
        return any(pat in k for k in keys)

    # Detect norm key pattern
    if has("input_layernorm.weight"):
        input_norm_tmpl = "model.layers.{}.input_layernorm.weight"
    elif has("attention_norm.weight"):
        input_norm_tmpl = "model.layers.{}.attention_norm.weight"
    else:
        input_norm_tmpl = None

    if has("post_attention_layernorm.weight"):
        post_attn_norm_tmpl = "model.layers.{}.post_attention_layernorm.weight"
    elif has("ffn_norm.weight"):
        post_attn_norm_tmpl = "model.layers.{}.ffn_norm.weight"
    else:
        post_attn_norm_tmpl = None

    # Detect attention key pattern
    if has("self_attn.q_proj"):
        attn_tmpl = "model.layers.{}.self_attn.{}_proj.weight"
    elif has("attention.wq"):
        attn_tmpl = "model.layers.{}.attention.w{}.weight"
    else:
        raise ValueError(f"Cannot detect attention key pattern from {first_keys}")

    # Detect QKV bias
    has_qkv_bias = has("q_proj.bias")

    # Detect MLP key pattern (gated vs non-gated)
    ffn_gated = has("mlp.gate_proj") or (has("mlp.gate") and not has("mlp.fc1"))
    if ffn_gated:
        if has("mlp.gate_proj"):
            mlp_tmpl = "model.layers.{}.mlp.{}_proj.weight"
        elif has("mlp.gate"):
            mlp_tmpl = "model.layers.{}.mlp.{}.weight"
        else:
            mlp_tmpl = None
    else:
        if has("mlp.fc1"):
            mlp_tmpl = "model.layers.{}.mlp.fc{}.weight"
        else:
            mlp_tmpl = None

    # Detect per-layer anomalies (missing v_proj = Gemma 4 global layers)
    per_layer = {}
    is_gemma4 = model_type == "gemma4"
    for layer_idx, layer_keys in entry.items():
        has_v = any("v_proj" in k for k in layer_keys)
        per_layer[layer_idx] = {
            "has_v_proj": has_v,
            "is_gemma4_global": is_gemma4 and not has_v,
        }
        # For Gemma 4 global layers, detect head_dim from k_proj shape
        if is_gemma4 and not has_v:
            k_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            if k_key in state:
                n_kv_local = getattr(cfg, "num_key_value_heads", None)
                if n_kv_local:
                    per_layer[layer_idx]["head_dim"] = state[k_key].shape[0] // n_kv_local
        elif is_gemma4 and has_v:
            # Sliding layers: use default or detect
            per_layer[layer_idx]["head_dim"] = None  # will use default from config

    embed_key = "model.embed_tokens.weight"
    lm_head_key = "lm_head.weight" if "lm_head.weight" in state else embed_key
    norm_key = "model.norm.weight" if "model.norm.weight" in state else None

    # Detect PLE (Per-Layer Embeddings) for Gemma 4
    has_ple = "model.embed_tokens_per_layer.weight" in state
    if has_ple:
        ple_keys = {}
        for layer_idx, lk in entry.items():
            has_ple_gate = any("per_layer_input_gate" in k for k in lk)
            ple_keys[layer_idx] = has_ple_gate
    else:
        ple_keys = {}

    return {
        "attn_tmpl": attn_tmpl,
        "mlp_tmpl": mlp_tmpl,
        "ffn_gated": ffn_gated,
        "per_layer": per_layer,
        "embed_key": embed_key,
        "lm_head_key": lm_head_key,
        "norm_key": norm_key,
        "input_norm_tmpl": input_norm_tmpl,
        "post_attn_norm_tmpl": post_attn_norm_tmpl,
        "has_ple": has_ple,
        "ple_keys": ple_keys,
        "has_qkv_bias": has_qkv_bias,
    }


def _load_layer_weights(
    state: Dict[str, torch.Tensor],
    layer_idx: int,
    key_info: dict,
) -> LayerWeightSet:
    tmpl = key_info["attn_tmpl"]
    q = _torch_to_b16(state[tmpl.format(layer_idx, "q")], transpose=True)
    k = _torch_to_b16(state[tmpl.format(layer_idx, "k")], transpose=True)

    has_v = key_info["per_layer"].get(layer_idx, {}).get("has_v_proj", True)
    v = None
    if has_v:
        v = _torch_to_b16(state[tmpl.format(layer_idx, "v")], transpose=True)

    o = _torch_to_b16(state[tmpl.format(layer_idx, "o")], transpose=True)

    mlp_tmpl = key_info["mlp_tmpl"]
    if key_info["ffn_gated"]:
        gate = _torch_to_b16(state[mlp_tmpl.format(layer_idx, "gate")], transpose=True)
        up = _torch_to_b16(state[mlp_tmpl.format(layer_idx, "up")], transpose=True)
        down = _torch_to_b16(state[mlp_tmpl.format(layer_idx, "down")], transpose=True)
    else:
        fc1 = _torch_to_b16(state[mlp_tmpl.format(layer_idx, "1")], transpose=True)
        fc2 = _torch_to_b16(state[mlp_tmpl.format(layer_idx, "2")], transpose=True)
        gate = fc1
        up = fc1
        down = fc2

    # QKV biases (float32 tiny arrays)
    q_bias = k_bias = v_bias = None
    if key_info.get("has_qkv_bias"):
        q_bias = _torch_to_np(state[tmpl.replace(".weight", ".bias").format(layer_idx, "q")], transpose=False)
        k_bias = _torch_to_np(state[tmpl.replace(".weight", ".bias").format(layer_idx, "k")], transpose=False)
        v_bias = _torch_to_np(state[tmpl.replace(".weight", ".bias").format(layer_idx, "v")], transpose=False)

    # Per-layer norm weights (always float32, tiny arrays)
    input_norm = None
    post_attn_norm = None
    input_norm_tmpl = key_info.get("input_norm_tmpl")
    if input_norm_tmpl:
        input_norm = state[input_norm_tmpl.format(layer_idx)].to(torch.float32).cpu().numpy()
    post_attn_norm_tmpl = key_info.get("post_attn_norm_tmpl")
    if post_attn_norm_tmpl:
        post_attn_norm = state[post_attn_norm_tmpl.format(layer_idx)].to(torch.float32).cpu().numpy()

    # PLE per-layer weights (float32, tiny)
    ple_gate = None
    ple_proj = None
    ple_post_norm = None
    if key_info.get("has_ple") and key_info["ple_keys"].get(layer_idx, False):
        prefix = f"model.layers.{layer_idx}."
        ple_gate = _torch_to_np(state[prefix + "per_layer_input_gate.weight"], transpose=True)
        ple_proj = _torch_to_np(state[prefix + "per_layer_projection.weight"], transpose=True)
        ple_post_norm = state[prefix + "post_per_layer_input_norm.weight"].to(torch.float32).cpu().numpy()

    return LayerWeightSet(q=q, k=k, v=v, ffn_gate=gate, ffn_up=up, ffn_down=down,
                          has_v_proj=has_v, o=o,
                          q_bias=q_bias, k_bias=k_bias, v_bias=v_bias,
                          input_layernorm=input_norm, post_attention_layernorm=post_attn_norm,
                          ple_gate=ple_gate, ple_proj=ple_proj, ple_post_norm=ple_post_norm)


def _load_full_weights(model_id: str, num_layers: Optional[int] = None) -> FullWeights:
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    state = model.state_dict()
    del model

    model_type = getattr(cfg, "model_type", "llama")
    key_info = _infer_weight_keys(state, model_type=model_type, cfg=cfg)

    hidden_dim = getattr(cfg, "hidden_size", getattr(cfg, "hidden_dim", None))
    n_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", None))
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    default_head_dim = getattr(cfg, "head_dim", None) or (hidden_dim // n_heads)
    ffn_dim = getattr(cfg, "intermediate_size", getattr(cfg, "ffn_dim", None))
    n_layers = num_layers or getattr(cfg, "num_hidden_layers",
                                     getattr(cfg, "num_layers",
                                             max(key_info["per_layer"].keys()) + 1))

    # Share lm_head/embedding when tied (saves 50% memory for large vocabs)
    if key_info["lm_head_key"] == key_info["embed_key"]:
        embedding = _ensure_f32(_torch_to_np(state[key_info["embed_key"]], transpose=False))
        lm_head = embedding  # same array, no copy
    else:
        lm_head = _ensure_f32(_torch_to_np(state[key_info["lm_head_key"]], transpose=False))
        embedding = _ensure_f32(_torch_to_np(state[key_info["embed_key"]], transpose=False))
    final_norm = None
    if key_info["norm_key"]:
        final_norm = state[key_info["norm_key"]].to(torch.float32).cpu().numpy()
    vocab_size = embedding.shape[0]

    layer_weights = {}
    layer_props = {}
    is_gemma4 = model_type == "gemma4"

    # Detect KV shared layers (last N layers that don't compute own KV)
    num_kv_shared = getattr(cfg, "num_kv_shared_layers", 0)
    layer_types_list = getattr(cfg, "layer_types", None)
    kv_shared_map = {}
    if num_kv_shared > 0 and is_gemma4:
        for shared_i in range(n_layers - num_kv_shared, n_layers):
            stype = layer_types_list[shared_i] if layer_types_list and shared_i < len(layer_types_list) else None
            for src in range(shared_i - 1, -1, -1):
                src_type = layer_types_list[src] if layer_types_list and src < len(layer_types_list) else None
                if src_type == stype or stype is None:
                    kv_shared_map[shared_i] = src
                    break
            if shared_i not in kv_shared_map:
                kv_shared_map[shared_i] = shared_i - 1  # fallback: previous layer

    for i in range(n_layers):
        if i in key_info["per_layer"]:
            lw = _load_layer_weights(state, i, key_info)
            pl = key_info["per_layer"][i]
        else:
            lw = _load_layer_weights(state, min(key_info["per_layer"].keys()), key_info)
            pl = key_info["per_layer"].get(min(key_info["per_layer"].keys()), {})

        if is_gemma4 and not pl.get("has_v_proj", True):
            hd = pl.get("head_dim", default_head_dim * 2)
            lw.props = LayerProperties.gemma4_global(hd)
        elif is_gemma4 and pl.get("has_v_proj", True):
            hd = pl.get("head_dim", default_head_dim)
            lw.props = LayerProperties.gemma4_sliding(hd)
        else:
            lw.props = LayerProperties.standard(default_head_dim)

        # For KV shared layers, copy k/v weights from source layer
        if i in kv_shared_map:
            src_i = kv_shared_map[i]
            if src_i in layer_weights:
                src_lw = layer_weights[src_i]
                if isinstance(src_lw.k, torch.Tensor):
                    lw.k = src_lw.k.clone()
                    lw.v = src_lw.v.clone() if src_lw.v is not None else None
                else:
                    lw.k = src_lw.k.copy()
                    lw.v = src_lw.v.copy() if src_lw.v is not None else None
            lw.props.kv_source_layer = src_i

        layer_weights[i] = lw
        layer_props[i] = lw.props

    # PLE model-level weights
    ple_embedding = None
    ple_projection = None
    ple_projection_norm = None
    ple_dim_val = 0
    ple_vocab_size_val = 0
    if key_info.get("has_ple"):
        ple_embedding = _torch_to_np(state["model.embed_tokens_per_layer.weight"], transpose=False)
        ple_projection = _torch_to_np(state["model.per_layer_model_projection.weight"], transpose=True)
        ple_projection_norm = state["model.per_layer_projection_norm.weight"].to(torch.float32).cpu().numpy()
        ple_dim_val = getattr(cfg, "hidden_size_per_layer_input", 0) or 0
        ple_vocab_size_val = getattr(cfg, "vocab_size_per_layer_input", 0) or 0

    del state

    return FullWeights(
        layer_weights=layer_weights,
        layer_props=layer_props,
        embedding=embedding,
        lm_head=lm_head,
        final_norm=final_norm,
        num_layers=n_layers,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=default_head_dim,
        ffn_dim=ffn_dim,
        vocab_size=vocab_size,
        model_type=model_type,
        ple_embedding=ple_embedding,
        ple_projection=ple_projection,
        ple_projection_norm=ple_projection_norm,
        ple_dim=ple_dim_val,
        ple_vocab_size=ple_vocab_size_val,
    )


def _slice_attn_for_node(
    full: FullWeights,
    layer_idx: int,
    head_start: int,
    head_end: int,
    copy_weights: bool = True,
) -> dict:
    lw = full.layer_weights[layer_idx]
    hd = lw.props.head_dim if lw.props else full.head_dim
    q_end = head_end * hd
    kv_dim = full.n_kv_heads * hd
    o_dim = full.n_heads * hd

    def _c(arr, slc):
        if arr is None:
            return None
        if isinstance(arr, torch.Tensor):
            return np.ascontiguousarray(slc.to(torch.float32).cpu().numpy())
        return slc.copy() if copy_weights else _ensure_f32(slc)

    result = {
        "q": _c(lw.q, lw.q[:, head_start * hd:q_end]),
        "k": _c(lw.k, lw.k[:, :kv_dim]),
        "v": None,
        "o": _c(lw.o, lw.o[:, :o_dim]) if lw.o is not None else None,
        "has_v_proj": lw.has_v_proj,
    }
    if lw.v is not None:
        result["v"] = _c(lw.v, lw.v[:, :kv_dim])
    elif lw.has_v_proj:
        if isinstance(lw.k, torch.Tensor):
            result["v"] = _c(lw.k, lw.k[:, :kv_dim])
        else:
            result["v"] = result["k"].copy() if copy_weights else result["k"]
    return result


def _slice_ffn_for_node(full: FullWeights, layer_idx: int, ffn_start: int, ffn_end: int,
                         copy_weights: bool = True) -> dict:
    lw = full.layer_weights[layer_idx]

    def _c(arr, slc):
        if arr is None:
            return None
        if isinstance(arr, torch.Tensor):
            return np.ascontiguousarray(slc.to(torch.float32).cpu().numpy())
        return slc.copy() if copy_weights else _ensure_f32(slc)

    return {
        "gate": _c(lw.ffn_gate, lw.ffn_gate[:, ffn_start:ffn_end]),
        "up": _c(lw.ffn_up, lw.ffn_up[:, ffn_start:ffn_end]),
        "down": _c(lw.ffn_down, lw.ffn_down[ffn_start:ffn_end, :]),
    }


class WeightProvider:
    """Provides sliced weights for each node and layer."""

    def __init__(self, model_id: str, partitions: Dict[int, dict], num_layers: int = 0):
        self.model_id = model_id
        self.partitions = partitions
        self.full = _load_full_weights(model_id, num_layers if num_layers > 0 else None)
        self._layer_props = self.full.layer_props

    def get_layer_props(self, layer_idx: int) -> LayerProperties:
        return self._layer_props.get(layer_idx, LayerProperties.standard(self.full.head_dim))

    def get_unique_layer_types(self) -> Dict[tuple, int]:
        types = {}
        for i, p in self._layer_props.items():
            key = (p.head_dim, p.attention_type, p.rope_fraction, p.use_v_norm)
            if key not in types:
                types[key] = i
        return types

    @property
    def hidden_dim(self) -> int:
        return self.full.hidden_dim

    @property
    def n_heads(self) -> int:
        return self.full.n_heads

    @property
    def n_kv_heads(self) -> int:
        return self.full.n_kv_heads

    @property
    def head_dim(self) -> int:
        return self.full.head_dim

    @property
    def ffn_dim(self) -> int:
        return self.full.ffn_dim

    @property
    def vocab_size(self) -> int:
        return self.full.vocab_size

    @property
    def num_layers(self) -> int:
        return self.full.num_layers

    def _layer_weights_for_node(self, layer_idx: int, p: dict, full_q: bool, copy_weights: bool = True) -> dict:
        lw = self.full.layer_weights[layer_idx]
        props = self._layer_props.get(layer_idx, LayerProperties.standard(self.full.head_dim))
        hd = props.head_dim

        def _tslice(arr, slc):
            """Slice a torch.Tensor or np.ndarray, returning float32 numpy."""
            if arr is None:
                return None
            if isinstance(arr, torch.Tensor):
                return np.ascontiguousarray(slc.to(torch.float32).cpu().numpy())
            return slc.copy() if copy_weights else _ensure_f32(slc)

        def _slice(arr, key_end, key_start=0):
            if arr is None:
                return None
            if isinstance(arr, torch.Tensor):
                slc = arr[:, key_start:key_end] if arr.ndim > 1 else arr[key_start:key_end]
            else:
                slc = arr[:, key_start:key_end] if arr.ndim > 1 else arr[key_start:key_end]
            return _tslice(arr, slc)

        def _slice_row(arr, key_start, key_end):
            if arr is None:
                return None
            if isinstance(arr, torch.Tensor):
                slc = arr[key_start:key_end, :]
            else:
                slc = arr[key_start:key_end, :]
            return _tslice(arr, slc)

        if full_q:
            attn = {
                "q": _slice(lw.q, self.full.n_heads * hd),
                "k": _slice(lw.k, self.full.n_kv_heads * hd),
                "v": _slice(lw.v, self.full.n_kv_heads * hd) if lw.v is not None else None,
                "o": _slice(lw.o, self.full.n_heads * hd) if lw.o is not None else None,
                "has_v_proj": lw.has_v_proj,
                "q_bias": _slice(lw.q_bias, self.full.n_heads * hd) if lw.q_bias is not None else None,
                "k_bias": _slice(lw.k_bias, self.full.n_kv_heads * hd) if lw.k_bias is not None else None,
                "v_bias": _slice(lw.v_bias, self.full.n_kv_heads * hd) if lw.v_bias is not None else None,
            }
        else:
            ffn_width = p["ffn_end"] - p["ffn_start"]
            n_q_local = ffn_width // hd if ffn_width > 0 else self.full.n_heads
            attn = _slice_attn_for_node(self.full, layer_idx, 0, n_q_local,
                                        copy_weights=copy_weights)
        ffn = _slice_ffn_for_node(self.full, layer_idx, p["ffn_start"], p["ffn_end"],
                                   copy_weights=copy_weights)
        props_dict = {
            "head_dim": hd,
            "has_v_proj": props.has_v_proj,
            "rope_fraction": props.rope_fraction,
            "use_v_norm": props.use_v_norm,
            "attention_type": props.attention_type,
        }
        if props.kv_source_layer is not None:
            props_dict["kv_source_layer"] = props.kv_source_layer
        result = {
            "attn": attn,
            "ffn": ffn,
            "_props": props_dict,
            "input_layernorm": _ensure_f32(lw.input_layernorm) if lw.input_layernorm is not None else None,
            "post_attention_layernorm": _ensure_f32(lw.post_attention_layernorm) if lw.post_attention_layernorm is not None else None,
        }
        # Include PLE per-layer weights (root only)
        ple_g = lw.ple_gate
        if ple_g is not None:
            result["ple_gate"] = _ensure_f32(ple_g) if not copy_weights else ple_g.copy()
            result["ple_proj"] = _ensure_f32(lw.ple_proj) if not copy_weights else lw.ple_proj.copy()
            result["ple_post_norm"] = _ensure_f32(lw.ple_post_norm) if not copy_weights else lw.ple_post_norm.copy() if lw.ple_post_norm is not None else None
        return result

    def get_node_weights(self, node_id: int, num_nodes: int) -> Dict[int, dict]:
        """Returns {layer_idx: {attn: ..., ffn: ...}} for a worker node (full Q, with copies)."""
        p = self.partitions[node_id]
        node_layers = {}
        for layer_idx in range(self.full.num_layers):
            node_layers[layer_idx] = self._layer_weights_for_node(layer_idx, p, full_q=True, copy_weights=True)
        return node_layers

    def get_root_weights(self, node_id: int) -> Dict[int, dict]:
        """Returns {layer_idx: {attn: ..., ffn: ...}} for the root node. Uses safe copies."""
        p = self.partitions[node_id]
        node_layers = {}
        for layer_idx in range(self.full.num_layers):
            node_layers[layer_idx] = self._layer_weights_for_node(layer_idx, p, full_q=True, copy_weights=True)
        return node_layers

    def get_embedding(self) -> np.ndarray:
        return self.full.embedding

    def get_lm_head(self) -> np.ndarray:
        return self.full.lm_head

    def get_final_norm(self) -> Optional[np.ndarray]:
        return self.full.final_norm

    def get_ple_embedding(self) -> Optional[np.ndarray]:
        return self.full.ple_embedding

    def get_ple_projection(self) -> Optional[np.ndarray]:
        return self.full.ple_projection

    def get_ple_projection_norm(self) -> Optional[np.ndarray]:
        return self.full.ple_projection_norm

    @property
    def ple_dim(self) -> int:
        return self.full.ple_dim

    @property
    def ple_vocab_size(self) -> int:
        return self.full.ple_vocab_size


SMOLM_135M_CONFIG = {
    "hidden_dim": 576,
    "n_heads": 9,
    "n_kv_heads": 3,
    "head_dim": 64,
    "ffn_dim": 1536,
    "vocab_size": 49152,
    "num_layers": 30,
}


def get_smollm_config() -> dict:
    return dict(SMOLM_135M_CONFIG)


def load_real_weights(model_name: str = "HuggingFaceTB/SmolLM-135M", layer_idx: int = 0) -> dict:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name)
    state = model.state_dict()

    def _w(name):
        return _ensure_f32(_torch_to_np(state[name], transpose=True))

    def _norm(name):
        return state[name].to(torch.float32).cpu().numpy()

    weights = {
        "q_weight": _w(f"model.layers.{layer_idx}.self_attn.q_proj.weight"),
        "k_weight": _w(f"model.layers.{layer_idx}.self_attn.k_proj.weight"),
        "v_weight": _w(f"model.layers.{layer_idx}.self_attn.v_proj.weight"),
        "ffn_gate": _w(f"model.layers.{layer_idx}.mlp.gate_proj.weight"),
        "ffn_up": _w(f"model.layers.{layer_idx}.mlp.up_proj.weight"),
        "ffn_down": _w(f"model.layers.{layer_idx}.mlp.down_proj.weight"),
        "input_layernorm": _norm(f"model.layers.{layer_idx}.input_layernorm.weight"),
        "post_attention_layernorm": _norm(f"model.layers.{layer_idx}.post_attention_layernorm.weight"),
        "lm_head": _ensure_f32(_torch_to_np(
            state.get("lm_head.weight", state["model.embed_tokens.weight"]), transpose=False,
        )),
        "embedding": _ensure_f32(_torch_to_np(
            state["model.embed_tokens.weight"], transpose=False,
        )),
        "final_norm": _norm("model.norm.weight"),
    }
    return weights


def create_synthetic_weights(config: dict) -> dict:
    hidden_dim = config["hidden_dim"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = config["head_dim"]
    ffn_dim = config["ffn_dim"]
    vocab_size = config["vocab_size"]
    num_layers = config.get("num_layers", 8)
    ple_dim = config.get("ple_dim", 0)

    total_nodes = 3
    ffn_width_per_node = ffn_dim // total_nodes
    n_q_per_node = ffn_width_per_node // head_dim
    total_q_heads = n_q_per_node * total_nodes

    result = {
        "q_weight": np.random.randn(hidden_dim, total_q_heads * head_dim).astype(np.float32),
        "k_weight": np.random.randn(hidden_dim, n_kv_heads * head_dim).astype(np.float32),
        "v_weight": np.random.randn(hidden_dim, n_kv_heads * head_dim).astype(np.float32),
        "o_weight": np.random.randn(n_heads * head_dim, hidden_dim).astype(np.float32),
        "ffn_gate": np.random.randn(hidden_dim, ffn_dim).astype(np.float32),
        "ffn_up": np.random.randn(hidden_dim, ffn_dim).astype(np.float32),
        "ffn_down": np.random.randn(ffn_dim, hidden_dim).astype(np.float32),
        "lm_head": np.random.randn(vocab_size, hidden_dim).astype(np.float32),
        "embedding": np.random.randn(vocab_size, hidden_dim).astype(np.float32),
        "final_norm": np.random.randn(hidden_dim).astype(np.float32),
    }
    for i in range(num_layers):
        result[f"input_layernorm_{i}"] = np.random.randn(hidden_dim).astype(np.float32)
        result[f"post_attention_layernorm_{i}"] = np.random.randn(hidden_dim).astype(np.float32)
    if ple_dim > 0:
        ple_vocab = config.get("ple_vocab_size", vocab_size)
        result["ple_embedding"] = np.random.randn(ple_vocab, num_layers * ple_dim).astype(np.float32)
        result["ple_projection"] = np.random.randn(hidden_dim, num_layers * ple_dim).astype(np.float32)
        result["ple_projection_norm"] = np.random.randn(ple_dim).astype(np.float32)
        for i in range(num_layers):
            result[f"ple_gate_{i}"] = np.random.randn(hidden_dim, ple_dim).astype(np.float32)
            result[f"ple_proj_{i}"] = np.random.randn(ple_dim, hidden_dim).astype(np.float32)
            result[f"ple_post_norm_{i}"] = np.random.randn(hidden_dim).astype(np.float32)
    return result


def slice_ffn_weights(full_weights: dict, ffn_start: int, ffn_end: int) -> Dict[str, np.ndarray]:
    width = ffn_end - ffn_start
    return {
        "ffn_gate": _ensure_f32(full_weights["ffn_gate"][:, ffn_start:ffn_end]),
        "ffn_up": _ensure_f32(full_weights["ffn_up"][:, ffn_start:ffn_end]),
        "ffn_down": _ensure_f32(full_weights["ffn_down"][ffn_start:ffn_end, :]),
    }


def slice_attention_weights(full_weights: dict, head_start: int, head_end: int, n_kv_heads: int, head_dim: int) -> Dict[str, np.ndarray]:
    q_dim_start = head_start * head_dim
    q_dim_end = head_end * head_dim
    kv_dim = n_kv_heads * head_dim
    return {
        "q_slice": _ensure_f32(full_weights["q_weight"][:, q_dim_start:q_dim_end]),
        "k_full": _ensure_f32(full_weights["k_weight"][:, :kv_dim]),
        "v_full": _ensure_f32(full_weights["v_weight"][:, :kv_dim]),
    }


def slice_weights_for_node(node_shard: dict, full_weights: dict, total_nodes: int, config: dict) -> dict:
    head_dim = config["head_dim"]
    n_kv_heads = config["n_kv_heads"]
    hidden_dim = config["hidden_dim"]
    ffn_width = node_shard["ffn_end"] - node_shard["ffn_start"]
    n_q_local = ffn_width // head_dim
    q_dim_start = 0
    q_dim_end = n_q_local * head_dim
    kv_dim = n_kv_heads * head_dim
    attn = {
        "q": _ensure_f32(full_weights["q_weight"][:, q_dim_start:q_dim_end]),
        "k": _ensure_f32(full_weights["k_weight"][:, :kv_dim]),
        "v": _ensure_f32(full_weights["v_weight"][:, :kv_dim]),
    }
    ffn = {
        "gate": _ensure_f32(full_weights["ffn_gate"][:, node_shard["ffn_start"]:node_shard["ffn_end"]]),
        "up": _ensure_f32(full_weights["ffn_up"][:, node_shard["ffn_start"]:node_shard["ffn_end"]]),
        "down": _ensure_f32(full_weights["ffn_down"][node_shard["ffn_start"]:node_shard["ffn_end"], :]),
    }
    return {"attn": attn, "ffn": ffn}


def validate_weight_shapes(node_weights: Dict[int, dict], partitions: Dict[int, dict], config: dict) -> bool:
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
        assert w["gate"].shape == (hidden_dim, width), (
            f"Node {node_id} ffn_gate shape {w['gate'].shape} != ({hidden_dim}, {width})"
        )
        assert w["up"].shape == (hidden_dim, width), (
            f"Node {node_id} ffn_up shape {w['up'].shape} != ({hidden_dim}, {width})"
        )
        assert w["down"].shape == (width, hidden_dim), (
            f"Node {node_id} ffn_down shape {w['down'].shape} != ({width}, {hidden_dim})"
        )

        a = weights["attn"]
        assert a["q"].shape == (hidden_dim, n_q_local * head_dim), (
            f"Node {node_id} q_slice shape {a['q'].shape} != ({hidden_dim}, {n_q_local * head_dim})"
        )
        assert a["k"].shape == (hidden_dim, n_kv_local * head_dim), (
            f"Node {node_id} k_full shape {a['k'].shape} != ({hidden_dim}, {n_kv_local * head_dim})"
        )
        assert a["v"].shape == (hidden_dim, n_kv_local * head_dim), (
            f"Node {node_id} v_full shape {a['v'].shape} != ({hidden_dim}, {n_kv_local * head_dim})"
        )

    return True
