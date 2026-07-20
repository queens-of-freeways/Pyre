from __future__ import annotations

import math
import pickle
import socket
import struct
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.driver import CPU

from src.attention.builder import build_ulysses_attention_graph, ShardSpec as AttentionShardSpec
from src.ffn.builder import build_ffn_graph, ShardSpec as FFNShardSpec
from src.orchestrator.cluster import ModelConfig
from src.orchestrator.protocol import (
    MSG_SHARD_SPEC, MSG_READY, MSG_FORWARD_DATA, MSG_FORWARD_RESULT,
    MSG_SHUTDOWN, MSG_ATTN_OUTPUT, MSG_FFN_RESULT, MSG_INIT_WEIGHTS,
)


def _recv_exact(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


class RootNode:
    def __init__(
        self,
        worker_addrs: List[Tuple[str, int]],
        config: ModelConfig,
        all_layer_weights: Dict[int, Dict[int, dict]] = None,
        ple_embedding: Optional[np.ndarray] = None,
        ple_projection: Optional[np.ndarray] = None,
        ple_projection_norm: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.all_layer_weights = all_layer_weights or {}
        self.device = DeviceRef.CPU()
        self.session = InferenceSession(devices=[CPU()])
        self.ple_embedding = ple_embedding
        self.ple_projection = ple_projection
        self.ple_projection_norm = ple_projection_norm
        self.ple_dim = config.ple_dim if hasattr(config, 'ple_dim') else 0

        total_nodes = 1 + len(worker_addrs)
        self.partitions = self._solve_partitions(total_nodes)
        self.worker_addrs = worker_addrs

        p0 = self.partitions[0]
        seq_len = p0["seq_end"] - p0["seq_start"]

        needed_hds = {config.head_dim}
        if self.all_layer_weights and 0 in self.all_layer_weights:
            for lidx, lw in self.all_layer_weights[0].items():
                p = lw.get("_props", {})
                needed_hds.add(p.get("head_dim", config.head_dim))

        attn_shard_base = AttentionShardSpec(
            ffn_dim_start=p0["ffn_start"], ffn_dim_end=p0["ffn_end"],
            seq_start=p0["seq_start"], seq_end=p0["seq_end"],
        )
        self.attn_models = {}
        for hd in needed_hds:
            attn_graph = build_ulysses_attention_graph(
                attn_shard_base, config.hidden_dim, config.n_heads, config.n_kv_heads,
                hd, self.device, full_q_weights=True,
            )
            self.attn_models[hd] = self.session.load(attn_graph)

        ffn_shard = FFNShardSpec(
            ffn_dim_start=p0["ffn_start"], ffn_dim_end=p0["ffn_end"],
        )
        ffn_graph = build_ffn_graph(
            ffn_shard, config.hidden_dim, self.device,
            seq_len=seq_len, gated=True,
        )
        self.ffn_model = self.session.load(ffn_graph)

        self.worker_conns = []
        self.worker_ids = []
        for i, (host, port) in enumerate(worker_addrs):
            worker_id = i + 1
            conn = self._connect_worker(host, port)
            p = self.partitions[worker_id]
            shard_spec = AttentionShardSpec(
                ffn_dim_start=p["ffn_start"], ffn_dim_end=p["ffn_end"],
                seq_start=p["seq_start"], seq_end=p["seq_end"],
            )
            self._send_msg(conn, MSG_SHARD_SPEC, (shard_spec, config))
            msg_type, _ = self._recv_msg(conn)
            if msg_type != MSG_READY:
                raise RuntimeError(f"Expected READY from worker {worker_id}, got {msg_type}")

            worker_data = {"weights": {}, "props": {}, "graph_key": {}}
            if all_layer_weights and worker_id in all_layer_weights:
                wl = all_layer_weights[worker_id]
                worker_data["weights"] = {k: v for k, v in wl.items()}
                for lidx, lw in wl.items():
                    lp = lw.get("_props", {})
                    worker_data["props"][lidx] = lp
                    hd = lp.get("head_dim", config.head_dim)
                    worker_data["graph_key"][lidx] = (hd, 0, 0, 0, 0)
            self._send_msg(conn, MSG_INIT_WEIGHTS, worker_data)

            self.worker_conns.append(conn)
            self.worker_ids.append(worker_id)

    def _solve_partitions(self, n):
        ffn_dim = self.config.ffn_dim
        seq_len = 64
        partitions = {}
        ffn_chunk = ffn_dim // n
        seq_chunk = seq_len // n
        for i in range(n):
            ffn_start = i * ffn_chunk
            ffn_end = (i + 1) * ffn_chunk if i < n - 1 else ffn_dim
            seq_start = i * seq_chunk
            seq_end = (i + 1) * seq_chunk if i < n - 1 else seq_len
            partitions[i] = {
                "ffn_start": ffn_start,
                "ffn_end": ffn_end,
                "seq_start": seq_start,
                "seq_end": seq_end,
            }
        return partitions

    def _connect_worker(self, host, port, max_retries=10, delay=0.2):
        for attempt in range(max_retries):
            try:
                conn = socket.create_connection((host, port), timeout=10)
                conn.settimeout(120.0)
                return conn
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    raise

    @staticmethod
    def _softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        exp = np.exp(x - x_max)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    @staticmethod
    def _rms_norm(x, weight, eps=1e-6):
        variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
        x_norm = x / np.sqrt(variance + eps)
        return (x_norm * weight).astype(np.float32)

    @staticmethod
    def _apply_rope(x, rope_fraction=1.0, theta=10000.0):
        batch, n_heads, seq_len, head_dim = x.shape
        dims = int(head_dim * rope_fraction)
        if dims < 2:
            return x
        pos = np.arange(seq_len, dtype=np.float32)
        freq = 1.0 / (theta ** (np.arange(0, dims, 2, dtype=np.float32) / dims))
        cos = np.cos(pos[:, None] * freq[None, :])
        sin = np.sin(pos[:, None] * freq[None, :])
        x_rot = x[:, :, :, :dims]
        x_left = x_rot[..., ::2] * cos - x_rot[..., 1::2] * sin
        x_right = x_rot[..., ::2] * sin + x_rot[..., 1::2] * cos
        out = x.copy()
        out[:, :, :, :dims:2] = x_left
        out[:, :, :, 1:dims:2] = x_right
        return out

    @staticmethod
    def _apply_v_norm(v):
        rms = np.sqrt(np.mean(v.astype(np.float64) ** 2, axis=-1, keepdims=True) + 1e-6)
        return (v / rms).astype(np.float32)

    def _compute_attention(self, all_qkv, layer_idx: int = 0,
                            kv_cache: Optional[dict] = None):
        head_dim = self.config.head_dim
        n_heads = self.config.n_heads
        n_kv = self.config.n_kv_heads

        # Check if this layer shares KV from another (KV shared cache)
        props = self.all_layer_weights.get(0, {}).get(layer_idx, {}).get("_props", {})
        kv_source = props.get("kv_source_layer", None)
        if kv_source is not None and kv_cache is not None and kv_source in kv_cache:
            # Reuse cached K,V from source layer
            k_cached, v_cached = kv_cache[kv_source]
            ids = [0] + self.worker_ids
            q_full = np.concatenate([all_qkv[i][0] for i in ids], axis=1)
            k_full = k_cached
            v_full = v_cached
        else:
            ids = [0] + self.worker_ids
            q_full = np.concatenate([all_qkv[i][0] for i in ids], axis=1)
            k_full = np.concatenate([all_qkv[i][1] for i in ids], axis=1)
            v_full = np.concatenate([all_qkv[i][2] for i in ids], axis=1)

        full_seq = q_full.shape[1]
        head_dim = q_full.shape[3]
        q = q_full.transpose(0, 2, 1, 3)
        k = k_full.transpose(0, 2, 1, 3)
        v = v_full.transpose(0, 2, 1, 3)

        rope_frac = props.get("rope_fraction", 1.0)
        use_vn = props.get("use_v_norm", False)

        if rope_frac > 0:
            theta = getattr(self.config, 'rope_theta', 10000.0)
            q = self._apply_rope(q, rope_fraction=rope_frac, theta=theta)
            k = self._apply_rope(k, rope_fraction=rope_frac, theta=theta)

        q = np.clip(q, -1000, 1000)
        k = np.clip(k, -1000, 1000)
        v = np.clip(v, -1000, 1000)

        if use_vn:
            v = self._apply_v_norm(v)

        n_q_per_kv = n_heads // n_kv
        k_exp = k[:, :, None, :, :].repeat(n_q_per_kv, axis=2).reshape(1, n_heads, full_seq, head_dim)
        v_exp = v[:, :, None, :, :].repeat(n_q_per_kv, axis=2).reshape(1, n_heads, full_seq, head_dim)

        scale = np.float32(np.sqrt(head_dim))
        scores = q @ k_exp.transpose(0, 1, 3, 2) / scale
        scores = np.clip(scores, -500, 500)
        probs = self._softmax(scores, axis=-1)
        attn = probs @ v_exp
        attn = attn.transpose(0, 2, 1, 3).reshape(1, full_seq, n_heads * head_dim)

        # Cache this layer's K,V if it might be shared by later layers
        if kv_source is None and kv_cache is not None:
            kv_cache[layer_idx] = (k_full, v_full)

        return attn.astype(np.float32)

    def _compute_ple_signal(self, input_ids: np.ndarray, x: np.ndarray) -> Optional[np.ndarray]:
        """Compute per-layer PLE signal: [batch, seq, num_layers, ple_dim]."""
        if self.ple_embedding is None:
            return None
        # Token-identity component
        input_ids_clipped = np.clip(input_ids, 0, self.ple_embedding.shape[0] - 1)
        ple_token = self.ple_embedding[input_ids_clipped]  # [batch, seq, num_layers * ple_dim]
        # Context-aware component
        ple_context = x @ self.ple_projection  # [batch, seq, num_layers * ple_dim]
        ple_context *= (self.config.hidden_dim ** -0.5)
        # Combine
        ple_all = (ple_context + ple_token) * (2.0 ** -0.5)
        # Reshape to separate layers
        ple_all = ple_all.reshape(*x.shape[:2], self.config.num_layers, self.ple_dim)
        # Apply RMSNorm along ple_dim
        if self.ple_projection_norm is not None:
            mean_sq = np.mean(ple_all.astype(np.float64) ** 2, axis=-1, keepdims=True)
            ple_all = (ple_all / np.sqrt(mean_sq + 1e-6) * self.ple_projection_norm).astype(np.float32)
        return ple_all

    @staticmethod
    def _apply_ple_to_hidden(hidden: np.ndarray, ple_signal: np.ndarray,
                              ple_gate: np.ndarray, ple_proj: np.ndarray,
                              ple_post_norm: np.ndarray) -> np.ndarray:
        """Apply Per-Layer Embeddings to hidden states (Gemma 4 decoder layer style)."""
        # Define gelu activation (clamp input to prevent overflow in x**3)
        def gelu(x):
            x_clamped = np.clip(x, -100, 100)
            return 0.5 * x_clamped * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x_clamped + 0.044715 * x_clamped ** 3)))
        residual = hidden.copy()
        gated = hidden @ ple_gate  # [batch, seq, hidden] -> [batch, seq, ple_dim]
        gated = gelu(gated)
        gated = gated * ple_signal
        hidden_add = gated @ ple_proj  # [batch, seq, ple_dim] -> [batch, seq, hidden]
        # RMSNorm
        mean_sq = np.mean(hidden_add.astype(np.float64) ** 2, axis=-1, keepdims=True)
        hidden_add = (hidden_add / np.sqrt(mean_sq + 1e-6) * ple_post_norm).astype(np.float32)
        return residual + hidden_add

    def run(self, x: np.ndarray, input_ids: Optional[np.ndarray] = None) -> np.ndarray:
        batch, seq_len, hidden_dim = x.shape
        assert hidden_dim == self.config.hidden_dim

        if self.all_layer_weights and 0 in self.all_layer_weights:
            num_layers = len(self.all_layer_weights[0])
        else:
            num_layers = 1

        # Pre-compute PLE signal once
        ple_all = self._compute_ple_signal(input_ids, x) if (
            self.ple_embedding is not None and input_ids is not None
        ) else None

        kv_cache = {}
        for layer_idx in range(num_layers):
            ple_slice = ple_all[:, :, layer_idx, :] if ple_all is not None else None
            h = self._run_single_layer(x, layer_idx, ple_slice, kv_cache)
            x = h

        return x

    def _run_single_layer(self, x: np.ndarray, layer_idx: int,
                           ple_slice: Optional[np.ndarray] = None,
                           kv_cache: Optional[dict] = None) -> np.ndarray:
        batch, seq_len, hidden_dim = x.shape
        head_dim = self.config.head_dim
        n_heads = self.config.n_heads
        n_kv = self.config.n_kv_heads

        root_w = {}
        if self.all_layer_weights and 0 in self.all_layer_weights:
            root_w = self.all_layer_weights[0].get(layer_idx, {})
        rw_attn = root_w.get("attn", {})

        # ---- Pre-attention RMSNorm ----
        input_ln = root_w.get("input_layernorm")
        if input_ln is not None:
            x_norm = self._rms_norm(x, input_ln)
        else:
            variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
            x_norm = (x / np.sqrt(variance + 1e-6)).astype(np.float32)

        # ---- QKV projection on normed input ----
        p0 = self.partitions[0]
        x_root = x_norm[:, p0["seq_start"]:p0["seq_end"], :]

        layer_hd = root_w.get("_props", {}).get("head_dim", head_dim)
        wq_r = rw_attn.get("q", np.random.randn(hidden_dim, n_heads * layer_hd).astype(np.float32))
        wk_r = rw_attn.get("k", np.random.randn(hidden_dim, n_kv * layer_hd).astype(np.float32))
        wv_r = rw_attn.get("v", np.random.randn(hidden_dim, n_kv * layer_hd).astype(np.float32))

        hd = root_w.get("_props", {}).get("head_dim", self.config.head_dim)
        attn_model = self.attn_models.get(hd, list(self.attn_models.values())[0])

        v_arr = wv_r if wv_r is not None else wk_r
        (q_root, k_root, v_root) = attn_model.execute(
            np.ascontiguousarray(x_root),
            np.ascontiguousarray(wq_r),
            np.ascontiguousarray(wk_r),
            np.ascontiguousarray(v_arr),
        )
        all_qkv = {0: (q_root.to_numpy(), k_root.to_numpy(), v_root.to_numpy())}

        # ---- Send normed slices to workers for QKV ----
        for idx, worker_id in enumerate(self.worker_ids):
            p = self.partitions[worker_id]
            x_worker = x_norm[:, p["seq_start"]:p["seq_end"], :]
            self._send_msg(
                self.worker_conns[idx], MSG_FORWARD_DATA,
                (layer_idx, x_worker),
            )

        for idx, worker_id in enumerate(self.worker_ids):
            _, qkv = self._recv_msg(self.worker_conns[idx])
            all_qkv[worker_id] = qkv

        # ---- Attention (with RoPE applied inside) ----
        attn_out = self._compute_attention(all_qkv, layer_idx, kv_cache)

        # Output projection
        o_weight = root_w.get("attn", {}).get("o")
        if o_weight is not None:
            attn_out = attn_out @ o_weight

        # ---- Residual: h = x + attn_out ----
        h = x + attn_out

        # ---- Pre-FFN RMSNorm ----
        post_attn_ln = root_w.get("post_attention_layernorm")
        if post_attn_ln is not None:
            h_norm = self._rms_norm(h, post_attn_ln)
        else:
            variance = np.mean(h.astype(np.float64) ** 2, axis=-1, keepdims=True)
            h_norm = (h / np.sqrt(variance + 1e-6)).astype(np.float32)

        # ---- Send normed slices to workers for FFN ----
        for idx, worker_id in enumerate(self.worker_ids):
            p = self.partitions[worker_id]
            self._send_msg(self.worker_conns[idx], MSG_ATTN_OUTPUT, (h_norm[:, p["seq_start"]:p["seq_end"], :],))

        partials = {}
        for idx, worker_id in enumerate(self.worker_ids):
            _, partial = self._recv_msg(self.worker_conns[idx])
            partials[worker_id] = partial

        # ---- Root FFN on normed hidden ----
        attn_root = h_norm[:, p0["seq_start"]:p0["seq_end"], :]
        width0 = p0["ffn_end"] - p0["ffn_start"]

        rw_ffn = root_w.get("ffn", {})
        ffn_gate_r = rw_ffn.get("gate", np.random.randn(hidden_dim, width0).astype(np.float32))
        ffn_up_r = rw_ffn.get("up", np.random.randn(hidden_dim, width0).astype(np.float32))
        ffn_down_r = rw_ffn.get("down", np.random.randn(width0, hidden_dim).astype(np.float32))

        (partial_root,) = self.ffn_model.execute(
            np.ascontiguousarray(attn_root),
            np.ascontiguousarray(ffn_gate_r),
            np.ascontiguousarray(ffn_up_r),
            np.ascontiguousarray(ffn_down_r),
        )
        partial_root = partial_root.to_numpy()

        # ---- Sum FFN partials ----
        ffn_out = np.zeros((batch, seq_len, hidden_dim), dtype=np.float32)
        ffn_out[:, p0["seq_start"]:p0["seq_end"], :] += partial_root
        for worker_id in self.worker_ids:
            p = self.partitions[worker_id]
            ffn_out[:, p["seq_start"]:p["seq_end"], :] += partials[worker_id]

        # ---- Residual: output = h + ffn_out ----
        final_output = h + ffn_out

        # ---- Apply PLE after FFN + residual (Gemma 4 pattern) ----
        if ple_slice is not None:
            ple_gate = root_w.get("ple_gate")
            if ple_gate is not None:
                final_output = self._apply_ple_to_hidden(
                    final_output, ple_slice,
                    ple_gate,
                    root_w["ple_proj"],
                    root_w["ple_post_norm"],
                )

        return final_output

    def shutdown(self):
        for conn in self.worker_conns:
            try:
                self._send_msg(conn, MSG_SHUTDOWN)
                conn.close()
            except Exception:
                pass

    def _send_msg(self, conn, msg_type, obj=None):
        payload = pickle.dumps(obj) if obj is not None else b""
        header = struct.pack("!II", msg_type, len(payload))
        conn.sendall(header + payload)

    def _recv_msg(self, conn):
        header = _recv_exact(conn, 8)
        msg_type, payload_len = struct.unpack("!II", header)
        payload = _recv_exact(conn, payload_len)
        return msg_type, pickle.loads(payload) if payload else None
