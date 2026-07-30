from __future__ import annotations

import socket
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.driver import CPU

from src.attention.builder import build_ulysses_attention_graph, ShardSpec as AttentionShardSpec
from src.ffn.builder import build_ffn_graph, ShardSpec as FFNShardSpec
from src.orchestrator.cluster import ModelConfig, AdaptivePartitioner
from src.orchestrator.net import send_msg, recv_msg, recv_exact
from src.orchestrator.quantizer import quantize_weights_dict, dequantize_weights_dict
from src.orchestrator.protocol import (
    MSG_SHARD_SPEC, MSG_READY, MSG_FORWARD_DATA, MSG_FORWARD_RESULT,
    MSG_SHUTDOWN, MSG_ATTN_OUTPUT, MSG_FFN_RESULT, MSG_DECODE_STEP, MSG_DECODE_QKV,
)


_PYRE_KERNELS = None

def _arch_suffix():
    import platform
    m = platform.machine()
    return {"AMD64": "x86_64", "x86_64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(m, m)

def _project_root():
    import os
    d = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return d  # .../distributed-llama-mojo

def _get_pyre_kernels():
    global _PYRE_KERNELS
    if _PYRE_KERNELS is not None:
        return _PYRE_KERNELS
    import os, sys, subprocess, importlib.util
    root = _project_root()
    kernels_dir = os.path.join(root, 'src', 'kernels')
    arch = _arch_suffix()
    so_name = f'pyre_kernels_{arch}.so'
    so_path = os.path.join(kernels_dir, so_name)
    if os.path.isfile(so_path):
        spec = importlib.util.spec_from_file_location("pyre_kernels", so_path)
        if spec is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["pyre_kernels"] = mod
            _PYRE_KERNELS = mod
            return _PYRE_KERNELS
    # Not found — try to compile from source.
    mojo_src = os.path.join(kernels_dir, 'mojo', 'pyre_kernels.mojo')
    if not os.path.isfile(mojo_src):
        raise ImportError(
            f"Mojo kernel .so not found for arch '{arch}' and source "
            f"missing at {mojo_src}. Cannot run without compiled kernels."
        )
    print(f"[pyre] Compiling Mojo kernels for {arch}...")
    ret = subprocess.run(
        ["mojo", "build", "--emit", "shared-lib", "-I", root,
         "-o", so_path, mojo_src],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        raise ImportError(
            f"Mojo compilation failed (exit {ret.returncode}):\n"
            f"{ret.stderr.strip()}"
        )
    if not os.path.isfile(so_path):
        raise ImportError("Mojo compilation succeeded but output .so not found.")
    spec = importlib.util.spec_from_file_location("pyre_kernels", so_path)
    if spec is None:
        raise ImportError(f"Could not create module spec for {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["pyre_kernels"] = mod
    _PYRE_KERNELS = mod
    return _PYRE_KERNELS


class RootNode:
    def __init__(
        self,
        worker_addrs: List[Tuple[str, int]],
        config: ModelConfig,
        model_id: str = "",
        weight_provider: Optional["WeightProvider"] = None,
        ple_embedding: Optional[np.ndarray] = None,
        ple_projection: Optional[np.ndarray] = None,
        ple_projection_norm: Optional[np.ndarray] = None,
        num_layers: int = 0,
    ):
        self.config = config
        self.num_layers = num_layers if num_layers > 0 else config.num_layers
        self.device = DeviceRef.CPU()
        self.session = InferenceSession(devices=[CPU()])
        self.ple_embedding = ple_embedding
        self.ple_projection = ple_projection
        self.ple_projection_norm = ple_projection_norm
        self.ple_dim = config.ple_dim if hasattr(config, 'ple_dim') else 0
        self._wp = weight_provider
        self._weight_cache: Dict[int, dict] = {}

        total_nodes = 1 + len(worker_addrs)
        self.partitions = self._solve_partitions(total_nodes)
        self.worker_addrs = worker_addrs
        self._use_head_partitioning = total_nodes <= config.n_kv_heads
        # Disable head partitioning if any layer has kv_source (cross-attention)
        if self._use_head_partitioning and weight_provider is not None:
            for lidx in range(self.num_layers):
                lp = weight_provider.get_layer_props(lidx)
                if lp is not None and lp.kv_source_layer is not None:
                    self._use_head_partitioning = False
                    break
        self.attn_models = {}
        self.ffn_model = None

        p0 = self.partitions[0]
        seq_len = p0["seq_end"] - p0["seq_start"]

        if not self._use_head_partitioning:
            needed_hds = {config.head_dim}
            if weight_provider is not None:
                for lidx in range(self.num_layers):
                    lp = weight_provider.get_layer_props(lidx)
                    if lp is not None:
                        needed_hds.add(lp.head_dim)

            attn_shard_base = AttentionShardSpec(
                ffn_dim_start=p0["ffn_start"], ffn_dim_end=p0["ffn_end"],
                seq_start=p0["seq_start"], seq_end=p0["seq_end"],
            )
            for hd in needed_hds:
                attn_graph = build_ulysses_attention_graph(
                    attn_shard_base, config.hidden_dim, config.n_heads, config.n_kv_heads,
                    hd, self.device, full_q_weights=True,
                )
                self.attn_models[hd] = self.session.load(attn_graph)

        full_seq_len = 64
        ffn_shard = FFNShardSpec(
            ffn_dim_start=p0["ffn_start"], ffn_dim_end=p0["ffn_end"],
        )
        ffn_graph = build_ffn_graph(
            ffn_shard, config.hidden_dim, self.device,
            seq_len=full_seq_len, gated=True,
        )
        self.ffn_model = self.session.load(ffn_graph)

        self.worker_conns = []
        self.worker_ids = []
        self.worker_info = []  # (conn, worker_id, partition, is_local)

        # ── Adaptive Partitioning ──────────────────────────────────────
        total_nodes = 1 + len(worker_addrs)
        self.adaptive_partitioner = AdaptivePartitioner(
            config.ffn_dim, total_nodes, seq_len=64,
        )

        # ── Phase 1: Connect to all workers and send shard specs ───────
        for i, (host, port) in enumerate(worker_addrs):
            worker_id = i + 1
            conn = self._connect_worker(host, port)
            p = self.partitions[worker_id]
            shard_spec = AttentionShardSpec(
                ffn_dim_start=p["ffn_start"], ffn_dim_end=p["ffn_end"],
                seq_start=p["seq_start"], seq_end=p["seq_end"],
                n_q_heads=p["n_q_heads"], n_kv_heads=p["n_kv_heads"],
                q_head_start=p["q_head_start"], kv_head_start=p["kv_head_start"],
            )
            send_msg(conn, MSG_SHARD_SPEC, (shard_spec, config, model_id))
            msg_type, _ = recv_msg(conn)
            if msg_type != MSG_READY:
                raise RuntimeError(f"Expected READY from worker {worker_id}, got {msg_type}")

            self.worker_conns.append(conn)
            self.worker_ids.append(worker_id)

    def _solve_partitions(self, n):
        ffn_dim = self.config.ffn_dim
        seq_len = 64
        n_heads = self.config.n_heads
        n_kv = self.config.n_kv_heads
        partitions = {}
        ffn_chunk = ffn_dim // n
        seq_chunk = seq_len // n
        nq_per_kv = n_heads // n_kv
        nkv_base = n_kv // n
        nkv_rem = n_kv % n
        kv_head_idx = 0
        q_head_idx = 0
        for i in range(n):
            ffn_start = i * ffn_chunk
            ffn_end = (i + 1) * ffn_chunk if i < n - 1 else ffn_dim
            seq_start = i * seq_chunk
            seq_end = (i + 1) * seq_chunk if i < n - 1 else seq_len
            nkv_local = nkv_base + (1 if i < nkv_rem else 0)
            n_ql = nq_per_kv * nkv_local
            partitions[i] = {
                "ffn_start": ffn_start, "ffn_end": ffn_end,
                "seq_start": seq_start, "seq_end": seq_end,
                "n_q_heads": n_ql, "n_kv_heads": nkv_local,
                "q_head_start": q_head_idx, "kv_head_start": kv_head_idx,
            }
            q_head_idx += n_ql
            kv_head_idx += nkv_local
        return partitions

    def _connect_worker(self, host, port, max_retries=10, delay=0.2):
        for attempt in range(max_retries):
            try:
                conn = socket.create_connection((host, port), timeout=10)
                conn.settimeout(600.0)
                return conn
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    raise

    @staticmethod
    def _softmax(x, axis=-1):
        out = np.empty_like(x)
        _get_pyre_kernels().softmax(x, out)
        return out

    @staticmethod
    def _rms_norm(x, weight, eps=1e-6):
        out = np.empty_like(x)
        _get_pyre_kernels().rms_norm(x, weight, out)
        return out

    @staticmethod
    def _apply_rope(x, rope_fraction=1.0, theta=10000.0, start_pos=0):
        x = np.ascontiguousarray(x)
        out = np.empty_like(x)
        _get_pyre_kernels().apply_rope(x, out, np.float32(theta), np.int32(start_pos), np.float32(rope_fraction))
        return out

    @staticmethod
    def _apply_v_norm(v):
        v = np.ascontiguousarray(v)
        out = np.empty_like(v)
        _get_pyre_kernels().apply_v_norm(v, out)
        return out

    def _compute_attention(self, all_qkv, layer_idx: int = 0,
                            kv_cache: Optional[dict] = None,
                            decode_cache: Optional[dict] = None,
                            props: Optional[dict] = None,
                            concat_by_head: bool = False) -> np.ndarray:
        head_dim = self.config.head_dim
        n_heads = self.config.n_heads
        n_kv = self.config.n_kv_heads

        axis = 2 if concat_by_head else 1

        kv_source = (props or {}).get("kv_source_layer", None)
        if kv_source is not None and kv_cache is not None and kv_source in kv_cache:
            k_cached, v_cached = kv_cache[kv_source]
            ids = [0] + self.worker_ids
            q_full = np.concatenate([all_qkv[i][0] for i in ids], axis=axis)
            k_full = k_cached
            v_full = v_cached
        else:
            ids = [0] + self.worker_ids
            q_full = np.concatenate([all_qkv[i][0] for i in ids], axis=axis)
            k_full = np.concatenate([all_qkv[i][1] for i in ids], axis=axis)
            v_full = np.concatenate([all_qkv[i][2] for i in ids], axis=axis)

        full_seq = q_full.shape[1]
        head_dim = q_full.shape[3]
        q = np.ascontiguousarray(q_full.transpose(0, 2, 1, 3))
        k = np.ascontiguousarray(k_full.transpose(0, 2, 1, 3))
        v = np.ascontiguousarray(v_full.transpose(0, 2, 1, 3))

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

        mask = np.triu(np.full((full_seq, full_seq), -np.inf, dtype=np.float32), k=1)
        scores = scores + mask

        probs = self._softmax(scores, axis=-1)
        attn = probs @ v_exp
        attn = attn.transpose(0, 2, 1, 3).reshape(1, full_seq, n_heads * head_dim)

        if kv_source is None and kv_cache is not None:
            kv_cache[layer_idx] = (k_full, v_full)
        if kv_source is None and decode_cache is not None:
            decode_cache[layer_idx] = (k, v)

        return attn.astype(np.float32)

    def _compute_ple_signal(self, input_ids: np.ndarray, x: np.ndarray) -> Optional[np.ndarray]:
        if self.ple_embedding is None:
            return None
        input_ids_clipped = np.clip(input_ids, 0, self.ple_embedding.shape[0] - 1)
        ple_token = self.ple_embedding[input_ids_clipped]
        ple_context = x @ self.ple_projection
        ple_context *= (self.config.hidden_dim ** -0.5)
        ple_all = (ple_context + ple_token) * (2.0 ** -0.5)
        ple_all = ple_all.reshape(*x.shape[:2], self.config.num_layers, self.ple_dim)
        if self.ple_projection_norm is not None:
            mean_sq = np.mean(ple_all.astype(np.float64) ** 2, axis=-1, keepdims=True)
            ple_all = (ple_all / np.sqrt(mean_sq + 1e-6) * self.ple_projection_norm).astype(np.float32)
        return ple_all

    @staticmethod
    def _apply_ple_to_hidden(hidden: np.ndarray, ple_signal: np.ndarray,
                              ple_gate: np.ndarray, ple_proj: np.ndarray,
                              ple_post_norm: np.ndarray) -> np.ndarray:
        residual = hidden.copy()
        gated = hidden @ ple_gate
        gated_out = np.empty_like(gated)
        _get_pyre_kernels().gelu(gated, gated_out)
        gated = gated_out * ple_signal
        hidden_add = gated @ ple_proj
        mean_sq = np.mean(hidden_add.astype(np.float64) ** 2, axis=-1, keepdims=True)
        hidden_add = (hidden_add / np.sqrt(mean_sq + 1e-6) * ple_post_norm).astype(np.float32)
        return residual + hidden_add

    def run(self, x: np.ndarray, input_ids: Optional[np.ndarray] = None,
            kv_cache: Optional[dict] = None, prefill: bool = True) -> np.ndarray:
        batch, seq_len, hidden_dim = x.shape
        assert hidden_dim == self.config.hidden_dim

        num_layers = self.num_layers

        self.adaptive_partitioner.reset_pass()

        if not prefill:
            return self._decode_step(x, kv_cache, input_ids)

        ple_all = self._compute_ple_signal(input_ids, x) if (
            self.ple_embedding is not None and input_ids is not None
        ) else None

        kv_cache_internal = {}
        decode_cache = {}
        for layer_idx in range(num_layers):
            ple_slice = ple_all[:, :, layer_idx, :] if ple_all is not None else None
            h = self._run_single_layer(x, layer_idx, ple_slice, kv_cache_internal, decode_cache)
            x = h

        if kv_cache is not None:
            kv_cache.clear()
            kv_cache.update(decode_cache)

        if self.adaptive_partitioner.drift_detected():
            new_parts = self.adaptive_partitioner.get_partitions()
            # Log but don't apply mid-run — apply on next init for now
            print(f"[Adaptive] Drift detected. New root FFN: "
                  f"[{new_parts[0]['ffn_start']}:{new_parts[0]['ffn_end']}) "
                  f"(was [{self.partitions[0]['ffn_start']}:{self.partitions[0]['ffn_end']}))")

        return x

    def _run_single_layer(self, x: np.ndarray, layer_idx: int,
                           ple_slice: Optional[np.ndarray] = None,
                           kv_cache: Optional[dict] = None,
                           decode_cache: Optional[dict] = None) -> np.ndarray:
        batch, seq_len, hidden_dim = x.shape
        head_dim = self.config.head_dim
        n_heads = self.config.n_heads
        n_kv = self.config.n_kv_heads

        root_w = self._get_root_layer_weights(layer_idx)
        rw_attn = root_w.get("attn", {})

        input_ln = root_w.get("input_layernorm")
        if input_ln is not None:
            x_norm = self._rms_norm(x, input_ln)
        else:
            variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
            x_norm = (x / np.sqrt(variance + 1e-6)).astype(np.float32)

        attn_type = root_w.get("_props", {}).get("attention_type", "full")

        if attn_type == "linear":
            # ── Linear attention (SSM) — skip QKV exchange ──
            prev_state = decode_cache.get(f"ssm_{layer_idx}") if decode_cache else None
            attn_out, new_state = self._run_linear_attention(x_norm, rw_attn, prev_state)
            if decode_cache is not None:
                decode_cache[f"ssm_{layer_idx}"] = new_state
        else:
            layer_hd = root_w.get("_props", {}).get("head_dim", head_dim)

            has_kv_source = (root_w.get("_props", {}) or {}).get("kv_source_layer") is not None
            if self._use_head_partitioning and not has_kv_source:
                # ── Head-partitioned attention — local QKV for ALL positions ──
                p0 = self.partitions[0]
                n_q_local = p0["n_q_heads"]
                n_kv_local = p0["n_kv_heads"]
                wq_r = rw_attn.get("q")
                wk_r = rw_attn.get("k")
                wv_r = rw_attn.get("v")
                if wq_r is None:
                    wq_r = np.random.randn(hidden_dim, n_q_local * layer_hd).astype(np.float32)
                    rw_attn["q"] = wq_r

                x_2d = x_norm.reshape(-1, hidden_dim)
                q_vals = np.empty((batch * seq_len, n_q_local * layer_hd), dtype=np.float32)
                _get_pyre_kernels().matmul_f32(x_2d, wq_r, q_vals)

                if wk_r is None:
                    wk_r = np.random.randn(hidden_dim, n_kv_local * layer_hd).astype(np.float32)
                    rw_attn["k"] = wk_r
                k_vals = np.empty((batch * seq_len, n_kv_local * layer_hd), dtype=np.float32)
                _get_pyre_kernels().matmul_f32(x_2d, wk_r, k_vals)

                if wv_r is not None:
                    v_vals = np.empty((batch * seq_len, n_kv_local * layer_hd), dtype=np.float32)
                    _get_pyre_kernels().matmul_f32(x_2d, wv_r, v_vals)
                else:
                    v_vals = k_vals.copy()

                q_vals = q_vals.reshape(batch, seq_len, n_q_local, layer_hd)
                k_vals = k_vals.reshape(batch, seq_len, n_kv_local, layer_hd)
                v_vals = v_vals.reshape(batch, seq_len, n_kv_local, layer_hd)

                q_bias = rw_attn.get("q_bias")
                k_bias = rw_attn.get("k_bias")
                v_bias = rw_attn.get("v_bias")
                if q_bias is not None:
                    q_vals += q_bias.reshape(1, 1, n_q_local, layer_hd)
                if k_bias is not None:
                    k_vals += k_bias.reshape(1, 1, n_kv_local, layer_hd)
                if v_bias is not None and wv_r is not None:
                    v_vals += v_bias.reshape(1, 1, n_kv_local, layer_hd)

                all_qkv = {0: (q_vals, k_vals, v_vals)}

                for idx, worker_id in enumerate(self.worker_ids):
                    send_msg(
                        self.worker_conns[idx], MSG_FORWARD_DATA,
                        (layer_idx, x_norm),
                    )

                for idx, worker_id in enumerate(self.worker_ids):
                    _, qkv = recv_msg(self.worker_conns[idx])
                    all_qkv[worker_id] = qkv

                attn_out = self._compute_attention(all_qkv, layer_idx, kv_cache, decode_cache,
                                                     props=root_w.get("_props", {}),
                                                     concat_by_head=True)

                # Split the full KV cache into per-node slices for decode
                if decode_cache is not None and layer_idx in decode_cache:
                    full_k, full_v = decode_cache.pop(layer_idx)
                    for node_id in [0] + self.worker_ids:
                        p = self.partitions[node_id]
                        kv_start = p.get("kv_head_start", 0)
                        n_kv = p.get("n_kv_heads", 0)
                        if n_kv > 0:
                            decode_cache[f"hp_{layer_idx}_{node_id}"] = (
                                np.ascontiguousarray(full_k[:, kv_start:kv_start+n_kv, :, :]),
                                np.ascontiguousarray(full_v[:, kv_start:kv_start+n_kv, :, :]),
                            )
            else:
                # ── Seq-partitioned attention (Ulysses) ──
                p0 = self.partitions[0]
                x_root = x_norm[:, p0["seq_start"]:p0["seq_end"], :]

                wq_r = rw_attn.get("q")
                if wq_r is None:
                    wq_r = np.random.randn(hidden_dim, n_heads * layer_hd).astype(np.float32)
                    rw_attn["q"] = wq_r
                wk_r = rw_attn.get("k")
                if wk_r is None:
                    wk_r = np.random.randn(hidden_dim, n_kv * layer_hd).astype(np.float32)
                    rw_attn["k"] = wk_r
                wv_r = rw_attn.get("v")
                if wv_r is None:
                    wv_r = np.random.randn(hidden_dim, n_kv * layer_hd).astype(np.float32)
                    rw_attn["v"] = wv_r

                hd = root_w.get("_props", {}).get("head_dim", self.config.head_dim)
                attn_model = self.attn_models.get(hd, list(self.attn_models.values())[0])

                v_arr = wv_r if wv_r is not None else wk_r
                (q_root, k_root, v_root) = attn_model.execute(
                    np.ascontiguousarray(x_root),
                    np.ascontiguousarray(wq_r),
                    np.ascontiguousarray(wk_r),
                    np.ascontiguousarray(v_arr),
                )
                q_vals = q_root.to_numpy()
                k_vals = k_root.to_numpy()
                v_vals = v_root.to_numpy()
                q_bias = rw_attn.get("q_bias")
                k_bias = rw_attn.get("k_bias")
                v_bias = rw_attn.get("v_bias")
                if q_bias is not None:
                    q_vals += q_bias.reshape(1, 1, n_heads, layer_hd)
                if k_bias is not None:
                    k_vals += k_bias.reshape(1, 1, n_kv, layer_hd)
                if v_bias is not None and wv_r is not None:
                    v_vals += v_bias.reshape(1, 1, n_kv, layer_hd)
                all_qkv = {0: (q_vals, k_vals, v_vals)}

                for idx, worker_id in enumerate(self.worker_ids):
                    p = self.partitions[worker_id]
                    x_worker = x_norm[:, p["seq_start"]:p["seq_end"], :]
                    send_msg(
                        self.worker_conns[idx], MSG_FORWARD_DATA,
                        (layer_idx, x_worker),
                    )

                for idx, worker_id in enumerate(self.worker_ids):
                    _, qkv = recv_msg(self.worker_conns[idx])
                    q_w, k_w, v_w = qkv
                    if q_bias is not None:
                        q_w = q_w + q_bias.reshape(1, 1, n_heads, layer_hd)
                    if k_bias is not None:
                        k_w = k_w + k_bias.reshape(1, 1, n_kv, layer_hd)
                    if v_bias is not None and wv_r is not None:
                        v_w = v_w + v_bias.reshape(1, 1, n_kv, layer_hd)
                    all_qkv[worker_id] = (q_w, k_w, v_w)

                attn_out = self._compute_attention(all_qkv, layer_idx, kv_cache, decode_cache,
                                                    props=root_w.get("_props", {}))

            o_weight = root_w.get("attn", {}).get("o")
            if o_weight is not None:
                attn_out = attn_out @ o_weight

        h = x + attn_out

        post_attn_ln = root_w.get("post_attention_layernorm")
        if post_attn_ln is not None:
            h_norm = self._rms_norm(h, post_attn_ln)
        else:
            variance = np.mean(h.astype(np.float64) ** 2, axis=-1, keepdims=True)
            h_norm = (h / np.sqrt(variance + 1e-6)).astype(np.float32)

        rw_ffn = root_w.get("ffn", {})
        p0 = self.partitions[0]
        width0 = p0["ffn_end"] - p0["ffn_start"]
        ffn_gate_r = rw_ffn.get("gate")
        if ffn_gate_r is None:
            ffn_gate_r = np.random.randn(hidden_dim, width0).astype(np.float32)
            rw_ffn["gate"] = ffn_gate_r
        ffn_up_r = rw_ffn.get("up")
        if ffn_up_r is None:
            ffn_up_r = np.random.randn(hidden_dim, width0).astype(np.float32)
            rw_ffn["up"] = ffn_up_r
        ffn_down_r = rw_ffn.get("down")
        if ffn_down_r is None:
            ffn_down_r = np.random.randn(width0, hidden_dim).astype(np.float32)
            rw_ffn["down"] = ffn_down_r

        (partial_root,) = self.ffn_model.execute(
            np.ascontiguousarray(h_norm),
            np.ascontiguousarray(ffn_gate_r),
            np.ascontiguousarray(ffn_up_r),
            np.ascontiguousarray(ffn_down_r),
        )
        ffn_out = partial_root.to_numpy()

        # Broadcast FFN input to all workers (true parallelism), then collect
        for idx, worker_id in enumerate(self.worker_ids):
            send_msg(self.worker_conns[idx], MSG_ATTN_OUTPUT, (h_norm, layer_idx))

        worker_timings = {}
        for idx, worker_id in enumerate(self.worker_ids):
            t0 = time.time()
            _, partial = recv_msg(self.worker_conns[idx])
            elapsed = time.time() - t0
            worker_timings[worker_id] = elapsed
            ffn_out += partial

        self.adaptive_partitioner.update(worker_timings)

        final_output = h + ffn_out

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

    # ── Linear attention (Qwen3.5 Mamba2‑style SSM) ─────────────────────
    @staticmethod
    def _run_linear_attention(
        x: np.ndarray,
        w: dict,
        prev_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run one layer of linear attention (conv1d + SSM scan).

        Args:
            x: [batch, seq, hidden_dim] float32 input (pre‑norm).
            w: Linear-attention weight dict from ``attn`` field.
            prev_state: [batch, hidden_dim, d_state] state from previous
                        decode step, or None for prefill.

        Returns:
            (output, new_state) where output is [batch, seq, hidden_dim]
            after the ``out_proj``, and new_state is the final SSM state.
        """
        batch, seq, d_model = x.shape
        d_state = w["A_log"].shape[0]
        d_conv = w["conv1d"].shape[-1]
        d_inner = w["in_proj_z"].shape[-1]  # internal SSM dim from weight shape

        # 1. Linear projections via Mojo matmul_f32
        x_2d = x.reshape(-1, d_model)
        qkv = np.empty((batch * seq, 2 * d_inner), dtype=np.float32)
        _get_pyre_kernels().matmul_f32(x_2d, w["in_proj_qkv"], qkv)
        qkv = qkv.reshape(batch, seq, 2 * d_inner)

        z_gate = np.empty((batch * seq, d_inner), dtype=np.float32)
        _get_pyre_kernels().matmul_f32(x_2d, w["in_proj_z"], z_gate)
        z_gate = z_gate.reshape(batch, seq, d_inner)
        z_gate = z_gate / (1.0 + np.exp(-z_gate))  # SiLU

        a_proj = np.empty((batch * seq, d_state), dtype=np.float32)
        _get_pyre_kernels().matmul_f32(x_2d, w["in_proj_a"], a_proj)
        a_proj = a_proj.reshape(batch, seq, d_state)

        b_proj = np.empty((batch * seq, d_state), dtype=np.float32)
        _get_pyre_kernels().matmul_f32(x_2d, w["in_proj_b"], b_proj)
        b_proj = b_proj.reshape(batch, seq, d_state)

        # 2. Conv1d along sequence dimension (depthwise) — Mojo kernel
        d_conv_inner = 2 * d_inner
        pad = np.repeat(qkv[:, :1, :], d_conv - 1, axis=1)
        qkv_pad = np.concatenate([pad, qkv], axis=1)
        conv_w = w["conv1d"]
        qkv_conv = np.empty_like(qkv)
        _get_pyre_kernels().linear_attn_conv1d(qkv_pad, conv_w, qkv_conv)

        # Split → x_ssm (first d_inner) and gate_conv (second d_inner)
        x_ssm = qkv_conv[..., :d_inner]
        gate_conv = qkv_conv[..., d_inner:]

        # 3. SSM parameters
        dt = np.log(1.0 + np.exp(w["dt_bias"]))
        A = -np.exp(w["A_log"])
        A_bar = np.exp(A * dt)

        # 4. SSM scan — Mojo kernel, h updated in-place
        if prev_state is not None:
            h = prev_state.copy()
        else:
            h = np.zeros((batch, d_inner, d_state), dtype=np.float32)
        y = np.empty((batch, seq, d_inner), dtype=np.float32)
        _get_pyre_kernels().linear_attn_ssm_scan(x_ssm, b_proj, a_proj, A_bar, h, y)

        # 5. Gate: conv_gate * z_gate (both SiLU)
        y = y * gate_conv * z_gate

        # 6. RMS norm (per-head, weight shape = head_dim)
        if w.get("norm") is not None:
            head_dim_v = w["norm"].shape[0]  # e.g. 128 for Qwen3.5 (linear_value_head_dim)
            n_vheads = d_inner // head_dim_v
            y_r = y.reshape(batch, seq, n_vheads, head_dim_v)
            eps = 1e-6
            variance = np.mean(y_r.astype(np.float64) ** 2, axis=-1, keepdims=True)
            y_r = (y_r / np.sqrt(variance + eps)).astype(np.float32) * w["norm"].reshape(1, 1, 1, head_dim_v)
            y = y_r.reshape(batch, seq, d_inner)

        # 7. Output projection via Mojo matmul_f32
        y_2d = y.reshape(-1, d_inner)
        out = np.empty((batch * seq, d_model), dtype=np.float32)
        _get_pyre_kernels().matmul_f32(y_2d, w["out_proj"], out)
        y = out.reshape(batch, seq, d_model)

        return y, h

    def _decode_step(self, x: np.ndarray, decode_cache: dict,
                      input_ids: Optional[np.ndarray] = None) -> np.ndarray:
        num_layers = self.num_layers

        ple_all = None
        if self.ple_embedding is not None and input_ids is not None:
            ple_all = self._compute_ple_signal(input_ids, x)
            ple_all = ple_all[:, -1:, :, :]

        if "_pos" not in decode_cache:
            if decode_cache:
                entry = next(v for k, v in decode_cache.items() if not k.startswith("_"))
                if isinstance(entry, tuple) and len(entry) >= 2:
                    decode_cache["_pos"] = entry[0].shape[2]
                else:
                    decode_cache["_pos"] = 1
            else:
                decode_cache["_pos"] = 1

        if "_prefill_len" not in decode_cache:
            decode_cache["_prefill_len"] = decode_cache.get("_pos", 0)
        if "_prefill_max_seq" not in decode_cache:
            for k, v in decode_cache.items():
                if isinstance(k, str) and not k.startswith("_"):
                    decode_cache["_prefill_max_seq"] = v[0].shape[2]
                    break

        for layer_idx in range(num_layers):
            ple_slice = ple_all[:, :, layer_idx, :] if ple_all is not None else None
            x = self._decode_single_layer(x, layer_idx, decode_cache, ple_slice)

        decode_cache["_pos"] = decode_cache.get("_pos", 0) + 1

        if self.adaptive_partitioner.drift_detected():
            new_parts = self.adaptive_partitioner.get_partitions()
            print(f"[Adaptive] Decode drift. New root FFN: "
                  f"[{new_parts[0]['ffn_start']}:{new_parts[0]['ffn_end']})")

        return x

    def _decode_single_layer(self, x: np.ndarray, layer_idx: int,
                              decode_cache: dict,
                              ple_slice: Optional[np.ndarray] = None) -> np.ndarray:
        batch, seq_len, hidden_dim = x.shape
        head_dim = self.config.head_dim
        n_heads = self.config.n_heads
        n_kv = self.config.n_kv_heads

        root_w = self._get_root_layer_weights(layer_idx)
        rw_attn = root_w.get("attn", {})

        input_ln = root_w.get("input_layernorm")
        if input_ln is not None:
            x_norm = self._rms_norm(x, input_ln)
        else:
            variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
            x_norm = (x / np.sqrt(variance + 1e-6)).astype(np.float32)

        attn_type = root_w.get("_props", {}).get("attention_type", "full")

        if attn_type == "linear":
            prev_state = decode_cache.get(f"ssm_{layer_idx}")
            attn_out, new_state = self._run_linear_attention(x_norm, rw_attn, prev_state)
            decode_cache[f"ssm_{layer_idx}"] = new_state
        elif self._use_head_partitioning and not (root_w.get("_props", {}) or {}).get("kv_source_layer"):
            # ── Head-partitioned decode — local QKV, distributed attention ──
            p0 = self.partitions[0]
            n_q_local = p0["n_q_heads"]
            n_kv_local = p0["n_kv_heads"]
            layer_hd = root_w.get("_props", {}).get("head_dim", head_dim)
            props = root_w.get("_props", {})
            rope_frac = props.get("rope_fraction", 1.0)
            theta = getattr(self.config, 'rope_theta', 10000.0)

            wq_r = rw_attn.get("q")
            wk_r = rw_attn.get("k")
            if wq_r is None:
                wq_r = np.random.randn(hidden_dim, n_q_local * layer_hd).astype(np.float32)
                rw_attn["q"] = wq_r
            if wk_r is None:
                wk_r = np.random.randn(hidden_dim, n_kv_local * layer_hd).astype(np.float32)
                rw_attn["k"] = wk_r
            wv_r = rw_attn.get("v")

            # Root local QKV
            xn_2d = x_norm.reshape(1, -1)
            q = np.empty((1, n_q_local * layer_hd), dtype=np.float32)
            k = np.empty((1, n_kv_local * layer_hd), dtype=np.float32)
            _get_pyre_kernels().matmul_f32(xn_2d, wq_r, q)
            _get_pyre_kernels().matmul_f32(xn_2d, wk_r, k)
            q_bias = rw_attn.get("q_bias")
            k_bias = rw_attn.get("k_bias")
            if q_bias is not None: q += q_bias.reshape(1, -1)
            if k_bias is not None: k += k_bias.reshape(1, -1)
            q = q.reshape(1, 1, n_q_local, layer_hd)
            k = k.reshape(1, 1, n_kv_local, layer_hd)

            if wv_r is not None:
                v = np.empty((1, n_kv_local * layer_hd), dtype=np.float32)
                _get_pyre_kernels().matmul_f32(xn_2d, wv_r, v)
                v_bias = rw_attn.get("v_bias")
                if v_bias is not None: v += v_bias.reshape(1, -1)
                v = v.reshape(1, 1, n_kv_local, layer_hd)
            else:
                v = k.copy()

            # RoPE
            q_rope = np.ascontiguousarray(q.transpose(0, 2, 1, 3))
            k_rope = np.ascontiguousarray(k.transpose(0, 2, 1, 3))
            v_out = np.ascontiguousarray(v.transpose(0, 2, 1, 3))
            cache_len = decode_cache.get("_pos", 0)
            if cache_len == 0 and layer_idx in decode_cache:
                entry = decode_cache[layer_idx]
                if isinstance(entry, tuple) and len(entry) >= 2:
                    cache_len = entry[0].shape[2]
            if rope_frac > 0:
                q_rope = self._apply_rope(q_rope, rope_fraction=rope_frac, theta=theta, start_pos=cache_len)
                k_rope = self._apply_rope(k_rope, rope_fraction=rope_frac, theta=theta, start_pos=cache_len)

            # Append to root local cache (per-node KV)
            cache_key = f"hp_{layer_idx}_0"
            if cache_key in decode_cache:
                cached_k, cached_v = decode_cache[cache_key]
                k_local = np.concatenate([cached_k, k_rope], axis=2)
                v_local = np.concatenate([cached_v, v_out], axis=2)
            else:
                k_local = k_rope
                v_local = v_out
            decode_cache[cache_key] = (k_local, v_local)

            # Root local GQA attention (on root's local KV slice)
            full_seq = k_local.shape[2]
            n_q_per_kv = n_q_local // n_kv_local
            k_exp = k_local[:, :, None, :, :].repeat(n_q_per_kv, axis=2).reshape(1, n_q_local, full_seq, layer_hd)
            v_exp = v_local[:, :, None, :, :].repeat(n_q_per_kv, axis=2).reshape(1, n_q_local, full_seq, layer_hd)
            scale = np.float32(np.sqrt(layer_hd))
            scores = q_rope @ k_exp.transpose(0, 1, 3, 2) / scale
            prefill_real = decode_cache.get("_prefill_len", 0)
            prefill_max_seq = decode_cache.get("_prefill_max_seq", 0)
            if prefill_max_seq > prefill_real:
                scores[:, :, :, prefill_real:prefill_max_seq] = -np.inf
            probs = self._softmax(scores, axis=-1)
            partial_attn = probs @ v_exp  # [1, n_q_local, 1, hd]
            all_partial = {0: partial_attn.transpose(0, 2, 1, 3).reshape(1, 1, n_q_local * layer_hd)}

            # Workers compute their local attention
            for idx, worker_id in enumerate(self.worker_ids):
                send_msg(self.worker_conns[idx], MSG_DECODE_QKV, (layer_idx, x_norm, cache_len))
            for idx, worker_id in enumerate(self.worker_ids):
                _, partial_w = recv_msg(self.worker_conns[idx])
                all_partial[worker_id] = partial_w

            # Concatenate partial attention outputs by head dim
            ids = [0] + self.worker_ids
            attn_out = np.concatenate([all_partial[i] for i in ids], axis=-1)  # [1, 1, n_heads * hd]
            o_weight = rw_attn.get("o")
            if o_weight is not None:
                ao = np.empty((1, hidden_dim), dtype=np.float32)
                _get_pyre_kernels().matmul_f32(attn_out.reshape(1, -1), o_weight, ao)
                attn_out = ao.reshape(1, 1, hidden_dim)
        else:
            # ── Full-attention (Ulysses) decode — all heads at root ──
            layer_hd = root_w.get("_props", {}).get("head_dim", head_dim)
            wq_r = rw_attn.get("q")
            if wq_r is None:
                wq_r = np.random.randn(hidden_dim, n_heads * layer_hd).astype(np.float32)
                rw_attn["q"] = wq_r
            wk_r = rw_attn.get("k")
            if wk_r is None:
                wk_r = np.random.randn(hidden_dim, n_kv * layer_hd).astype(np.float32)
                rw_attn["k"] = wk_r
            wv_r = rw_attn.get("v")
            if wv_r is None:
                wv_r = np.random.randn(hidden_dim, n_kv * layer_hd).astype(np.float32)
                rw_attn["v"] = wv_r

            xn_2d = x_norm.reshape(1, -1)
            q = np.empty((1, n_heads * layer_hd), dtype=np.float32)
            k = np.empty((1, n_kv * layer_hd), dtype=np.float32)
            _get_pyre_kernels().matmul_f32(xn_2d, wq_r, q)
            _get_pyre_kernels().matmul_f32(xn_2d, wk_r, k)

            q_bias = rw_attn.get("q_bias")
            k_bias = rw_attn.get("k_bias")
            if q_bias is not None:
                q += q_bias.astype(np.float32).reshape(1, -1)
            if k_bias is not None:
                k += k_bias.astype(np.float32).reshape(1, -1)

            q = q.reshape(1, 1, n_heads, layer_hd)
            k = k.reshape(1, 1, n_kv, layer_hd)
            if wv_r is not None:
                v = np.empty((1, n_kv * layer_hd), dtype=np.float32)
                _get_pyre_kernels().matmul_f32(xn_2d, wv_r, v)
                v_bias = rw_attn.get("v_bias")
                if v_bias is not None:
                    v += v_bias.astype(np.float32).reshape(1, -1)
                v = v.reshape(1, 1, n_kv, layer_hd)
            else:
                v = k.copy()

            q_rope = np.ascontiguousarray(q.transpose(0, 2, 1, 3))
            k_rope = np.ascontiguousarray(k.transpose(0, 2, 1, 3))
            v_out = np.ascontiguousarray(v.transpose(0, 2, 1, 3))

            props = root_w.get("_props", {})
            rope_frac = props.get("rope_fraction", 1.0)
            theta = getattr(self.config, 'rope_theta', 10000.0)
            cache_len = decode_cache.get("_pos", 0)
            if cache_len == 0 and layer_idx in decode_cache:
                entry = decode_cache[layer_idx]
                if isinstance(entry, tuple) and len(entry) >= 2:
                    cache_len = entry[0].shape[2]
            if rope_frac > 0:
                q_rope = self._apply_rope(q_rope, rope_fraction=rope_frac,
                                           theta=theta, start_pos=cache_len)
                k_rope = self._apply_rope(k_rope, rope_fraction=rope_frac,
                                           theta=theta, start_pos=cache_len)

            # ---- Append to KV cache (concatenation) ----
            if layer_idx in decode_cache:
                if len(decode_cache[layer_idx]) == 3:
                    del decode_cache[layer_idx]
            if layer_idx in decode_cache:
                cached_k, cached_v = decode_cache[layer_idx]
                k_full = np.concatenate([cached_k, k_rope], axis=2)
                v_full = np.concatenate([cached_v, v_out], axis=2)
            else:
                k_full = k_rope
                v_full = v_out
            decode_cache[layer_idx] = (k_full, v_full)

            q_rope = np.clip(q_rope, -1000, 1000)
            k_full = np.clip(k_full, -1000, 1000)
            v_full = np.clip(v_full, -1000, 1000)

            prefill_real = decode_cache.get("_prefill_len", 0)
            prefill_max_seq = decode_cache.get("_prefill_max_seq", 0)
            attn_out = self._compute_attention_decode(q_rope, k_full, v_full, layer_idx,
                                                        prefill_real=prefill_real,
                                                        prefill_max_seq=prefill_max_seq)

            o_weight = root_w.get("attn", {}).get("o")
            if o_weight is not None:
                ao = np.empty((1, hidden_dim), dtype=np.float32)
                _get_pyre_kernels().matmul_f32(attn_out.reshape(1, -1), o_weight, ao)
                attn_out = ao.reshape(*attn_out.shape[:-1], hidden_dim)

        h = x + attn_out

        post_attn_ln = root_w.get("post_attention_layernorm")
        if post_attn_ln is not None:
            h_norm = self._rms_norm(h, post_attn_ln)
        else:
            variance = np.mean(h.astype(np.float64) ** 2, axis=-1, keepdims=True)
            h_norm = (h / np.sqrt(variance + 1e-6)).astype(np.float32)

        rw_ffn = root_w.get("ffn", {})
        p0 = self.partitions[0]
        width0 = p0["ffn_end"] - p0["ffn_start"]
        ffn_gate_r = rw_ffn.get("gate")
        if ffn_gate_r is None:
            ffn_gate_r = np.random.randn(hidden_dim, width0).astype(np.float32)
            rw_ffn["gate"] = ffn_gate_r
        ffn_up_r = rw_ffn.get("up")
        if ffn_up_r is None:
            ffn_up_r = np.random.randn(hidden_dim, width0).astype(np.float32)
            rw_ffn["up"] = ffn_up_r
        ffn_down_r = rw_ffn.get("down")
        if ffn_down_r is None:
            ffn_down_r = np.random.randn(width0, hidden_dim).astype(np.float32)
            rw_ffn["down"] = ffn_down_r

        gate = np.empty((1, width0), dtype=np.float32)
        up = np.empty((1, width0), dtype=np.float32)
        _get_pyre_kernels().matmul_f32(h_norm.reshape(1, -1), ffn_gate_r, gate)
        _get_pyre_kernels().matmul_f32(h_norm.reshape(1, -1), ffn_up_r, up)
        hidden = np.empty_like(gate)
        _get_pyre_kernels().silu_mul(gate, up, hidden)
        ffn_out = np.empty((1, hidden_dim), dtype=np.float32)
        _get_pyre_kernels().matmul_f32(hidden, ffn_down_r, ffn_out)
        ffn_out = ffn_out.reshape(h_norm.shape)

        # Broadcast decode step to all workers, then collect
        for idx, worker_id in enumerate(self.worker_ids):
            send_msg(self.worker_conns[idx], MSG_DECODE_STEP,
                           (layer_idx, h_norm))

        decode_timings = {}
        for idx, worker_id in enumerate(self.worker_ids):
            t0 = time.time()
            _, partial = recv_msg(self.worker_conns[idx])
            elapsed = time.time() - t0
            decode_timings[worker_id] = elapsed
            ffn_out += partial

        self.adaptive_partitioner.update(decode_timings)

        final_output = h + ffn_out

        if ple_slice is not None:
            ple_gate = root_w.get("ple_gate")
            if ple_gate is not None:
                final_output = self._apply_ple_to_hidden(
                    final_output, ple_slice,
                    ple_gate, root_w["ple_proj"], root_w["ple_post_norm"],
                )

        return final_output

    @staticmethod
    def _compute_attention_decode(q, k, v, layer_idx=0,
                                    prefill_real=0, prefill_max_seq=0):
        n_heads = q.shape[1]
        n_kv = k.shape[1]
        head_dim = q.shape[3]
        full_seq = k.shape[2]

        n_q_per_kv = n_heads // n_kv
        k_exp = k[:, :, None, :, :].repeat(n_q_per_kv, axis=2).reshape(1, n_heads, full_seq, head_dim)
        v_exp = v[:, :, None, :, :].repeat(n_q_per_kv, axis=2).reshape(1, n_heads, full_seq, head_dim)

        scale = np.float32(np.sqrt(head_dim))
        scores = q @ k_exp.transpose(0, 1, 3, 2) / scale

        if prefill_max_seq > prefill_real:
            scores[:, :, :, prefill_real:prefill_max_seq] = -np.inf

        probs = RootNode._softmax(scores, axis=-1)
        attn = probs @ v_exp
        attn = attn.transpose(0, 2, 1, 3).reshape(1, 1, n_heads * head_dim)

        return attn.astype(np.float32)

    def _get_root_layer_weights(self, layer_idx: int) -> dict:
        if self._wp is not None:
            if layer_idx not in self._weight_cache:
                lp = self._wp.get_layer_props(layer_idx)
                has_kv_source = lp is not None and lp.kv_source_layer is not None
                full_q = not self._use_head_partitioning or has_kv_source
                raw = self._wp._layer_weights_for_node(
                    layer_idx, self.partitions[0],
                    full_q=full_q, copy_weights=True,
                )
                # Store everything in f16 (2× memory savings)
                self._weight_cache[layer_idx] = quantize_weights_dict(raw, mode="f16")
            return dequantize_weights_dict(self._weight_cache[layer_idx])
        if self.all_layer_weights and 0 in self.all_layer_weights:
            return self.all_layer_weights[0].get(layer_idx, {})
        return {}

    def _get_root_props(self, layer_idx: int) -> dict:
        """Get just the _props dict for a layer (used by _compute_attention)."""
        if self._wp is not None:
            # LayerProperties are cheap — get them from the partitioner
            lp = self._wp.get_layer_props(layer_idx)
            return {
                "head_dim": lp.head_dim,
                "has_v_proj": lp.has_v_proj,
                "rope_fraction": lp.rope_fraction,
                "use_v_norm": lp.use_v_norm,
                "attention_type": lp.attention_type,
                "kv_source_layer": lp.kv_source_layer,
            }
        if self.all_layer_weights and 0 in self.all_layer_weights:
            return self.all_layer_weights[0].get(layer_idx, {}).get("_props", {})
        return {}

    def shutdown(self):
        for conn in self.worker_conns:
            try:
                send_msg(conn, MSG_SHUTDOWN)
                conn.close()
            except Exception:
                pass
