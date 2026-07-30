from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.driver import CPU

from src.attention.builder import build_ulysses_attention_graph, ShardSpec as AttentionShardSpec
from src.ffn.builder import build_ffn_graph
from src.orchestrator.net import send_msg, recv_msg, recv_exact
from src.orchestrator.quantizer import quantize_weights_dict, dequantize_weights_dict
from src.orchestrator.root_node import _get_pyre_kernels
from src.orchestrator.protocol import (
    MSG_SHARD_SPEC, MSG_READY, MSG_FORWARD_DATA, MSG_FORWARD_RESULT,
    MSG_SHUTDOWN, MSG_ATTN_OUTPUT, MSG_FFN_RESULT, MSG_DECODE_STEP, MSG_DECODE_QKV,
)


def _make_attention_graph_key(layer_props, shard, hidden_dim, n_heads, n_kv_heads, device):
    hd = layer_props["head_dim"]
    local_seq = shard.local_seq_len()
    n_q = n_heads
    return (hd, local_seq, hidden_dim, n_heads, n_kv_heads)


class WorkerNode:
    def __init__(self, host="localhost", port=9000, use_mdns=True):
        self.host = host
        self.port = port
        self.use_mdns = use_mdns
        self._registrar = None
        self.device = DeviceRef.CPU()
        self.session = InferenceSession(devices=[CPU()])
        self.attn_models = {}
        self.ffn_model = None
        self.ffn_decode_model = None
        self.shard = None
        self.config = None
        self.local_seq_len = None
        self.hidden_dim = None
        self.layer_props = {}
        self._current_layer_idx: int | None = None
        self._loaded = {}  # layer_idx → weights dict (lazy loaded)
        self._wp = None
        self._model_id = None
        self._partition = None
        self._decode_cache = {}  # layer_idx → (k, v) for decode attention

    def _load_layer_weights(self, layer_idx):
        """Load one layer's weights from HuggingFace via WeightProvider (Q8_0 cached)."""
        if layer_idx in self._loaded:
            return dequantize_weights_dict(self._loaded[layer_idx])
        n_qh = self._partition.get("n_q_heads", 0)
        use_head_partitioning = n_qh > 0 and n_qh < self.config.n_heads if self.config else False
        lw = self._wp._layer_weights_for_node(
            layer_idx, self._partition, full_q=not use_head_partitioning, copy_weights=False,
        )
        self._loaded[layer_idx] = quantize_weights_dict(lw)
        return dequantize_weights_dict(self._loaded[layer_idx])

    def _compile_attention(self, head_dim: int):
        if self.shard is None:
            return None
        attn_graph = build_ulysses_attention_graph(
            self.shard, self.config.hidden_dim,
            self.config.n_heads, self.config.n_kv_heads,
            head_dim, self.device,
            full_q_weights=True,
        )
        return self.session.load(attn_graph)

    def start(self, ready_event=None):
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            self.port = server.getsockname()[1]
            server.listen(1)

            if ready_event:
                ready_event.set()

            if self.use_mdns:
                try:
                    from src.orchestrator.mdns import WorkerRegistrar
                    self._registrar = WorkerRegistrar(host=self.host, port=self.port)
                    self._registrar.start()
                    print(f"mDNS: registered on port {self.port}")
                except Exception as e:
                    print(f"Warning: mDNS registration failed: {e}")
            print(f"Worker listening on {self.host}:{self.port}")

            # Accept connections in a loop so stray probes don't kill the worker
            while True:
                conn, addr = server.accept()
                conn.settimeout(600.0)
                print(f"Accepted connection from {addr}", flush=True)
                try:
                    msg_type, obj = recv_msg(conn)
                    if msg_type != MSG_SHARD_SPEC:
                        print(f"Expected SHARD_SPEC, got {msg_type} — ignoring", flush=True)
                        conn.close()
                        continue

                    shard_spec, model_config, model_id = obj
                    self.shard = shard_spec
                    self.config = model_config
                    self._model_id = model_id
                    self.local_seq_len = shard_spec.local_seq_len()
                    self.hidden_dim = model_config.hidden_dim

                    full_seq_len = 64
                    self.ffn_model = self.session.load(build_ffn_graph(
                        shard_spec, model_config.hidden_dim, self.device,
                        seq_len=full_seq_len, gated=True,
                    ))
                    self.ffn_decode_model = self.session.load(build_ffn_graph(
                        shard_spec, model_config.hidden_dim, self.device,
                        seq_len=1, gated=True,
                    ))

                    # ── Load weights from HuggingFace independently ─────────
                    from src.orchestrator.llama_loader import WeightProvider
                    self._partition = {
                        "ffn_start": shard_spec.ffn_dim_start,
                        "ffn_end": shard_spec.ffn_dim_end,
                        "seq_start": shard_spec.seq_start,
                        "seq_end": shard_spec.seq_end,
                        "n_q_heads": shard_spec.n_q_heads,
                        "n_kv_heads": shard_spec.n_kv_heads,
                        "q_head_start": shard_spec.q_head_start,
                        "kv_head_start": shard_spec.kv_head_start,
                    }
                    partitions = {0: self._partition}
                    t0 = time.time()
                    self._wp = WeightProvider(
                        model_id, partitions,
                        num_layers=model_config.num_layers, use_cache=True,
                    )
                    self.layer_props = {
                        lidx: {
                            "head_dim": p.head_dim,
                            "has_v_proj": p.has_v_proj,
                            "rope_fraction": p.rope_fraction,
                            "use_v_norm": p.use_v_norm,
                            "attention_type": p.attention_type,
                            "kv_source_layer": p.kv_source_layer,
                        }
                        for lidx, p in self._wp._layer_props.items()
                    }

                    needed_hds = set()
                    for lp in self.layer_props.values():
                        needed_hds.add(lp.get("head_dim", model_config.head_dim))
                    if not needed_hds:
                        needed_hds.add(model_config.head_dim)

                    for hd in needed_hds:
                        self.attn_models[hd] = self._compile_attention(hd)

                    print(f"  [worker] weights ready in {time.time()-t0:.1f}s", flush=True)
                    send_msg(conn, MSG_READY)

                    while True:
                        msg_type, data = recv_msg(conn)
                        if msg_type == MSG_SHUTDOWN:
                            break
                        if msg_type == MSG_FORWARD_DATA:
                            layer_idx, x_data = data
                            self._current_layer_idx = layer_idx
                            lw = self._load_layer_weights(layer_idx)
                            aw = lw["attn"]
                            lp = self.layer_props.get(layer_idx, {})
                            hd = lp.get("head_dim", self.config.head_dim)
                            n_qh = self._partition.get("n_q_heads", 0)
                            use_hp = n_qh > 0 and n_qh < self.config.n_heads

                            if use_hp:
                                # Head-partitioned: matmul_f32 with local heads for ALL positions
                                batch, seq_len, hidden_dim = x_data.shape
                                x_2d = np.ascontiguousarray(x_data.reshape(-1, hidden_dim))
                                n_q_local = self._partition["n_q_heads"]
                                n_kv_local = self._partition["n_kv_heads"]
                                q = np.empty((batch * seq_len, n_q_local * hd), dtype=np.float32)
                                k = np.empty((batch * seq_len, n_kv_local * hd), dtype=np.float32)
                                _get_pyre_kernels().matmul_f32(x_2d, np.ascontiguousarray(aw["q"]), q)
                                _get_pyre_kernels().matmul_f32(x_2d, np.ascontiguousarray(aw["k"]), k)
                                v_arr = aw.get("v")
                                if v_arr is not None:
                                    v = np.empty((batch * seq_len, n_kv_local * hd), dtype=np.float32)
                                    _get_pyre_kernels().matmul_f32(x_2d, np.ascontiguousarray(v_arr), v)
                                else:
                                    v = k.copy()
                                q_bias = aw.get("q_bias")
                                k_bias = aw.get("k_bias")
                                v_bias = aw.get("v_bias")
                                if q_bias is not None:
                                    q += q_bias.reshape(1, n_q_local * hd)
                                if k_bias is not None:
                                    k += k_bias.reshape(1, n_kv_local * hd)
                                if v_bias is not None and v_arr is not None:
                                    v += v_bias.reshape(1, n_kv_local * hd)
                                q = q.reshape(batch, seq_len, n_q_local, hd)
                                k = k.reshape(batch, seq_len, n_kv_local, hd)
                                v = v.reshape(batch, seq_len, n_kv_local, hd)
                                send_msg(conn, MSG_FORWARD_RESULT, (q, k, v))
                            else:
                                # Seq-partitioned: MAX Graph with seq slice
                                attn_model = self.attn_models.get(hd)
                                if attn_model is None:
                                    attn_model = list(self.attn_models.values())[0]

                                v_arr = aw.get("v")
                                if v_arr is None:
                                    v_arr = aw["k"]
                                (q, k, v) = attn_model.execute(
                                    np.ascontiguousarray(x_data),
                                    np.ascontiguousarray(aw["q"]),
                                    np.ascontiguousarray(aw["k"]),
                                    np.ascontiguousarray(v_arr),
                                )
                                send_msg(
                                    conn, MSG_FORWARD_RESULT,
                                    (q.to_numpy(), k.to_numpy(), v.to_numpy()),
                                )
                        elif msg_type == MSG_DECODE_QKV:
                            layer_idx, x_norm, cache_len = data
                            lw = self._load_layer_weights(layer_idx)
                            aw = lw["attn"]
                            lp = self.layer_props.get(layer_idx, {})
                            hd = lp.get("head_dim", self.config.head_dim)
                            q_bias = aw.get("q_bias")
                            k_bias = aw.get("k_bias")
                            v_bias = aw.get("v_bias")
                            n_q_local = self._partition["n_q_heads"]
                            n_kv_local = self._partition["n_kv_heads"]

                            # QKV projection
                            xn_2d = x_norm.reshape(1, -1)
                            q = np.empty((1, n_q_local * hd), dtype=np.float32)
                            k = np.empty((1, n_kv_local * hd), dtype=np.float32)
                            _get_pyre_kernels().matmul_f32(xn_2d, aw["q"], q)
                            _get_pyre_kernels().matmul_f32(xn_2d, aw["k"], k)
                            v_arr = aw.get("v")
                            if v_arr is not None:
                                v = np.empty((1, n_kv_local * hd), dtype=np.float32)
                                _get_pyre_kernels().matmul_f32(xn_2d, v_arr, v)
                            else:
                                v = k.copy()
                            if q_bias is not None: q += q_bias.reshape(1, -1)
                            if k_bias is not None: k += k_bias.reshape(1, -1)
                            if v_bias is not None and v_arr is not None: v += v_bias.reshape(1, -1)

                            q = q.reshape(1, 1, n_q_local, hd)
                            k = k.reshape(1, 1, n_kv_local, hd)
                            v = v.reshape(1, 1, n_kv_local, hd)

                            # RoPE
                            q_rope = np.ascontiguousarray(q.transpose(0, 2, 1, 3))
                            k_rope = np.ascontiguousarray(k.transpose(0, 2, 1, 3))
                            v_out = np.ascontiguousarray(v.transpose(0, 2, 1, 3))
                            rope_frac = lp.get("rope_fraction", 1.0)
                            theta = getattr(self.config, 'rope_theta', 10000.0)
                            if rope_frac > 0:
                                q_rope_out = np.empty_like(q_rope)
                                k_rope_out = np.empty_like(k_rope)
                                _get_pyre_kernels().apply_rope(q_rope, q_rope_out, np.float32(theta), np.int32(cache_len), np.float32(rope_frac))
                                _get_pyre_kernels().apply_rope(k_rope, k_rope_out, np.float32(theta), np.int32(cache_len), np.float32(rope_frac))
                                q_rope, k_rope = q_rope_out, k_rope_out

                            # Cache append
                            if layer_idx in self._decode_cache:
                                cached_k, cached_v = self._decode_cache[layer_idx]
                                k_full = np.concatenate([cached_k, k_rope], axis=2)
                                v_full = np.concatenate([cached_v, v_out], axis=2)
                            else:
                                k_full = k_rope
                                v_full = v_out
                            self._decode_cache[layer_idx] = (k_full, v_full)

                            # Local GQA attention
                            full_seq = k_full.shape[2]
                            n_q_per_kv = n_q_local // n_kv_local
                            k_exp = k_full[:, :, None, :, :].repeat(n_q_per_kv, axis=2).reshape(1, n_q_local, full_seq, hd)
                            v_exp = v_full[:, :, None, :, :].repeat(n_q_per_kv, axis=2).reshape(1, n_q_local, full_seq, hd)
                            scale = np.float32(np.sqrt(hd))
                            scores = q_rope @ k_exp.transpose(0, 1, 3, 2) / scale
                            probs = np.empty_like(scores)
                            _get_pyre_kernels().softmax(scores, probs)
                            attn = probs @ v_exp
                            attn = attn.transpose(0, 2, 1, 3).reshape(1, 1, n_q_local * hd)
                            send_msg(conn, MSG_FFN_RESULT, attn.astype(np.float32))
                        elif msg_type == MSG_ATTN_OUTPUT:
                            (h_norm_full, lidx) = data
                            if lidx is not None:
                                lw = self._load_layer_weights(lidx)
                                fw = lw["ffn"]
                                (partial,) = self.ffn_model.execute(
                                    np.ascontiguousarray(h_norm_full),
                                    np.ascontiguousarray(fw["gate"]),
                                    np.ascontiguousarray(fw["up"]),
                                    np.ascontiguousarray(fw["down"]),
                                )
                                send_msg(conn, MSG_FFN_RESULT, partial.to_numpy())
                        elif msg_type == MSG_DECODE_STEP:
                            layer_idx, h_norm = data
                            lw = self._load_layer_weights(layer_idx)
                            fw = lw["ffn"]
                            (partial,) = self.ffn_decode_model.execute(
                                np.ascontiguousarray(h_norm),
                                np.ascontiguousarray(fw["gate"]),
                                np.ascontiguousarray(fw["up"]),
                                np.ascontiguousarray(fw["down"]),
                            )
                            send_msg(conn, MSG_FFN_RESULT, partial.to_numpy())
                        elif msg_type == MSG_ALL_LAYERS_DONE:
                            pass
                except (ConnectionError, BrokenPipeError, EOFError, OSError):
                    print(f"Connection lost from {addr}, waiting for new connection...", flush=True)
                except Exception:
                    import traceback
                    traceback.print_exc()
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            try:
                server.close()
            except Exception:
                pass
            if self._registrar:
                self._registrar.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Llama Worker Node")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9000, help="Port to listen on (0 = auto)")
    parser.add_argument("--no-mdns", action="store_true", help="Disable mDNS registration")
    args = parser.parse_args()
    worker = WorkerNode(host=args.host, port=args.port, use_mdns=not args.no_mdns)
    worker.start()
