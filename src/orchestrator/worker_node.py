from __future__ import annotations

import argparse
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.driver import CPU

from src.attention.builder import build_ulysses_attention_graph, ShardSpec as AttentionShardSpec
from src.ffn.builder import build_ffn_graph
from src.orchestrator.net import send_msg, recv_msg, recv_exact
from src.orchestrator.protocol import (
    MSG_SHARD_SPEC, MSG_READY, MSG_FORWARD_DATA, MSG_FORWARD_RESULT,
    MSG_SHUTDOWN, MSG_ATTN_OUTPUT, MSG_FFN_RESULT, MSG_INIT_WEIGHTS,
    MSG_DECODE_STEP, MSG_LAYER_WEIGHTS,
)
from src.orchestrator.quantizer import dequantize_weights_dict


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
        self.shard = None
        self.config = None
        self.local_seq_len = None
        self.hidden_dim = None
        self.layer_weights = {}
        self.layer_graph_key = {}
        self.layer_props = {}
        self._current_ffn_gate = None
        self._current_ffn_up = None
        self._current_ffn_down = None
        self._fallback_cache = {}  # {layer_idx: weights_dict}

    def _get_fallback_weights(self, layer_idx):
        """Lazily generate and cache fallback weights per layer (avoids repeated random alloc)."""
        if layer_idx in self._fallback_cache:
            return self._fallback_cache[layer_idx]
        width = self.shard.ffn_dim_end - self.shard.ffn_dim_start
        hd = self.config.head_dim
        fw = {
            "attn": {
                "q": np.random.randn(self.hidden_dim, self.config.n_heads * hd).astype(np.float32),
                "k": np.random.randn(self.hidden_dim, self.config.n_kv_heads * hd).astype(np.float32),
                "v": np.random.randn(self.hidden_dim, self.config.n_kv_heads * hd).astype(np.float32),
                "o": np.random.randn(self.config.n_heads * hd, self.hidden_dim).astype(np.float32),
                "has_v_proj": True,
            },
            "ffn": {
                "gate": np.random.randn(self.hidden_dim, width).astype(np.float32),
                "up": np.random.randn(self.hidden_dim, width).astype(np.float32),
                "down": np.random.randn(width, self.hidden_dim).astype(np.float32),
            },
        }
        self._fallback_cache[layer_idx] = fw
        return fw

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

            conn, addr = server.accept()
            conn.settimeout(180.0)

            msg_type, obj = recv_msg(conn)
            if msg_type != MSG_SHARD_SPEC:
                raise ValueError(f"Expected SHARD_SPEC, got {msg_type}")

            shard_spec, model_config = obj
            self.shard = shard_spec
            self.config = model_config
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

            send_msg(conn, MSG_READY)

            # Receive layers one at a time via MSG_LAYER_WEIGHTS (streaming)
            self.layer_weights = {}
            self.layer_props = {}
            num_layers = self.config.num_layers
            for layer_idx in range(num_layers):
                msg_type, payload = recv_msg(conn)
                if msg_type == MSG_LAYER_WEIGHTS:
                    lidx, layer_payload = payload
                elif msg_type == MSG_INIT_WEIGHTS:
                    # Fallback: batch receive all layers at once
                    layer_payload = payload
                    if isinstance(layer_payload, dict):
                        layer_payload = dequantize_weights_dict(layer_payload)
                    self.layer_weights = layer_payload.get("weights", {})
                    self.layer_props = layer_payload.get("props", {})
                    break
                else:
                    raise ValueError(
                        f"Expected LAYER_WEIGHTS ({MSG_LAYER_WEIGHTS}) "
                        f"or INIT_WEIGHTS ({MSG_INIT_WEIGHTS}), got {msg_type}"
                    )

                if isinstance(layer_payload, dict):
                    layer_payload = dequantize_weights_dict(layer_payload)

                self.layer_weights[lidx] = layer_payload
                self.layer_props[lidx] = layer_payload.get("_props", {})

            needed_hds = set()
            for lp in self.layer_props.values():
                needed_hds.add(lp.get("head_dim", model_config.head_dim))
            if not needed_hds:
                needed_hds.add(model_config.head_dim)

            for hd in needed_hds:
                self.attn_models[hd] = self._compile_attention(hd)

            while True:
                msg_type, data = recv_msg(conn)
                if msg_type == MSG_SHUTDOWN:
                    break
                if msg_type == MSG_FORWARD_DATA:
                    layer_idx, x_slice = data
                    lw = self.layer_weights.get(layer_idx) if self.layer_weights else None
                    if lw is None:
                        lw = self._get_fallback_weights(layer_idx)

                    lp = self.layer_props.get(layer_idx, {})
                    hd = lp.get("head_dim", self.config.head_dim)
                    aw = lw["attn"]
                    fw = lw["ffn"]
                    self._current_ffn_gate = fw["gate"]
                    self._current_ffn_up = fw["up"]
                    self._current_ffn_down = fw["down"]

                    attn_model = self.attn_models.get(hd)
                    if attn_model is None:
                        attn_model = list(self.attn_models.values())[0]

                    v_arr = aw.get("v")
                    if v_arr is None:
                        v_arr = aw["k"]
                    (q, k, v) = attn_model.execute(
                        np.ascontiguousarray(x_slice),
                        np.ascontiguousarray(aw["q"]),
                        np.ascontiguousarray(aw["k"]),
                        np.ascontiguousarray(v_arr),
                    )

                    send_msg(
                        conn, MSG_FORWARD_RESULT,
                        (q.to_numpy(), k.to_numpy(), v.to_numpy()),
                    )
                elif msg_type == MSG_ATTN_OUTPUT:
                    (h_norm_full,) = data
                    (partial,) = self.ffn_model.execute(
                        np.ascontiguousarray(h_norm_full),
                        np.ascontiguousarray(self._current_ffn_gate),
                        np.ascontiguousarray(self._current_ffn_up),
                        np.ascontiguousarray(self._current_ffn_down),
                    )
                    send_msg(conn, MSG_FFN_RESULT, partial.to_numpy())
                elif msg_type == MSG_DECODE_STEP:
                    layer_idx, h_norm = data
                    lw = self.layer_weights.get(layer_idx) if self.layer_weights else None
                    if lw is None:
                        lw = self._get_fallback_weights(layer_idx)
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

        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            try:
                conn.close()
            except Exception:
                pass
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
