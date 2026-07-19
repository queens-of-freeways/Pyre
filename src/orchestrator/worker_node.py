from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.driver import CPU

from src.attention.builder import build_ulysses_attention_graph, ShardSpec as AttentionShardSpec
from src.ffn.builder import build_ffn_graph


MSG_SHARD_SPEC = 0
MSG_READY = 1
MSG_FORWARD_DATA = 2
MSG_FORWARD_RESULT = 3
MSG_SHUTDOWN = 4


def _recv_exact(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


class WorkerNode:
    def __init__(self, host="localhost", port=9000):
        self.host = host
        self.port = port
        self.device = DeviceRef.CPU()
        self.session = InferenceSession(devices=[CPU()])
        self.attn_model = None
        self.ffn_model = None
        self.shard = None
        self.config = None
        self.local_seq_len = None
        self.hidden_dim = None

    def start(self):
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)

            conn, addr = server.accept()
            conn.settimeout(60.0)

            msg_type, obj = self._recv_msg(conn)
            if msg_type != MSG_SHARD_SPEC:
                raise ValueError(f"Expected SHARD_SPEC, got {msg_type}")

            shard_spec, model_config = obj
            self.shard = shard_spec
            self.config = model_config
            self.local_seq_len = shard_spec.local_seq_len()
            self.hidden_dim = model_config.hidden_dim

            attn_graph = build_ulysses_attention_graph(
                shard_spec, model_config.hidden_dim,
                model_config.n_heads, model_config.n_kv_heads,
                model_config.head_dim, self.device,
            )
            ffn_graph = build_ffn_graph(
                shard_spec, model_config.hidden_dim, self.device,
                seq_len=self.local_seq_len,
            )
            self.attn_model = self.session.load(attn_graph)
            self.ffn_model = self.session.load(ffn_graph)

            self._send_msg(conn, MSG_READY)

            while True:
                msg_type, arrays = self._recv_msg(conn)
                if msg_type == MSG_SHUTDOWN:
                    break
                if msg_type != MSG_FORWARD_DATA:
                    continue

                x_slice, wq_slice, wk_full, wv_full, ffn_up_slice, ffn_down_slice = arrays

                self.attn_model.execute(
                    np.ascontiguousarray(x_slice),
                    np.ascontiguousarray(wq_slice),
                    np.ascontiguousarray(wk_full),
                    np.ascontiguousarray(wv_full),
                )

                mock_attn = np.random.randn(1, self.local_seq_len, self.hidden_dim).astype(np.float32)

                (partial,) = self.ffn_model.execute(
                    np.ascontiguousarray(mock_attn),
                    np.ascontiguousarray(ffn_up_slice),
                    np.ascontiguousarray(ffn_down_slice),
                )

                self._send_msg(conn, MSG_FORWARD_RESULT, partial.to_numpy())
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

    def _send_msg(self, conn, msg_type, obj=None):
        payload = pickle.dumps(obj) if obj is not None else b""
        header = struct.pack("!II", msg_type, len(payload))
        conn.sendall(header + payload)

    def _recv_msg(self, conn):
        header = _recv_exact(conn, 8)
        msg_type, payload_len = struct.unpack("!II", header)
        payload = _recv_exact(conn, payload_len)
        return msg_type, pickle.loads(payload) if payload else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Llama Worker Node")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9000, help="Port to listen on")
    args = parser.parse_args()
    worker = WorkerNode(host=args.host, port=args.port)
    worker.start()
