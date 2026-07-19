from __future__ import annotations

import pickle
import socket
import struct
import time
from typing import List, Tuple

import numpy as np
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.driver import CPU

from src.attention.builder import build_ulysses_attention_graph, ShardSpec as AttentionShardSpec
from src.ffn.builder import build_ffn_graph, ShardSpec as FFNShardSpec
from src.orchestrator.cluster import ModelConfig


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


class RootNode:
    def __init__(self, worker_addrs: List[Tuple[str, int]], config: ModelConfig):
        self.config = config
        self.device = DeviceRef.CPU()
        self.session = InferenceSession(devices=[CPU()])

        total_nodes = 1 + len(worker_addrs)
        self.partitions = self._solve_partitions(total_nodes)
        self.worker_addrs = worker_addrs

        p0 = self.partitions[0]
        attn_shard = AttentionShardSpec(
            ffn_dim_start=p0["ffn_start"], ffn_dim_end=p0["ffn_end"],
            seq_start=p0["seq_start"], seq_end=p0["seq_end"],
        )
        attn_graph = build_ulysses_attention_graph(
            attn_shard, config.hidden_dim, config.n_heads, config.n_kv_heads,
            config.head_dim, self.device,
        )
        ffn_shard = FFNShardSpec(
            ffn_dim_start=p0["ffn_start"], ffn_dim_end=p0["ffn_end"],
        )
        ffn_graph = build_ffn_graph(
            ffn_shard, config.hidden_dim, self.device,
            seq_len=p0["seq_end"] - p0["seq_start"],
        )
        self.attn_model = self.session.load(attn_graph)
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
                conn = socket.create_connection((host, port), timeout=5)
                return conn
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    raise

    def run(self, x: np.ndarray) -> np.ndarray:
        batch, seq_len, hidden_dim = x.shape
        assert hidden_dim == self.config.hidden_dim

        p0 = self.partitions[0]
        x_root = x[:, p0["seq_start"]:p0["seq_end"], :]
        n_q_local = (p0["ffn_end"] - p0["ffn_start"]) // self.config.head_dim
        wq_slice = np.random.randn(hidden_dim, n_q_local * self.config.head_dim).astype(np.float32)
        wk_full = np.random.randn(hidden_dim, self.config.n_kv_heads * self.config.head_dim).astype(np.float32)
        wv_full = np.random.randn(hidden_dim, self.config.n_kv_heads * self.config.head_dim).astype(np.float32)
        self.attn_model.execute(
            np.ascontiguousarray(x_root),
            np.ascontiguousarray(wq_slice),
            np.ascontiguousarray(wk_full),
            np.ascontiguousarray(wv_full),
        )

        attn_out = np.random.randn(batch, seq_len, hidden_dim).astype(np.float32)

        for idx, worker_id in enumerate(self.worker_ids):
            p = self.partitions[worker_id]
            x_slice = x[:, p["seq_start"]:p["seq_end"], :]
            n_q_local = (p["ffn_end"] - p["ffn_start"]) // self.config.head_dim
            wq_slice_w = np.random.randn(hidden_dim, n_q_local * self.config.head_dim).astype(np.float32)
            wk_full_w = np.random.randn(hidden_dim, self.config.n_kv_heads * self.config.head_dim).astype(np.float32)
            wv_full_w = np.random.randn(hidden_dim, self.config.n_kv_heads * self.config.head_dim).astype(np.float32)
            width = p["ffn_end"] - p["ffn_start"]
            ffn_up_slice = np.random.randn(hidden_dim, width).astype(np.float32)
            ffn_down_slice = np.random.randn(width, hidden_dim).astype(np.float32)

            self._send_msg(
                self.worker_conns[idx], MSG_FORWARD_DATA,
                (x_slice, wq_slice_w, wk_full_w, wv_full_w, ffn_up_slice, ffn_down_slice),
            )

        partials = {}
        for idx, worker_id in enumerate(self.worker_ids):
            _, partial = self._recv_msg(self.worker_conns[idx])
            partials[worker_id] = partial

        attn_root = attn_out[:, p0["seq_start"]:p0["seq_end"], :]
        width0 = p0["ffn_end"] - p0["ffn_start"]
        ffn_up_root = np.random.randn(hidden_dim, width0).astype(np.float32)
        ffn_down_root = np.random.randn(width0, hidden_dim).astype(np.float32)
        (partial_root,) = self.ffn_model.execute(
            np.ascontiguousarray(attn_root),
            np.ascontiguousarray(ffn_up_root),
            np.ascontiguousarray(ffn_down_root),
        )
        partial_root = partial_root.to_numpy()

        final_output = np.zeros((batch, seq_len, hidden_dim), dtype=np.float32)
        final_output[:, p0["seq_start"]:p0["seq_end"], :] += partial_root
        for worker_id in self.worker_ids:
            p = self.partitions[worker_id]
            final_output[:, p["seq_start"]:p["seq_end"], :] += partials[worker_id]

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
