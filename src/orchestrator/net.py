"""Network utilities: chunked I/O, heartbeats, reconnection."""
from __future__ import annotations

import pickle
import socket
import struct
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks
HEARTBEAT_INTERVAL = 5.0
HEARTBEAT_TIMEOUT = 30.0


def send_msg(conn: socket.socket, msg_type: int, obj: Any = None):
    """Send a message with optional pickle payload using chunked transfer."""
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL) if obj is not None else b""
    n_chunks = (len(payload) + CHUNK_SIZE - 1) // CHUNK_SIZE
    header = struct.pack("!III", msg_type, len(payload), n_chunks)
    conn.sendall(header)
    for i in range(n_chunks):
        chunk = payload[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        conn.sendall(chunk)


def recv_exact(conn: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


def recv_msg(conn: socket.socket):
    """Receive a message sent by send_msg."""
    header = recv_exact(conn, 12)
    msg_type, total_len, n_chunks = struct.unpack("!III", header)
    payload = b""
    for _ in range(n_chunks):
        remaining = total_len - len(payload)
        chunk_size = min(CHUNK_SIZE, remaining)
        payload += recv_exact(conn, chunk_size)
    return msg_type, pickle.loads(payload) if payload else None


class HeartbeatMonitor:
    """Monitors connection health with periodic heartbeats."""

    def __init__(self, conn: socket.socket, on_lost: Optional[Callable] = None):
        self.conn = conn
        self.on_lost = on_lost
        self._last_activity = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            now = time.time()
            if now - self._last_activity > HEARTBEAT_TIMEOUT:
                if self.on_lost:
                    self.on_lost()
                break
            time.sleep(HEARTBEAT_INTERVAL)

    def touch(self):
        self._last_activity = time.time()

    def stop(self):
        self._running = False


class ReconnectingClient:
    """Wraps a socket with auto-reconnection and heartbeat."""

    def __init__(self, host: str, port: int, max_retries: int = 5, retry_delay: float = 1.0):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.conn: Optional[socket.socket] = None
        self.monitor: Optional[HeartbeatMonitor] = None

    def connect(self):
        for attempt in range(self.max_retries):
            try:
                self.conn = socket.create_connection((self.host, self.port), timeout=30)
                self.conn.settimeout(120.0)
                self.monitor = HeartbeatMonitor(self.conn)
                return True
            except (ConnectionRefusedError, socket.timeout, OSError):
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
        return False

    def send(self, msg_type: int, obj: Any = None):
        if self.conn is None:
            raise ConnectionError("Not connected")
        try:
            send_msg(self.conn, msg_type, obj)
            if self.monitor:
                self.monitor.touch()
        except (ConnectionError, BrokenPipeError):
            raise

    def recv(self):
        if self.conn is None:
            raise ConnectionError("Not connected")
        result = recv_msg(self.conn)
        if self.monitor:
            self.monitor.touch()
        return result

    def close(self):
        if self.monitor:
            self.monitor.stop()
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
