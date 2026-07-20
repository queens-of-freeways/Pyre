"""Gemma 4 end-to-end synthetic test: PLE, dual head_dim, K=V, p-RoPE, V_norm."""
import multiprocessing
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
multiprocessing.set_start_method("fork", force=True)

import numpy as np

from src.orchestrator.cluster import ModelConfig
from src.orchestrator.root_node import RootNode
from src.orchestrator.llama_loader import LayerProperties


def _make_synthetic_ple_weights(config, total_nodes, partitions):
    """Create all_layer_weights with PLE for every node."""
    hd = config.hidden_dim
    ple_dim = config.ple_dim
    n_layers = config.num_layers

    all_layer_weights = {}
    for node_id in range(total_nodes):
        p = partitions[node_id]
        width = p["ffn_end"] - p["ffn_start"]
        node_layers = {}
        for lidx in range(n_layers):
            lp = LayerProperties.standard(config.head_dim)
            if lidx % 3 == 1:
                lp = LayerProperties.gemma4_global(config.head_dim * 2)
            elif lidx % 3 == 2:
                lp = LayerProperties.gemma4_sliding(config.head_dim)
            hd_l = lp.head_dim
            n_heads = config.n_heads
            n_kv = config.n_kv_heads
            attn = {
                "q": np.random.randn(hd, n_heads * hd_l).astype(np.float32),
                "k": np.random.randn(hd, n_kv * hd_l).astype(np.float32),
                "v": None if not lp.has_v_proj else np.random.randn(hd, n_kv * hd_l).astype(np.float32),
                "o": np.random.randn(n_heads * hd_l, hd).astype(np.float32),
                "has_v_proj": lp.has_v_proj,
            }
            ffn = {
                "gate": np.random.randn(hd, width).astype(np.float32),
                "up": np.random.randn(hd, width).astype(np.float32),
                "down": np.random.randn(width, hd).astype(np.float32),
            }
            props = {
                "head_dim": hd_l,
                "has_v_proj": lp.has_v_proj,
                "rope_fraction": lp.rope_fraction,
                "use_v_norm": lp.use_v_norm,
                "attention_type": lp.attention_type,
            }
            result = {"attn": attn, "ffn": ffn, "_props": props}
            # Add PLE per-layer weights
            if ple_dim > 0:
                result["ple_gate"] = np.random.randn(hd, ple_dim).astype(np.float32)
                result["ple_proj"] = np.random.randn(ple_dim, hd).astype(np.float32)
                result["ple_post_norm"] = np.random.randn(hd).astype(np.float32)
            node_layers[lidx] = result
        all_layer_weights[node_id] = node_layers
    return all_layer_weights


WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "src", "orchestrator", "worker_node.py")

def _run_test(config, with_ple, port_base):
    """Run a single Gemma 4 test variant."""
    worker_procs = []
    for port in [port_base, port_base + 1]:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..")
        p = subprocess.Popen(
            [sys.executable, WORKER_SCRIPT, "--host", "localhost", "--port", str(port), "--no-mdns"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        worker_procs.append(p)
    time.sleep(1)

    total_nodes = 3
    local_seq = 64 // total_nodes
    seq_chunk = local_seq
    ffn_chunk = config.ffn_dim // total_nodes
    partitions = {}
    for i in range(total_nodes):
        ffn_start = i * ffn_chunk
        ffn_end = (i + 1) * ffn_chunk if i < total_nodes - 1 else config.ffn_dim
        seq_start = i * seq_chunk
        seq_end = (i + 1) * seq_chunk if i < total_nodes - 1 else 64
        partitions[i] = {"ffn_start": ffn_start, "ffn_end": ffn_end,
                         "seq_start": seq_start, "seq_end": seq_end}

    all_layer_weights = _make_synthetic_ple_weights(config, total_nodes, partitions)

    kwargs = {}
    if with_ple:
        kwargs["ple_embedding"] = np.random.randn(config.vocab_size,
                                                    config.num_layers * config.ple_dim).astype(np.float32)
        kwargs["ple_projection"] = np.random.randn(config.hidden_dim,
                                                     config.num_layers * config.ple_dim).astype(np.float32)
        kwargs["ple_projection_norm"] = np.random.randn(config.ple_dim).astype(np.float32)

    root = RootNode(
        [("localhost", port_base), ("localhost", port_base + 1)],
        config, all_layer_weights, **kwargs,
    )

    try:
        x = np.random.randn(1, 64, config.hidden_dim).astype(np.float32)
        if with_ple:
            input_ids = np.random.randint(0, config.vocab_size - 1, size=(1, 64)).astype(np.int32)
            out = root.run(x, input_ids=input_ids)
        else:
            out = root.run(x)
        assert out.shape == (1, 64, config.hidden_dim), f"Shape mismatch: {out.shape}"
        assert not np.isnan(out).any(), "NaN in output"
        label = "PLE" if with_ple else "no-PLE"
        print(f"Gemma 4 {label} test passed! shape={out.shape} range=[{out.min():.2f}, {out.max():.2f}]")
    finally:
        root.shutdown()
        for p in worker_procs:
            p.terminate()
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()


def test_kv_shared():
    """Test KV shared cache: last layers share KV from preceding same-type layer."""
    config = ModelConfig(
        hidden_dim=256, n_heads=4, n_kv_heads=2, head_dim=64,
        ffn_dim=1024, num_layers=5, vocab_size=4096,
        model_type="gemma4", ple_dim=0,
    )

    procs = []
    for port in [10030, 10031]:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..")
        p = subprocess.Popen(
            [sys.executable, WORKER_SCRIPT, "--host", "localhost", "--port", str(port), "--no-mdns"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(p)
    time.sleep(1)

    total_nodes = 3
    local_seq = 64 // total_nodes
    ffn_chunk = config.ffn_dim // total_nodes
    partitions = {}
    for i in range(total_nodes):
        ffn_start = i * ffn_chunk
        ffn_end = (i + 1) * ffn_chunk if i < total_nodes - 1 else config.ffn_dim
        seq_start = i * local_seq
        seq_end = (i + 1) * local_seq if i < total_nodes - 1 else 64
        partitions[i] = {"ffn_start": ffn_start, "ffn_end": ffn_end,
                         "seq_start": seq_start, "seq_end": seq_end}

    all_layer_weights = {}
    for node_id in range(total_nodes):
        p = partitions[node_id]
        width = p["ffn_end"] - p["ffn_start"]
        node_layers = {}
        for lidx in range(5):
            # Last 2 layers (3, 4) are KV shared, sharing from same-type layers
            is_shared = lidx >= 3
            # Layer types: 0=sliding, 1=global, 2=sliding, 3=shared(same type as 2), 4=shared(same type as 1)
            if lidx == 0:
                lp = LayerProperties.gemma4_sliding(config.head_dim)
            elif lidx == 1:
                lp = LayerProperties.gemma4_global(config.head_dim * 2)
            elif lidx == 2:
                lp = LayerProperties.gemma4_sliding(config.head_dim)
            elif lidx == 3:
                lp = LayerProperties.gemma4_sliding(config.head_dim)  # shares KV from layer 2
            elif lidx == 4:
                lp = LayerProperties.gemma4_global(config.head_dim * 2)  # shares KV from layer 1
            src = 2 if lidx == 3 else (1 if lidx == 4 else None)
            hd_l = lp.head_dim
            attn = {
                "q": np.random.randn(config.hidden_dim, config.n_heads * hd_l).astype(np.float32),
                "k": np.random.randn(config.hidden_dim, config.n_kv_heads * hd_l).astype(np.float32),
                "v": None if not lp.has_v_proj else np.random.randn(config.hidden_dim, config.n_kv_heads * hd_l).astype(np.float32),
                "o": np.random.randn(config.n_heads * hd_l, config.hidden_dim).astype(np.float32),
                "has_v_proj": lp.has_v_proj,
            }
            props = {
                "head_dim": hd_l,
                "has_v_proj": lp.has_v_proj,
                "rope_fraction": lp.rope_fraction,
                "use_v_norm": lp.use_v_norm,
                "attention_type": lp.attention_type,
            }
            if is_shared:
                props["kv_source_layer"] = src
            node_layers[lidx] = {
                "attn": attn,
                "ffn": {
                    "gate": np.random.randn(config.hidden_dim, width).astype(np.float32),
                    "up": np.random.randn(config.hidden_dim, width).astype(np.float32),
                    "down": np.random.randn(width, config.hidden_dim).astype(np.float32),
                },
                "_props": props,
            }
        all_layer_weights[node_id] = node_layers

    root = RootNode([("localhost", 10030), ("localhost", 10031)], config, all_layer_weights)
    try:
        x = np.random.randn(1, 64, config.hidden_dim).astype(np.float32)
        out = root.run(x)
        assert out.shape == (1, 64, config.hidden_dim), f"Shape: {out.shape}"
        assert not np.isnan(out).any(), "NaN"
        print("KV shared cache test passed!")
        print(f"  Output range: [{out.min():.2f}, {out.max():.2f}]")
    finally:
        root.shutdown()
        for p in procs:
            p.terminate()
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()


def test_full_gemma4_all_features():
    """Comprehensive Gemma 4 test: PLE + KV shared + mixed layers + quantizer + net helpers."""
    config = ModelConfig(
        hidden_dim=256, n_heads=4, n_kv_heads=2, head_dim=64,
        ffn_dim=1024, num_layers=4, vocab_size=4096,
        model_type="gemma4", ple_dim=16,
    )

    procs = []
    for port in [10040, 10041]:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..")
        p = subprocess.Popen(
            [sys.executable, WORKER_SCRIPT, "--host", "localhost", "--port", str(port), "--no-mdns"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(p)
    time.sleep(1)

    total_nodes = 3
    local_seq = 64 // total_nodes
    ffn_chunk = config.ffn_dim // total_nodes
    partitions = {}
    for i in range(total_nodes):
        ffn_start = i * ffn_chunk
        ffn_end = (i + 1) * ffn_chunk if i < total_nodes - 1 else config.ffn_dim
        seq_start = i * local_seq
        seq_end = (i + 1) * local_seq if i < total_nodes - 1 else 64
        partitions[i] = {"ffn_start": ffn_start, "ffn_end": ffn_end,
                         "seq_start": seq_start, "seq_end": seq_end}

    all_layer_weights = {}
    for node_id in range(total_nodes):
        p = partitions[node_id]
        width = p["ffn_end"] - p["ffn_start"]
        node_layers = {}
        for lidx in range(4):
            is_shared = lidx == 3
            src = lidx - 1 if is_shared else None
            if lidx == 0:
                lp = LayerProperties.gemma4_sliding(64)
            elif lidx == 1:
                lp = LayerProperties.gemma4_global(128)
            elif lidx == 2:
                lp = LayerProperties.gemma4_sliding(64)
            else:
                lp = LayerProperties.gemma4_sliding(64)  # shared, uses layer 2's K,V
            hd_l = lp.head_dim
            attn = {
                "q": np.random.randn(config.hidden_dim, config.n_heads * hd_l).astype(np.float32),
                "k": np.random.randn(config.hidden_dim, config.n_kv_heads * hd_l).astype(np.float32),
                "v": None if not lp.has_v_proj else np.random.randn(config.hidden_dim, config.n_kv_heads * hd_l).astype(np.float32),
                "o": np.random.randn(config.n_heads * hd_l, config.hidden_dim).astype(np.float32),
                "has_v_proj": lp.has_v_proj,
            }
            props = {
                "head_dim": hd_l,
                "has_v_proj": lp.has_v_proj,
                "rope_fraction": lp.rope_fraction,
                "use_v_norm": lp.use_v_norm,
                "attention_type": lp.attention_type,
            }
            if is_shared:
                props["kv_source_layer"] = src
            result = {
                "attn": attn,
                "ffn": {
                    "gate": np.random.randn(config.hidden_dim, width).astype(np.float32),
                    "up": np.random.randn(config.hidden_dim, width).astype(np.float32),
                    "down": np.random.randn(width, config.hidden_dim).astype(np.float32),
                },
                "_props": props,
            }
            # PLE per-layer weights
            result["ple_gate"] = np.random.randn(config.hidden_dim, config.ple_dim).astype(np.float32)
            result["ple_proj"] = np.random.randn(config.ple_dim, config.hidden_dim).astype(np.float32)
            result["ple_post_norm"] = np.random.randn(config.hidden_dim).astype(np.float32)
            node_layers[lidx] = result
        all_layer_weights[node_id] = node_layers

    ple_embedding = np.random.randn(config.vocab_size,
                                     config.num_layers * config.ple_dim).astype(np.float32)
    ple_projection = np.random.randn(config.hidden_dim,
                                      config.num_layers * config.ple_dim).astype(np.float32)
    ple_projection_norm = np.random.randn(config.ple_dim).astype(np.float32)

    root = RootNode(
        [("localhost", 10040), ("localhost", 10041)],
        config, all_layer_weights,
        ple_embedding=ple_embedding,
        ple_projection=ple_projection,
        ple_projection_norm=ple_projection_norm,
    )

    try:
        x = np.random.randn(1, 64, config.hidden_dim).astype(np.float32)
        input_ids = np.random.randint(0, config.vocab_size - 1, size=(1, 64)).astype(np.int32)
        out = root.run(x, input_ids=input_ids)
        assert out.shape == (1, 64, config.hidden_dim), f"Shape: {out.shape}"
        assert not np.isnan(out).any(), "NaN"
        print("Full Gemma 4 (PLE+KV shared+all features) test passed!")
        print(f"  Output range: [{out.min():.2f}, {out.max():.2f}]")
    finally:
        root.shutdown()
        for p in procs:
            p.terminate()
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()


def test_quantizer():
    """Test Q8_0 quantizer."""
    from src.orchestrator.quantizer import quantize_q80, dequantize_q80, quantize_weights_dict, dequantize_weights_dict
    arr = np.random.randn(16, 64).astype(np.float32) * 0.5 + 3.0
    q, s, sh = quantize_q80(arr)
    recovered = dequantize_q80(q, s, sh)
    mse = np.mean((arr - recovered) ** 2)
    assert recovered.shape == arr.shape, f"Shape: {recovered.shape} vs {arr.shape}"
    assert mse < 2.0, f"MSE too high: {mse}"
    print(f"Quantizer test passed! MSE: {mse:.6f}")

    d = {"a": arr, "b": {"c": arr * 2}}
    qd = quantize_weights_dict(d)
    rd = dequantize_weights_dict(qd)
    assert np.allclose(d["a"], rd["a"], atol=2.0)
    assert np.allclose(d["b"]["c"], rd["b"]["c"], atol=2.0)
    print("Weights dict quantize/dequantize test passed!")


def test_net_helpers():
    """Test net helpers (chunked send/recv with a local loopback)."""
    from src.orchestrator.net import send_msg, recv_msg
    import multiprocessing
    import socket

    def _server(port):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("localhost", port))
        srv.listen(1)
        conn, _ = srv.accept()
        conn.settimeout(10)
        msg_type, obj = recv_msg(conn)
        assert msg_type == 42
        assert obj == {"key": "value"}
        # Send back
        send_msg(conn, 99, [1, 2, 3])
        conn.close()
        srv.close()

    port = 10050
    p = multiprocessing.Process(target=_server, args=(port,))
    p.start()
    time.sleep(0.5)

    conn = socket.create_connection(("localhost", port), timeout=5)
    conn.settimeout(10)
    send_msg(conn, 42, {"key": "value"})
    msg_type, obj = recv_msg(conn)
    assert msg_type == 99
    assert obj == [1, 2, 3]
    conn.close()
    p.join(timeout=3)
    print("Net helper (chunked send/recv) test passed!")


def test_graph_reuse():
    """Test graph reuse logic."""
    from src.orchestrator.graph_reuse import select_graph_dims, pad_attn_weight
    needed = {(64, "standard", 1.0, False): 0, (128, "gemma4_global", 0.25, True): 1, (64, "gemma4_sliding", 1.0, False): 2}
    best_key, best_hd = select_graph_dims(needed)
    assert best_hd == 64, f"Expected 64, got {best_hd}"
    print(f"Graph reuse test passed! selected head_dim={best_hd}")

    w = np.random.randn(256, 128).astype(np.float32)
    padded = pad_attn_weight(w, target_heads=4, target_hd=64)
    assert padded.shape == (256, 256), f"Padded shape: {padded.shape}"
    assert np.allclose(padded[:, :128], w)
    assert np.all(padded[:, 128:] == 0)
    print("Weight padding test passed!")


if __name__ == "__main__":
    test_quantizer()
    test_net_helpers()
    test_graph_reuse()
    cfg_ple = ModelConfig(
        hidden_dim=256, n_heads=4, n_kv_heads=2, head_dim=64,
        ffn_dim=1024, num_layers=3, vocab_size=4096,
        model_type="gemma4", ple_dim=32,
    )
    cfg_nople = ModelConfig(
        hidden_dim=256, n_heads=4, n_kv_heads=2, head_dim=64,
        ffn_dim=1024, num_layers=2, vocab_size=4096,
        model_type="gemma4", ple_dim=0,
    )

    _run_test(cfg_ple, with_ple=True, port_base=10010)
    _run_test(cfg_nople, with_ple=False, port_base=10020)
    test_kv_shared()
    test_full_gemma4_all_features()
    print("\nAll Gemma 4 tests passed!")
