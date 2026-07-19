import sys
import os
import time
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transformers import AutoTokenizer

from src.orchestrator.cluster import ModelConfig
from src.orchestrator.root_node import RootNode
from src.orchestrator.worker_node import WorkerNode
from src.orchestrator.generator import Generator
from src.orchestrator.llama_loader import (
    get_smollm_config,
    create_synthetic_weights,
    slice_weights_for_node,
    validate_weight_shapes,
)


def _run_worker(port):
    worker = WorkerNode(host="localhost", port=port)
    worker.start()


def test_weight_slicing():
    config = get_smollm_config()
    full_weights = create_synthetic_weights(config)
    hidden_dim = config["hidden_dim"]
    head_dim = config["head_dim"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    ffn_dim = config["ffn_dim"]

    total_nodes = 3
    ffn_width_per_node = ffn_dim // total_nodes
    n_q_per_node = ffn_width_per_node // head_dim
    total_q_heads = n_q_per_node * total_nodes
    assert full_weights["q_weight"].shape == (hidden_dim, total_q_heads * head_dim)
    assert full_weights["k_weight"].shape == (hidden_dim, n_kv_heads * head_dim)
    assert full_weights["v_weight"].shape == (hidden_dim, n_kv_heads * head_dim)
    assert full_weights["ffn_gate"].shape == (hidden_dim, ffn_dim)
    assert full_weights["ffn_up"].shape == (hidden_dim, ffn_dim)
    assert full_weights["ffn_down"].shape == (ffn_dim, hidden_dim)
    assert full_weights["lm_head"].shape == (config["vocab_size"], hidden_dim)
    assert full_weights["embedding"].shape == (config["vocab_size"], hidden_dim)

    print("test_weight_slicing: full weight shapes OK")
    print("test_weight_slicing passed!")


def test_distributed_generation():
    procs = []
    for port in [9101, 9102]:
        p = multiprocessing.Process(target=_run_worker, args=(port,))
        p.start()
        procs.append(p)

    time.sleep(1)

    try:
        model_config = ModelConfig(
            hidden_dim=576, n_heads=9, n_kv_heads=3,
            head_dim=64, ffn_dim=1536,
        )
        root = RootNode([("localhost", 9101), ("localhost", 9102)], model_config)

        smollm_config = get_smollm_config()
        full_weights = create_synthetic_weights(smollm_config)

        node_weights = {}
        for node_id in root.partitions:
            p = root.partitions[node_id]
            shard = {"id": node_id, "ffn_start": p["ffn_start"], "ffn_end": p["ffn_end"]}
            sliced = slice_weights_for_node(shard, full_weights, 3, smollm_config)
            node_weights[node_id] = sliced

        validate_weight_shapes(node_weights, root.partitions, smollm_config)
        print("Weight shape validation passed!")

        lm_head = full_weights["lm_head"]
        embedding = full_weights["embedding"]

        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        gen = Generator(root, tokenizer, lm_head, embedding, seq_len=64)
        output = gen.generate("Hello, my name is", max_tokens=1)

        assert isinstance(output, str), f"Expected string, got {type(output)}"
        assert len(output) > 0, "Expected non-empty output"
        print(f"Generated output: {repr(output)}")
        print("test_distributed_generation passed!")

    finally:
        root.shutdown()
        for p in procs:
            p.terminate()
            p.join(timeout=5)


if __name__ == "__main__":
    test_weight_slicing()
    test_distributed_generation()
    print("All generation tests passed!")
