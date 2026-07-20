import sys
import os
import time
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.orchestrator.cluster import ModelConfig
from src.orchestrator.root_node import RootNode
from src.orchestrator.worker_node import WorkerNode


def _run_worker(port):
    worker = WorkerNode(host="localhost", port=port, use_mdns=False)
    worker.start()


def test_distributed_forward_pass():
    procs = []
    for port in [9001, 9002]:
        p = multiprocessing.Process(target=_run_worker, args=(port,))
        p.start()
        procs.append(p)

    time.sleep(1)

    try:
        config = ModelConfig(hidden_dim=512, n_heads=8, n_kv_heads=8, head_dim=64, ffn_dim=2048)
        root = RootNode([("localhost", 9001), ("localhost", 9002)], config)
        x = np.random.randn(1, 64, 512).astype(np.float32)
        out = root.run(x)
        assert out.shape == (1, 64, 512), f"Expected (1, 64, 512), got {out.shape}"
        print("test_distributed_forward_pass passed!")
    finally:
        root.shutdown()
        for p in procs:
            p.terminate()
            p.join(timeout=5)


if __name__ == "__main__":
    test_distributed_forward_pass()
    print("All distributed cluster tests passed!")
