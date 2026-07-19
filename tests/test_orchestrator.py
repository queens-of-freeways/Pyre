import numpy as np
from src.orchestrator.cluster import ClusterOrchestrator, NodeCap, ModelConfig

def test_orchestrator_run():
    nodes = [
        NodeCap(id=0, flops_gflops=100.0, mem_bytes=1000000, net_bps=10000.0, barrier_latency_us=10.0),
        NodeCap(id=1, flops_gflops=100.0, mem_bytes=1000000, net_bps=10000.0, barrier_latency_us=10.0),
        NodeCap(id=2, flops_gflops=100.0, mem_bytes=1000000, net_bps=10000.0, barrier_latency_us=10.0),
    ]
    config = ModelConfig(hidden_dim=512, n_heads=8, n_kv_heads=8, head_dim=64, ffn_dim=2048)
    
    orchestrator = ClusterOrchestrator(nodes, config)
    
    x = np.random.randn(1, 64, 512).astype(np.float32)
    output = orchestrator.run(x)
    
    assert output.shape == (1, 64, 512), f"Expected (1, 64, 512), got {output.shape}"
    print("test_orchestrator_run passed!")

def test_detect_drift():
    nodes = [
        NodeCap(id=0, flops_gflops=100.0, mem_bytes=1000000, net_bps=10000.0, barrier_latency_us=10.0),
        NodeCap(id=1, flops_gflops=100.0, mem_bytes=1000000, net_bps=10000.0, barrier_latency_us=10.0),
        NodeCap(id=2, flops_gflops=100.0, mem_bytes=1000000, net_bps=10000.0, barrier_latency_us=10.0),
    ]
    config = ModelConfig(hidden_dim=512, n_heads=8, n_kv_heads=8, head_dim=64, ffn_dim=2048)
    orchestrator = ClusterOrchestrator(nodes, config)
    
    # 13.5 is >15% slower than average of [10.0, 10.1, 13.5] = 11.2
    assert orchestrator.detect_drift([10.0, 10.1, 13.5]) == True
    print("test_detect_drift passed!")

if __name__ == "__main__":
    test_orchestrator_run()
    test_detect_drift()
    print("All orchestrator tests passed!")
