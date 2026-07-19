from collections import Dict
from collections import List

struct NodeCap:
    var id: Int
    var flops_gflops: Float64
    var mem_bytes: Int
    var net_bps: Float64
    var barrier_latency_us: Float64

    fn __init__(inout self, id: Int, flops_gflops: Float64, mem_bytes: Int, net_bps: Float64, barrier_latency_us: Float64):
        self.id = id
        self.flops_gflops = flops_gflops
        self.mem_bytes = mem_bytes
        self.net_bps = net_bps
        self.barrier_latency_us = barrier_latency_us

struct ShardSpec:
    var start: Int
    var end: Int

    fn __init__(inout self, start: Int, end: Int):
        self.start = start
        self.end = end

    fn size(self) -> Int:
        return self.end - self.start

fn solve_partitions(nodes: List[NodeCap], ffn_dim: Int) -> Dict[Int, ShardSpec]:
    var total_flops = 0.0
    for i in range(len(nodes)):
        total_flops += nodes[i].flops_gflops

    var partitions = Dict[Int, ShardSpec]()
    var current_start = 0

    for i in range(len(nodes)):
        var node = nodes[i]
        var is_last = i == len(nodes) - 1
        
        var size: Int
        if is_last:
            size = ffn_dim - current_start
        else:
            # Proportional allocation based on flops_gflops
            var fraction = node.flops_gflops / total_flops
            size = max(1, int(fraction * Float64(ffn_dim)))
            
        var end = current_start + size
        partitions[node.id] = ShardSpec(current_start, end)
        current_start = end

    return partitions
