struct NodeCap(Movable):
    var id: Int
    var flops_gflops: Float64
    var mem_bytes: Int
    var net_bps: Float64
    var barrier_latency_us: Float64

    def __init__(out self, id: Int, flops_gflops: Float64, mem_bytes: Int, net_bps: Float64, barrier_latency_us: Float64):
        self.id = id
        self.flops_gflops = flops_gflops
        self.mem_bytes = mem_bytes
        self.net_bps = net_bps
        self.barrier_latency_us = barrier_latency_us

struct ShardSpec(Movable):
    var start: Int
    var end: Int

    def __init__(out self, start: Int, end: Int):
        self.start = start
        self.end = end

    def size(self) -> Int:
        return self.end - self.start

def solve_partitions(nodes: List[NodeCap], ffn_dim: Int) -> Dict[Int, ShardSpec]:
    var total_flops: Float64 = 0.0
    for i in range(len(nodes)):
        total_flops += nodes[i].flops_gflops

    var partitions = Dict[Int, ShardSpec]()
    var current_start = 0

    for i in range(len(nodes)):
        var is_last = i == len(nodes) - 1

        var size: Int
        if is_last:
            size = ffn_dim - current_start
        else:
            var fraction = nodes[i].flops_gflops / total_flops
            size = max(1, Int(fraction * Float64(ffn_dim)))

        var end = current_start + size
        partitions[nodes[i].id] = ShardSpec(current_start, end)
        current_start = end

    return partitions^
