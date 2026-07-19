from testing import assert_true, assert_equal
from collections import List
from collections import Dict
from src.partitioner.solver import NodeCap, ShardSpec, solve_partitions

fn test_3_nodes() raises:
    var nodes = List[NodeCap]()
    nodes.append(NodeCap(0, 100.0, 1000, 1000.0, 10.0))
    nodes.append(NodeCap(1, 200.0, 2000, 2000.0, 20.0))
    nodes.append(NodeCap(2, 300.0, 3000, 3000.0, 30.0))
    
    var ffn_dim = 1024
    var result = solve_partitions(nodes, ffn_dim)
    
    # Check that all nodes are present
    assert_true(result.contains(0))
    assert_true(result.contains(1))
    assert_true(result.contains(2))
    
    # Check that ranges sum to ffn_dim
    var total_size = 0
    for i in range(len(nodes)):
        var spec = result[nodes[i].id]
        total_size += spec.size()
        
    assert_equal(total_size, ffn_dim, "Total size must equal ffn_dim")

fn test_5_nodes() raises:
    var nodes = List[NodeCap]()
    nodes.append(NodeCap(0, 50.0, 1000, 1000.0, 10.0))
    nodes.append(NodeCap(1, 100.0, 2000, 2000.0, 20.0))
    nodes.append(NodeCap(2, 150.0, 3000, 3000.0, 30.0))
    nodes.append(NodeCap(3, 200.0, 4000, 4000.0, 40.0))
    nodes.append(NodeCap(4, 250.0, 5000, 5000.0, 50.0))
    
    var ffn_dim = 2048
    var result = solve_partitions(nodes, ffn_dim)
    
    # Check that all nodes are present
    assert_true(result.contains(0))
    assert_true(result.contains(1))
    assert_true(result.contains(2))
    assert_true(result.contains(3))
    assert_true(result.contains(4))
    
    # Check that ranges sum to ffn_dim
    var total_size = 0
    for i in range(len(nodes)):
        var spec = result[nodes[i].id]
        total_size += spec.size()
        
    assert_equal(total_size, ffn_dim, "Total size must equal ffn_dim")

fn main() raises:
    test_3_nodes()
    test_5_nodes()
    print("All solver tests passed!")
