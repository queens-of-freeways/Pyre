from std.testing import assert_equal, assert_true, TestSuite
from src.partitioner.solver import NodeCap, ShardSpec, solve_partitions

def test_3_nodes() raises:
    var nodes = List[NodeCap]()
    nodes.append(NodeCap(0, 100.0, 1000, 1000.0, 10.0))
    nodes.append(NodeCap(1, 200.0, 2000, 2000.0, 20.0))
    nodes.append(NodeCap(2, 300.0, 3000, 3000.0, 30.0))

    var ffn_dim = 1024
    var result = solve_partitions(nodes, ffn_dim)

    assert_true(0 in result)
    assert_true(1 in result)
    assert_true(2 in result)

    var total_size = 0
    for i in range(len(nodes)):
        total_size += result[nodes[i].id].size()

    assert_equal(total_size, ffn_dim)

def test_5_nodes() raises:
    var nodes = List[NodeCap]()
    nodes.append(NodeCap(0, 50.0, 1000, 1000.0, 10.0))
    nodes.append(NodeCap(1, 100.0, 2000, 2000.0, 20.0))
    nodes.append(NodeCap(2, 150.0, 3000, 3000.0, 30.0))
    nodes.append(NodeCap(3, 200.0, 4000, 4000.0, 40.0))
    nodes.append(NodeCap(4, 250.0, 5000, 5000.0, 50.0))

    var ffn_dim = 2048
    var result = solve_partitions(nodes, ffn_dim)

    assert_true(0 in result)
    assert_true(1 in result)
    assert_true(2 in result)
    assert_true(3 in result)
    assert_true(4 in result)

    var total_size = 0
    for i in range(len(nodes)):
        total_size += result[nodes[i].id].size()

    assert_equal(total_size, ffn_dim)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
