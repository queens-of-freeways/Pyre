# Project: distributed-llama-mojo

Rewrite of github.com/b4rtaz/distributed-llama in Mojo + Modular MAX,
breaking two original limits:
  - Option C: Ulysses-style sequence-parallel attention (breaks KV-head cap)
  - Level 2:  Adaptive non-uniform re-partitioning (breaks power-of-2 + homogeneity)

## Build & test
- Build:   `pixi run mojo build src/`
- Test:    `pixi run mojo run tests/`
- Format:  `pixi run mojo format src/`

## Critical architectural constraint
The Mojo Driver/Graph/Engine APIs were DEPRECATED in MAX v25.X.
Do NOT use `from max.driver import Graph` etc. in Mojo.
Use the PYTHON max.graph API (max.graph.Graph, ops.matmul, TensorType, DeviceRef)
and write Mojo ONLY for custom ops and kernels, plugged in via:
    Graph(..., custom_extensions=[Path(__file__).parent / "kernels"])
Reference: docs/llms-python.txt and docs/llms-max-guides.txt.

## Phase discipline
Work strictly in phases. Do not start Phase N+1 until Phase N's tests pass.
Phase 1: NodeCap + solve_partitions (pure Mojo math, 3-node and 5-node tests)
Phase 2: Python max.graph FFN builder with non-uniform ShardSpec
Phase 3: Mojo comm kernels (ring_all_reduce, all_to_all) + threaded mock test
Phase 4: Ulysses attention graph (Q normal, KV sequence-sharded)
Phase 5: Orchestrator integration + Level 2 drift detector
