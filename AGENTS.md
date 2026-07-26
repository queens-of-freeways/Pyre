# Pyre — distributed LLM inference

Rewrite of github.com/b4rtaz/distributed-llama. Python-first (MAX graph API).
Mojo only for custom kernels (`src/kernels/mojo/`, `src/partitioner/`).

## Current state
All 7 original phases are substantially implemented. The system runs real
HuggingFace models (any causal LM) across TCP-connected workers with mDNS
auto-discovery, Ulysses sequence-parallel attention, non-uniform FFN
partitioning, and an adaptive drift detector.

## Build & test commands
- Mojo tests:    `pixi run mojo run -I . tests/test_solver.mojo`
  `pixi run mojo run -I . tests/test_comm.mojo`
- Python tests:  `pixi run python tests/test_ffn_builder.py`
  `pixi run python tests/test_attention_builder.py`
  `pixi run python tests/test_distributed_cluster.py`
  `pixi run python tests/test_generation.py`
- Run app:       `pixi run python -m pyre run --model HuggingFaceTB/SmolLM-135M --prompt "Hello"`
- Install:       `pixi install` (editable pip install of `pyre/` package)

## Critical architectural constraint
MAX Driver/Graph/Engine APIs DEPRECATED in Mojo. **Use Python** for graph
construction (`max.graph.Graph`, `max.graph.ops`, `TensorType`, `DeviceRef`).
Mojo is ONLY for custom ops/kernels loaded via `custom_extensions`.
Reference: `docs/llms-python.txt` and `docs/llms-max-guides.txt`.

## Key gotchas
- `multiprocessing.set_start_method("fork", force=True)` must run before
  importing orchestrator modules (done in `pyre/core.py` and test files).
- Tests use `sys.path.insert(0, ...)` because the `pyre/` CLI package is
  separate from `src/` library package.
- CLI auto-forwards to pixi when called outside pixi shell (checks
  `PIXI_IN_SHELL` / `PIXI_PROJECT_ROOT` env vars).
- Weight cache lives at `~/.cache/pyre/full/` and `~/.cache/pyre/sliced/`.
  Use `--reload` to force re-download from HuggingFace.
- Mojo & Python packages coexist. `src/__init__.py` and `src/__init__.mojo`
  both exist; keep them in sync.
- Python graph compilation takes 20-30s on first run (cached per session).
- `src/orchestrator/graph_reuse.py` pads weights to share compiled attention
  graphs across layers with different head_dims.

## Package boundaries
- `pyre/` — Click CLI (`cli.py`:`main`) + core entry points (`core.py`)
- `src/orchestrator/` — RootNode, WorkerNode, Generator, ClusterOrchestrator,
  WeightProvider (llama_loader.py), network (net.py, protocol.py, mdns.py),
  quantizer, graph_reuse
- `src/ffn/builder.py` — `build_ffn_graph()` using `max.graph`
- `src/attention/builder.py` — `build_ulysses_attention_graph()` using
  `max.graph`
- `src/partitioner/` — Mojo `NodeCap`/`solve_partitions` (pure math)
- `src/kernels/mojo/` — Mojo mock comm kernels (q80 compression,
  ring_all_reduce_accumulate, all_to_all_dispatch)
