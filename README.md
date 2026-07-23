# Pyre — Distributed LLM Inference

Run large language models across multiple machines, breaking the memory and compute limits of a single node. Supports any HuggingFace causal LM, including Gemma 4's advanced architecture (dual head_dim, K=V sharing, p-RoPE, V_norm, Per-Layer Embeddings, Shared KV Cache).

## Quickstart

```bash
# One command — auto-starts a local worker, no separate terminal needed
pyre run --model HuggingFaceTB/SmolLM-135M --prompt "Hello" --max-tokens 10

# Chat mode — auto-stops at EOS token
pyre run --model Qwen/Qwen2.5-0.5B-Instruct --prompt "What is 2+2?" --chat

# With remote workers
pyre run --model mistralai/Mistral-7B-v0.3 --workers 10.0.0.5:9001,10.0.0.6:9002

# List workers via mDNS
pyre ps
```

## Install

### Prerequisites
- Linux x86-64 (root node with Mojo/MAX requires this)
- Linux aarch64 (workers only — Python-only, no Mojo/MAX needed)
- [pixi](https://pixi.sh) (package manager)

### Setup
```bash
git clone https://github.com/queens-of-freeways/Pyre.git
cd Pyre
pixi install
```

That's it — pixi pulls Mojo, Python, PyTorch, transformers, zeroconf, and the MAX SDK automatically. The editable install creates a `pyre` entry point at `~/.local/bin/pyre` that auto-forwards to pixi, so you can run it from anywhere without prefixing every command with `pixi run`.

## Usage

### `pyre run` — Run generation

Starts a local worker by default (the root machine also contributes compute).

| Flag | Default | Description |
|---|---|---|---|
| `--model` | `HuggingFaceTB/SmolLM-135M` | Any HuggingFace model ID |
| `--workers` | auto (mDNS) | Comma-separated `host:port` list |
| `--prompt` | `Hello, my name is` | Input text |
| `--max-tokens` | 10 | Tokens to generate (ignored when `--chat`) |
| `--chat` | false | Generate until EOS token; no `--max-tokens` needed |
| `--temperature` | 0.7 | Sampling temperature (0 = greedy) |
| `--layers` | auto (all) | Number of transformer layers |
| `--no-local` | false | Skip local worker (root only orchestrates) |
| `--discover-timeout` | 3.0 | Seconds to wait for mDNS discovery |
| `--expect-workers` | — | Return as soon as N workers found |
| `--reload` | false | Force re-download model from HuggingFace, bypassing disk cache |

```bash
# Local only
pyre run --model HuggingFaceTB/SmolLM-135M --prompt "Once upon a time"

# Chat mode (auto-stop at EOS)
pyre run --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Hello" --chat

# Greedy decoding
pyre run --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Hi!" --temperature 0

# Explicit remote workers
pyre run --model Qwen/Qwen2.5-0.5B --workers 192.168.1.5:9001 --prompt "Hello"

# mDNS auto-discovery + local worker
pyre run --model mistralai/Mistral-7B-v0.3 --prompt "The answer is"
```

### `pyre worker` — Start a remote worker node

Run on each machine that contributes compute:

```bash
pyre worker --host 0.0.0.0 --port 9001
```

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | 9000 | Listen port (0 = auto-assign) |
| `--no-mdns` | false | Disable mDNS registration |

### `pyre ps` — List running workers

```bash
pyre ps
```

### HuggingFace Login (for gated models)

Models like Gemma 4, Llama 3, and Mistral require authentication:

```bash
huggingface-cli login
```

## Weight Cache

Model weights are cached to disk after the first load to accelerate subsequent runs:

- **Full weights** (`~/.cache/pyre/full/`) — The complete model is saved after loading from HuggingFace. Subsequent runs skip `from_pretrained()` and load directly from cache.
- **Sliced weights** (`~/.cache/pyre/sliced/`) — Per-node weight slices are cached after the first partition computation. Subsequent runs skip the torch→float32 conversion and slicing overhead.

Clear a model's cache with `pyre run --reload` to force re-downloading from HuggingFace.

## Architecture

Pyre distributes a single forward pass across nodes using two complementary parallelization strategies:

### Ulysses Sequence-Parallel Attention
The sequence dimension is split across workers. Each worker computes Q, K, V for its subsequence. The root gathers all QKV, computes full softmax attention, then scatters attention output back. This breaks the KV-head-per-worker limit — any number of workers can participate regardless of the model's KV head count.

### Non-Uniform FFN Partitioning
The FFN intermediate dimension is split across workers (including the root). Each worker computes its partial FFN output. The root sums partials to produce the final hidden state. Partition boundaries can adapt at runtime via the drift detector.

### Supported Architectures
Pyre reads `AutoConfig` from HuggingFace and adapts to any model using standard Q/K/V/O projections + gated MLP, including:
- Llama 2/3
- Mistral / Mixtral
- Qwen2 / Qwen2.5
- Phi-3 / Phi-4
- Gemma 2/3
- **Gemma 4** (dual head_dim, K=V sharing, p-RoPE, V_norm, PLE, Shared KV Cache)
- Any model with QKV bias (Qwen, some fine-tuned models)
- DeepSeek
- Any model using `q_proj/k_proj/v_proj/o_proj` + gated FFN

### Gemma 4 Features
- **Dual head_dim**: sliding window (small) and global (large) attention layers auto-detected from config
- **K=V sharing**: layers with `attention_k_eq_v` skip loading `v_proj` weights
- **p-RoPE**: partial rotary embeddings applied only to the first `rope_fraction` dimensions
- **V_norm**: RMSNorm applied to V before attention in specified layers
- **PLE (Per-Layer Embeddings)**: layer-specific token+context signals applied after FFN
- **Shared KV Cache**: layers with `num_kv_shared_layers` reuse K,V from earlier layers of the same type

### Network Protocol
- TCP binary protocol with msgpack-style framing (type + length + payload)
- Chunked transfer for large weight payloads (8 MB chunks)
- Heartbeat monitoring (30s timeout, 5s interval) with auto-reconnect
- mDNS service registration (`_dl-worker._tcp.local.`) for zero-config discovery
- Q8_0 block-wise weight quantization for 4x transfer compression

## Project Structure

```
pyre/                   # CLI package (click commands)
  __main__.py           # python -m pyre
  cli.py                # run, worker, ps commands
  core.py               # orchestrator wrapping logic
src/
  orchestrator/         # Distributed orchestration
    root_node.py        # Root orchestrator (attention + FFN aggregation)
    worker_node.py      # Worker node (QKV compute + FFN partials)
    generator.py        # Generation loop with streaming
    cluster.py          # ClusterOrchestrator, ModelConfig, NodeCap
    llama_loader.py     # WeightProvider — auto-detects any HF model weights
    graph_reuse.py      # Pad weights to share compiled attention graphs
    mdns.py             # mDNS auto-discovery
    net.py              # Chunked TCP, HeartbeatMonitor, ReconnectingClient
    protocol.py         # Message type constants
    quantizer.py        # Q8_0 block-wise quantization
  attention/
    builder.py          # Ulysses sequence-parallel attention graph
  ffn/
    builder.py          # Gated FFN graph with ShardSpec
tests/                  # Test suites (all pass)
```

## Development

```bash
# Run tests
pixi run python tests/test_generation.py
pixi run python tests/test_orchestrator.py
pixi run python tests/test_gemma4_e2e.py

# Build Mojo components
pixi run mojo build src/
```

## Caveats

- **Compilation delay**: MAX graph compilation takes 20–30s for small models, 1–2 min for larger ones (happens once on startup)
- **GPU**: Currently CPU-only. GPU execution via MAX is a future enhancement.
- **Gemma 4 weights**: All Gemma 4 models on HF are gated — you need a HuggingFace token with granted access
