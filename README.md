# Pyre

**A model-agnostic, distributed LLM inference engine.** Run any GGUF quantization (or dllama `.m`) model, chat with it from a friendly CLI, serve it over an OpenAI-style API, and scale across multiple machines — with pluggable compute backends (CPU AVX2, OpenCL GPU, Mojo) that all produce **bit-identical** results.

Pyre is a full Rust rewrite and extension of [distributed-llama](https://github.com/b4rtaz/distributed-llama) (the C++ reference lives on in `src/` as the parity oracle).

---

## Quick start

**1. Build** (needs the Rust toolchain):
```bash
cargo build --release          # from the repo root: cd rust && cargo build --release
```

**2. Drop a model into `./models/`** — any `.gguf` file works. For example grab a Qwen3 GGUF from Hugging Face:
```bash
mkdir models
# e.g. bartowski/Qwen_Qwen3-0.6B-GGUF → Qwen_Qwen3-0.6B-Q4_K_M.gguf
```

**3. Chat:**
```bash
rust/target/release/dllama-rs chat 0.6B        # fuzzy name match — no paths to type
```

That's it. Type a message, get a streamed answer, empty line to quit.

---

## The CLI

| Command | What it does |
|---|---|
| `dllama-rs chat [model]` | interactive chat — streams tokens, multi-turn, remembers context |
| `dllama-rs ask [model] "question"` | one-shot answer |
| `dllama-rs models` | list discovered models |
| `dllama-rs ppl [model] --prompt "text"` | perplexity of a text |
| `dllama-rs serve [model] [--port 8080]` | OpenAI-style `/v1` API server |
| `dllama-rs nodes` | list auto-discovered cluster nodes |
| `dllama-rs perplexity-dist --workers auto ...` | distributed perplexity |
| `dllama-rs worker` | start a cluster worker (auto-discoverable) |

You never type paths: models are found automatically in `./models`, `$DLLAMA_MODELS_DIR`, or next to the binary. Names match fuzzily (`chat qwen3`, `ask q40`), and with exactly one model present you can omit the name entirely: `dllama-rs chat`.

### Useful flags

```
--threads N            CPU threads for matmuls (default: all cores)
--temperature 0.7      sampling (0 = greedy, the deterministic default)
--top-k 40 --top-p 0.9 sampling filters (with --temperature)
--seed N               reproducible sampling
--steps N              max tokens to generate
--system "text"        custom system prompt
```

### Environment variables

| Variable | Effect |
|---|---|
| `DLLAMA_RUST_GPU=0` | disable the OpenCL GPU accelerant |
| `DLLAMA_KERNEL_LIB=path` | load an alternate kernel backend (`.dll`/`.so`) |
| `DLLAMA_RS_THREADS=N` | like `--threads`, works everywhere |
| `DLLAMA_PREFILL=0` | disable batched prompt processing |

---

## Models

- **GGUF** (recommended): llama-3, mistral, qwen2/qwen3, qwen3-MoE, smollm, exa families — Q4_0, Q4_K, Q6_K, Q8_0, F16, F32. Tokenizer and chat template are embedded — just drop the file in.
- **dllama `.m`**: the original format with its `.t` tokenizer (Qwen3 builds available from the upstream converter). Place the `.t` next to the `.m`.

Both formats work identically through every command.

## Cluster: run a model across machines

Every machine runs a worker; the root discovers them automatically (UDP broadcast, port 9990 — no config files, no address lists):

```bash
# on each worker machine:
dllama-rs worker                     # listens + answers discovery probes

# on the root machine:
dllama-rs nodes                                                    # see who's out there
dllama-rs perplexity-dist --workers auto --model <name> --prompt "text"   # ppl
dllama-rs inference-dist  --workers auto --model <name> \
    --prompt "The capital of France is" --steps 64                  # generation
```
Generation works on the cluster too: the root samples each token and
broadcasts it, so every node's KV cache stays identical — same sampler
flags (`--temperature`, `--seed`, ...) as local inference.

Tensor-parallel sharding: each node holds 1/n of the weights and activations; nodes sync activations per layer over TCP. Requirements: node count is a power of two, and ≤ the model's KV-head count (GQA). With n nodes, a machine needs ~1/n the RAM.

Notes: same-machine roots find workers on loopback (works across WSL2); remote workers answer from their LAN IP. Explicit addresses still work: `--workers 192.168.1.10:9998 192.168.1.11:9998`.

## API server

```bash
dllama-rs serve qwen3 --port 8080
```
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hi"}],"temperature":0.7,"max_tokens":50}'
```
`temperature`, `top_p`, `top_k`, `seed` and `stream` (SSE) are supported per request.

## Backends (all bit-exact with each other)

Pyre's kernel ABI makes compute pluggable. The engine picks automatically, and every backend reproduces the same logits **bit-for-bit**:

| Backend | Where | How it engages |
|---|---|---|
| Rust AVX2 CPU | everywhere (built-in) | default |
| OpenCL GPU | NVIDIA, AMD, Intel GPUs | automatic when a GPU is found; `DLLAMA_RUST_GPU=0` to disable |
| Mojo CPU | Linux/WSL | drop `libdllama_mojo.so` next to the binary (see `rust/mojo/`) |

Example A/B: `DLLAMA_KERNEL_LIB=libdllama_mojo.so dllama-rs chat ...` — the engine logs which backend it picked.

## Building

```bash
cd rust
cargo build --release
```
- **Windows**: MSVC toolchain (standard `rustup` install). If a freshly built `dllama-rs.exe` is blocked by Device Guard/Smart App Control, rebuild once — the hash changes.
- **Linux/WSL**: same. For the optional Mojo backend: `rust/wsl/setup-mojo.sh` (rootless uv install, no sudo).
- The C++ oracle binary (`dllama.exe`) is only needed for parity development — see `rust/DESIGN.md` §10.

## Troubleshooting

- **Big model is slow** — the 30B-class MoE needs ~17 GB of weights streamed per token; on a machine where it doesn't fit in RAM you are disk-bound. Use a cluster (RAM splits across nodes), or a smaller quant.
- **`model is ambiguous`** — type a longer substring, or run `dllama-rs models` to see names.
- **No nodes found** — workers must be running; check the discovery port (UDP 9990) isn't firewalled on your LAN.
- **Different answer with the same seed?** Only `--temperature 0` (default) is deterministic; sampled runs vary by seed.

## Project layout

```
rust/                  the Pyre engine (Rust)
  DESIGN.md            full engineering log — architecture, wire protocol,
                       and the validation record for every gate (G0–G7)
  crates/dllama-*      kernel, IR, executors, loaders, cluster, API, CLI
  mojo/                Mojo kernel backend + GPU kernels
src/                   the C++ distributed-llama reference (parity oracle)
models/                your models (git-ignored)
```

`rust/DESIGN.md` is the deep-dive: model-agnostic graph IR, the DLRS cluster
protocol, quantization layouts, bit-exactness methodology, and the full gate
history with measurements.

## Credits & license

MIT — same as upstream. Pyre is a derivative of
[distributed-llama](https://github.com/b4rtaz/distributed-llama) by
Bartłomiej Tadych (b4rtaz); the C++ tree in `src/` is kept as the numerical
reference the Rust engine was validated against.
