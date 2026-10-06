# dllama Mojo kernel backend (C-ABI v1)

`kernels.mojo` implements the engine's kernel ABI v1 (see
`../crates/dllama-kernel/src/abi.rs` and DESIGN.md §10.9–§10.12) in Mojo 1.1,
bit-exact with the Rust AVX2 reference backend.

**OS-aware selection**: the engine auto-detects this plugin at startup —
place `libdllama_mojo.so` next to `dllama-rs` (or in the repo `mojo/` dir)
and it loads with no env vars on Linux/macOS. `DLLAMA_KERNEL_LIB` still
overrides. On Windows (no Mojo toolchain yet) the built-in Rust backend
runs, optionally accelerated by the OpenCL GPU path
(`../crates/dllama-kernel/src/opencl.rs`, `DLLAMA_RUST_GPU=0` to disable).

## Build (Linux / WSL2)

```sh
# one-time (rootless):
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/mojo-env
VIRTUAL_ENV=~/mojo-env uv pip install mojo --index https://whl.modular.com/simple/

# build the backend:
~/mojo-env/bin/mojo build --emit shared-lib --fp-mode contract=off \
    kernels.mojo -o libdllama_mojo.so
```

**`--fp-mode contract=off` is mandatory**: Mojo defaults to fp-contract=fast
(`sum + a*b` becomes one fma), while the Rust reference never contracts —
without this flag the K-quant matmuls drift by 1 ULP (see DESIGN.md §10.10).

## Run with the engine

```sh
DLLAMA_KERNEL_LIB=./libdllama_mojo.so dllama-rs perplexity --model ... --prompt ...
# stderr: [kernel] backend "mojo-1.1 (wsl)" loaded from ... (ABI v1)
```

## GPU kernels (`gpu_kernels.mojo`)

Device matmul kernels (Q40/Q8_0/Q4_K/Q6_K) in portable `max.gpu` intrinsics —
bit-exact by construction (thread-per-row, sequential fma per block; K-quant
terms plain mul+add). Compile-verified; runtime execution needs an
NVIDIA/AMD GPU (`DLLAMA_MOJO_GPU=1` opt-in, `DeviceContext.number_of_devices()`
gate). This repo's dev box has an Intel iGPU (not a Mojo target), so the
weight-residency cache + launch wiring are documented in the file header as
the next hardware session. See DESIGN.md §10.11.

## Bit-parity status

Full op-trace diffs (8719 lines) are byte-identical with the built-in Rust
backend on both the dllama `.m` format (Q40/Q80 paths) and GGUF Q4_K_M
(Q4_K/Q6_K/F16/Q8_0 paths). Generation works end-to-end. Perf is ~2.7× behind
the hand-AVX2 Rust kernels (scalar lane emulation; SIMD-izing the fma lane
chains is future work — the lane structure is already spelled out per lane,
so a `SIMD[DType.float32, 8]` port can preserve bit-exactness).

Windows: compile the same file with a future Windows Mojo toolchain
(`mojo build --emit shared-lib` → `dllama_mojo.dll`); no engine change needed.
