#!/bin/bash
set -e
mkdir -p ~/dllama
cd ~/dllama
if [ ! -d rust ]; then
  tar -C /mnt/c/Users/kodur/distributed-llama --exclude=rust/target -cf - rust | tar xf -
fi
mkdir -p models
cp -n /mnt/c/Users/kodur/distributed-llama/models/dllama_model_qwen3_0.6b_q40.m models/ || true
cp -n /mnt/c/Users/kodur/distributed-llama/models/dllama_tokenizer_qwen3_0.6b.t models/ || true
cp -n /mnt/c/Users/kodur/distributed-llama/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf models/ || true
cp /mnt/c/Users/kodur/distributed-llama/rust/mojo/libdllama_mojo.so rust/mojo/ 2>/dev/null || true
cd rust
cargo build --release --workspace 2>&1 | grep -E "^error|warning: unused|Finished" | head -10
ls -la target/release/dllama-rs target/release/libdllama_kernel_c.so 2>/dev/null | head -4
