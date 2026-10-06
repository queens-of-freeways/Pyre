#!/bin/bash
set -e
W=/mnt/c/Users/kodur/distributed-llama/rust
cd ~/dllama/rust
cp -f $W/crates/dllama-kernel/src/lib.rs crates/dllama-kernel/src/lib.rs
cp -f $W/crates/dllama-kernel/src/opencl.rs crates/dllama-kernel/src/opencl.rs
cargo build --release --workspace 2>&1 | grep -E "^error|Finished" | head -5
cp -f $W/mojo/libdllama_mojo.so target/release/libdllama_mojo.so
echo "=== run 1: auto-detect (mojo .so next to exe, env unset)"
env -u DLLAMA_KERNEL_LIB -u DLLAMA_RUST_GPU ./target/release/dllama-rs perplexity \
  --model ../models/dllama_model_qwen3_0.6b_q40.m \
  --tokenizer ../models/dllama_tokenizer_qwen3_0.6b.t \
  --prompt "The capital of France is Paris. The capital of Italy is" 2>&1 | grep -E "\[kernel|perplexity"
echo "=== run 2: plugin removed -> builtin Rust, OpenCL probe degrades gracefully"
rm target/release/libdllama_mojo.so
env -u DLLAMA_KERNEL_LIB -u DLLAMA_RUST_GPU ./target/release/dllama-rs perplexity \
  --model ../models/dllama_model_qwen3_0.6b_q40.m \
  --tokenizer ../models/dllama_tokenizer_qwen3_0.6b.t \
  --prompt "The capital of France is Paris. The capital of Italy is" 2>&1 | grep -E "\[kernel|perplexity"
