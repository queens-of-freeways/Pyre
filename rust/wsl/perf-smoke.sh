#!/bin/bash
set -e
cd ~/dllama/rust
PROMPT="The capital of France is"
echo "=== generation smoke (mojo backend, 16 steps):"
DLLAMA_KERNEL_LIB=$PWD/mojo/libdllama_mojo.so ./target/release/dllama-rs inference \
  --model ../models/dllama_model_qwen3_0.6b_q40.m \
  --tokenizer ../models/dllama_tokenizer_qwen3_0.6b.t \
  --prompt "$PROMPT" --steps 16 2>&1 | tail -6
echo
echo "=== timing: builtin vs mojo (ppl, 11 evals, wall seconds)"
unset DLLAMA_KERNEL_LIB
T0=$(date +%s.%N)
./target/release/dllama-rs perplexity --model ../models/dllama_model_qwen3_0.6b_q40.m \
  --tokenizer ../models/dllama_tokenizer_qwen3_0.6b.t --prompt "$PROMPT. The capital of Italy is" > /dev/null 2>&1
T1=$(date +%s.%N)
echo "builtin: $(echo "$T1 - $T0" | bc)s"
export DLLAMA_KERNEL_LIB=$PWD/mojo/libdllama_mojo.so
T0=$(date +%s.%N)
./target/release/dllama-rs perplexity --model ../models/dllama_model_qwen3_0.6b_q40.m \
  --tokenizer ../models/dllama_tokenizer_qwen3_0.6b.t --prompt "$PROMPT. The capital of Italy is" > /dev/null 2>&1
T1=$(date +%s.%N)
echo "mojo:   $(echo "$T1 - $T0" | bc)s"
