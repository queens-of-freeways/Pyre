#!/bin/bash
cd /mnt/c/Users/kodur/distributed-llama/rust/mojo
sed -i 's/def load_u16_le(b: Pointer\[Scalar\[DType.uint8\], MutUntrackedOrigin\],/def load_u16_le(b: Pointer[Scalar[DType.uint8], _],/' gpu_kernels.mojo
~/mojo-env/bin/mojo build --emit shared-lib --fp-mode contract=off gpu_kernels.mojo -o libdllama_gpu_check.so 2>&1 | grep -viE 'crashpad|^$' | head -14
ls -la libdllama_gpu_check.so 2>/dev/null | head -2
