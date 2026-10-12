"""Evidence check: which kernel backend do the Pi nodes actually use?

The Kernels::load() selection chain is:
  1. DLLAMA_KERNEL_LIB env var (explicit plugin override)
  2. platform default plugin probe (Mojo: dllama_mojo.so / libdllama_mojo.so
     next to the exe, in <exe>/../../mojo/, or in the cwd)
  3. built-in Rust kernels (compiled into the binary)

This checks that (1) and (2) are empty on the nodes, so (3) is the answer.
Credentials come from NODE_PASS in the environment (never hardcoded).
"""
import os
import paramiko

PASS = os.environ['NODE_PASS']
NODES = [('S1', 'n1', '192.168.0.71'), ('S2', 'n2', '192.168.0.97')]

CHECK = (
    'echo "-- Mojo shared libs anywhere under ~/pyre:"; '
    'find ~/pyre -iname "*mojo*" \\( -name "*.so" -o -name "*.so.*" \\) 2>/dev/null; '
    'echo "(none listed = none present)"; '
    'echo "-- next to the binary:"; '
    'ls ~/pyre/rust/target/release/ 2>/dev/null | grep -i mojo || echo none; '
    'echo "-- mojo source dir (source only, not compiled):"; '
    'ls ~/pyre/rust/mojo/ 2>/dev/null || echo absent; '
    'echo "-- DLLAMA env overrides:"; '
    'env | grep DLLAMA || echo "no DLLAMA vars set"; '
    'echo "-- OpenCL linked at build time (expect not-linked, loaded via dlopen):"; '
    'ldd ~/pyre/rust/target/release/dllama-rs 2>/dev/null | grep -i opencl || echo not-linked; '
    'echo "-- CPU SIMD:"; '
    'grep -m1 Features /proc/cpuinfo'
)

for name, user, host in NODES:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=PASS, timeout=15, banner_timeout=15)
    _, out, err = c.exec_command(CHECK, timeout=30)
    print(f'===== {name} ({user}@{host}) =====')
    print(out.read().decode('utf-8', errors='replace'))
    e = err.read().decode('utf-8', errors='replace').strip()
    if e:
        print('[stderr]', e)
    c.close()
print('kernel evidence check done')
