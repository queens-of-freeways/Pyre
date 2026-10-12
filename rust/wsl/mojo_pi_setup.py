"""Try to install the Mojo toolchain on the Pi nodes (aarch64) and, if it
exists, build the Mojo kernel backend into the engine's plugin probe path.

Rootless install via uv — same recipe as the WSL setup (DESIGN 10.10):
  curl -LsSf https://astral.sh/uv/install.sh | sh        -> ~/.local/bin/uv
  uv venv ~/mojo-env
  VIRTUAL_ENV=~/mojo-env uv pip install mojo --index https://whl.modular.com/simple/
  upload kernels.mojo -> ~/mojo/
  mojo build --emit shared-lib --fp-mode contract=off ... -o <exe_dir>/libdllama_mojo.so

The key question this answers: does whl.modular.com publish linux-aarch64
wheels at all? If not, Mojo-on-Pi is a dead end for now (report honestly).

Credentials come from NODE_PASS in the environment (never hardcoded).
"""
import io
import os
import paramiko

PASS = os.environ['NODE_PASS']
LOCAL_MOJO = '/mnt/c/Users/kodur/distributed-llama/rust/mojo/kernels.mojo'
NODES = [('S1', 'n1', '192.168.0.71'), ('S2', 'n2', '192.168.0.97')]


def connect(host, user):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=PASS, timeout=15, banner_timeout=15)
    return c


def run(c, cmd, timeout=120):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    o = out.read().decode('utf-8', errors='replace')
    e = err.read().decode('utf-8', errors='replace')
    return o, e


for name, user, host in NODES:
    print(f'===== {name} ({user}@{host}) =====')
    c = connect(host, user)

    o, e = run(c, 'uname -m; ~/.local/bin/uv --version 2>/dev/null || echo no-uv')
    print('arch/uv:', o.strip(), ('| ' + e.strip() if e.strip() else ''))

    if 'no-uv' in o:
        o, e = run(c, 'curl -LsSf https://astral.sh/uv/install.sh | sh', timeout=300)
        print('uv install:', (o + e).strip()[-200:])

    run(c, '~/.local/bin/uv venv ~/mojo-env 2>&1 | tail -2', timeout=120)

    # THE aarch64 test:
    o, e = run(c, 'VIRTUAL_ENV=~/mojo-env ~/.local/bin/uv pip install mojo '
                  '--index https://whl.modular.com/simple/ 2>&1 | tail -12', timeout=900)
    print('mojo install tail:\n' + (o + e).strip())

    o, e = run(c, 'test -x ~/mojo-env/bin/mojo && echo mojo-present || echo mojo-missing', timeout=30)
    if 'mojo-present' not in o:
        print(f'!! [{name}] NO MOJO TOOLCHAIN for this platform — cannot build the Mojo backend here')
        c.close()
        continue

    o, e = run(c, '~/mojo-env/bin/mojo --version 2>&1', timeout=120)
    print('mojo version:', o.strip() or e.strip())

    run(c, 'mkdir -p ~/mojo', timeout=30)
    sftp = c.open_sftp()
    data = open(LOCAL_MOJO, 'rb').read()
    sftp.putfo(io.BytesIO(data), f'/home/{user}/mojo/kernels.mojo')
    sftp.close()
    print(f'[{name}] kernels.mojo uploaded ({len(data)} bytes)')

    o, e = run(c, '~/mojo-env/bin/mojo build --emit shared-lib --fp-mode contract=off '
                  '~/mojo/kernels.mojo -o ~/pyre/rust/target/release/libdllama_mojo.so 2>&1 | tail -8',
               timeout=900)
    print('mojo build tail:\n' + (o + e).strip())

    o, e = run(c, 'ls -la ~/pyre/rust/target/release/libdllama_mojo.so 2>/dev/null || echo BUILD-FAILED', timeout=30)
    print(f'[{name}]', o.strip() or e.strip())
    c.close()

print('mojo-on-pi setup done')
