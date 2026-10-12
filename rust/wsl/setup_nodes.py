import os
"""Setup both SBCs: upload source, install rustup, start release build in background."""
import paramiko, threading, time

NODES = [("S1", "192.168.0.71", "n1"), ("S2", "192.168.0.97", "n2")]
PASS = os.environ["NODE_PASS"]
SRC = "/tmp/pyre-src.tgz"

def setup(name, host, user):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=PASS, timeout=20)
    def run(cmd, timeout=300):
        _, o, e = c.exec_command(cmd, timeout=timeout)
        rc = o.channel.recv_exit_status()
        out = o.read().decode().strip()
        err = e.read().decode().strip()
        return rc, out, err

    # 1. upload source
    sftp = c.open_sftp()
    sftp.put(SRC, "/tmp/pyre-src.tgz")
    sftp.close()
    print(f"[{name}] source uploaded")

    # 2. rustup install (idempotent; ~1-2 min each)
    rc, out, err = run(
        "test -x ~/.cargo/bin/cargo && echo HAVE_RUST || "
        "(curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal 2>&1 | tail -2)"
    )
    print(f"[{name}] rustup: {out or err}")
    rc, _, _ = run("test -x ~/.cargo/bin/cargo && echo OK")
    if rc != 0:
        print(f"[{name}] RUSTUP FAILED")
        c.close()
        return

    # 3. extract + start the build in background with a completion marker
    run("rm -rf ~/pyre && mkdir -p ~/pyre && tar xzf /tmp/pyre-src.tgz -C ~/pyre")
    run(
        "cd ~/pyre/rust && "
        "nohup sh -c '~/.cargo/bin/cargo build --release --bin dllama-rs > build.log 2>&1 && echo done > build.done || echo failed > build.done' >/dev/null 2>&1 &"
    )
    print(f"[{name}] build launched")
    c.close()

threads = [threading.Thread(target=setup, args=(n, h, u)) for n, h, u in NODES]
for t in threads: t.start()
for t in threads: t.join()
print("BOTH NODES: build running in background")
