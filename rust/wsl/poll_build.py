import os
"""Poll cluster build status on both nodes."""
import paramiko

NODES = [("S1", "192.168.0.71", "n1"), ("S2", "192.168.0.97", "n2")]
PASS = os.environ["NODE_PASS"]

for name, host, user in NODES:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=PASS, timeout=15)
    _, o, _ = c.exec_command(
        "cd ~/pyre/rust && "
        "if [ -f build.done ]; then "
        "  echo \"DONE_MARKER=$(cat build.done)\"; "
        "  tail -3 build.log; "
        "  ls -la target/release/dllama-rs 2>/dev/null || echo NO_BINARY; "
        "else "
        "  echo BUILDING; "
        "  grep -c '^   Compiling' build.log 2>/dev/null | xargs -I{} echo compiled={} crates; "
        "  tail -1 build.log | cut -c1-120; "
        "fi"
    )
    print(f"===== {name}:")
    print(o.read().decode().strip())
    c.close()
