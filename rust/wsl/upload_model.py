import os
"""Upload the 0.6B model + tokenizer to S1 (root node) and sanity-check both binaries."""
import paramiko, time, sys

PASS = os.environ["NODE_PASS"]
LOCAL_M = "/mnt/c/Users/kodur/distributed-llama/models/dllama_model_qwen3_0.6b_q40.m"
LOCAL_T = "/mnt/c/Users/kodur/distributed-llama/models/dllama_tokenizer_qwen3_0.6b.t"

def sanity(host, user, label):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=PASS, timeout=15)
    _, o, _ = c.exec_command("cd ~/pyre/rust && ./target/release/dllama-rs help | head -3")
    print(f"[{label}] binary:", o.read().decode().strip().splitlines()[0])
    c.close()

sanity("192.168.0.71", "n1", "S1")
sanity("192.168.0.97", "n2", "S2")

# upload to S1
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.0.71", username="n1", password=PASS, timeout=15)
c.exec_command("mkdir -p ~/models")
time.sleep(1)

for local, remote in [(LOCAL_T, "~/models/dllama_tokenizer_qwen3_0.6b.t"), (LOCAL_M, "~/models/dllama_model_qwen3_0.6b_q40.m")]:
    t0 = time.time()
    sftp = c.open_sftp()
    sftp.put(local, remote.replace("~", "/home/n1"))
    sftp.close()
    dt = time.time() - t0
    print(f"[S1] uploaded {remote} in {dt:.0f}s")

_, o, _ = c.exec_command("ls -la ~/models/")
print(o.read().decode().strip())
c.close()
