import os
"""Cluster test v2: explicit worker address (AP suppresses broadcast)."""
import paramiko, time

PASS = os.environ["NODE_PASS"]
PROMPT = "The capital of France is Paris. The capital of Italy is"
ARGS = ("--model /home/n1/models/dllama_model_qwen3_0.6b_q40.m "
        "--tokenizer /home/n1/models/dllama_tokenizer_qwen3_0.6b.t "
        f'--prompt "{PROMPT}" --max-seq-len 2048 --threads 4 '
        "--workers 192.168.0.97:9998")

# worker state on S2
c2 = paramiko.SSHClient()
c2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c2.connect("192.168.0.97", username="n2", password=PASS, timeout=15)
_, o, _ = c2.exec_command("cat ~/worker.log; echo ---; ss -tlnp | grep 9998 || echo NOT_LISTENING")
print("[S2] worker state:")
print(o.read().decode().strip())
c2.close()

# distributed perplexity from S1
c1 = paramiko.SSHClient()
c1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c1.connect("192.168.0.71", username="n1", password=PASS, timeout=15)
print("[S1] perplexity-dist (explicit workers)...")
_, o, e = c1.exec_command(
    f"cd ~/pyre/rust && time ./target/release/dllama-rs perplexity-dist {ARGS}",
    timeout=900)
print(o.read().decode())
err = e.read().decode().strip()
if err:
    print("[stderr]", err[-400:])
c1.close()
