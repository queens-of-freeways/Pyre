import os
"""Cluster test v3: properly detached worker (setsid) + held-open session."""
import paramiko, time

PASS = os.environ["NODE_PASS"]
PROMPT = "The capital of France is Paris. The capital of Italy is"
ARGS = ("--model /home/n1/models/dllama_model_qwen3_0.6b_q40.m "
        "--tokenizer /home/n1/models/dllama_tokenizer_qwen3_0.6b.t "
        f'--prompt "{PROMPT}" --max-seq-len 2048 --threads 4 '
        "--workers 192.168.0.97:9998")

# --- S2: worker with setsid (full session detach) ---
c2 = paramiko.SSHClient()
c2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c2.connect("192.168.0.97", username="n2", password=PASS, timeout=15)
c2.exec_command("pkill -f 'dllama-rs worker' 2>/dev/null; sleep 1; true")
time.sleep(2)
c2.exec_command(
    "cd ~/pyre/rust && setsid nohup ./target/release/dllama-rs worker "
    "--host 0.0.0.0 --port 9998 > ~/worker.log 2>&1 < /dev/null &"
)
time.sleep(3)
_, o, _ = c2.exec_command("cat ~/worker.log; echo ---; ss -tln | grep 9998 && echo LISTENING")
print("[S2] worker:")
print(o.read().decode().strip())

# --- S1: perplexity-dist ---
c1 = paramiko.SSHClient()
c1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c1.connect("192.168.0.71", username="n1", password=PASS, timeout=15)
print("[S1] perplexity-dist ...")
_, o, e = c1.exec_command(
    f"cd ~/pyre/rust && time ./target/release/dllama-rs perplexity-dist {ARGS}",
    timeout=900)
print(o.read().decode())
err = e.read().decode().strip()
if err:
    print("[stderr]", err[-400:])
c1.close()

# --- S1: inference-dist (worker accepts a second session) ---
c1 = paramiko.SSHClient()
c1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c1.connect("192.168.0.71", username="n1", password=PASS, timeout=15)
print("[S1] inference-dist ...")
_, o, e = c1.exec_command(
    f"cd ~/pyre/rust && time ./target/release/dllama-rs inference-dist {ARGS} --steps 24",
    timeout=900)
print(o.read().decode())
err = e.read().decode().strip()
if err:
    print("[stderr]", err[-400:])
c1.close()

# --- S2: final worker state ---
_, o, _ = c2.exec_command("tail -4 ~/worker.log")
print("[S2] worker log tail:")
print(o.read().decode().strip())
c2.close()
