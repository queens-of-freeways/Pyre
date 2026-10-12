import os
"""The real 2-node cluster test: worker on S2, root on S1."""
import paramiko, time

PASS = os.environ["NODE_PASS"]
PROMPT = "The capital of France is Paris. The capital of Italy is"
ARGS = ("--model /home/n1/models/dllama_model_qwen3_0.6b_q40.m "
        "--tokenizer /home/n1/models/dllama_tokenizer_qwen3_0.6b.t "
        f'--prompt "{PROMPT}" --max-seq-len 2048 --threads 4')

# 1. start the worker on S2 (binds all interfaces so S1 can reach it)
c2 = paramiko.SSHClient()
c2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c2.connect("192.168.0.97", username="n2", password=PASS, timeout=15)
c2.exec_command(
    "cd ~/pyre/rust && "
    "pkill -f 'dllama-rs worker' 2>/dev/null; sleep 1; "
    "nohup ./target/release/dllama-rs worker --host 0.0.0.0 --port 9998 "
    "> ~/worker.log 2>&1 &"
)
time.sleep(2)
_, o, _ = c2.exec_command("tail -1 ~/worker.log; pgrep -f 'dllama-rs worker' | head -1")
print("[S2] worker:", o.read().decode().strip() or "NOT RUNNING")
c2.close()

# 2. discovery from S1
c1 = paramiko.SSHClient()
c1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c1.connect("192.168.0.71", username="n1", password=PASS, timeout=15)

_, o, _ = c1.exec_command(
    "cd ~/pyre/rust && ./target/release/dllama-rs nodes --timeout 2000", timeout=30)
print("[S1] nodes:", o.read().decode().strip())
c1.close()

# 3. distributed perplexity (worker shard streams from S1 -> S2 first)
c1 = paramiko.SSHClient()
c1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c1.connect("192.168.0.71", username="n1", password=PASS, timeout=15)
print("[S1] perplexity-dist running (shard sync + eval)...")
_, o, e = c1.exec_command(
    f"cd ~/pyre/rust && time ./target/release/dllama-rs perplexity-dist --workers auto {ARGS}",
    timeout=900)
out = o.read().decode()
print(out)
err = e.read().decode().strip()
if err:
    print("[stderr]", err[-500:])
c1.close()

# 4. distributed inference (restart the worker: the ppl run ends its session;
#    serve_worker accepts a new connection)
c2 = paramiko.SSHClient()
c2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c2.connect("192.168.0.97", username="n2", password=PASS, timeout=15)
_, o, _ = c2.exec_command("tail -1 ~/worker.log")
print("[S2] after ppl:", o.read().decode().strip())
c2.close()

c1 = paramiko.SSHClient()
c1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c1.connect("192.168.0.71", username="n1", password=PASS, timeout=15)
print("[S1] inference-dist running...")
_, o, e = c1.exec_command(
    f"cd ~/pyre/rust && ./target/release/dllama-rs inference-dist --workers auto "
    f'{ARGS} --steps 24',
    timeout=900)
print(o.read().decode())
err = e.read().decode().strip()
if err:
    print("[stderr]", err[-500:])
c1.close()
