"""Deploy the clean-stop fixes to both Pi nodes, rebuild, and verify.

Fixes deployed (after local cargo build):
  - crates/dllama-cluster/src/lib.rs : worker treats clean EOF at a control
    packet boundary as a normal session end ("root disconnected")
  - crates/dllama-cli/src/g1.rs      : perplexity-dist sends the stop packet
    (batch_size 0) so workers end their session gracefully

Verification: ppl over the direct cable -> expect 7.727886 and a clean
"stop packet received" in the worker log (no error line).

Credentials come from NODE_PASS in the environment (never hardcoded).
"""
import io
import os
import time
import paramiko

PASS = os.environ['NODE_PASS']
LOCAL = '/mnt/c/Users/kodur/distributed-llama/rust'
FILES = [
    'crates/dllama-cluster/src/lib.rs',
    'crates/dllama-cli/src/g1.rs',
]
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


clients = {}
for name, user, host in NODES:
    clients[name] = connect(host, user)

# 1. upload the two changed files
for name, user, host in NODES:
    sftp = clients[name].open_sftp()
    for rel in FILES:
        data = open(f'{LOCAL}/{rel}', 'rb').read()
        sftp.putfo(io.BytesIO(data), f'/home/{user}/pyre/rust/{rel}')
        print(f'[{name}] {rel} ({len(data)} bytes)')
    sftp.close()

# 2. rebuild on each node
for name, user, host in NODES:
    print(f'[{name}] cargo build --release ...')
    o, e = run(clients[name], 'cd ~/pyre/rust && ~/.cargo/bin/cargo build --release 2>&1 | tail -3', timeout=600)
    print((o + e).strip())

# 3. verify: fresh worker + ppl over the cable -> clean stop + 7.727886
c2 = clients['S2']
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; sleep 1; true", timeout=30)
time.sleep(2)
c2.exec_command("cd ~/pyre/rust && setsid nohup ./target/release/dllama-rs worker "
                "--host 0.0.0.0 --port 9998 > ~/worker.log 2>&1 < /dev/null &")
time.sleep(3)
o, e = run(c2, "ss -tln | grep 9998 && echo LISTENING", timeout=30)
print('[S2] worker:', o.strip())

o, e = run(clients['S1'],
           "cd ~/pyre/rust && time ./target/release/dllama-rs perplexity-dist "
           "--model /home/n1/models/dllama_model_qwen3_0.6b_q40.m "
           "--tokenizer /home/n1/models/dllama_tokenizer_qwen3_0.6b.t "
           '--prompt "The capital of France is Paris. The capital of Italy is" '
           "--max-seq-len 2048 --threads 4 --workers 10.0.0.2:9998", timeout=600)
print('[S1] ppl over cable:\n' + o.strip())
if e.strip():
    print('[stderr]', e.strip()[-400:])

time.sleep(2)
o, e = run(c2, 'tail -8 ~/worker.log', timeout=30)
print('[S2] worker log tail:\n' + o)
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; true", timeout=30)

for c in clients.values():
    c.close()
print('polish deploy + verify done')
