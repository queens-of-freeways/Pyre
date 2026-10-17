"""Final Mojo-on-Pi confirmation pass.

1. Rebuild the plugin on both nodes from the updated kernels.mojo
   (backend name string fix; numerics untouched — re-validated anyway).
2. Thermal readings before the run.
3. Single-node ppl sanity after the rebuild (expect 7.913573).
4. Full 24-step inference-dist over the cable with Mojo on BOTH nodes,
   wrapped in a remote `timeout` so the SSH channel can never hang.

Credentials come from NODE_PASS in the environment (never hardcoded).
"""
import io
import os
import time
import paramiko

PASS = os.environ['NODE_PASS']
LOCAL_MOJO = '/mnt/c/Users/kodur/distributed-llama/rust/mojo/kernels.mojo'
MODEL_M = '/home/n1/models/dllama_model_qwen3_0.6b_q40.m'
TOK_M = '/home/n1/models/dllama_tokenizer_qwen3_0.6b.t'
PROMPT = 'The capital of France is Paris. The capital of Italy is'
BIN = './target/release/dllama-rs'
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


def fire(c, cmd):
    c.exec_command(cmd)


clients = {}
for name, user, host in NODES:
    c = connect(host, user)
    clients[name] = (c, user)

    sftp = c.open_sftp()
    data = open(LOCAL_MOJO, 'rb').read()
    sftp.putfo(io.BytesIO(data), f'/home/{user}/mojo/kernels.mojo')
    sftp.close()

    o, e = run(c, '~/mojo-env/bin/mojo build --emit shared-lib --fp-mode contract=off '
                  '~/mojo/kernels.mojo -o ~/pyre/rust/target/release/libdllama_mojo.so 2>&1 | tail -2',
               timeout=600)
    print(f'[{name}] plugin rebuild:', (o + e).strip() or 'ok')
    o, e = run(c, '(vcgencmd measure_temp 2>/dev/null || '
                  'awk \'{printf "%.1f C\\n", $1/1000}\' /sys/class/thermal/thermal_zone0/temp)',
               timeout=30)
    print(f'[{name}] temp before:', o.strip())

c1, _ = clients['S1']
c2, _ = clients['S2']

# single-node ppl sanity after the rebuild (numerics must be untouched)
o, e = run(c1, f'cd ~/pyre/rust && timeout 300 {BIN} perplexity --model {MODEL_M} '
               f'--tokenizer {TOK_M} --prompt "{PROMPT}" --max-seq-len 2048 --threads 4 '
               f'2>&1 | grep -E "perplexity|backend"; echo RC=$?', timeout=360)
print('\n[post-rebuild ppl sanity, expect 7.913573 + new backend name]')
print(o.strip() or e.strip())

# full 24-step dist generation, Mojo on both nodes
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; sleep 1; true", timeout=30)
time.sleep(2)
fire(c2, f"cd ~/pyre/rust && setsid nohup {BIN} worker "
         "--host 0.0.0.0 --port 9998 > ~/worker.log 2>&1 < /dev/null &")
time.sleep(3)
o, e = run(c2, "ss -tln | grep 9998 && echo LISTENING", timeout=30)
print('\n[S2] worker:', o.strip())

o, e = run(c1, f'cd ~/pyre/rust && timeout 900 {BIN} inference-dist --model {MODEL_M} '
               f'--tokenizer {TOK_M} --workers 10.0.0.2:9998 --prompt "{PROMPT}" '
               f'--steps 24 --max-seq-len 2048 --threads 4; echo RC=$?', timeout=960)
print('\n=== 24-step inference-dist over CABLE, Mojo on BOTH nodes ===')
print((o + e).strip()[-1500:])

for name in ('S1', 'S2'):
    c, _ = clients[name]
    o, e = run(c, '(vcgencmd measure_temp 2>/dev/null || '
                  'awk \'{printf "%.1f C\\n", $1/1000}\' /sys/class/thermal/thermal_zone0/temp)',
               timeout=30)
    print(f'[{name}] temp after:', o.strip())

o, e = run(c2, 'tail -4 ~/worker.log', timeout=30)
print('[S2] worker log tail:\n' + o.strip())
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; true", timeout=30)

for c, _ in clients.values():
    c.close()
print('\nfinal confirmation done')
