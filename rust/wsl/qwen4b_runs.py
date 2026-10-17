"""Qwen3-4B proof runs on the 2-node Pi cluster (after qwen4b_download.py).

1. Single-node ppl on S1 (1-node anchor for the 4B).
2. 2-node ppl-dist over the cable (2-node anchor).
3. 2-node inference-dist over the cable (coherence + s/tok).

All runs force the built-in Rust backend (DLLAMA_KERNEL_LIB=/nonexistent)
on both root and worker; remote `timeout` wrappers so SSH can never hang.
--max-seq-len 2048 keeps KV within the 4GB node (~300 MB/node at 2 nodes).

Credentials come from NODE_PASS in the environment (never hardcoded).
"""
import os
import time
import paramiko

PASS = os.environ['NODE_PASS']
MODEL_S1 = '/home/n1/models/Qwen3-4B-Q4_K_M.gguf'
EXPECT = 2497280256
PROMPT = 'The capital of France is Paris. The capital of Italy is'
BIN = './target/release/dllama-rs'
ENV = 'DLLAMA_KERNEL_LIB=/nonexistent'


def connect(host, user):
    for attempt in range(5):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(host, username=user, password=PASS, timeout=20, banner_timeout=20)
            return c
        except Exception as ex:
            print(f'  connect retry {attempt + 1}: {ex}')
            time.sleep(5)
    raise SystemExit('cannot connect')


def run(c, cmd, timeout=120):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    o = out.read().decode('utf-8', errors='replace')
    e = err.read().decode('utf-8', errors='replace')
    return o, e


def fire(c, cmd):
    c.exec_command(cmd)


def temp(c):
    o, e = run(c, "(vcgencmd measure_temp 2>/dev/null || "
                   "awk '{printf \"temp=%.1f'C\\n\", $1/1000}' /sys/class/thermal/thermal_zone0/temp)", timeout=30)
    return o.strip()


def summarize(out, keep=('perplexity', 'Results', 'tokens/s', 'generated', 'Arch', 'nKvHeads', 'nodes',
                         'RC=', 'Prediction', 'Evaluating', 'capital', 'listened', 'stop packet',
                         'Sharding', 'estimated', 'worker 1')):
    return '\n'.join(ln for ln in out.splitlines() if any(k in ln for k in keep))


c1 = connect('192.168.0.71', 'n1')
c2 = connect('192.168.0.97', 'n2')

o, _ = run(c1, f'stat -c %s {MODEL_S1}', timeout=30)
if o.strip() != str(EXPECT):
    raise SystemExit(f'GGUF not ready on S1: {o.strip()} (expected {EXPECT})')

print('== hardware state ==')
for c, tag in ((c1, 'S1'), (c2, 'S2')):
    o, _ = run(c, 'free -h | head -2', timeout=30)
    print(f'[{tag}]', ' | '.join(o.strip().splitlines()), '|', temp(c))

# 1. single-node ppl anchor
o, e = run(c1, f'cd ~/pyre/rust && {ENV} timeout 900 {BIN} perplexity --model {MODEL_S1} '
               f'--prompt "{PROMPT}" --max-seq-len 2048 --threads 4; echo RC=$?', timeout=960)
print('\n=== S1 single-node ppl, Qwen3-4B Q4_K_M (Rust backend) ===')
print(summarize(o + e))

# 2. 2-node ppl-dist over the cable
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; sleep 1; true", timeout=30)
time.sleep(2)
fire(c2, f"cd ~/pyre/rust && {ENV} setsid nohup {BIN} worker "
         "--host 0.0.0.0 --port 9998 > ~/worker.log 2>&1 < /dev/null &")
time.sleep(3)
o, e = run(c2, "ss -tln | grep 9998 && echo LISTENING", timeout=30)
print('\n[S2] worker:', o.strip())

o, e = run(c1, f'cd ~/pyre/rust && {ENV} timeout 900 {BIN} perplexity-dist --model {MODEL_S1} '
               f'--workers 10.0.0.2:9998 --prompt "{PROMPT}" --max-seq-len 2048 --threads 4; echo RC=$?',
           timeout=960)
print('\n=== 2-node ppl-dist over CABLE, Qwen3-4B (Rust on both) ===')
print(summarize(o + e))

# 3. 2-node inference-dist over the cable
o, e = run(c1, f'cd ~/pyre/rust && {ENV} timeout 1500 {BIN} inference-dist --model {MODEL_S1} '
               f'--workers 10.0.0.2:9998 --prompt "{PROMPT}" --steps 24 '
               f'--max-seq-len 2048 --threads 4; echo RC=$?', timeout=1560)
print('\n=== 2-node inference-dist over CABLE, Qwen3-4B (Rust on both) ===')
print(summarize(o + e))
print('\n(full tail)')
print((o + e).strip()[-800:])

print('\n== after ==')
for c, tag in ((c1, 'S1'), (c2, 'S2')):
    o, _ = run(c, 'free -h | head -2', timeout=30)
    print(f'[{tag}]', ' | '.join(o.strip().splitlines()), '|', temp(c))
o, e = run(c2, 'tail -4 ~/worker.log', timeout=30)
print('[S2] worker tail:\n' + o.strip())
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; true", timeout=30)

c1.close()
c2.close()
print('\nqwen3-4b proof runs done')
