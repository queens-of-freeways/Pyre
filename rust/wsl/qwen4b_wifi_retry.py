"""One careful retry of the 2-node Qwen3-4B proof over WiFi.

Context: both Pis show power-delivery problems (S2: undervoltage + throttling
flags; S1: spontaneous reboot = hard brownout, flags wiped at boot). The
worker on S2 is still listening. Fresh connect to S1 (it may have rebooted
again — wait up to ~2 min for it), then ppl-dist and, if that succeeds,
inference-dist, over the WiFi path. If the stream dies again we stop and
leave the hardware fix (proper 27W PSUs) to the user.

Credentials come from NODE_PASS in the environment (never hardcoded).
"""
import os
import time
import paramiko

PASS = os.environ['NODE_PASS']
MODEL_S1 = '/home/n1/models/Qwen3-4B-Q4_K_M.gguf'
PROMPT = 'The capital of France is Paris. The capital of Italy is'
BIN = './target/release/dllama-rs'
ENV = 'DLLAMA_KERNEL_LIB=/nonexistent'


def connect(host, user, tries=12, delay=10):
    for attempt in range(1, tries + 1):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(host, username=user, password=PASS, timeout=20, banner_timeout=20)
            if attempt > 1:
                print(f'  connected after {attempt} attempts')
            return c
        except Exception as ex:
            print(f'  connect attempt {attempt}: {type(ex).__name__}')
            time.sleep(delay)
    return None


def run(c, cmd, timeout=120):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    o = out.read().decode('utf-8', errors='replace')
    e = err.read().decode('utf-8', errors='replace')
    return o, e


def fire(c, cmd):
    c.exec_command(cmd)


c1 = connect('192.168.0.71', 'n1')
if c1 is None:
    raise SystemExit('S1 unreachable for ~2 min — likely rebooted again; stopping (hardware fix needed)')
o, _ = run(c1, 'uptime -p; vcgencmd measure_temp 2>/dev/null', timeout=30)
print('[S1]', o.strip().replace('\n', ' | '))

c2 = connect('192.168.0.97', 'n2')
o, e = run(c2, "ss -tln | grep 9998 && echo WORKER-STILL-UP || echo WORKER-DOWN", timeout=30)
print('[S2]', o.strip())
if 'WORKER-DOWN' in o:
    run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; true", timeout=30)
    time.sleep(2)
    fire(c2, f"cd ~/pyre/rust && {ENV} setsid nohup {BIN} worker "
             "--host 0.0.0.0 --port 9998 > ~/worker.log 2>&1 < /dev/null &")
    time.sleep(3)
    o, e = run(c2, "ss -tln | grep 9998 && echo LISTENING", timeout=30)
    print('[S2] worker restarted:', o.strip())


def summarize(out, keep=('perplexity', 'Results', 'tokens/s', 'generated', 'nodes', 'RC=',
                         'Prediction', 'Evaluating', 'capital', 'stop packet', 'worker 1')):
    return '\n'.join(ln for ln in out.splitlines() if any(k in ln for k in keep))


o, e = run(c1, f'cd ~/pyre/rust && {ENV} timeout 900 {BIN} perplexity-dist --model {MODEL_S1} '
               f'--workers 192.168.0.97:9998 --prompt "{PROMPT}" --max-seq-len 2048 --threads 4; echo RC=$?',
           timeout=960)
print('\n=== 2-node ppl-dist over WIFI, Qwen3-4B (Rust on both) ===')
print(summarize(o + e))

if 'RC=0' in (o + e):
    o, e = run(c1, f'cd ~/pyre/rust && {ENV} timeout 1500 {BIN} inference-dist --model {MODEL_S1} '
                   f'--workers 192.168.0.97:9998 --prompt "{PROMPT}" --steps 24 '
                   f'--max-seq-len 2048 --threads 4; echo RC=$?', timeout=1560)
    print('\n=== 2-node inference-dist over WIFI, Qwen3-4B (Rust on both) ===')
    print(summarize(o + e))
    print('\n(full tail)')
    print((o + e).strip()[-700:])
else:
    print('\n!! stream failed again — stopping. Hardware fix (power) required before the 4B dist proof.')

o, e = run(c2, 'tail -4 ~/worker.log', timeout=30)
print('\n[S2] worker tail:\n' + o.strip())
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; true", timeout=30)
o, e = run(c2, 'vcgencmd get_throttled 2>/dev/null; vcgencmd measure_temp 2>/dev/null', timeout=30)
print('[S2] final:', o.strip().replace('\n', ' | '))
if c1:
    o, e = run(c1, 'vcgencmd get_throttled 2>/dev/null; vcgencmd measure_temp 2>/dev/null', timeout=30)
    print('[S1] final:', o.strip().replace('\n', ' | '))

c1.close()
c2.close()
print('\nretry done')
