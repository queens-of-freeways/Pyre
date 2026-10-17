"""Undervoltage evidence + 4B two-node proof over WiFi.

The cable is physically down (carrier=0 both ends; S1 spontaneously
rebooted right after a sustained WiFi download). Checking the Pi
throttle/undervoltage flags, then running the 2-node Qwen3-4B proof over
the WiFi path (Pi<->Pi flows have been reliable all session).

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
NODES = [('S1', 'n1', '192.168.0.71'), ('S2', 'n2', '192.168.0.97')]

THROTTLE_BITS = {
    0: 'under-voltage NOW',
    1: 'arm freq capped NOW',
    2: 'throttled NOW',
    3: 'soft temp limit NOW',
    16: 'under-voltage OCCURRED since boot',
    17: 'arm cap OCCURRED since boot',
    18: 'throttling OCCURRED since boot',
    19: 'soft temp limit OCCURRED since boot',
}


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


cs = {}
for name, user, host in NODES:
    cs[name] = connect(host, user)

print('== power/thermal evidence ==')
for name, user, host in NODES:
    c = cs[name]
    o, _ = run(c, 'vcgencmd get_throttled 2>/dev/null; vcgencmd measure_temp 2>/dev/null', timeout=30)
    raw = o.strip()
    print(f'[{name}] {raw}')
    for line in raw.splitlines():
        if line.startswith('throttled='):
            try:
                v = int(line.split('=')[1], 16 if line.split('=')[1].startswith('0x') else 10)
                flags = [desc for bit, desc in THROTTLE_BITS.items() if v & (1 << bit)]
                print(f'[{name}] decoded: {", ".join(flags) if flags else "clean — no throttle/UV flags"}')
            except ValueError:
                pass
    o, _ = run(c, f"echo '{PASS}' | sudo -S dmesg 2>/dev/null | grep -icE 'under-?voltage'", timeout=60)
    print(f'[{name}] dmesg under-voltage lines: {o.strip()}')

# ---- 4B two-node proof over WiFi ----
c2 = cs['S2']
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; sleep 1; true", timeout=30)
time.sleep(2)
fire(c2, f"cd ~/pyre/rust && {ENV} setsid nohup {BIN} worker "
         "--host 0.0.0.0 --port 9998 > ~/worker.log 2>&1 < /dev/null &")
time.sleep(3)
o, e = run(c2, "ss -tln | grep 9998 && echo LISTENING", timeout=30)
print('\n[S2] worker:', o.strip())


def summarize(out, keep=('perplexity', 'Results', 'tokens/s', 'generated', 'nodes', 'RC=',
                        'Prediction', 'Evaluating', 'capital', 'stop packet', 'worker 1')):
    return '\n'.join(ln for ln in out.splitlines() if any(k in ln for k in keep))


o, e = run(cs['S1'], f'cd ~/pyre/rust && {ENV} timeout 900 {BIN} perplexity-dist --model {MODEL_S1} '
                     f'--workers 192.168.0.97:9998 --prompt "{PROMPT}" --max-seq-len 2048 --threads 4; echo RC=$?',
           timeout=960)
print('\n=== 2-node ppl-dist over WIFI, Qwen3-4B (Rust on both) ===')
print(summarize(o + e))

o, e = run(cs['S1'], f'cd ~/pyre/rust && {ENV} timeout 1500 {BIN} inference-dist --model {MODEL_S1} '
                     f'--workers 192.168.0.97:9998 --prompt "{PROMPT}" --steps 24 '
                     f'--max-seq-len 2048 --threads 4; echo RC=$?', timeout=1560)
print('\n=== 2-node inference-dist over WIFI, Qwen3-4B (Rust on both) ===')
print(summarize(o + e))
print('\n(full tail)')
print((o + e).strip()[-700:])

o, e = run(c2, 'tail -4 ~/worker.log', timeout=30)
print('\n[S2] worker tail:\n' + o.strip())
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; true", timeout=30)
o, e = run(c2, "pgrep -a dllama-rs || echo worker-stopped", timeout=30)
print('[S2]', o.strip())

for c in cs.values():
    c.close()
print('\nundervoltage check + wifi 4B proof done')
