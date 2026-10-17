"""Diagnose the cable connect-timeout, repair it, and finish the 4B proof.

Symptoms: 2-node runs failed with `connect 10.0.0.2:9998: timed out` even
though the worker listened on 0.0.0.0 (verified with ss) — S1's SYNs got no
answer. Single-node 4B ppl worked (20.033398), so the engine is fine; this
is a link-state problem on the direct cable.

Steps:
 1. eth0 diagnostics on both nodes (address, carrier, NM profiles, dmesg)
 2. repair if needed (nmcli con up direct, fallback ip addr add)
 3. ping verify both ways
 4. single-node 4B generation on S1 — coherence check for the ppl=20 result
 5. 2-node ppl-dist over the cable
 6. 2-node inference-dist over the cable

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
NODES = [('S1', 'n1', '192.168.0.71', '10.0.0.1'), ('S2', 'n2', '192.168.0.97', '10.0.0.2')]


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


cs = {}
for name, user, host, eth_ip in NODES:
    cs[name] = connect(host, user)

# ---- 1. diagnostics ----
print('== eth0 diagnostics ==')
for name, user, host, eth_ip in NODES:
    c = cs[name]
    o, e = run(c, 'ip -brief addr show eth0; echo carrier=$(cat /sys/class/net/eth0/carrier 2>/dev/null); '
                  'nmcli -t -f NAME,DEVICE,STATE connection show --active; uptime -p', timeout=30)
    print(f'[{name}] {o.strip()}')
    o, e = run(c, f"echo '{PASS}' | sudo -S dmesg 2>/dev/null | grep -iE 'eth0|bcmgenet|link' | tail -4",
               timeout=60)
    print(f'[{name}] dmesg: {o.strip() or "(none)"}')

o, e = run(cs['S1'], 'ip route get 10.0.0.2 2>&1; ping -c 3 -W 2 10.0.0.2 2>&1 | tail -2', timeout=30)
print('[S1->S2]', o.strip().replace('\n', ' | '))

# ---- 2. repair (idempotent) ----
print('\n== repair if needed ==')
for name, user, host, eth_ip in NODES:
    c = cs[name]
    o, _ = run(c, 'ip -brief addr show eth0', timeout=30)
    if eth_ip in o:
        print(f'[{name}] eth0 already has {eth_ip}')
        continue
    print(f'[{name}] eth0 missing {eth_ip} -> bringing up the direct profile')
    o, e = run(c, f"echo '{PASS}' | sudo -S nmcli connection up direct 2>&1 | tail -1", timeout=60)
    print(f'[{name}] nmcli up: {o.strip()}')
    time.sleep(2)
    o, _ = run(c, 'ip -brief addr show eth0', timeout=30)
    if eth_ip not in o:
        o2, e2 = run(c, f"echo '{PASS}' | sudo -S ip addr add {eth_ip}/24 dev eth0 2>&1", timeout=30)
        print(f'[{name}] fallback ip addr add: {(o2 + e2).strip() or "ok"}')
    o, _ = run(c, 'ip -brief addr show eth0', timeout=30)
    print(f'[{name}] eth0 now: {o.strip()}')

# ---- 3. ping verify ----
print('\n== ping verify ==')
ok = True
for src, dst, target in (('S1', 'S2', '10.0.0.2'), ('S2', 'S1', '10.0.0.1')):
    o, e = run(cs[src], f'ping -c 3 -W 2 {target} 2>&1 | tail -2', timeout=30)
    print(f'[{src}->{dst}]', o.strip().replace('\n', ' | '))
    if '0% packet loss' not in o and ' 0% ' not in o:
        ok = 'ttl=' in o
print('cable path:', 'UP' if ok else 'STILL DOWN')

# ---- 4. coherence check: single-node 4B generation on S1 ----
o, e = run(cs['S1'], f'cd ~/pyre/rust && {ENV} timeout 600 {BIN} inference --model {MODEL_S1} '
                     f'--prompt "{PROMPT}" --steps 12 --max-seq-len 2048 --threads 4; echo RC=$?', timeout=660)
print('\n=== S1 single-node generation, Qwen3-4B (coherence check) ===')
print((o + e).strip()[-1200:])

if not ok:
    print('\n!! cable still down — stopping before the dist runs')
    for c in cs.values():
        c.close()
    raise SystemExit(1)

# ---- 5-6. two-node runs over the cable ----
run(cs['S2'], "pkill -f 'dllama-rs worker' 2>/dev/null; sleep 1; true", timeout=30)
time.sleep(2)
fire(cs['S2'], f"cd ~/pyre/rust && {ENV} setsid nohup {BIN} worker "
               "--host 0.0.0.0 --port 9998 > ~/worker.log 2>&1 < /dev/null &")
time.sleep(3)
o, e = run(cs['S2'], "ss -tln | grep 9998 && echo LISTENING", timeout=30)
print('\n[S2] worker:', o.strip())


def summarize(out, keep=('perplexity', 'Results', 'tokens/s', 'generated', 'Arch', 'nKvHeads', 'nodes',
                         'RC=', 'Prediction', 'Evaluating', 'capital', 'stop packet', 'worker 1',
                         'tensors')):
    return '\n'.join(ln for ln in out.splitlines() if any(k in ln for k in keep))


o, e = run(cs['S1'], f'cd ~/pyre/rust && {ENV} timeout 900 {BIN} perplexity-dist --model {MODEL_S1} '
                     f'--workers 10.0.0.2:9998 --prompt "{PROMPT}" --max-seq-len 2048 --threads 4; echo RC=$?',
           timeout=960)
print('\n=== 2-node ppl-dist over CABLE, Qwen3-4B (Rust on both) ===')
print(summarize(o + e))

o, e = run(cs['S1'], f'cd ~/pyre/rust && {ENV} timeout 1500 {BIN} inference-dist --model {MODEL_S1} '
                     f'--workers 10.0.0.2:9998 --prompt "{PROMPT}" --steps 24 '
                     f'--max-seq-len 2048 --threads 4; echo RC=$?', timeout=1560)
print('\n=== 2-node inference-dist over CABLE, Qwen3-4B (Rust on both) ===')
print(summarize(o + e))
print('\n(full tail)')
print((o + e).strip()[-800:])

print('\n== after ==')
for name, user, host, eth_ip in NODES:
    c = cs[name]
    o, _ = run(c, 'free -h | head -2', timeout=30)
    print(f'[{name}]', ' | '.join(o.strip().splitlines()), '|', temp(c))
o, e = run(cs['S2'], 'tail -4 ~/worker.log', timeout=30)
print('[S2] worker tail:\n' + o.strip())
run(cs['S2'], "pkill -f 'dllama-rs worker' 2>/dev/null; true", timeout=30)

for c in cs.values():
    c.close()
print('\nfix + proof done')
