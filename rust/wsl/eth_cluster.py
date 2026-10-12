"""Cluster validation over the direct 1 Gbps Ethernet link (S1 <-> S2 cable).

1. perplexity-dist over the cable  -> must reproduce ppl 7.727886 exactly.
2. inference-dist over WiFi        -> same-session s/tok baseline.
3. inference-dist over the cable   -> s/tok comparison vs WiFi.
4. `nodes` UDP discovery over the direct link (fresh worker)      [G3.2 follow-up]
5. perplexity-dist --workers auto  -> end-to-end auto-discovery run.

Note: background daemon launches use fire() (no output read) — a detached
setsid process never closes the exec channel, so reading would block.

Credentials come from NODE_PASS in the environment (never hardcoded).
"""
import os
import time
import paramiko

PASS = os.environ['NODE_PASS']
MODEL = '/home/n1/models/dllama_model_qwen3_0.6b_q40.m'
TOK = '/home/n1/models/dllama_tokenizer_qwen3_0.6b.t'
PROMPT = 'The capital of France is Paris. The capital of Italy is'
BASE = (f'--model {MODEL} --tokenizer {TOK} '
        f'--prompt "{PROMPT}" --max-seq-len 2048 --threads 4 ')


def connect(host, user):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=PASS, timeout=15, banner_timeout=15)
    return c


def run(c, cmd, timeout=900):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    o = out.read().decode('utf-8', errors='replace')
    e = err.read().decode('utf-8', errors='replace')
    return o, e


def fire(c, cmd):
    """Launch a detached background command without reading its output."""
    c.exec_command(cmd)


def start_worker(c2):
    run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; sleep 1; true", timeout=30)
    time.sleep(2)
    fire(c2, "cd ~/pyre/rust && setsid nohup ./target/release/dllama-rs worker "
            "--host 0.0.0.0 --port 9998 > ~/worker.log 2>&1 < /dev/null &")
    time.sleep(3)
    o, e = run(c2, "cat ~/worker.log; echo ---; ss -tln | grep 9998 && echo LISTENING", timeout=30)
    print('[S2] worker startup:')
    print(o)
    if 'LISTENING' not in o:
        print('[S2] !! worker not listening — aborting')
        raise SystemExit(1)


# --- S2: fresh worker (setsid, fully detached) ---
c2 = connect('192.168.0.97', 'n2')
start_worker(c2)

c1 = connect('192.168.0.71', 'n1')


def run_root(label, subcmd, extra):
    print(f'\n=== {label} ===')
    o, e = run(c1, f'cd ~/pyre/rust && time ./target/release/dllama-rs {subcmd} {BASE}{extra}')
    print(o.strip())
    tail = e.strip()
    if tail:
        print('[stderr]', tail[-600:])
    return o


# 1. perplexity over the cable (bit-exactness gate)
run_root('perplexity-dist over CABLE (10.0.0.2)', 'perplexity-dist', '--workers 10.0.0.2:9998')

# 2. inference over WiFi (same-session baseline)
run_root('inference-dist over WIFI (192.168.0.97)', 'inference-dist', '--workers 192.168.0.97:9998 --steps 24')

# 3. inference over the cable
run_root('inference-dist over CABLE (10.0.0.2)', 'inference-dist', '--workers 10.0.0.2:9998 --steps 24')

# 4. discovery over the direct link (fresh worker)
start_worker(c2)
for attempt in (1, 2):
    o, e = run(c1, 'cd ~/pyre/rust && ./target/release/dllama-rs nodes --timeout 2000', timeout=30)
    print(f'\n=== nodes discovery (attempt {attempt}) ===')
    print((o + e).strip())
    time.sleep(1)

# 5. end-to-end auto-discovery run
run_root('perplexity-dist with --workers AUTO', 'perplexity-dist', '--workers auto')

# --- final worker state + stop ---
o, e = run(c2, 'tail -6 ~/worker.log', timeout=30)
print('\n[S2] worker log tail:')
print(o)
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; true", timeout=30)
time.sleep(1)
o, e = run(c2, "pgrep -a dllama-rs || echo worker-stopped", timeout=30)
print('[S2]', o.strip())

c1.close()
c2.close()
print('\ncluster cable test done')
