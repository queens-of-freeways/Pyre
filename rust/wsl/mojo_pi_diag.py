"""Diagnose the Mojo-backend hang: inference-dist timed out at 1200s while
all ppl runs (single-node .m / GGUF + 2-node dist) were fast and bit-exact.

 A. post-mortem: leftover processes, worker log, kernel dmesg (segv/oom)
 B. kill leftovers
 C. single-node generation with Mojo   (remote `timeout` so the SSH channel
    can never hang; RC=124 = the process itself hung, RC=139 = SIGSEGV ...)
 D. if C failed: same with DLLAMA_PREFILL=0 (isolate the prefill batch path)
 E. control: single-node generation with the Rust built-in backend
 F. fresh worker + inference-dist retry over the cable (timeout 420)

Credentials come from NODE_PASS in the environment (never hardcoded).
"""
import os
import time
import paramiko

PASS = os.environ['NODE_PASS']
MODEL_M = '/home/n1/models/dllama_model_qwen3_0.6b_q40.m'
TOK_M = '/home/n1/models/dllama_tokenizer_qwen3_0.6b.t'
PROMPT = 'The capital of France is Paris. The capital of Italy is'
BIN = './target/release/dllama-rs'


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


c1 = connect('192.168.0.71', 'n1')
c2 = connect('192.168.0.97', 'n2')

print('== A. post-mortem ==')
for c, tag in ((c1, 'S1'), (c2, 'S2')):
    o, e = run(c, 'pgrep -a dllama-rs || echo none', timeout=30)
    print(f'[{tag}] procs:', o.strip())
    o, e = run(c, 'free -h | head -2', timeout=30)
    print(f'[{tag}] mem:', ' | '.join(o.strip().splitlines()))
    o, e = run(c, f"echo '{PASS}' | sudo -S dmesg 2>/dev/null | "
                  "grep -iE 'segv|sigill|sigabrt|out of memory|oom|dllama' | tail -5",
               timeout=60)
    print(f'[{tag}] dmesg hits:', o.strip() or '(none)')
o, e = run(c2, 'tail -8 ~/worker.log', timeout=30)
print('[S2] worker.log tail:\n' + o.strip())

print('\n== B. kill leftovers ==')
run(c1, 'pkill -f dllama-rs 2>/dev/null; true', timeout=30)
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; true", timeout=30)
time.sleep(1)


def gen(env, label, steps=24, to=150):
    cmd = (f'cd ~/pyre/rust && {env} timeout {to} {BIN} inference --model {MODEL_M} '
           f'--tokenizer {TOK_M} --prompt "{PROMPT}" --steps {steps} '
           f'--max-seq-len 2048 --threads 4; echo RC=$?')
    o, e = run(c1, cmd, timeout=to + 60)
    out = (o + e).strip()
    print(f'\n== single-node generation [{label}] ==')
    print(out[-1400:])
    return out


out_mojo = gen('', 'MOJO backend, prefill on')
if 'RC=0' not in out_mojo:
    gen('DLLAMA_PREFILL=0', 'MOJO backend, prefill OFF (DLLAMA_PREFILL=0)')
gen('DLLAMA_KERNEL_LIB=/nonexistent', 'RUST built-in control')

print('\n== F. inference-dist retry (fresh worker, Mojo on both) ==')
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; sleep 1; true", timeout=30)
time.sleep(2)
fire(c2, f"cd ~/pyre/rust && setsid nohup {BIN} worker "
         "--host 0.0.0.0 --port 9998 > ~/worker.log 2>&1 < /dev/null &")
time.sleep(3)
o, e = run(c1, f'cd ~/pyre/rust && timeout 420 {BIN} inference-dist --model {MODEL_M} '
               f'--tokenizer {TOK_M} --workers 10.0.0.2:9998 --prompt "{PROMPT}" '
               f'--steps 12 --max-seq-len 2048 --threads 4; echo RC=$?', timeout=480)
print((o + e).strip()[-1600:])
time.sleep(2)
o, e = run(c2, 'pgrep -a dllama-rs || echo worker-gone; echo ---; tail -6 ~/worker.log', timeout=30)
print('[S2] after dist attempt:\n' + o.strip())
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; true", timeout=30)

c1.close()
c2.close()
print('\ndiag done')
