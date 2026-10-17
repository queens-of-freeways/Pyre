"""Validate the Mojo kernel backend on the real Pi cluster (aarch64).

1. Upload the 0.6B Q4_K_M GGUF to S1 (K-quant path coverage on ARM).
2. Single-node ppl (.m)        -> expect backend line + ppl 7.913573
3. Single-node ppl (GGUF)      -> expect ppl 13.472507 (K-quant + contract=off)
4. 2-node ppl-dist over the cable (Mojo on BOTH nodes) -> expect 7.727886
5. A/B inference-dist: Mojo vs Rust built-in (DLLAMA_KERNEL_LIB=/nonexistent
   forces the fallback on both root and worker) — same-session timing.

Credentials come from NODE_PASS in the environment (never hardcoded).
"""
import os
import time
import paramiko

PASS = os.environ['NODE_PASS']
LOCAL_GGUF = '/mnt/c/Users/kodur/distributed-llama/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf'
MODEL_M = '/home/n1/models/dllama_model_qwen3_0.6b_q40.m'
TOK_M = '/home/n1/models/dllama_tokenizer_qwen3_0.6b.t'
GGUF_S1 = '/home/n1/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf'
PROMPT = 'The capital of France is Paris. The capital of Italy is'
BIN = './target/release/dllama-rs'


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
    c.exec_command(cmd)


def start_worker(c2, env=''):
    run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; sleep 1; true", timeout=30)
    time.sleep(2)
    fire(c2, f"cd ~/pyre/rust && {env} setsid nohup {BIN} worker "
            "--host 0.0.0.0 --port 9998 > ~/worker.log 2>&1 < /dev/null &")
    time.sleep(3)
    o, e = run(c2, "ss -tln | grep 9998 && echo LISTENING; grep -m2 'kernel' ~/worker.log", timeout=30)
    print('[S2] worker:', o.strip() or e.strip())


c1 = connect('192.168.0.71', 'n1')
c2 = connect('192.168.0.97', 'n2')

# 0. sanity: plugin present on both nodes
for c, tag in ((c1, 'S1'), (c2, 'S2')):
    o, e = run(c, 'ls -la ~/pyre/rust/target/release/libdllama_mojo.so', timeout=30)
    print(f'[{tag}]', o.strip() or e.strip())

# 1. upload the GGUF to S1
t0 = time.time()
sftp = c1.open_sftp()
sftp.put(LOCAL_GGUF, GGUF_S1)
sftp.close()
print(f'[S1] uploaded Q4_K_M GGUF ({os.path.getsize(LOCAL_GGUF)/1e6:.0f} MB) '
      f'in {time.time()-t0:.0f}s')

# 2. single-node ppl, .m format, Mojo auto-picked
o, e = run(c1, f'cd ~/pyre/rust && time {BIN} perplexity --model {MODEL_M} '
               f'--tokenizer {TOK_M} --prompt "{PROMPT}" --max-seq-len 2048 --threads 4',
           timeout=900)
print('\n=== S1 single-node ppl .m (expect Mojo backend line + 7.913573) ===')
print((o + e).strip()[-1800:])

# 3. single-node ppl, GGUF Q4_K_M (K-quant paths on ARM)
o, e = run(c1, f'cd ~/pyre/rust && time {BIN} perplexity --model {GGUF_S1} '
               f'--prompt "{PROMPT}" --max-seq-len 2048 --threads 4',
           timeout=900)
print('\n=== S1 single-node ppl GGUF Q4_K_M (expect 13.472507) ===')
print((o + e).strip()[-1800:])

# 4. 2-node ppl-dist over the cable, Mojo on both nodes
start_worker(c2)
o, e = run(c1, f'cd ~/pyre/rust && time {BIN} perplexity-dist --model {MODEL_M} '
               f'--tokenizer {TOK_M} --workers 10.0.0.2:9998 '
               f'--prompt "{PROMPT}" --max-seq-len 2048 --threads 4', timeout=900)
print('\n=== 2-node ppl-dist over CABLE, Mojo on BOTH nodes (expect 7.727886) ===')
print((o + e).strip()[-1800:])
o, e = run(c2, 'grep -m2 kernel ~/worker.log', timeout=30)
print('[S2] worker kernel lines:', o.strip())


def infer(env, label):
    o, e = run(c1, f'cd ~/pyre/rust && {env} time {BIN} inference-dist --model {MODEL_M} '
                   f'--tokenizer {TOK_M} --workers 10.0.0.2:9998 '
                   f'--prompt "{PROMPT}" --steps 24 --max-seq-len 2048 --threads 4',
               timeout=1200)
    print(f'\n=== inference-dist {label} ===')
    print((o + e).strip()[-1500:])


# 5a. Mojo on both (worker default env, root default env)
start_worker(c2)
infer('', 'MOJO (auto-picked on both nodes)')

# 5b. Rust built-in on both (env override forces the documented fallback)
start_worker(c2, env='DLLAMA_KERNEL_LIB=/nonexistent')
infer('DLLAMA_KERNEL_LIB=/nonexistent', 'RUST BUILT-IN (forced fallback)')

# cleanup
run(c2, "pkill -f 'dllama-rs worker' 2>/dev/null; true", timeout=30)
time.sleep(1)
o, e = run(c2, "pgrep -a dllama-rs || echo worker-stopped", timeout=30)
print('\n[S2]', o.strip())
c1.close()
c2.close()
print('\nmojo-on-pi validation done')
