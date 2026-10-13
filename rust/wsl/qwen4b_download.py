"""Fix the WiFi transfer issue + get the 4B GGUF onto S1.

Problem: sustained ~6 MB/s SFTP from the orchestration box to S1 gets its
SSH session reset after 20-350 MB (8/8 attempts). Short transfers are fine.

Layer 1 (link mitigation): check + disable wlan0 power_save on BOTH Pis
(the classic Pi cause of exactly this flakiness), persist it via
NetworkManager (802-11-wireless.powersave 2), report signal.

Layer 2 (sidestep, guaranteed): S1 downloads the file from HuggingFace
itself with curl -C - --retry 100 --retry-all-errors --speed-time 60 —
every reset resumes, progress is monotonic, the download converges no
matter how flaky the link is. The existing 1.79 GB SFTP partial is only
trusted after an md5 prefix check; otherwise it is deleted.

Credentials come from NODE_PASS in the environment (never hardcoded).
"""
import hashlib
import os
import time
import paramiko

PASS = os.environ['NODE_PASS']
LOCAL_GGUF = '/mnt/c/Users/kodur/distributed-llama/models/Qwen3-4B-Q4_K_M.gguf'
URL = 'https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf'
DEST = '/home/n1/models/Qwen3-4B-Q4_K_M.gguf'
EXPECT = 2497280256
NODES = [('S1', 'n1', '192.168.0.71'), ('S2', 'n2', '192.168.0.97')]


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


# ---- Layer 1: WiFi power_save off + signal report, both nodes ----------
print('== Layer 1: WiFi power_save + signal ==')
for name, user, host in NODES:
    c = connect(host, user)
    o, _ = run(c, 'iw dev wlan0 get power_save; iw dev wlan0 link | grep -E "signal|tx bitrate" | head -2', timeout=30)
    print(f'[{name}] before: {o.strip()}')
    run(c, f"echo '{PASS}' | sudo -S iw dev wlan0 set power_save off", timeout=30)
    o, e = run(c, f"echo '{PASS}' | sudo -S nmcli con modify "
                  '"$(nmcli -t -f NAME,DEVICE connection show --active | grep wlan0 | cut -d: -f1)" '
                  '802-11-wireless.powersave 2', timeout=30)
    print(f'[{name}] nmcli persist: {(o + e).strip()[:80] or "ok"}')
    o, _ = run(c, 'iw dev wlan0 get power_save', timeout=30)
    print(f'[{name}] after: {o.strip()}')
    c.close()

# ---- Layer 2: verified-resume download on S1 ----------------------------
c1 = connect('192.168.0.71', 'n1')
o, _ = run(c1, f'test -f {DEST} && stat -c %s {DEST} || echo 0', timeout=30)
try:
    remote_size = int(o.strip().splitlines()[-1])
except ValueError:
    remote_size = 0
print(f'\n== Layer 2: download on S1 (existing partial: {remote_size/1e9:.2f} GB) ==')

resume = False
if remote_size >= EXPECT:
    print('file already complete')
    resume = True
elif remote_size > 0:
    print('verifying the SFTP partial is a clean prefix (md5)...')
    h = hashlib.md5()
    with open(LOCAL_GGUF, 'rb') as f:
        remaining = remote_size
        while remaining > 0:
            buf = f.read(min(8 << 20, remaining))
            if not buf:
                break
            h.update(buf)
            remaining -= len(buf)
    local_md5 = h.hexdigest()
    o, _ = run(c1, f'md5sum {DEST}', timeout=600)
    remote_md5 = o.strip().split()[0] if o.strip() else ''
    print(f'  local prefix md5: {local_md5}')
    print(f'  remote file  md5: {remote_md5}')
    resume = (local_md5 == remote_md5)
    if resume:
        print('  clean prefix -> curl will resume from it')
    else:
        print('  NOT a clean prefix -> deleting, fresh download')
        run(c1, f'rm -f {DEST}', timeout=30)

if remote_size < EXPECT:
    flag = '-C -' if resume else ''
    o, e = run(c1, f'cd /home/n1/models && timeout 1400 curl -fL {flag} --retry 100 --retry-delay 2 '
                   f'--retry-all-errors --connect-timeout 15 --speed-limit 102400 --speed-time 60 '
                   f'-sS -o {DEST.rsplit("/", 1)[-1]} "{URL}"; echo RC=$?', timeout=1500)
    print((o + e).strip()[-400:])
    o, _ = run(c1, f'stat -c %s {DEST} 2>/dev/null || echo missing', timeout=30)
    got = o.strip()
    print(f'size on S1: {got} (expected {EXPECT})')
    print('SIZE OK' if got == str(EXPECT) else 'SIZE MISMATCH — rerun this script to resume further')

c1.close()
print('\ndownload phase done')
