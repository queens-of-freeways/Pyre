"""Ethernet direct-link setup + link measurements for the 2-node Pi cluster.

1. Creates a persistent NetworkManager profile `direct` (ipv4 manual) on
   eth0 of each Pi: S1=10.0.0.1/24, S2=10.0.0.2/24.
2. Verifies with ping in both directions (eth + wifi RTT comparison).
3. Measures raw TCP bandwidth S1 -> S2 over the direct cable and over WiFi.

Credentials come from NODE_PASS in the environment (never hardcoded).
"""
import io
import os
import time
import paramiko

PASS = os.environ['NODE_PASS']
NODES = [
    ('S1', 'n1', '192.168.0.71', '10.0.0.1'),
    ('S2', 'n2', '192.168.0.97', '10.0.0.2'),
]

BW_RECV = '''import socket, sys, time
bind = sys.argv[1]
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((bind, 9999))
s.listen(1)
c, a = s.accept()
t = time.time(); n = 0
while True:
    d = c.recv(1 << 20)
    if not d:
        break
    n += len(d)
dt = time.time() - t
c.close()
print(f'recv {n/1e6:.1f} MB from {a[0]} in {dt:.2f}s = {n/dt/1e6:.1f} MB/s')
'''

BW_SEND = '''import socket, sys, time
host = sys.argv[1]; mb = int(sys.argv[2])
c = socket.create_connection((host, 9999), timeout=10)
buf = b'x' * (1 << 20)
t = time.time(); n = 0
for _ in range(mb):
    c.sendall(buf)
    n += len(buf)
dt = time.time() - t; c.close()
print(f'sent {n/1e6:.1f} MB in {dt:.2f}s = {n/dt/1e6:.1f} MB/s')
'''


def connect(host, user):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=PASS, timeout=15, banner_timeout=15)
    return c


def run(c, cmd, timeout=90):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    o = out.read().decode('utf-8', errors='replace').strip()
    e = err.read().decode('utf-8', errors='replace').strip()
    return o, e


def sudo(c, cmd, timeout=90):
    return run(c, f"echo '{PASS}' | sudo -S {cmd}", timeout)


# ---- 1. configure eth0 on both nodes --------------------------------------
for name, user, host, ip in NODES:
    c = connect(host, user)
    print(f'== {name}: configuring eth0 = {ip}/24 ==')
    o, e = sudo(c, f"nmcli connection add type ethernet con-name direct ifname eth0 ipv4.method manual ipv4.addresses {ip}/24")
    print(' add:', (o or 'ok'), ('| ' + e if e else ''))
    o, e = sudo(c, "nmcli connection up direct")
    print(' up:', (o or 'ok'), ('| ' + e if e else ''))
    o, e = run(c, "ip -brief addr show eth0")
    print(' eth0 now:', o or e)
    c.close()

# ---- 2. ping verification --------------------------------------------------
s1 = NODES[0]
s2 = NODES[1]
c1 = connect(s1[2], s1[1])
c2 = connect(s2[2], s2[1])

print('\n== ping S1 -> S2 over cable (10.0.0.2) ==')
o, e = run(c1, 'ping -c 5 -W 2 10.0.0.2')
print(o or e)
print('\n== ping S2 -> S1 over cable (10.0.0.1) ==')
o, e = run(c2, 'ping -c 5 -W 2 10.0.0.1')
print(o or e)
print('\n== ping S1 -> S2 over WiFi (192.168.0.97, via AP) ==')
o, e = run(c1, 'ping -c 5 -W 2 192.168.0.97')
print(o or e)

# ---- 3. raw TCP bandwidth --------------------------------------------------
sftp2 = c2.open_sftp()
sftp2.putfo(io.StringIO(BW_RECV), '/tmp/bw_recv.py')
sftp1 = c1.open_sftp()
sftp1.putfo(io.StringIO(BW_SEND), '/tmp/bw_send.py')


def bandwidth(label, bind_addr, connect_addr, mb):
    run(c2, 'pkill -f bw_recv.py 2>/dev/null; sleep 1; true')
    time.sleep(1.5)
    run(c2, f"setsid nohup python3 /tmp/bw_recv.py {bind_addr} > /tmp/bw_recv.log 2>&1 < /dev/null &")
    time.sleep(1.5)
    o, e = run(c1, f'python3 /tmp/bw_send.py {connect_addr} {mb}', timeout=180)
    print(f'\n== bandwidth {label}: S1 -> {connect_addr} ({mb} MB) ==')
    print(' sender:', o or e)
    time.sleep(0.5)
    o, e = run(c2, 'cat /tmp/bw_recv.log')
    print(' receiver:', o or e)
    run(c2, 'pkill -f bw_recv.py 2>/dev/null; true')


bandwidth('over direct cable', '10.0.0.2', '10.0.0.2', 200)
bandwidth('over WiFi (via AP)', '192.168.0.97', '192.168.0.97', 100)

c1.close()
c2.close()
print('\nlink setup + measurements done')
