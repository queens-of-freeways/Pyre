"""Ethernet direct-link recon for the 2-node Pi cluster.

Checks SSH reachability, link state, addressing, network manager, sudo
groups, and leftover dllama processes on both nodes.
Credentials come from NODE_PASS in the environment (never hardcoded).
"""
import os
import paramiko

PASS = os.environ['NODE_PASS']
NODES = [
    ('S1', 'n1', '192.168.0.71'),
    ('S2', 'n2', '192.168.0.97'),
]


def connect(host, user):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=PASS, timeout=15, banner_timeout=15)
    return c


def run(c, cmd, timeout=30):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    o = out.read().decode('utf-8', errors='replace').strip()
    e = err.read().decode('utf-8', errors='replace').strip()
    return o, e


def show(label, o, e):
    print(f'-- {label} --')
    if o:
        print(o)
    if e:
        print(f'[stderr] {e}')


for name, user, host in NODES:
    print(f'===== {name}: {user}@{host} =====')
    try:
        c = connect(host, user)
    except Exception as ex:
        print(f'!! SSH FAILED: {ex}')
        continue
    peer = [p for _, _, p in NODES if p != host][0]
    checks = [
        ('os', 'uname -a'),
        ('id', 'id'),
        ('netlinks', 'for f in /sys/class/net/*; do n=$(basename "$f"); w=no; test -d "$f/phy80211" && w=yes; car=$(cat "$f/carrier" 2>/dev/null); sp=$(cat "$f/speed" 2>/dev/null); op=$(cat "$f/operstate" 2>/dev/null); echo "$n state=$op carrier=$car speed=$sp wireless=$w"; done'),
        ('addrs', 'ip -brief addr'),
        ('routes', 'ip route'),
        ('manager', 'systemctl is-active NetworkManager 2>/dev/null; systemctl is-active systemd-networkd 2>/dev/null; test -f /etc/network/interfaces && echo ifupdown-file-present; command -v nmcli >/dev/null && nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device'),
        ('peer-wifi-ping', f'ping -c 2 -W 2 {peer} >/dev/null 2>&1 && echo peer-ok || echo peer-unreachable'),
        ('leftovers', 'pgrep -a dllama-rs || echo none'),
    ]
    for label, cmd in checks:
        try:
            o, e = run(c, cmd)
        except Exception as ex:
            o, e = '', f'exception: {ex}'
        show(label, o, e)
    c.close()
print('recon done')
