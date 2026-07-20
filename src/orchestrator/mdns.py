from __future__ import annotations

import socket
import time
from typing import Callable, List, Optional, Tuple

from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf


SERVICE_TYPE = "_dl-worker._tcp.local."


def _resolve_ip(host: str) -> str:
    if host in ("0.0.0.0", "localhost", "127.0.0.1"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    return host


class WorkerRegistrar:
    def __init__(self, host: str, port: int):
        self._zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
        ip = _resolve_ip(host)
        self._info = ServiceInfo(
            SERVICE_TYPE,
            f"worker-{port}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=port,
        )

    def start(self):
        self._zeroconf.register_service(self._info)

    def stop(self):
        try:
            self._zeroconf.unregister_service(self._info)
        except Exception:
            pass
        self._zeroconf.close()


class _WorkerListener:
    def __init__(self, callback: Callable[[str, int], None]):
        self.callback = callback

    def add_service(self, zeroconf, service_type, name):
        info = zeroconf.get_service_info(service_type, name)
        if info:
            host = socket.inet_ntoa(info.addresses[0])
            port = info.port
            self.callback(host, port)

    def update_service(self, zeroconf, service_type, name):
        self.add_service(zeroconf, service_type, name)

    def remove_service(self, zeroconf, service_type, name):
        pass


def discover_workers(
    timeout: float = 3.0,
    expect: Optional[int] = None,
    quiet: bool = False,
) -> List[Tuple[str, int]]:
    discovered: List[Tuple[str, int]] = []

    def on_service(host, port):
        addr = (host, port)
        if addr not in discovered:
            discovered.append(addr)

    zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
    listener = _WorkerListener(on_service)
    browser = ServiceBrowser(zeroconf, SERVICE_TYPE, listener)

    start = time.monotonic()
    try:
        while time.monotonic() - start < timeout:
            if expect is not None and len(discovered) >= expect:
                break
            time.sleep(0.1)
    finally:
        browser.cancel()
        zeroconf.close()

    if not quiet:
        if discovered:
            print(f"Discovered {len(discovered)} worker(s): {discovered}")
        else:
            print("No workers discovered via mDNS.")
    return discovered
