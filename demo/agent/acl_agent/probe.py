"""Can this container reach ``host:port`` directly? (Demo step 6: bypassing the gateway.)

Resolves the name, then opens a TCP connection with a short timeout. Nothing is sent.
"""

import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeResult:
    host: str
    port: int
    resolved: tuple[str, ...]  # empty: the name does not resolve from here
    connected: bool
    error: str | None = None


def probe(host: str, port: int, *, timeout_s: float = 2.0) -> ProbeResult:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return ProbeResult(host, port, (), connected=False, error=f"dns: {exc.strerror}")
    addresses = tuple(dict.fromkeys(str(info[4][0]) for info in infos))
    try:
        with socket.create_connection((addresses[0], port), timeout=timeout_s):
            return ProbeResult(host, port, addresses, connected=True)
    except TimeoutError:
        return ProbeResult(host, port, addresses, connected=False, error="timed out")
    except OSError as exc:
        return ProbeResult(host, port, addresses, connected=False, error=exc.strerror or str(exc))
