"""`mcp-fetch`: an untrusted web fetcher. Lives alone on `mcp_untrusted`.

SSRF defence at connect time, independent of the gateway's egress control:

* only http(s) on ports 80/443;
* the server resolves the host itself and rejects the request if ANY answer is not a public
  address (loopback, RFC 1918, CGNAT, link-local incl. 169.254.169.254, ULA, multicast,
  unspecified, reserved, and IPv4-mapped / 6to4 forms of those);
* the connection is pinned to the validated IP: the request URL carries the IP, while the
  Host header and the TLS SNI/certificate check keep the original host name. A second DNS
  answer (rebinding) is never consulted;
* proxies from the environment are ignored, so nothing re-resolves the name elsewhere.

Redirects are not followed: the gateway's egress control validates every destination, so a
redirect is reported back and the agent must fetch the new URL explicitly (through the gateway).
"""

from __future__ import annotations

import socket
from collections.abc import Sequence
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Protocol

import anyio
import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

_DEFAULT_PORTS = {"http": 80, "https": 443}
_ALLOWED_PORTS = frozenset(_DEFAULT_PORTS.values())
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


type IPAddress = IPv4Address | IPv6Address


class UnsupportedUrlError(ToolError):
    def __init__(self) -> None:
        super().__init__("only absolute http(s) URLs on port 80 or 443 are supported")


class DisallowedDestinationError(ToolError):
    def __init__(self, host: str) -> None:
        super().__init__(f"destination not allowed: {host!r} is not a public address")


class UnresolvableHostError(ToolError):
    def __init__(self, host: str) -> None:
        super().__init__(f"cannot resolve host {host!r}")


class HostResolver(Protocol):
    async def __call__(self, host: str, port: int) -> Sequence[IPAddress]: ...


async def system_resolver(host: str, port: int) -> Sequence[IPAddress]:
    """Resolve `host` with the system resolver (A and AAAA)."""
    try:
        infos = await anyio.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return []
    return [ip_address(str(info[4][0]).split("%", 1)[0]) for info in infos]


def _embedded_ipv4(address: IPv6Address) -> IPv4Address | None:
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address.sixtofour is not None:
        return address.sixtofour
    if address.teredo is not None:
        return address.teredo[1]
    return None


def is_public_address(address: IPAddress) -> bool:
    """True only for globally routable unicast addresses (and their IPv6 wrappings)."""
    if isinstance(address, IPv6Address):
        embedded = _embedded_ipv4(address)
        if embedded is not None and not is_public_address(embedded):
            return False
    return address.is_global and not (
        address.is_multicast or address.is_reserved or address.is_unspecified
    )


class RedirectNotFollowedError(ToolError):
    def __init__(self, status: int, location: str) -> None:
        super().__init__(f"HTTP {status} redirect to {location!r}; not followed")


class UpstreamHTTPError(ToolError):
    def __init__(self, status: int, snippet: str) -> None:
        super().__init__(f"HTTP {status}: {snippet}")


class FetchSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_bytes: int = Field(default=256 * 1024, gt=0)
    timeout_s: float = Field(default=10.0, gt=0)


class Fetcher:
    def __init__(
        self,
        settings: FetchSettings,
        *,
        resolver: HostResolver = system_resolver,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._resolver = resolver
        self._transport = transport

    async def fetch(self, url: str) -> str:
        """Fetch a public web page with HTTP GET and return its body as text (size-capped)."""
        target, headers, extensions = await self._pin(url)
        async with (
            httpx.AsyncClient(
                transport=self._transport or httpx.AsyncHTTPTransport(),
                timeout=httpx.Timeout(self._settings.timeout_s),
                follow_redirects=False,
                trust_env=False,
            ) as client,
            client.stream("GET", target, headers=headers, extensions=extensions) as response,
        ):
            if response.status_code in _REDIRECT_STATUSES:
                location = response.headers.get("location", "")
                raise RedirectNotFollowedError(response.status_code, location)
            body = await self._read_capped(response)
        text = body.decode(response.encoding or "utf-8", errors="replace")
        if response.is_error:
            raise UpstreamHTTPError(response.status_code, text[:500])
        return text

    async def _pin(self, url: str) -> tuple[httpx.URL, dict[str, str], dict[str, str]]:
        """Validate the destination and return the IP-pinned URL, Host header and extensions."""
        try:
            parsed = httpx.URL(url)
        except httpx.InvalidURL as exc:
            raise UnsupportedUrlError from exc
        default_port = _DEFAULT_PORTS.get(parsed.scheme)
        port = parsed.port or default_port
        if default_port is None or not parsed.host or port not in _ALLOWED_PORTS:
            raise UnsupportedUrlError
        host = parsed.raw_host.decode("ascii")
        try:
            addresses: Sequence[IPAddress] = [ip_address(host)]
            literal = True
        except ValueError:
            addresses = await self._resolver(host, port)
            literal = False
        if not addresses:
            raise UnresolvableHostError(host)
        if not all(is_public_address(address) for address in addresses):
            raise DisallowedDestinationError(host)
        pinned = parsed.copy_with(host=str(addresses[0]))
        headers = {"Host": parsed.netloc.decode("ascii")}
        extensions = {} if literal else {"sni_hostname": host}
        return pinned, headers, extensions

    async def _read_capped(self, response: httpx.Response) -> bytes:
        limit = self._settings.max_bytes
        chunks: list[bytes] = []
        received = 0
        async for chunk in response.aiter_bytes():
            chunks.append(chunk[: limit - received])
            received += len(chunk)
            if received >= limit:
                break
        return b"".join(chunks)


def build_fetch_server(settings: FetchSettings | None = None) -> MCPServer:
    fetcher = Fetcher(settings or FetchSettings())
    server = MCPServer(name="mcp-fetch")
    server.tool(
        name="fetch",
        annotations=ToolAnnotations(
            title="Fetch URL",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
    )(fetcher.fetch)
    return server
