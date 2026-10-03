"""``egress``: where a call to an MCP server with ``adapter: http`` may connect.

Pre stage, MCP channel, only for ``adapter: http`` servers (any other server: not applicable).
The control reads the ``url`` argument of the interaction's payload. That is the canonical URL
the http adapter derived the resource from and that is forwarded upstream
(`gateway.adapters.mcp.canonical_url`), so the host checked here is the host authorized and
the host fetched. Checks, in order:

1. scheme ``http`` or ``https`` (``egress_scheme_not_allowed``);
2. port, explicit or the scheme's default, in ``allowed_ports`` (``egress_port_not_allowed``);
3. host matching one of ``allow_hosts`` when configured (``egress_host_not_allowed``);
4. an IP literal, or every address the host resolves to, is public: loopback, RFC 1918,
   CGNAT, link-local (169.254.169.254 included), ULA, multicast, unspecified and reserved
   addresses are refused, and so are IPv6 forms embedding such an IPv4 address (mapped,
   6to4, Teredo) (``egress_private_address``). One non-public answer is enough to refuse;
5. a host that does not resolve within ``resolve_timeout_s`` fails closed
   (``egress_unresolvable``).

The gateway's resolution is defense in depth. The fetch server resolves again, re-validates
and pins its connection to the validated address, so DNS rebinding between the two
resolutions is caught there. The gateway cannot pin the upstream server's connection itself.
Taint is not this control's concern: ``risk_rules`` remove or hold egress per session mode.
"""

import asyncio
import socket
from collections.abc import Mapping, Sequence
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Any, ClassVar, Final, Protocol, cast, override

import httpx

from gateway.controls.scope import current_scope
from gateway.core.envelope import Interaction, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import ControlKind, ControlMode, Decision, Stage
from gateway.policy.permissions import glob_match
from gateway.policy.schema import EgressConfig

EGRESS_ALLOWED: Final = "egress_allowed"
EGRESS_NOT_APPLICABLE: Final = "egress_not_applicable"
EGRESS_PRIVATE_ADDRESS: Final = "egress_private_address"
EGRESS_PORT_NOT_ALLOWED: Final = "egress_port_not_allowed"
EGRESS_HOST_NOT_ALLOWED: Final = "egress_host_not_allowed"
EGRESS_SCHEME_NOT_ALLOWED: Final = "egress_scheme_not_allowed"
EGRESS_UNRESOLVABLE: Final = "egress_unresolvable"
EGRESS_UNVERIFIABLE: Final = "egress_unverifiable"  # no call scope, or no URL to check

_HOLDABLE: Final = frozenset(
    {EGRESS_PRIVATE_ADDRESS, EGRESS_PORT_NOT_ALLOWED, EGRESS_HOST_NOT_ALLOWED}
)

URL_ARGUMENT: Final = "url"
HTTP_ADAPTER: Final = "http"
DEFAULT_PORTS: Final[Mapping[str, int]] = {"http": 80, "https": 443}

type IPAddress = IPv4Address | IPv6Address


class HostResolver(Protocol):
    """Resolves ``host`` (A and AAAA) for ``port``; an empty answer means unresolvable."""

    async def __call__(self, host: str, port: int) -> Sequence[IPAddress]: ...


async def system_resolver(host: str, port: int) -> Sequence[IPAddress]:
    """The system resolver, off the event loop (``loop.getaddrinfo``)."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        return []
    # A scoped IPv6 answer carries "%<zone>"; the zone never makes an address public.
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
    """True only for globally routable unicast addresses (and IPv6 forms wrapping one)."""
    if isinstance(address, IPv6Address):
        embedded = _embedded_ipv4(address)
        if embedded is not None and not is_public_address(embedded):
            return False
    return address.is_global and not (
        address.is_multicast or address.is_reserved or address.is_unspecified
    )


def _url_of(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    arguments = cast("Mapping[str, Any]", payload).get("arguments")
    if not isinstance(arguments, Mapping):
        return None
    url = cast("Mapping[str, Any]", arguments).get(URL_ARGUMENT)
    return url if isinstance(url, str) else None


class _RefusedError(Exception):
    def __init__(self, reason_code: str, reason: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.reason = reason


def _destination(url: str, settings: EgressConfig) -> tuple[str, int]:
    """The host and port ``url`` connects to; `_RefusedError` for a scheme, port or host the
    settings do not allow (checks 1 to 3 of the module docstring)."""
    try:
        parsed = httpx.URL(url)
        host = parsed.raw_host.decode("ascii")
    except (httpx.InvalidURL, UnicodeError):
        raise _RefusedError(EGRESS_UNVERIFIABLE, "the url does not parse") from None
    default_port = DEFAULT_PORTS.get(parsed.scheme)
    if default_port is None or not host:
        raise _RefusedError(EGRESS_SCHEME_NOT_ALLOWED, "only http(s) URLs")
    port = parsed.port or default_port
    if port not in settings.allowed_ports:
        raise _RefusedError(EGRESS_PORT_NOT_ALLOWED, f"port {port} is not allowed")
    if settings.allow_hosts is not None and not any(
        glob_match(pattern, host) for pattern in settings.allow_hosts
    ):
        raise _RefusedError(EGRESS_HOST_NOT_ALLOWED, "the host is not allowlisted")
    return host, port


class EgressControl(Control):
    id: ClassVar[str] = "egress"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE})
    kind: ClassVar[ControlKind] = ControlKind.DETERMINISTIC

    def __init__(self, resolver: HostResolver = system_resolver) -> None:
        self._resolver = resolver

    @override
    async def evaluate(self, interaction: Interaction, stage: Stage, cfg: ControlConfig) -> Verdict:
        scope = current_scope()
        if scope is None or interaction.server is None:
            return self._refuse(EGRESS_UNVERIFIABLE, "no call scope or server to check", cfg)
        server = scope.snapshot.policy.upstreams.mcp.get(interaction.server)
        if server is None:
            return self._refuse(EGRESS_UNVERIFIABLE, "the server is not in the policy", cfg)
        if server.adapter != HTTP_ADAPTER:
            return Verdict(
                decision=Decision.ALLOW, control_id=self.id, reason_code=EGRESS_NOT_APPLICABLE
            )
        settings = cfg if isinstance(cfg, EgressConfig) else EgressConfig()
        url = _url_of(interaction.payload)
        if url is None:
            return self._refuse(EGRESS_UNVERIFIABLE, "no url argument to check", cfg)
        return await self._check(url, settings, cfg)

    async def _check(self, url: str, settings: EgressConfig, cfg: ControlConfig) -> Verdict:
        try:
            host, port = _destination(url, settings)
        except _RefusedError as refused:
            return self._refuse(refused.reason_code, refused.reason, cfg)
        try:
            addresses: Sequence[IPAddress] = [ip_address(host.strip("[]"))]
        except ValueError:
            addresses = await self._resolve(host, port, settings.resolve_timeout_s)
        if not addresses:
            return self._refuse(EGRESS_UNRESOLVABLE, "the host did not resolve in time", cfg)
        if not all(is_public_address(address) for address in addresses):
            return self._refuse(EGRESS_PRIVATE_ADDRESS, "a non-public destination", cfg)
        return Verdict(decision=Decision.ALLOW, control_id=self.id, reason_code=EGRESS_ALLOWED)

    async def _resolve(self, host: str, port: int, timeout_s: float) -> Sequence[IPAddress]:
        try:
            async with asyncio.timeout(timeout_s):
                return await self._resolver(host, port)
        except TimeoutError:
            return []

    def _refuse(self, reason_code: str, reason: str, cfg: ControlConfig) -> Verdict:
        """``require_approval`` mode holds a destination an operator can judge; a call the
        control cannot check (no scope, no URL, unresolvable) is blocked in every mode."""
        holdable = reason_code in _HOLDABLE and cfg.mode is ControlMode.REQUIRE_APPROVAL
        decision = Decision.REQUIRE_APPROVAL if holdable else Decision.BLOCK
        return Verdict(
            decision=decision,
            control_id=self.id,
            reason_code=reason_code,
            reason=reason,
            risk_delta=cfg.risk_delta or 0.0,
        )
