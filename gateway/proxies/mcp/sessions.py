"""Downstream MCP sessions and the upstream session each one owns.

A downstream session is created by the agent's ``initialize`` on ``/mcp/{server}`` and is bound
to exactly what authenticated it: the gateway session, principal, agent and server. A request
naming the id with any other binding is refused as if the id did not exist. Each downstream
session lazily opens its own `MCPUpstream`, so upstream sessions are never shared across
principals (or even across two sessions of the same principal).

Upstream sessions end when the downstream session is deleted, when its gateway session ends,
when the per-gateway-session cap evicts the oldest one, and on gateway shutdown.
"""

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from typing import Final

from gateway.errors import RejectionError
from gateway.identity import TokenClaims
from gateway.policy.loader import PolicySnapshot
from gateway.proxies.mcp.upstream import MCPConnector, MCPUpstream

logger = logging.getLogger(__name__)

MAX_SESSIONS_PER_GATEWAY_SESSION: Final = 8


class MCPSessionNotFoundError(RejectionError):
    """Unknown, ended, or bound to someone else: indistinguishable on purpose (HTTP 404)."""

    status_code = 404

    def __init__(self) -> None:
        super().__init__("mcp_session_not_found", "MCP session not found; initialize a new one")


class UnknownServerError(RejectionError):
    status_code = 404

    def __init__(self) -> None:
        super().__init__("mcp_server_not_found", "no such MCP server")


@dataclass(frozen=True, slots=True)
class DownstreamBinding:
    """What a downstream MCP session belongs to."""

    gateway_session: str
    principal: str
    agent: str
    server: str

    @classmethod
    def of(cls, claims: TokenClaims, server: str) -> "DownstreamBinding":
        return cls(
            gateway_session=claims.session_id,
            principal=claims.sub,
            agent=claims.agent,
            server=server,
        )


@dataclass(eq=False, slots=True)
class DownstreamSession:
    id: str
    binding: DownstreamBinding
    upstream: MCPUpstream | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MCPSessionRegistry:
    """Maps downstream session ids to their binding and upstream session."""

    def __init__(
        self,
        connector: MCPConnector,
        *,
        per_gateway_session: int = MAX_SESSIONS_PER_GATEWAY_SESSION,
    ) -> None:
        self._connector = connector
        self._per_gateway_session = per_gateway_session
        self._sessions: dict[str, DownstreamSession] = {}  # insertion order = age

    def __len__(self) -> int:
        return len(self._sessions)

    async def open(self, claims: TokenClaims, server: str) -> DownstreamSession:
        """A new downstream session; the oldest one of the same gateway session is evicted
        when the cap is reached, so an agent cannot pile up upstream connections."""
        binding = DownstreamBinding.of(claims, server)
        siblings = [
            s
            for s in self._sessions.values()
            if s.binding.gateway_session == binding.gateway_session
        ]
        for evicted in siblings[: max(len(siblings) - self._per_gateway_session + 1, 0)]:
            await self.close(evicted.id)
        session = DownstreamSession(id=secrets.token_urlsafe(24), binding=binding)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str, claims: TokenClaims, server: str) -> DownstreamSession:
        """The session ``session_id`` if it belongs to exactly this caller and server."""
        session = self._sessions.get(session_id)
        if session is None or session.binding != DownstreamBinding.of(claims, server):
            raise MCPSessionNotFoundError
        return session

    async def upstream(self, session: DownstreamSession, snapshot: PolicySnapshot) -> MCPUpstream:
        """The session's upstream, opened on first use. A policy reload that moves the server
        to another URL or trust level replaces it: the old session belongs to the old server."""
        config = snapshot.policy.upstreams.mcp.get(session.binding.server)
        if config is None:
            raise UnknownServerError
        async with session.lock:
            current = session.upstream
            if current is not None and (current.url, current.trust) != (config.url, config.trust):
                await current.aclose()
                current = None
            if current is None:
                current = self._connector.open(
                    session.binding.server, config, session.binding.principal
                )
                session.upstream = current
            return current

    async def close(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        if session.upstream is not None:
            await session.upstream.aclose()
        return True

    async def close_gateway_session(self, gateway_session: str) -> int:
        """End every downstream session of a gateway session that has ended."""
        ids = [
            s.id for s in self._sessions.values() if s.binding.gateway_session == gateway_session
        ]
        for session_id in ids:
            await self.close(session_id)
        return len(ids)

    async def aclose(self) -> None:
        for session_id in list(self._sessions):
            await self.close(session_id)
