"""Agent-facing MCP endpoint: ``/mcp/{server}`` over streamable HTTP, protocol ``2025-06-18``.

Hand-rolled instead of the SDK's server: the gateway answers a fixed tools-only subset and
must decide, per request and before anything reaches an upstream, who is calling and whether
the session id they present is theirs. Every answer is ``application/json``; the endpoint never
streams, so ``GET`` (the server-to-client stream) is 405.

Order of checks on ``POST``: Origin, bearer token (401 before anything about servers is
revealed), server name (404), media types, body size, one JSON-RPC message (batches were
removed in 2025-06-18), then ``initialize`` or a session id bound to this exact gateway
session, principal, agent and server, plus the ``MCP-Protocol-Version`` header.

Methods: ``initialize``, ``notifications/initialized`` (and any other notification: 202),
``ping``, ``tools/list`` (filtered), ``tools/call`` (the full pipeline). Everything else gets
``-32601``. Policy refusals of a tool call are tool errors (``isError: true``) carrying only a
reason code; a held call also carries its ``approval_id``.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Final

from pydantic import ValidationError

from gateway.adapters.mcp import mcp_adapter
from gateway.clock import Clock, utc_now
from gateway.core.types import Action, Channel, Decision
from gateway.errors import RejectionError
from gateway.identity import TokenClaims
from gateway.pipeline import CallRequest, ChannelRoute, Pipeline, PipelineOutcome, SessionGate
from gateway.policy.loader import PolicySnapshot
from gateway.policy.schema import McpServer
from gateway.proxies.mcp import wire
from gateway.proxies.mcp.pins import PinnedSchemas
from gateway.proxies.mcp.sessions import (
    DownstreamSession,
    MCPSessionRegistry,
    UnknownServerError,
)
from gateway.sessions import SessionError, SessionReason
from gateway.upstream import UpstreamError

logger = logging.getLogger(__name__)

SERVER_VERSION: Final = "0.1.0"
_JSON_MEDIA: Final = frozenset({"application/json", "application/*", "*/*"})


@dataclass(frozen=True, slots=True)
class MCPHttpRequest:
    """What the endpoint reads from an HTTP request; the web layer fills it in."""

    headers: Mapping[str, str]  # lower-case names
    token: str | None
    body: bytes = b""


@dataclass(frozen=True, slots=True)
class MCPReply:
    """What the web layer sends back: a status, an optional JSON body, extra headers."""

    status: int
    body: dict[str, Any] | None = None
    headers: Mapping[str, str] = field(default_factory=dict[str, str])


class _ProtocolError(Exception):
    """A request the endpoint cannot accept, answered with a JSON-RPC error."""

    def __init__(
        self, status: int, code: int, message: str, request_id: wire.RequestId | None = None
    ) -> None:
        super().__init__(message)
        self.reply = MCPReply(status, wire.error(request_id, code, message))


class MCPProxy:
    """Routes downstream MCP messages to the pipeline and the per-session upstreams."""

    def __init__(  # noqa: PLR0913 -- the collaborators of one endpoint, wired once at startup
        self,
        *,
        gate: SessionGate,
        pipeline: Pipeline,
        registry: MCPSessionRegistry,
        pins: PinnedSchemas,
        allowed_origins: frozenset[str] = frozenset(),
        clock: Clock = utc_now,
    ) -> None:
        self._gate = gate
        self._pipeline = pipeline
        self._registry = registry
        self._pins = pins
        self._allowed_origins = allowed_origins
        self._clock = clock

    @property
    def registry(self) -> MCPSessionRegistry:
        return self._registry

    # ------------------------------------------------------------------- HTTP verbs

    async def post(
        self, server: str, request: MCPHttpRequest, snapshot: PolicySnapshot
    ) -> MCPReply:
        claims: TokenClaims | None = None
        try:
            claims, config = self._admit_http(server, request, snapshot)
            _check_media_types(request.headers)
            message = _parse_body(request.body)
            if isinstance(message, wire.JsonRpcRequest) and message.method == "initialize":
                return await self._initialize(message, request, claims, server, snapshot)
            session = self._session(request.headers, claims, server)
            _check_protocol_version(request.headers)
            if isinstance(message, wire.JsonRpcNotification):
                return MCPReply(HTTPStatus.ACCEPTED)  # initialized, cancelled, progress, ...
            return await self._dispatch(message, request, session, config, snapshot)
        except _ProtocolError as exc:
            return exc.reply
        except RejectionError as exc:
            return await self._refused(exc, claims)

    async def delete(
        self, server: str, request: MCPHttpRequest, snapshot: PolicySnapshot
    ) -> MCPReply:
        """End the downstream session and, with it, its upstream session."""
        claims: TokenClaims | None = None
        try:
            claims, _config = self._admit_http(server, request, snapshot)
            session = self._session(request.headers, claims, server)
            _check_protocol_version(request.headers)
            await self._registry.close(session.id)
            return MCPReply(HTTPStatus.NO_CONTENT)
        except _ProtocolError as exc:
            return exc.reply
        except RejectionError as exc:
            return await self._refused(exc, claims)

    @staticmethod
    def get() -> MCPReply:
        """No server-to-client stream: everything is answered on the POST that asked."""
        reply = wire.error(None, wire.INVALID_REQUEST, "this endpoint does not stream")
        return MCPReply(HTTPStatus.METHOD_NOT_ALLOWED, reply, {"allow": "POST, DELETE"})

    async def end_gateway_session(self, gateway_session: str) -> None:
        await self._registry.close_gateway_session(gateway_session)

    async def aclose(self) -> None:
        await self._registry.aclose()

    # ---------------------------------------------------------------------- methods

    async def _initialize(
        self,
        message: wire.JsonRpcRequest,
        request: MCPHttpRequest,
        claims: TokenClaims,
        server: str,
        snapshot: PolicySnapshot,
    ) -> MCPReply:
        try:
            wire.InitializeParams.model_validate(message.params or {})
        except ValidationError:
            return _rpc_error(message, wire.INVALID_PARAMS, "invalid initialize params")
        async with self._gate.admit(request.token, snapshot):  # the gateway session is live
            session = await self._registry.open(claims, server)
        payload = {
            "protocolVersion": wire.PROTOCOL_VERSION,  # the only one; the client decides
            "capabilities": wire.server_capabilities(),
            "serverInfo": {"name": f"ai-control-layer/{server}", "version": SERVER_VERSION},
        }
        return MCPReply(
            HTTPStatus.OK, wire.result(message.id, payload), {wire.SESSION_HEADER: session.id}
        )

    async def _dispatch(
        self,
        message: wire.JsonRpcRequest,
        request: MCPHttpRequest,
        session: DownstreamSession,
        config: McpServer,
        snapshot: PolicySnapshot,
    ) -> MCPReply:
        match message.method:
            case "ping":
                async with self._gate.admit(request.token, snapshot):
                    return MCPReply(HTTPStatus.OK, wire.result(message.id, {}))
            case "tools/list":
                return await self._list_tools(message, request, session, config, snapshot)
            case "tools/call":
                return await self._call_tool(message, request, session, snapshot)
            case _:  # resources/*, prompts/*, logging/*, completion/*, sampling, elicitation...
                return _rpc_error(message, wire.METHOD_NOT_FOUND, "Method not found")

    async def _list_tools(
        self,
        message: wire.JsonRpcRequest,
        request: MCPHttpRequest,
        session: DownstreamSession,
        config: McpServer,
        snapshot: PolicySnapshot,
    ) -> MCPReply:
        """Only tools that are mapped, within the agent's ``max_actions`` and not currently
        removed by session restrictions. Resource checks wait for ``tools/call``."""
        async with self._gate.admit(request.token, snapshot) as (claims, ctx):
            upstream = await self._registry.upstream(session, snapshot)
            try:
                tools = await upstream.list_tools(snapshot)
            except UpstreamError as exc:
                return _rpc_error(
                    message, wire.INTERNAL_ERROR, exc.message, {"reason_code": exc.reason_code}
                )
            agent = snapshot.policy.agents.get(claims.agent)
            removed = self._pipeline.evaluator.removed_actions(snapshot, ctx, self._clock())
            usable = frozenset(agent.max_actions) - removed if agent else frozenset[Action]()
            listed = [
                tool.as_wire()
                for tool in tools
                if (mapped := config.tools.get(tool.name)) is not None and mapped.action in usable
            ]
        return MCPReply(HTTPStatus.OK, wire.result(message.id, {"tools": listed}))

    async def _call_tool(
        self,
        message: wire.JsonRpcRequest,
        request: MCPHttpRequest,
        session: DownstreamSession,
        snapshot: PolicySnapshot,
    ) -> MCPReply:
        try:
            params = wire.CallToolParams.model_validate(message.params or {})
        except ValidationError:
            return _rpc_error(message, wire.INVALID_PARAMS, "invalid tools/call params")
        server = session.binding.server
        config = snapshot.policy.upstreams.mcp[server]
        try:
            upstream = await self._registry.upstream(session, snapshot)
            schemas = self._pins.lookup(server)
            if schemas is None:
                schemas = await upstream.advertised_schemas(snapshot)
        except RejectionError as exc:  # unreadable pin file, upstream down: no decision made
            logger.warning("MCP %s: tool schemas unavailable (%s)", server, exc.reason_code)
            return MCPReply(
                HTTPStatus.OK, wire.result(message.id, wire.tool_error(exc.reason_code))
            )
        call = CallRequest(
            channel=Channel.MCP,
            token=request.token,
            body=json.dumps(params.payload()).encode(),
            server=server,
        )
        route = ChannelRoute(adapter=mcp_adapter(server, config, schemas), upstream=upstream)
        outcome = await self._pipeline.handle(call, snapshot, route=route)
        if outcome.status_code == HTTPStatus.UNAUTHORIZED:  # token or session gone mid-call
            await self._end_if_session_over(outcome.reason_code, outcome.session_id)
            return _refusal_reply(outcome.status_code, outcome.reason_code, outcome.message)
        return MCPReply(HTTPStatus.OK, wire.result(message.id, _tool_result(outcome)))

    # -------------------------------------------------------------------- admission

    def _admit_http(
        self, server: str, request: MCPHttpRequest, snapshot: PolicySnapshot
    ) -> tuple[TokenClaims, McpServer]:
        origin = request.headers.get("origin")
        if origin is not None and origin not in self._allowed_origins:
            raise RejectionError("origin_not_allowed", "requests from this origin are refused")
        claims = self._gate.authenticate(request.token, snapshot)  # 401 before any 404
        config = snapshot.policy.upstreams.mcp.get(server)
        if config is None:
            raise UnknownServerError
        return claims, config

    def _session(
        self, headers: Mapping[str, str], claims: TokenClaims, server: str
    ) -> DownstreamSession:
        session_id = headers.get(wire.SESSION_HEADER)
        if not session_id:
            raise _ProtocolError(400, wire.INVALID_REQUEST, "Mcp-Session-Id header is required")
        return self._registry.get(session_id, claims, server)

    async def _refused(self, exc: RejectionError, claims: TokenClaims | None) -> MCPReply:
        if isinstance(exc, SessionError) and claims is not None:
            await self._end_if_session_over(exc.reason_code, claims.session_id)
        return _refusal_reply(exc.status_code, exc.reason_code, exc.message)

    async def _end_if_session_over(self, reason_code: str, gateway_session: str | None) -> None:
        if reason_code == SessionReason.ENDED and gateway_session is not None:
            await self._registry.close_gateway_session(gateway_session)


# ------------------------------------------------------------------------- helpers


def _parse_body(body: bytes) -> wire.JsonRpcRequest | wire.JsonRpcNotification:
    """One request or notification. Batches (removed in 2025-06-18) and responses (the
    endpoint never sends requests to answer) are refused."""
    try:
        document: object = json.loads(body)
    except ValueError:
        raise _ProtocolError(400, wire.PARSE_ERROR, "Parse error") from None
    if isinstance(document, list):
        raise _ProtocolError(400, wire.INVALID_REQUEST, "JSON-RPC batches are not supported")
    try:
        message = wire.parse_message(document)
    except ValidationError:
        raise _ProtocolError(400, wire.INVALID_REQUEST, "Invalid Request") from None
    if isinstance(message, wire.JsonRpcResult | wire.JsonRpcError):
        raise _ProtocolError(400, wire.INVALID_REQUEST, "unexpected JSON-RPC response")
    return message


def _media_type(value: str) -> str:
    return value.partition(";")[0].strip().lower()


def _check_media_types(headers: Mapping[str, str]) -> None:
    if _media_type(headers.get("content-type", "")) != "application/json":
        raise _ProtocolError(415, wire.INVALID_REQUEST, "Content-Type must be application/json")
    accept = headers.get("accept")
    if accept is not None and not {_media_type(part) for part in accept.split(",")} & _JSON_MEDIA:
        raise _ProtocolError(406, wire.INVALID_REQUEST, "the response is application/json")


def _check_protocol_version(headers: Mapping[str, str]) -> None:
    if headers.get(wire.PROTOCOL_HEADER) != wire.PROTOCOL_VERSION:
        message = f"MCP-Protocol-Version must be {wire.PROTOCOL_VERSION}"
        raise _ProtocolError(400, wire.INVALID_REQUEST, message)


def _rpc_error(message: wire.JsonRpcRequest, code: int, text: str, data: object = None) -> MCPReply:
    return MCPReply(HTTPStatus.OK, wire.error(message.id, code, text, data))


def _refusal_reply(status: int, reason_code: str, message: str) -> MCPReply:
    body = wire.error(None, wire.GATEWAY_REFUSAL, message, {"reason_code": reason_code})
    headers = {"www-authenticate": "Bearer"} if status == HTTPStatus.UNAUTHORIZED else {}
    return MCPReply(status, body, headers)


def _tool_result(outcome: PipelineOutcome) -> dict[str, Any]:
    if outcome.released:
        return outcome.result
    if outcome.decision is Decision.REQUIRE_APPROVAL:
        return wire.tool_error(outcome.reason_code, approval_id=outcome.approval_id)
    return wire.tool_error(outcome.reason_code)
