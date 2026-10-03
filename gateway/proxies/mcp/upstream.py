"""Upstream MCP client: one `MCPUpstream` per downstream (agent) MCP session.

Hand-rolled on httpx rather than the SDK's ``ClientSession``, because the gateway needs, per
HTTP request, things the SDK client keeps out of reach:

- a fresh ``X-ACL-Principal`` assertion on every request (minted with the internal key, never
  the agent's bearer token), sent to ``trust: internal`` servers only;
- the policy snapshot's ``limits``: a total timeout and a byte cap on every response;
- buffered handling of ``text/event-stream`` answers, stopping at the response it waits for;
- no background tasks: the SDK client owns an anyio task group that would have to live in one
  long-running task per session, outside the request that uses it.

Protocol: streamable HTTP, pinned to ``2025-06-18``. The session id the upstream issues on
``initialize`` is sent on every later request together with ``MCP-Protocol-Version``. Server
requests arriving on a response stream (sampling, elicitation, roots, ...) are answered with
"method not found". Upstream failures become `UpstreamError` with a generic message; the
upstream's own text is logged, never returned.
"""

import asyncio
import codecs
import itertools
import json
import logging
import time
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any, Final, Literal, cast
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from gateway.clock import Clock, utc_now
from gateway.identity import mint_principal_assertion
from gateway.policy.loader import PolicySnapshot
from gateway.policy.schema import McpServer
from gateway.proxies.mcp import wire
from gateway.upstream import Upstream, UpstreamError, UpstreamResult

logger = logging.getLogger(__name__)

CLIENT_INFO: Final = {"name": "ai-control-layer", "version": "0.1.0"}
MAX_TOOL_PAGES: Final = 20  # tools/list pagination bound
CLOSE_TIMEOUT_S: Final = 5.0
_LOG_TEXT_CHARS: Final = 200

type Trust = Literal["internal", "untrusted"]


class UpstreamSessionLostError(Exception):
    """The upstream answered 404 to our session id: it forgot the session (e.g. restarted)."""


class MCPConnector:
    """Owns the HTTP connection pool and the assertion key every upstream session uses."""

    def __init__(
        self,
        internal_key: bytes,
        *,
        clock: Clock = utc_now,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._key = internal_key
        self._clock = clock
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is None:
            # trust_env=False: no proxy or netrc settings from the environment.
            self._client = httpx.AsyncClient(
                transport=self._transport, follow_redirects=False, trust_env=False
            )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = "MCPConnector.start() was not awaited"
            raise RuntimeError(msg)
        return self._client

    def open(self, server: str, config: McpServer, principal: str) -> "MCPUpstream":
        """A new, not yet initialized upstream session for one principal."""
        return MCPUpstream(
            self, server=server, url=config.url, trust=config.trust, principal=principal
        )

    def assertion(self, audience: str, principal: str) -> str:
        return mint_principal_assertion(
            self._key, audience=audience, principal=principal, clock=self._clock
        )


class _SseDecoder:
    """Incremental ``text/event-stream`` decoder yielding each event's joined ``data``."""

    def __init__(self) -> None:
        self._text = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._buffer = ""
        self._data: list[str] = []

    def feed(self, chunk: bytes) -> list[str]:
        self._buffer += self._text.decode(chunk)
        # A trailing CR may be the first half of a CRLF split across chunks: keep it back,
        # or the LF in the next chunk would read as an empty line and end the event early.
        complete, held = (
            (self._buffer[:-1], "\r") if self._buffer.endswith("\r") else (self._buffer, "")
        )
        *lines, rest = complete.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self._buffer = rest + held
        events: list[str] = []
        for line in lines:
            if not line:
                if self._data:
                    events.append("\n".join(self._data))
                    self._data = []
                continue
            field, _, value = line.partition(":")
            if field == "data":
                self._data.append(value.removeprefix(" "))
            # `event`, `id`, `retry` and comments carry nothing the gateway needs.
        return events


class MCPUpstream(Upstream):
    """One upstream MCP session, bound to one server and one principal. Never shared."""

    def __init__(
        self, connector: MCPConnector, *, server: str, url: str, trust: Trust, principal: str
    ) -> None:
        self._connector = connector
        self.server = server
        self.url = url
        self.trust: Trust = trust
        self.principal = principal
        self._audience = urlsplit(url).hostname or server
        self._session_id: str | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._ids = itertools.count(1)
        self._advertised: wire.ToolSchemas | None = None
        self._closed = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ----------------------------------------------------------------- operations

    async def execute(self, payload: object, snapshot: PolicySnapshot) -> UpstreamResult:
        """``tools/call`` with the final payload ``{"name", "arguments"}``; runs once."""
        if not isinstance(payload, dict):
            raise UpstreamError("upstream_invalid_request")
        params = cast("dict[str, Any]", payload)  # the adapter's {"name", "arguments"}
        started = time.perf_counter()
        # Not retried after a lost session: a call that may have run is never sent twice.
        body = await self._call("tools/call", params, snapshot, retry_lost_session=False)
        elapsed = time.perf_counter() - started
        try:
            wire.CallToolResult.model_validate(body)
        except ValidationError:
            raise UpstreamError("upstream_invalid_response") from None
        return UpstreamResult(body=body, elapsed_s=elapsed, untrusted=self.trust == "untrusted")

    async def list_tools(self, snapshot: PolicySnapshot) -> list[wire.ToolDefinition]:
        """Every tool the upstream advertises (all pages). The first listing's input schemas
        are kept as this session's argument schemas; later listings never replace them."""
        tools: list[wire.ToolDefinition] = []
        cursor: str | None = None
        for _ in range(MAX_TOOL_PAGES):
            params = {"cursor": cursor} if cursor is not None else None
            body = await self._call("tools/list", params, snapshot, retry_lost_session=True)
            try:
                page = wire.ListToolsResult.model_validate(body)
            except ValidationError:
                raise UpstreamError("upstream_invalid_response") from None
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                break
        else:
            logger.warning(
                "MCP server %s: tools/list exceeded %d pages", self.server, MAX_TOOL_PAGES
            )
            raise UpstreamError("upstream_invalid_response")
        if self._advertised is None:
            self._advertised = {tool.name: tool.input_schema for tool in tools}
        return tools

    async def advertised_schemas(self, snapshot: PolicySnapshot) -> wire.ToolSchemas:
        """Input schemas from this session's first ``tools/list``, fetching it if needed."""
        if self._advertised is None:
            await self.list_tools(snapshot)
        return self._advertised or {}

    async def aclose(self) -> None:
        """End the upstream session (best effort): ``DELETE`` with its session id."""
        if self._closed:
            return
        self._closed = True
        if self._session_id is None:
            return
        try:
            response = await self._connector.client.delete(
                self.url, headers=self._headers(), timeout=CLOSE_TIMEOUT_S
            )
            if not response.is_success and response.status_code not in {404, 405}:
                logger.info(
                    "MCP server %s: DELETE session -> %d", self.server, response.status_code
                )
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.info(
                "MCP server %s: closing session failed: %s", self.server, type(exc).__name__
            )
        finally:
            self._session_id = None
            self._initialized = False

    # ------------------------------------------------------------------- protocol

    async def _call(
        self,
        method: str,
        params: dict[str, Any] | None,
        snapshot: PolicySnapshot,
        *,
        retry_lost_session: bool,
    ) -> dict[str, Any]:
        limits = snapshot.policy.limits
        try:
            async with asyncio.timeout(limits.upstream_timeout_s):
                await self._ensure_initialized(snapshot)
                try:
                    return await self._request(method, params, snapshot)
                except UpstreamSessionLostError:
                    if not retry_lost_session:
                        raise UpstreamError("upstream_session_lost") from None
                    logger.info("MCP server %s lost the session; re-initializing", self.server)
                    self._session_id, self._initialized = None, False
                    await self._ensure_initialized(snapshot)
                    return await self._request(method, params, snapshot)
        except TimeoutError:
            raise UpstreamError("upstream_timeout") from None
        except UpstreamSessionLostError:
            raise UpstreamError("upstream_session_lost") from None

    async def _ensure_initialized(self, snapshot: PolicySnapshot) -> None:
        if self._closed:
            raise UpstreamError("upstream_session_closed")
        async with self._init_lock:
            if self._initialized:
                return
            self._session_id = None  # a half-finished handshake never leaks into the next one
            params = {
                "protocolVersion": wire.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            }
            body = await self._request("initialize", params, snapshot)
            try:
                init = wire.InitializeResult.model_validate(body)
            except ValidationError:
                raise UpstreamError("upstream_invalid_response") from None
            if init.protocol_version != wire.PROTOCOL_VERSION:
                logger.warning(
                    "MCP server %s negotiated protocol %r, gateway pins %s",
                    self.server,
                    init.protocol_version[:_LOG_TEXT_CHARS],
                    wire.PROTOCOL_VERSION,
                )
                raise UpstreamError("upstream_protocol_mismatch")
            self._initialized = True
            await self._notify("notifications/initialized", snapshot)

    async def _request(
        self, method: str, params: dict[str, Any] | None, snapshot: PolicySnapshot
    ) -> dict[str, Any]:
        request_id = next(self._ids)
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        answer = await self._exchange(message, snapshot, request_id=request_id)
        if isinstance(answer, wire.JsonRpcError):
            logger.warning(
                "MCP server %s answered %s with error %d: %r",
                self.server,
                method,
                answer.error.code,
                answer.error.message[:_LOG_TEXT_CHARS],
            )
            raise UpstreamError("upstream_rpc_error")
        return answer.result

    async def _notify(self, method: str, snapshot: PolicySnapshot) -> None:
        await self._post({"jsonrpc": "2.0", "method": method}, snapshot, request_id=None)

    async def _exchange(
        self, message: dict[str, Any], snapshot: PolicySnapshot, *, request_id: int
    ) -> wire.JsonRpcResult | wire.JsonRpcError:
        answer = await self._post(message, snapshot, request_id=request_id)
        if answer is None:
            logger.warning("MCP server %s accepted a request without answering it", self.server)
            raise UpstreamError("upstream_invalid_response")
        return answer

    async def _post(
        self, message: dict[str, Any], snapshot: PolicySnapshot, *, request_id: int | None
    ) -> wire.JsonRpcResult | wire.JsonRpcError | None:
        """POST one message; return the response to ``request_id`` (None for notifications)."""
        limits = snapshot.policy.limits
        try:
            async with self._connector.client.stream(
                "POST",
                self.url,
                json=message,
                headers=self._headers(),
                timeout=limits.upstream_timeout_s,
            ) as response:
                return await self._answer(response, snapshot, request_id)
        except httpx.TimeoutException:
            raise UpstreamError("upstream_timeout") from None
        except httpx.HTTPError as exc:
            logger.warning("MCP server %s unreachable: %s", self.server, type(exc).__name__)
            raise UpstreamError("upstream_unreachable") from None

    async def _answer(
        self, response: httpx.Response, snapshot: PolicySnapshot, request_id: int | None
    ) -> wire.JsonRpcResult | wire.JsonRpcError | None:
        if response.status_code == HTTPStatus.NOT_FOUND and self._session_id is not None:
            raise UpstreamSessionLostError
        if not response.is_success:
            logger.warning("MCP server %s answered HTTP %d", self.server, response.status_code)
            raise UpstreamError("upstream_error")
        if self._session_id is None and (issued := response.headers.get(wire.SESSION_HEADER)):
            self._session_id = issued
        if request_id is None:
            return None  # a notification: 202 Accepted, no body expected
        media = response.headers.get("content-type", "").partition(";")[0].strip().lower()
        max_bytes = snapshot.policy.limits.max_response_bytes
        if media == "application/json":
            raw = bytearray()
            async for chunk in _capped(response, max_bytes):
                raw += chunk
            return self._matching(_decode(bytes(raw)), request_id)
        if media == "text/event-stream":
            decoder = _SseDecoder()
            async for chunk in _capped(response, max_bytes):
                try:
                    events = decoder.feed(chunk)
                except UnicodeDecodeError:
                    raise UpstreamError("upstream_invalid_response") from None
                for data in events:
                    if answer := await self._on_event(data, snapshot, request_id):
                        return answer
            logger.warning("MCP server %s closed the stream without a response", self.server)
            raise UpstreamError("upstream_invalid_response")
        logger.warning("MCP server %s answered with media type %r", self.server, media[:64])
        raise UpstreamError("upstream_invalid_response")

    async def _on_event(
        self, data: str, snapshot: PolicySnapshot, request_id: int
    ) -> wire.JsonRpcResult | wire.JsonRpcError | None:
        message = _parse(data)
        if isinstance(message, wire.JsonRpcRequest):
            await self._refuse_server_request(message.id, snapshot)
            return None
        if isinstance(message, wire.JsonRpcNotification):
            return None  # progress, logging, list_changed: nothing the gateway relays
        return self._matching(message, request_id)

    def _matching(
        self, message: wire.JsonRpcMessage, request_id: int
    ) -> wire.JsonRpcResult | wire.JsonRpcError:
        if not isinstance(message, wire.JsonRpcResult | wire.JsonRpcError) or (
            message.id != request_id
        ):
            logger.warning("MCP server %s sent an unexpected message", self.server)
            raise UpstreamError("upstream_invalid_response")
        return message

    async def _refuse_server_request(
        self, request_id: wire.RequestId, snapshot: PolicySnapshot
    ) -> None:
        """Server-initiated requests (sampling, elicitation, roots, ...) are not supported."""
        refusal = wire.error(request_id, wire.METHOD_NOT_FOUND, "Method not found")
        await self._post(refusal, snapshot, request_id=None)

    def _headers(self) -> dict[str, str]:
        headers = {
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
        }
        if self._session_id is not None:
            headers[wire.SESSION_HEADER] = self._session_id
        if self._initialized:
            headers[wire.PROTOCOL_HEADER] = wire.PROTOCOL_VERSION
        if self.trust == "internal":
            # Fresh per request: the assertion lives 60 s, a session may live for hours.
            headers[wire.PRINCIPAL_HEADER] = self._connector.assertion(
                self._audience, self.principal
            )
        return headers


async def _capped(response: httpx.Response, max_bytes: int) -> AsyncIterator[bytes]:
    received = 0
    async for chunk in response.aiter_bytes():
        received += len(chunk)
        if received > max_bytes:
            raise UpstreamError("upstream_response_too_large")
        yield chunk


def _decode(raw: bytes) -> wire.JsonRpcMessage:
    try:
        return _parse(raw.decode("utf-8"))
    except UnicodeDecodeError:
        raise UpstreamError("upstream_invalid_response") from None


def _parse(text: str) -> wire.JsonRpcMessage:
    try:
        return wire.parse_message(json.loads(text))
    except (ValueError, ValidationError):
        raise UpstreamError("upstream_invalid_response") from None
