"""The MCP suites' harness: the stack under test and a minimal streamable-HTTP MCP client."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import count
from typing import Any

import httpx
from gateway_testkit import Harness, bearer
from upstreams import RoutingTransport, UpstreamLog

PROTOCOL = "2025-06-18"


@dataclass
class MCPStack:
    gateway: Harness
    transport: RoutingTransport
    log: UpstreamLog


@dataclass
class MCPClient:
    """A minimal streamable-HTTP MCP client speaking to the gateway's agent listener."""

    http: httpx.AsyncClient
    token: str | None
    server: str
    session_id: str | None = None
    _ids: Iterator[int] = field(default_factory=lambda: count(1))

    @property
    def path(self) -> str:
        return f"/mcp/{self.server}"

    def headers(self, **extra: str) -> dict[str, str]:
        headers = {
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
        }
        if self.token is not None:
            headers |= bearer(self.token)
        if self.session_id is not None:
            headers |= {"mcp-session-id": self.session_id, "mcp-protocol-version": PROTOCOL}
        return headers | extra

    async def post(self, message: object, **headers: str) -> httpx.Response:
        return await self.http.post(self.path, json=message, headers=self.headers(**headers))

    async def initialize(self) -> httpx.Response:
        response = await self.post(
            {
                "jsonrpc": "2.0",
                "id": next(self._ids),
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL,
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1"},
                },
            }
        )
        if response.status_code == 200:
            self.session_id = response.headers["mcp-session-id"]
            initialized = await self.post({"jsonrpc": "2.0", "method": "notifications/initialized"})
            assert initialized.status_code == 202
        return response

    async def request(self, method: str, params: dict[str, Any] | None = None) -> httpx.Response:
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": next(self._ids), "method": method}
        if params is not None:
            message["params"] = params
        return await self.post(message)

    async def tools(self) -> list[str]:
        response = await self.request("tools/list")
        assert response.status_code == 200, response.text
        return sorted(tool["name"] for tool in response.json()["result"]["tools"])

    async def call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        """The ``tools/call`` result (tool errors included)."""
        response = await self.request("tools/call", {"name": tool, "arguments": arguments})
        assert response.status_code == 200, response.text
        body = response.json()
        assert "result" in body, body
        return body["result"]


async def connect(stack: MCPStack, sub: str, server: str) -> MCPClient:
    """An initialized MCP session on ``server``, in a fresh gateway session of ``sub``."""
    [client] = await connect_all(stack, sub, server)
    return client


async def connect_all(stack: MCPStack, sub: str, *servers: str) -> tuple[MCPClient, ...]:
    """One MCP session per server, all in the same gateway session (one token): taint and
    risk set through one server apply to the others."""
    token = await stack.gateway.token(sub)
    clients = tuple(MCPClient(stack.gateway.agent, token, server) for server in servers)
    for client in clients:
        response = await client.initialize()
        assert response.status_code == 200, response.text
    return clients


def error_text(result: dict[str, Any]) -> str:
    assert result["isError"] is True, result
    return result["content"][0]["text"]
