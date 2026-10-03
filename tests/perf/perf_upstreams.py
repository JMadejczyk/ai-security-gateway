"""Instantly answering upstreams for the perf layer, and the time the gateway waits on them.

`InstantUpstreams` is one `httpx` transport for every upstream the gateway calls: the LLM
router (``ollama``) and the three MCP servers of ``policy.yaml``. It speaks just enough of each
protocol (OpenAI chat completions; MCP streamable HTTP with JSON answers) to drive the real
gateway clients, builds every answer from constants, and adds the wall time spent inside each
request to `busy_s`. The perf runner reads `busy_s` around a call: whatever the client saw
beyond it is gateway time, whether or not the pipeline accounted for it.

The MCP SDK servers of ``tests/mcp/upstreams.py`` are not reused: they run in this process and
on this event loop, so their CPU time would be billed to the gateway wherever the pipeline does
not subtract it (``tools/list``, ``sql_guard``'s ``explain`` round trip).
"""

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any, Final

import httpx
from gateway_testkit import INTERNAL_KEY, T0, completion

from gateway.policy.loader import PolicySnapshot
from gateway.proxies.mcp import wire
from gateway.proxies.mcp.admin import capture_pin
from gateway.proxies.mcp.pins import PinFile
from gateway.proxies.mcp.upstream import MCPConnector

LLM_HOST: Final = "ollama"
PLAN_COST: Final = 12.5  # far under policy.yaml's sql_guard max_cost
PIN_OPERATOR: Final = "root@demo"


def _tool(tool: str, description: str, **properties: str) -> dict[str, Any]:
    return {
        "name": tool,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {key: {"type": kind} for key, kind in properties.items()},
            "required": list(properties),
        },
    }


# Hosts as in policy.yaml (`upstreams.mcp.<server>.url`), with the tools each one advertises.
MCP_TOOLS: Final[dict[str, list[dict[str, Any]]]] = {
    "mcp-postgres": [
        _tool("query", "Run one read-only SQL SELECT against the sales database.", sql="string"),
        _tool("explain", "Planner cost of one SELECT (gateway only).", sql="string"),
    ],
    "mcp-fetch": [_tool("fetch", "Fetch a public web page and return its body.", url="string")],
    "mcp-files": [
        _tool(
            "write_report",
            "Create a new text report with the given name.",
            name="string",
            content="string",
        )
    ],
}


def _text_result(text: str, structured: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": False}
    if structured is not None:
        result["structuredContent"] = structured
    return result


def _default_page() -> str:
    return "<html><body><p>Quarterly outlook: stable.</p></body></html>"


@dataclass
class InstantUpstreams(httpx.AsyncBaseTransport):
    """Every gateway upstream in one transport; `busy_s` is the time spent inside it.

    ``fetch_page`` builds the body ``fetch`` returns for a URL (the perf runner varies it per
    call, so the classifier cache never answers for it).
    """

    fetch_page: Callable[[str], str] = field(default=lambda _url: _default_page())
    busy_s: float = 0.0
    requests: int = 0
    _sessions: Iterator[int] = field(default_factory=lambda: count(1))

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        started = time.perf_counter()
        try:
            return self._answer(request)
        finally:
            self.busy_s += time.perf_counter() - started
            self.requests += 1

    def _answer(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == LLM_HOST:
            return self._llm(request)
        if host in MCP_TOOLS:
            return self._mcp(host, request)
        raise httpx.ConnectError("no route to host", request=request)

    @staticmethod
    def _llm(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":  # /models
            return httpx.Response(200, json={"object": "list", "data": []}, request=request)
        model = json.loads(request.content)["model"]  # model_allowlist checks the answer's model
        return httpx.Response(200, json=completion(model=model), request=request)

    def _mcp(self, host: str, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(200, request=request)
        message = json.loads(request.content)
        if "id" not in message:  # notifications/initialized
            return httpx.Response(202, request=request)
        headers: dict[str, str] = {}
        match message["method"]:
            case "initialize":
                result: dict[str, Any] = {
                    "protocolVersion": wire.PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": host, "version": "perf"},
                }
                headers[wire.SESSION_HEADER] = f"{host}-{next(self._sessions)}"
            case "tools/list":
                result = {"tools": MCP_TOOLS[host]}
            case "tools/call":
                result = self._call(message["params"])
            case _:
                body = wire.error(message["id"], wire.METHOD_NOT_FOUND, "Method not found")
                return httpx.Response(200, json=body, request=request)
        body = {"jsonrpc": "2.0", "id": message["id"], "result": result}
        return httpx.Response(200, json=body, headers=headers, request=request)

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        arguments: dict[str, Any] = params.get("arguments", {})
        match params["name"]:
            case "query":
                rows = [{"count": 40}]
                return _text_result(json.dumps(rows), {"result": rows})
            case "explain":
                return _text_result(str(PLAN_COST), {"total_cost": PLAN_COST})
            case "fetch":
                return _text_result(self.fetch_page(str(arguments.get("url", ""))))
            case "write_report":
                size = len(str(arguments.get("content", "")).encode())
                return _text_result(f"wrote {size} bytes to reports/{arguments.get('name')}")
            case _:
                return {"content": [{"type": "text", "text": "unknown tool"}], "isError": True}


async def capture_pins(transport: InstantUpstreams, snapshot: PolicySnapshot) -> dict[str, PinFile]:
    """A ``tool_pinning`` baseline per policy server, captured the way ``acl pin`` does."""
    connector = MCPConnector(INTERNAL_KEY.encode(), transport=transport)
    await connector.start()
    try:
        return {
            server: await capture_pin(connector, server, snapshot, principal=PIN_OPERATOR, now=T0)
            for server in snapshot.policy.upstreams.mcp
        }
    finally:
        await connector.aclose()


def save_pins(pins_dir: Path, pins: dict[str, PinFile]) -> None:
    pins_dir.mkdir(parents=True, exist_ok=True)
    for server, pin in pins.items():
        (pins_dir / f"{server}.json").write_text(pin.to_json())
