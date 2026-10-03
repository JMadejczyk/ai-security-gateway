"""`MCPUpstream` edge cases a well-behaved SDK server never produces, on a scripted transport.

The happy path runs against real ``MCPServer`` instances in test_mcp_proxy.py; here a
handler scripts the upstream's answers to reach what those servers never send.
"""

import json
from collections.abc import Callable

import httpx
import pytest
from gateway_testkit import INTERNAL_KEY

from gateway.proxies.mcp.upstream import MCPConnector, MCPUpstream, _SseDecoder
from gateway.upstream import UpstreamError

URL = "http://mcp-postgres:8000/mcp"
type Handler = Callable[[dict, httpx.Request], httpx.Response]


def sse(*messages: dict) -> httpx.Response:
    body = "".join(f"event: message\r\ndata: {json.dumps(m)}\r\n\r\n" for m in messages)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def initialize_ok(message: dict, version: str = "2025-06-18") -> httpx.Response:
    result = {
        "protocolVersion": version,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "scripted", "version": "1"},
    }
    response = sse({"jsonrpc": "2.0", "id": message["id"], "result": result})
    response.headers["mcp-session-id"] = "up-1"
    return response


class Scripted:
    """Answers initialize and the initialized notification; ``on_call`` answers the rest."""

    def __init__(self, on_call: Handler, version: str = "2025-06-18") -> None:
        self.on_call = on_call
        self.version = version
        self.received: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        message = json.loads(request.content) if request.content else {}
        self.received.append(message)
        if message.get("method") == "initialize":
            return initialize_ok(message, self.version)
        if "id" not in message or "method" not in message:
            return httpx.Response(202)
        return self.on_call(message, request)


async def upstream_for(script: Scripted) -> tuple[MCPConnector, MCPUpstream]:
    connector = MCPConnector(INTERNAL_KEY.encode(), transport=httpx.MockTransport(script))
    await connector.start()
    session = MCPUpstream(
        connector, server="sales_db", url=URL, trust="internal", principal="anna@demo"
    )
    return connector, session


CALL = {"name": "query", "arguments": {"sql": "SELECT 1 FROM sales.orders"}}
TOOL_RESULT = {"content": [{"type": "text", "text": "ok"}], "isError": False}


def test_sse_decoder_handles_split_chunks_and_crlf():
    decoder = _SseDecoder()
    assert decoder.feed(b"event: message\r\nda") == []
    assert decoder.feed(b'ta: {"a":\r') == []
    assert decoder.feed(b"\n") == []
    assert decoder.feed(b"data: 1}\r\n\r\n: comment\n\ndata: x\n\n") == ['{"a":\n1}', "x"]


async def test_server_requests_on_the_stream_are_refused_and_the_response_still_arrives(
    snapshot,
):
    def on_call(message: dict, _request: httpx.Request) -> httpx.Response:
        sampling = {"jsonrpc": "2.0", "id": "srv-7", "method": "sampling/createMessage"}
        progress = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}
        return sse(
            sampling, progress, {"jsonrpc": "2.0", "id": message["id"], "result": TOOL_RESULT}
        )

    script = Scripted(on_call)
    connector, session = await upstream_for(script)
    result = await session.execute(CALL, snapshot)
    await connector.aclose()
    assert result.body == TOOL_RESULT
    assert result.untrusted is False
    refusal = next(m for m in script.received if m.get("id") == "srv-7")
    assert refusal["error"]["code"] == -32601


async def test_responses_over_the_byte_cap_are_refused(snapshot_from, policy_doc):
    policy_doc["limits"]["max_response_bytes"] = 2048
    small = snapshot_from(policy_doc)

    def on_call(message: dict, _request: httpx.Request) -> httpx.Response:
        big = {"content": [{"type": "text", "text": "x" * 4096}], "isError": False}
        return sse({"jsonrpc": "2.0", "id": message["id"], "result": big})

    connector, session = await upstream_for(Scripted(on_call))
    with pytest.raises(UpstreamError) as exc:
        await session.execute(CALL, small)
    await connector.aclose()
    assert exc.value.reason_code == "upstream_response_too_large"


async def test_a_server_negotiating_another_protocol_is_refused(snapshot):
    connector, session = await upstream_for(Scripted(lambda m, r: sse(), version="2025-03-26"))
    with pytest.raises(UpstreamError) as exc:
        await session.execute(CALL, snapshot)
    await connector.aclose()
    assert exc.value.reason_code == "upstream_protocol_mismatch"


async def test_upstream_rpc_errors_and_mismatched_ids_never_leak(snapshot):
    def rpc_error(message: dict, _request: httpx.Request) -> httpx.Response:
        error = {"code": -32602, "message": "SECRET internal detail"}
        return sse({"jsonrpc": "2.0", "id": message["id"], "error": error})

    def wrong_id(message: dict, _request: httpx.Request) -> httpx.Response:
        return sse({"jsonrpc": "2.0", "id": 999, "result": TOOL_RESULT})

    for on_call, reason in (
        (rpc_error, "upstream_rpc_error"),
        (wrong_id, "upstream_invalid_response"),
    ):
        connector, session = await upstream_for(Scripted(on_call))
        with pytest.raises(UpstreamError) as exc:
            await session.execute(CALL, snapshot)
        await connector.aclose()
        assert exc.value.reason_code == reason
        assert "SECRET" not in exc.value.message


async def test_a_lost_session_is_reinitialized_for_listing_but_never_replays_a_call(snapshot):
    calls = {"tools/list": 0, "tools/call": 0}

    def on_call(message: dict, request: httpx.Request) -> httpx.Response:
        calls[message["method"]] += 1
        if calls[message["method"]] == 1:
            return httpx.Response(404)  # the upstream forgot our session
        listing = {"tools": [{"name": "query", "inputSchema": {"type": "object"}}]}
        return sse({"jsonrpc": "2.0", "id": message["id"], "result": listing})

    connector, session = await upstream_for(Scripted(on_call))
    assert [t.name for t in await session.list_tools(snapshot)] == ["query"]
    with pytest.raises(UpstreamError) as exc:
        await session.execute(CALL, snapshot)
    await connector.aclose()
    assert exc.value.reason_code == "upstream_session_lost"
    assert calls == {"tools/list": 2, "tools/call": 1}
