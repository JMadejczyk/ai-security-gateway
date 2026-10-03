"""The demo agent's gateway client (``demo/agent/acl_agent``), against a mocked gateway."""

import json

import httpx
import pytest

from demo.agent.acl_agent.__main__ import main as agent_main
from demo.agent.acl_agent.client import (
    APPROVAL_META,
    REASON_META,
    MCPSession,
    Outcome,
    chat,
    parse_rpc,
    tool_outcome,
    unthrottled,
)
from demo.agent.acl_agent.probe import probe


def _result(**result: object) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": 2, "result": result}


def test_a_structured_result_is_the_rows():
    body = _result(content=[], isError=False, structuredContent={"result": [{"count": 40}]})
    outcome = tool_outcome(200, body)
    assert (outcome.ok, outcome.result) == (True, [{"count": 40}])


def test_a_text_result_is_joined():
    body = _result(content=[{"type": "text", "text": "wrote 6 bytes"}], isError=False)
    assert tool_outcome(200, body).result == "wrote 6 bytes"


def test_a_held_call_carries_its_reason_and_approval_id():
    meta = {REASON_META: "approval_required", APPROVAL_META: "apr-1"}
    body = _result(
        content=[{"type": "text", "text": "approval_required"}], isError=True, _meta=meta
    )
    outcome = tool_outcome(200, body)
    assert (outcome.reason, outcome.approval_id) == ("approval_required", "apr-1")


def test_a_refusal_before_the_tool_reads_the_reason_code_from_error_data():
    body = {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32001, "message": "rate limited", "data": {"reason_code": "throttled"}},
    }
    outcome = tool_outcome(429, body, retry_after_s=5.0)
    assert (outcome.reason, outcome.throttled, outcome.retry_after_s) == ("throttled", True, 5.0)


def test_an_rpc_error_without_data_falls_back_to_its_message():
    body = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "method not found"}}
    assert tool_outcome(200, body).reason == "method not found"


def test_parse_rpc_reads_one_sse_event():
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=b'event: message\ndata: {"jsonrpc": "2.0", "id": 1, "result": {}}\n\n',
    )
    assert parse_rpc(response) == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert parse_rpc(httpx.Response(202)) == {}


def test_unthrottled_waits_out_retry_after_and_reports_the_waits():
    answers = iter(
        [
            Outcome(status=429, reason="throttled", retry_after_s=5.0),
            Outcome(status=429, reason="throttled", retry_after_s=10.0),
            Outcome(status=200, reason="ok", result=[{"count": 50}]),
        ]
    )
    slept: list[float] = []
    outcome = unthrottled(lambda: next(answers), sleep=slept.append)
    assert outcome.ok
    assert outcome.result == [{"count": 50}]
    assert slept == [5.5, 10.5]
    assert outcome.throttle_waits == (5.5, 10.5)


def test_unthrottled_gives_up_after_its_retries():
    slept: list[float] = []
    throttled = Outcome(status=429, reason="throttled", retry_after_s=1.0)
    outcome = unthrottled(lambda: throttled, sleep=slept.append, retries=2)
    assert outcome.throttled
    assert len(slept) == 2


class FakeGateway:
    """Answers MCP initialize / notifications / tools/call / DELETE and chat completions."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            self.calls.append(body)
            if "AKIA" in body["messages"][0]["content"]:
                error = {"error": {"code": "secret_detected", "message": "blocked"}}
                return httpx.Response(403, json=error)
            message = {"role": "assistant", "content": "CALLER PESEL [REDACTED:PL_PESEL]"}
            return httpx.Response(200, json={"choices": [{"message": message}]})
        if request.method == "DELETE":
            return httpx.Response(204)
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {}}
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": result},
                headers={"mcp-session-id": "mcp-1"},
            )
        if "id" not in body:
            return httpx.Response(202)
        assert request.headers["mcp-session-id"] == "mcp-1"
        self.calls.append(body["params"])
        return httpx.Response(
            200, json=_result(content=[{"type": "text", "text": "ok"}], isError=False)
        )


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


def test_an_mcp_retry_sends_the_approval_id_in_meta(gateway):
    with (
        httpx.Client(base_url="http://gateway:8080", transport=httpx.MockTransport(gateway)) as c,
        MCPSession(c, "token", "reports") as session,
    ):
        outcome = session.call("write_report", approval_id="apr-1", name="a.md", content="x")
    assert outcome.ok
    [params] = gateway.calls
    assert params == {
        "name": "write_report",
        "arguments": {"name": "a.md", "content": "x"},
        "_meta": {APPROVAL_META: "apr-1"},
    }


def test_chat_disables_thinking_and_caps_tokens(gateway):
    with httpx.Client(base_url="http://gw", transport=httpx.MockTransport(gateway)) as client:
        answered = chat(client, "token", "upper case: PESEL 44051401359", max_tokens=40)
        refused = chat(client, "token", "key AKIAQ3EGRVW6XKZT4M7N")
    assert (answered.reason, answered.answer) == ("ok", "CALLER PESEL [REDACTED:PL_PESEL]")
    assert (refused.status, refused.reason) == (403, "secret_detected")
    assert gateway.calls[0]["reasoning_effort"] == "none"
    assert gateway.calls[0]["max_tokens"] == 40


def test_probe_reports_an_unresolvable_name():
    result = probe("no-such-host.invalid", 80, timeout_s=0.5)
    assert (result.connected, result.resolved) == (False, ())
    assert result.error is not None


def test_the_cli_prints_one_json_line(capsys):
    assert agent_main(["probe", "no-such-host.invalid", "5432"]) == 0
    reply = json.loads(capsys.readouterr().out)
    assert (reply["kind"], reply["connected"], reply["port"]) == ("probe", False, 5432)
