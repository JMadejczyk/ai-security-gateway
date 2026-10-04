"""What opencode 1.18.34 really sends (captured request shape), end to end through the app.

opencode talks to the gateway with the AI SDK's OpenAI-compatible provider: ``stream: true``
with ``stream_options.include_usage``, ``tool_choice: auto``, ``reasoning_effort: none``,
``max_tokens`` from the model's output limit, its own ``x-opencode-*`` headers, and parallel
tool calls in one answer. The buffered SSE re-emission must give the AI SDK what it parses:
one ``tool_calls`` delta per call with ``index``, ``id``, name and arguments, then
``finish_reason: tool_calls`` and a usage chunk.
"""

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from gateway_testkit import Harness, bearer, completion, running_gateway

CHAT = "/v1/chat/completions"
REMOTE = "https://openrouter.ai/api/v1"
KEY_ENV = "OPENROUTER_API_KEY"
DATABOT_PROMPT = Path(__file__).resolve().parents[2] / "demo" / "opencode" / "databot-prompt.md"
OPENCODE_HEADERS = {
    "user-agent": "opencode/1.18.34 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14",
    "x-opencode-session-id": "ses_test",
    "x-session-affinity": "ses_test",
    "x-session-id": "ses_test",
}
MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }
    for name, description in (
        ("reports_write_report", "Create a new text report `name` (a plain file name)."),
        ("sales_db_query", "Run one read-only SQL SELECT against the sales database."),
        ("web_fetch", "Fetch a public web page with HTTP GET and return its body as text."),
    )
]


COUNT_CUSTOMERS = "SELECT COUNT(*) FROM sales.customers"
COUNT_ORDERS = "SELECT COUNT(*) FROM sales.orders"


def _query(call_id: str, sql_text: str) -> dict[str, Any]:
    sql = json.dumps({"sql": sql_text})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "sales_db_query", "arguments": sql},
    }


PARALLEL = [_query("call_customers", COUNT_CUSTOMERS), _query("call_orders", COUNT_ORDERS)]


def opencode_request(
    model: str, question: str = "How many customers and orders?"
) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 2048,
        "reasoning_effort": "none",
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": MCP_TOOLS,
        "messages": [
            {"role": "system", "content": DATABOT_PROMPT.read_text()},
            {"role": "user", "content": question},
        ],
    }


def sse(text: str) -> tuple[list[dict[str, Any]], str]:
    lines = [line.removeprefix("data: ") for line in text.split("\n\n") if line]
    return [json.loads(line) for line in lines[:-1]], lines[-1]


@pytest.fixture
async def remote(tmp_path: Path) -> AsyncIterator[Harness]:
    env = {KEY_ENV: "remote-test-key-" + "x" * 16}
    async with running_gateway(tmp_path, llm_upstream="remote", env=env) as harness:
        yield harness


@pytest.fixture
def openrouter() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=REMOTE, assert_all_called=False) as router:
        yield router


async def ask(gateway: Harness, body: dict[str, Any]) -> httpx.Response:
    token = await gateway.token("anna@demo", agent="opencode")
    return await gateway.agent.post(CHAT, json=body, headers=bearer(token) | OPENCODE_HEADERS)


async def test_parallel_tool_calls_stream_as_ai_sdk_chunks(remote, openrouter):
    route = openrouter.post("/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json=completion(None, model="deepseek/deepseek-v4.1-flash", tool_calls=PARALLEL),
        )
    )
    response = await ask(remote, opencode_request("deepseek-v4.1-flash"))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    chunks, last = sse(response.text)
    assert last == "[DONE]"
    calls = [
        call
        for chunk in chunks
        for choice in chunk["choices"]
        for call in choice["delta"].get("tool_calls", ())
    ]
    assert [(c["index"], c["id"], c["function"]["name"]) for c in calls] == [
        (0, "call_customers", "sales_db_query"),
        (1, "call_orders", "sales_db_query"),
    ]
    assert all(json.loads(c["function"]["arguments"])["sql"] for c in calls)
    finishes = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]]
    assert finishes[-1] == "tool_calls"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 19  # stream_options.include_usage honoured
    assert all(c["model"] == "deepseek-v4.1-flash" for c in chunks)  # logical id only

    sent = json.loads(route.calls.last.request.content)
    assert (sent["stream"], sent["tool_choice"], sent["max_tokens"]) == (False, "auto", 2048)
    assert sent["reasoning"] == {"enabled": False}
    assert "stream_options" not in sent
    assert [t["function"]["name"] for t in sent["tools"]] == [
        t["function"]["name"] for t in MCP_TOOLS
    ]
    forwarded = route.calls.last.request.headers
    assert not {name for name in OPENCODE_HEADERS if name.startswith("x-")} & set(forwarded)
    (entry,) = remote.audit_entries()
    assert (entry["actor"], entry["decision"], entry["upstream"]) == ("opencode", "allow", "remote")


async def test_the_databot_system_prompt_is_not_flagged(remote, openrouter):
    """The launcher's prompt replaces opencode's stock one, which the injection judge confirmed
    as an injection (live finding). The test classifier flags nothing it lacks markers for, so
    this pins the wiring: an opencode request is classified and allowed, not refused."""
    openrouter.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(model="deepseek/deepseek-v4.1-flash"))
    )
    response = await ask(remote, opencode_request("deepseek-v4.1-flash"))
    assert response.status_code == 200
    (entry,) = remote.audit_entries()
    verdict = next(v for v in entry["verdicts"] if v["control"] == "prompt_injection")
    assert verdict["decision"] == "allow"


async def test_opencode_is_registered_for_anna_and_bartek_only(remote):
    for sub in ("anna@demo", "bartek@demo"):
        assert await remote.token(sub, agent="opencode")
    for sub in ("olga@demo", "root@demo"):
        refused = await remote.operator.post(
            "/auth/demo-token", json={"sub": sub, "agent": "opencode"}
        )
        assert refused.status_code == 403
        assert refused.json()["error"]["code"] == "principal_not_allowed"
