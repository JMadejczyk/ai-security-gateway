"""LLM entry point end to end through the agent app, with the upstream mocked by respx."""

import json

import httpx
import pytest
from gateway_testkit import bearer, chat, claims, completion, echo_completion, sign

from gateway.telemetry import REGISTRY

AUTHZ_ALLOW = pytest.mark.control("authz", "allow")
AUTHZ_DENY = pytest.mark.control("authz", "deny")
AUTHN_DENY = pytest.mark.control("authn", "deny")

CHAT = "/v1/chat/completions"
TOOL_CALLS = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "query", "arguments": '{"sql": "SELECT count(*) FROM customers"}'},
    },
    {
        "id": "call_2",
        "type": "function",
        "function": {"name": "write_report", "arguments": '{"name": "q3.md"}'},
    },
]


def sample(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def sse_chunks(text: str) -> tuple[list[dict], str]:
    """Parse an SSE body into its JSON chunks; also return the final data line."""
    lines = [line for line in text.split("\n\n") if line]
    assert all(line.startswith("data: ") for line in lines)
    payloads = [line.removeprefix("data: ") for line in lines]
    return [json.loads(p) for p in payloads[:-1]], payloads[-1]


@pytest.fixture
def upstream_ok(llm_upstream):
    return llm_upstream.post("/chat/completions").mock(side_effect=echo_completion)


@AUTHZ_ALLOW
async def test_anna_can_generate(gateway, upstream_ok):
    token = await gateway.token("anna@demo")
    response = await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "There are 40 customers."
    assert upstream_ok.call_count == 1


@AUTHZ_DENY
async def test_bartek_denied_a_model_outside_his_role(gateway, upstream_ok):
    token = await gateway.token("bartek@demo")
    response = await gateway.agent.post(CHAT, json=chat("llama3:70b"), headers=bearer(token))
    assert response.status_code == 403
    error = response.json()["error"]
    assert (error["code"], error["type"]) == ("outside_principal_scope", "permission_error")
    assert not upstream_ok.called


@AUTHZ_ALLOW
@AUTHZ_DENY
async def test_same_request_different_user_different_result(gateway, upstream_ok):
    anna, bartek = await gateway.token("anna@demo"), await gateway.token("bartek@demo")
    body = chat("llama3:70b")
    assert (await gateway.agent.post(CHAT, json=body, headers=bearer(anna))).status_code == 200
    assert (await gateway.agent.post(CHAT, json=body, headers=bearer(bartek))).status_code == 403


@AUTHN_DENY
async def test_no_token_is_401(gateway, upstream_ok):
    response = await gateway.agent.post(CHAT, json=chat())
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "token_missing"
    assert not upstream_ok.called


@AUTHN_DENY
async def test_garbage_token_is_401(gateway, upstream_ok):
    response = await gateway.agent.post(CHAT, json=chat(), headers=bearer("not.a.jwt"))
    assert (response.status_code, response.json()["error"]["code"]) == (401, "token_invalid")


async def test_agent_authorization_never_reaches_upstream(gateway, upstream_ok):
    token = await gateway.token("anna@demo")
    headers = {**bearer(token), "x-custom": "agent-header", "cookie": "a=b"}
    await gateway.agent.post(CHAT, json=chat(), headers=headers)
    sent = upstream_ok.calls.last.request
    assert "authorization" not in sent.headers
    assert token not in str(sent.headers)
    assert "x-custom" not in sent.headers
    assert "cookie" not in sent.headers


async def test_upstream_router_key_comes_from_the_gateway(gateway, upstream_ok, monkeypatch):
    doc = gateway.policy_path.read_text().replace("api_key_env: null", "api_key_env: ROUTER_KEY", 1)
    gateway.policy_path.write_text(doc)
    assert gateway.container.policy_store.reload().error is None
    monkeypatch.setattr(gateway.container.llm, "_env", {"ROUTER_KEY": "sk-router"})
    token = await gateway.token("anna@demo")
    await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))
    assert upstream_ok.calls.last.request.headers["authorization"] == "Bearer sk-router"


async def test_upstream_always_gets_stream_false(gateway, upstream_ok):
    token = await gateway.token("anna@demo")
    body = chat(stream=True, stream_options={"include_usage": True}, temperature=0.2, seed=7)
    await gateway.agent.post(CHAT, json=body, headers=bearer(token))
    sent = json.loads(upstream_ok.calls.last.request.content)
    assert sent["stream"] is False
    assert "stream_options" not in sent
    assert (sent["temperature"], sent["seed"]) == (0.2, 7)  # other fields pass through


async def test_stream_is_reemitted_as_sse(gateway, llm_upstream):
    content = "Forty customers are visible to you. " * 5  # several 64-char chunks
    llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(content))
    )
    token = await gateway.token("anna@demo")
    response = await gateway.agent.post(CHAT, json=chat(stream=True), headers=bearer(token))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    chunks, last = sse_chunks(response.text)
    assert last == "[DONE]"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert streamed == content
    assert chunks[-1]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "stop"}


async def test_stream_includes_usage_when_asked(gateway, upstream_ok):
    token = await gateway.token("anna@demo")
    body = chat(stream=True, stream_options={"include_usage": True})
    chunks, _ = sse_chunks((await gateway.agent.post(CHAT, json=body, headers=bearer(token))).text)
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 19


async def test_tool_calls_survive_sse(gateway, llm_upstream):
    llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(None, tool_calls=TOOL_CALLS))
    )
    token = await gateway.token("anna@demo")
    response = await gateway.agent.post(CHAT, json=chat(stream=True), headers=bearer(token))
    chunks, last = sse_chunks(response.text)
    assert last == "[DONE]"
    calls = [
        c["choices"][0]["delta"]["tool_calls"][0]
        for c in chunks
        if "tool_calls" in c["choices"][0]["delta"]
    ]
    assert [{k: v for k, v in c.items() if k != "index"} for c in calls] == TOOL_CALLS
    assert [c["index"] for c in calls] == [0, 1]
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


async def test_oversize_request_is_413(gateway, upstream_ok):
    token = await gateway.token("anna@demo")
    huge = chat(messages=[{"role": "user", "content": "x" * 1_100_000}])
    response = await gateway.agent.post(CHAT, json=huge, headers=bearer(token))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert not upstream_ok.called


async def test_oversize_streamed_body_without_length_is_413(gateway, upstream_ok):
    token = await gateway.token("anna@demo")

    async def body():
        for _ in range(20):
            yield b"x" * 65536

    response = await gateway.agent.post(CHAT, content=body(), headers=bearer(token))
    assert response.status_code == 413
    assert not upstream_ok.called


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b"{not json", "invalid_json"),
        (b"[1, 2]", "invalid_json"),
        (json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(), "invalid_request"),
        (json.dumps(chat("bad model")).encode(), "invalid_model"),
    ],
    ids=["syntax", "array", "no-model", "bad-model"],
)
async def test_invalid_requests_are_400(gateway, upstream_ok, body, code):
    token = await gateway.token("anna@demo")
    response = await gateway.agent.post(CHAT, content=body, headers=bearer(token))
    assert (response.status_code, response.json()["error"]["code"]) == (400, code)
    assert "hi" not in response.text
    assert not upstream_ok.called


@pytest.mark.parametrize(
    "upstream",
    [
        httpx.Response(500, text="Traceback: secret internal detail"),
        httpx.Response(200, text="<html>not json, internal detail</html>"),
        httpx.Response(200, json={"unexpected": "internal detail"}),
        httpx.ConnectError("connection refused: internal detail"),
    ],
    ids=["500", "not-json", "bad-shape", "unreachable"],
)
async def test_upstream_failures_are_generic_502(gateway, llm_upstream, upstream):
    route = llm_upstream.post("/chat/completions")
    if isinstance(upstream, Exception):
        route.mock(side_effect=upstream)
    else:
        route.mock(return_value=upstream)
    token = await gateway.token("anna@demo")
    response = await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"
    assert "internal detail" not in response.text
    assert "internal detail" not in gateway.audit.getvalue()


async def test_upstream_response_over_the_limit_is_502(gateway, llm_upstream):
    llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion("y" * 4_200_000))
    )
    token = await gateway.token("anna@demo")
    response = await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))
    assert (response.status_code, response.json()["error"]["code"]) == (
        502,
        "upstream_response_too_large",
    )


async def test_models_are_filtered_per_user(gateway, llm_upstream):
    llm_upstream.get("/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "qwen3:8b", "object": "model", "owned_by": "library"},
                    {"id": "llama3:70b", "object": "model", "owned_by": "library"},
                ],
            },
        )
    )
    anna, bartek = await gateway.token("anna@demo"), await gateway.token("bartek@demo")
    olga = await gateway.token("olga@demo")

    async def models(token: str) -> list[str]:
        response = await gateway.agent.get("/v1/models", headers=bearer(token))
        assert response.status_code == 200
        return [m["id"] for m in response.json()["data"]]

    assert await models(anna) == ["qwen3:8b", "llama3:70b"]
    assert await models(bartek) == ["qwen3:8b"]
    assert await models(olga) == []  # ops-team grants nothing
    unauthenticated = await gateway.agent.get("/v1/models")
    assert unauthenticated.status_code == 401


@AUTHN_DENY
async def test_deleted_session_cannot_be_reused(gateway, upstream_ok):
    token = await gateway.token("anna@demo")
    assert (await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))).status_code == 200
    ended = await gateway.agent.delete("/v1/session", headers=bearer(token))
    assert ended.status_code == 200
    assert ended.json()["ended"] is True
    again = await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))
    assert (again.status_code, again.json()["error"]["code"]) == (401, "session_ended")
    # The demo issuer never names a caller-chosen session, and a validly signed token naming
    # the ended one is refused too.
    session_id = ended.json()["session_id"]
    reissue = await gateway.operator.post(
        "/auth/demo-token", json={"sub": "anna@demo", "session_id": session_id}
    )
    assert reissue.status_code == 422
    refreshed = sign(claims(gateway.clock, session_id=session_id))
    reuse = await gateway.agent.post(CHAT, json=chat(), headers=bearer(refreshed))
    assert reuse.json()["error"]["code"] == "session_ended"
    assert upstream_ok.call_count == 1


@AUTHN_DENY
async def test_session_reuse_by_another_principal_is_refused(gateway, upstream_ok):
    anna = sign(claims(gateway.clock, session_id="s-shared"))
    bartek = sign(claims(gateway.clock, sub="bartek@demo", roles=["intern"], session_id="s-shared"))
    assert (await gateway.agent.post(CHAT, json=chat(), headers=bearer(anna))).status_code == 200
    stolen = await gateway.agent.post(CHAT, json=chat(), headers=bearer(bartek))
    assert (stolen.status_code, stolen.json()["error"]["code"]) == (
        403,
        "session_binding_mismatch",
    )


@AUTHZ_DENY
async def test_audit_entry_has_revision_and_no_content(gateway, upstream_ok):
    sensitive_prompt = "Customer PESEL 44051401359 and the merger plan"
    token = await gateway.token("bartek@demo")
    body = chat(messages=[{"role": "user", "content": sensitive_prompt}])
    await gateway.agent.post(CHAT, json=body, headers=bearer(token))
    await gateway.agent.post(CHAT, json={**body, "model": "llama3:70b"}, headers=bearer(token))

    allowed, denied = gateway.audit_entries()
    revision = gateway.container.policy_store.current.revision
    assert allowed["policy_revision"] == denied["policy_revision"] == revision
    assert (allowed["decision"], allowed["status"], allowed["resource"]) == (
        "redact",  # the pii control masked the PESEL before forwarding
        200,
        "model:qwen3:8b",
    )
    assert (allowed["principal"], allowed["actor"], allowed["channel"]) == (
        "bartek@demo",
        "databot",
        "llm",
    )
    assert allowed["feed_version"] is not None
    assert allowed["feed_version"] == gateway.container.feed_store.version
    assert len(allowed["payload_hmac"]) == 64
    assert allowed["latency_ms"]["upstream"] is not None
    assert "generate:model:qwen3:8b" in allowed["effective_scope"]
    assert denied["decision"] == "block"
    assert denied["verdicts"] == [
        {
            "control": "authz",
            "stage": "pre",
            "decision": "block",
            "enforced": True,
            "reason_code": "outside_principal_scope",
        }
    ]
    assert denied["risk"] == pytest.approx(0.2)  # pii (0.1, first call) + authz deny (0.1)
    raw = gateway.audit.getvalue()
    for leaked in ("44051401359", "merger", "messages", "There are 40 customers"):
        assert leaked not in raw


@AUTHN_DENY
async def test_unauthenticated_call_is_audited_without_identity(gateway, upstream_ok):
    await gateway.agent.post(CHAT, json=chat())
    (entry,) = gateway.audit_entries()
    assert (entry["decision"], entry["reason_code"], entry["status"]) == (
        "block",
        "token_missing",
        401,
    )
    assert entry["principal"] is None
    assert entry["policy_revision"] == gateway.container.policy_store.current.revision


async def test_metrics_move(gateway, upstream_ok):
    allowed = {"channel": "llm", "decision": "allow", "agent": "databot"}
    blocked = {"channel": "llm", "decision": "block", "agent": "databot"}
    tokens = {"user": "anna@demo", "agent": "databot", "model": "qwen3:8b"}
    before = (
        sample("acl_requests_total", allowed),
        sample("acl_requests_total", blocked),
        sample("acl_tokens_total", tokens),
        sample("acl_overhead_seconds_count", {"channel": "llm"}),
        sample("acl_session_risk_count", {}),
    )
    anna, bartek = await gateway.token("anna@demo"), await gateway.token("bartek@demo")
    await gateway.agent.post(CHAT, json=chat(), headers=bearer(anna))
    await gateway.agent.post(CHAT, json=chat("llama3:70b"), headers=bearer(bartek))
    after = (
        sample("acl_requests_total", allowed),
        sample("acl_requests_total", blocked),
        sample("acl_tokens_total", tokens),
        sample("acl_overhead_seconds_count", {"channel": "llm"}),
        sample("acl_session_risk_count", {}),
    )
    assert [b - a for a, b in zip(before, after, strict=True)] == [1, 1, 19, 2, 2]

    exposed = await gateway.operator.get("/metrics")
    assert exposed.status_code == 200
    assert "acl_requests_total" in exposed.text
    assert "acl_tainted_sessions" in exposed.text


async def test_unknown_agent_label_is_bucketed(gateway, upstream_ok):
    labels = {"channel": "llm", "decision": "block", "agent": "other"}
    before = sample("acl_requests_total", labels)
    await gateway.agent.post(CHAT, json=chat(), headers=bearer("garbage"))
    assert sample("acl_requests_total", labels) == before + 1


async def test_unknown_agent_route_uses_openai_errors(gateway):
    response = await gateway.agent.get("/v1/nothing-here")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"


async def test_operator_routes_are_not_on_the_agent_app(gateway):
    for path in ("/auth/demo-token", "/admin/reload"):
        assert (await gateway.agent.post(path, json={"sub": "root@demo"})).status_code == 404
    assert (await gateway.agent.get("/metrics")).status_code == 404


async def test_healthz(gateway):
    response = await gateway.operator.get("/healthz")
    assert response.json() == {
        "status": "ok",
        "policy_revision": gateway.container.policy_store.current.revision,
        "budget_store": "memory",  # the test kit's store; Redis reports up/down
        "llm_upstream": "local",  # the product default
        "llm_upstream_host": "ollama",
    }
