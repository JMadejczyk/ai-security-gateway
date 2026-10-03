"""``/mcp/{server}`` end to end, against real in-process MCP servers (mcp==2.3.0 ``MCPServer``).

Covers the transport (initialize, session id, protocol header, batches, unsupported methods),
filtered ``tools/list``, ``tools/call`` through the pipeline, the principal contract with
trusted upstreams, taint from the untrusted fetch server, approvals for autonomous agents,
session binding and teardown, and generic upstream failures.
"""

import json
from datetime import timedelta

import jwt
import pytest
from gateway_testkit import INTERNAL_KEY, bearer, claims, running_gateway, sign
from mcp_harness import PROTOCOL, MCPClient, MCPStack, connect, connect_all, error_text
from upstreams import RoutingTransport, running_upstreams

from gateway.sessions import SessionUpdate
from gateway.telemetry import REGISTRY

ANNA, BARTEK, ETL = "anna@demo", "bartek@demo", "svc:nightly_etl"
COUNT_CUSTOMERS = "SELECT COUNT(*) FROM sales.customers"


def requests_total(decision: str, agent: str) -> float:
    labels = {"channel": "mcp", "decision": decision, "agent": agent}
    return REGISTRY.get_sample_value("acl_requests_total", labels) or 0.0


def rows(result: dict) -> list[dict]:
    assert result["isError"] is False, result
    return result["structuredContent"]["result"]


# ----------------------------------------------------------------------- transport


async def test_initialize_issues_a_session_and_pins_the_protocol(stack: MCPStack):
    client = MCPClient(stack.gateway.agent, await stack.gateway.token(ANNA), "sales_db")
    response = await client.initialize()
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    result = response.json()["result"]
    assert result["protocolVersion"] == PROTOCOL
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert client.session_id
    assert (await client.request("ping")).json()["result"] == {}
    # Nothing reaches an upstream until a tool is listed or called.
    assert stack.transport.requests == []


async def test_protocol_header_is_required_after_initialize(stack: MCPStack):
    client = await connect(stack, ANNA, "sales_db")
    message = {"jsonrpc": "2.0", "id": 9, "method": "ping"}
    for version in ("2025-03-26", ""):
        response = await client.post(message, **{"mcp-protocol-version": version})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32600


async def test_session_id_is_required_after_initialize(stack: MCPStack):
    client = MCPClient(stack.gateway.agent, await stack.gateway.token(ANNA), "sales_db")
    response = await client.post(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        **{"mcp-protocol-version": PROTOCOL},
    )
    assert response.status_code == 400
    response = await client.post(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        **{"mcp-protocol-version": PROTOCOL, "mcp-session-id": "made-up"},
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    "method",
    [
        "resources/list",
        "resources/read",
        "prompts/list",
        "logging/setLevel",
        "completion/complete",
        "sampling/createMessage",
        "elicitation/create",
        "tools/frobnicate",
    ],
)
async def test_everything_outside_the_tools_subset_is_method_not_found(stack: MCPStack, method):
    client = await connect(stack, ANNA, "sales_db")
    response = await client.request(method, {})
    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32601


async def test_batches_are_rejected(stack: MCPStack):
    client = await connect(stack, ANNA, "sales_db")
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
    ]
    response = await client.post(batch)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32600


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b"{not json", -32700),
        (b'{"jsonrpc": "1.0", "id": 1, "method": "ping"}', -32600),
        (b'{"jsonrpc": "2.0", "id": 1, "result": {}}', -32600),
    ],
    ids=["parse", "version", "response"],
)
async def test_malformed_messages(stack: MCPStack, body, code):
    client = await connect(stack, ANNA, "sales_db")
    response = await stack.gateway.agent.post(client.path, content=body, headers=client.headers())
    assert (response.status_code, response.json()["error"]["code"]) == (400, code)


async def test_get_is_405_and_foreign_origins_are_refused(stack: MCPStack):
    client = await connect(stack, ANNA, "sales_db")
    assert (await stack.gateway.agent.get(client.path, headers=client.headers())).status_code == 405
    response = await client.post(
        {"jsonrpc": "2.0", "id": 1, "method": "ping"}, origin="https://evil.example"
    )
    assert response.status_code == 403
    assert response.json()["error"]["data"]["reason_code"] == "origin_not_allowed"


async def test_no_token_is_401_and_servers_stay_hidden(stack: MCPStack):
    anonymous = MCPClient(stack.gateway.agent, None, "sales_db")
    response = await anonymous.initialize()
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["data"]["reason_code"] == "token_missing"
    # Unknown and known servers look the same to an unauthenticated caller...
    assert (await MCPClient(stack.gateway.agent, None, "nope").initialize()).status_code == 401
    # ...and an authenticated one learns only that this one does not exist.
    unknown = MCPClient(stack.gateway.agent, await stack.gateway.token(ANNA), "nope")
    assert (await unknown.initialize()).status_code == 404


# ----------------------------------------------------------------------- tools/list


async def test_tools_list_shows_only_mapped_tools_with_hints_passed_through(stack: MCPStack):
    reports = await connect(stack, ANNA, "reports")
    response = await reports.request("tools/list")
    tools = {tool["name"]: tool for tool in response.json()["result"]["tools"]}
    assert set(tools) == {"write_report"}  # drop_reports is not in the operator mapping
    assert tools["write_report"]["annotations"]["readOnlyHint"] is True  # a hint, as sent
    assert tools["write_report"]["inputSchema"]["required"] == ["name", "content"]


async def test_databot_sees_its_tools_until_taint_removes_write(stack: MCPStack):
    web, sales, reports = await connect_all(stack, ANNA, "web", "sales_db", "reports")
    assert await web.tools() == ["fetch"]
    assert await sales.tools() == ["query"]
    assert await reports.tools() == ["write_report"]

    await web.call("fetch", url="https://example.com/outlook")  # untrusted: taints

    assert await reports.tools() == []  # interactive session: write removed until it ends
    assert await sales.tools() == ["query"]


async def test_autonomous_session_keeps_listing_write_after_taint(stack: MCPStack):
    web, reports = await connect_all(stack, ETL, "web", "reports")
    await web.call("fetch", url="https://example.com/outlook")
    assert await reports.tools() == ["write_report"]  # held for approval, not removed


# ----------------------------------------------------------------------- tools/call


async def test_same_query_different_principal_different_rows(stack: MCPStack):
    anna, bartek = await connect(stack, ANNA, "sales_db"), await connect(stack, BARTEK, "sales_db")
    assert rows(await anna.call("query", sql=COUNT_CUSTOMERS)) == [{"count": 40}]
    assert rows(await bartek.call("query", sql=COUNT_CUSTOMERS)) == [{"count": 7}]


async def test_trusted_upstream_gets_a_signed_principal_never_the_bearer(stack: MCPStack):
    anna = await connect(stack, ANNA, "sales_db")
    before = requests_total("allow", "databot")
    await anna.call("query", sql=COUNT_CUSTOMERS)

    [call] = stack.log.of("query")
    assert "authorization" not in call.headers
    assertion = call.headers["x-acl-principal"]
    assert assertion != anna.token
    claims = jwt.decode(
        assertion,
        INTERNAL_KEY,
        algorithms=["HS256"],
        audience="mcp-postgres",
        issuer="ai-control-layer",
        options={"verify_exp": False, "verify_iat": False},  # minted on the fixed test clock
    )
    assert claims["iat"] == int(stack.gateway.clock().timestamp())
    assert claims["sub"] == ANNA
    assert claims["exp"] - claims["iat"] <= 60
    # Every upstream request carried it fresh, and none carried the agent's token.
    sent = stack.transport.sent("mcp-postgres")
    assert sent
    assert all("x-acl-principal" in r.headers for r in sent)
    assert not any("authorization" in r.headers for r in stack.transport.requests)
    assert requests_total("allow", "databot") == before + 1


async def test_untrusted_upstream_gets_no_principal_assertion(stack: MCPStack):
    web = await connect(stack, ANNA, "web")
    result = await web.call("fetch", url="https://example.com/outlook")
    assert len(stack.log.of("fetch")) == 1
    # The page's hidden injection is withheld by the signatures control (post).
    assert error_text(result) == "signature_match"
    assert all("x-acl-principal" not in r.headers for r in stack.transport.sent("mcp-fetch"))


async def test_table_outside_the_role_is_denied_before_the_upstream(stack: MCPStack):
    bartek = await connect(stack, BARTEK, "sales_db")
    before = requests_total("block", "databot")
    result = await bartek.call("query", sql="SELECT SUM(amount) FROM sales.payments")
    assert error_text(result) == "outside_principal_scope"
    assert result["_meta"]["ai-control-layer/reason_code"] == "outside_principal_scope"
    assert stack.log.of("query") == []
    assert stack.transport.sent("mcp-postgres", "tools/call") == []
    assert requests_total("block", "databot") == before + 1
    [entry] = [e for e in stack.gateway.audit_entries() if e.get("resource")]
    assert (entry["channel"], entry["resource"], entry["decision"]) == (
        "mcp",
        "db:sales.payments",
        "block",
    )


async def test_join_needs_every_table(stack: MCPStack):
    bartek = await connect(stack, BARTEK, "sales_db")
    sql = "SELECT COUNT(*) FROM sales.customers c JOIN sales.payments p ON p.cid = c.id"
    assert error_text(await bartek.call("query", sql=sql)) == "outside_principal_scope"
    assert stack.log.of("query") == []


async def test_the_demo_story_taint_blocks_the_second_report(stack: MCPStack):
    """Write allowed → untrusted fetch taints → the same write is now refused."""
    web, reports = await connect_all(stack, ANNA, "web", "reports")

    first = await reports.call("write_report", name="q3.md", content="Q3 summary")
    assert first["isError"] is False
    assert len(stack.log.of("write_report")) == 1

    await web.call("fetch", url="https://example.com/outlook")
    session = await stack.gateway.container.sessions.get(
        stack.gateway.audit_entries()[-1]["session_id"]
    )
    assert session is not None
    assert session.taint

    second = await reports.call("write_report", name="q3-final.md", content="Q3 summary")
    assert error_text(second) == "action_removed_by_session_risk"
    assert len(stack.log.of("write_report")) == 1  # never reached the upstream
    entry = stack.gateway.audit_entries()[-1]
    assert (entry["action"], entry["decision"], entry["taint"]) == ("write", "block", True)
    assert not any(scope.startswith("write:") for scope in entry["effective_scope"])


async def test_autonomous_write_after_taint_waits_for_approval(stack: MCPStack):
    web, reports = await connect_all(stack, ETL, "web", "reports")
    await web.call("fetch", url="https://example.com/outlook")

    held = await reports.call("write_report", name="nightly.md", content="numbers")
    text = error_text(held)
    approval_id = held["_meta"]["ai-control-layer/approval_id"]
    assert text == f"approval_required approval_id={approval_id}"
    assert approval_id.startswith("apr-")
    assert stack.log.of("write_report") == []

    # Retrying the same pending operation names the same approval; another one does not.
    again = await reports.call("write_report", name="nightly.md", content="numbers")
    other = await reports.call("write_report", name="other.md", content="numbers")
    assert again["_meta"]["ai-control-layer/approval_id"] == approval_id
    assert other["_meta"]["ai-control-layer/approval_id"] != approval_id
    assert stack.gateway.audit_entries()[-1]["decision"] == "require_approval"


async def test_fs_traversal_is_denied(stack: MCPStack):
    reports = await connect(stack, ANNA, "reports")
    for name in ("../etc/cron.d/x", "/etc/passwd", "a\\b"):
        assert error_text(await reports.call("write_report", name=name, content="x")) == (
            "invalid_arguments"
        )
    assert stack.log.of("write_report") == []


async def test_unmapped_tool_is_denied_despite_its_annotation(stack: MCPStack):
    reports = await connect(stack, ANNA, "reports")
    assert error_text(await reports.call("drop_reports")) == "tool_not_mapped"
    assert stack.log.of("drop_reports") == []


async def test_invalid_arguments_and_unsupported_sql(stack: MCPStack):
    sales = await connect(stack, ANNA, "sales_db")
    assert (
        error_text(await sales.call("query", sql="DELETE FROM sales.orders")) == "unsupported_sql"
    )
    assert error_text(await sales.call("query", sql=1)) == "invalid_arguments"
    assert error_text(await sales.call("query", sql=COUNT_CUSTOMERS, extra="x")) == (
        "invalid_arguments"
    )
    assert stack.log.of("query") == []


async def test_malformed_call_params_are_a_protocol_error(stack: MCPStack):
    sales = await connect(stack, ANNA, "sales_db")
    response = await sales.request("tools/call", {"arguments": {}})
    assert response.json()["error"]["code"] == -32602


async def test_arguments_are_validated_against_the_pinned_schema(stack: MCPStack, tmp_path):
    """The stack's pins are the servers' own listings (`pin_kit`); a pin that differs from
    what the upstream advertises is a rug pull (tests/mcp/test_tool_pinning.py)."""
    sales = await connect(stack, ANNA, "sales_db")
    assert rows(await sales.call("query", sql=COUNT_CUSTOMERS)) == [{"count": 40}]
    assert error_text(await sales.call("query", sql=COUNT_CUSTOMERS, extra="x")) == (
        "invalid_arguments"
    )
    (tmp_path / "pins" / "sales_db.json").write_text("{broken")
    assert error_text(await sales.call("query", sql=COUNT_CUSTOMERS)) == "tool_pin_invalid"


# ------------------------------------------------------------------ session binding


async def test_session_id_of_another_principal_is_rejected(stack: MCPStack):
    anna = await connect(stack, ANNA, "sales_db")
    bartek = MCPClient(stack.gateway.agent, await stack.gateway.token(BARTEK), "sales_db")
    bartek.session_id = anna.session_id
    response = await bartek.request(
        "tools/call", {"name": "query", "arguments": {"sql": COUNT_CUSTOMERS}}
    )
    assert response.status_code == 404
    assert response.json()["error"]["data"]["reason_code"] == "mcp_session_not_found"
    assert stack.log.of("query") == []


async def test_session_id_is_bound_to_its_server(stack: MCPStack):
    sales = await connect(stack, ANNA, "sales_db")
    reports = MCPClient(stack.gateway.agent, sales.token, "reports", session_id=sales.session_id)
    assert (await reports.request("tools/list")).status_code == 404


async def test_each_downstream_session_has_its_own_upstream_session(stack: MCPStack):
    anna, bartek = await connect(stack, ANNA, "sales_db"), await connect(stack, BARTEK, "sales_db")
    await anna.call("query", sql=COUNT_CUSTOMERS)
    await bartek.call("query", sql=COUNT_CUSTOMERS)
    upstream_ids = {  # sql_guard's explain calls run in short-lived sessions of their own
        r.headers.get("mcp-session-id") for r in stack.transport.tool_calls("mcp-postgres", "query")
    }
    assert len(upstream_ids) == 2
    assert None not in upstream_ids


async def test_deleting_the_downstream_session_ends_the_upstream_one(stack: MCPStack):
    sales = await connect(stack, ANNA, "sales_db")
    await sales.call("query", sql=COUNT_CUSTOMERS)
    [call] = stack.transport.tool_calls("mcp-postgres", "query")
    upstream_id = call.headers["mcp-session-id"]

    def deletes() -> list:  # sql_guard's planner sessions end on their own; not this one
        return [
            r
            for r in stack.transport.sent("mcp-postgres")
            if r.method == "DELETE" and r.headers["mcp-session-id"] == upstream_id
        ]

    assert deletes() == []
    response = await stack.gateway.agent.delete(sales.path, headers=sales.headers())
    assert response.status_code == 204
    assert len(deletes()) == 1
    assert (await sales.request("ping")).status_code == 404
    assert len(stack.gateway.container.mcp.registry) == 0


async def test_ending_the_gateway_session_ends_its_mcp_sessions(stack: MCPStack):
    sales = await connect(stack, ANNA, "sales_db")
    await sales.call("query", sql=COUNT_CUSTOMERS)
    assert sales.token is not None
    ended = await stack.gateway.agent.delete("/v1/session", headers=bearer(sales.token))
    assert ended.status_code == 200
    assert [r for r in stack.transport.sent("mcp-postgres") if r.method == "DELETE"]
    assert len(stack.gateway.container.mcp.registry) == 0
    assert (await sales.request("ping")).status_code == 404  # the MCP session is gone...
    again = await MCPClient(stack.gateway.agent, sales.token, "sales_db").initialize()
    assert again.status_code == 401  # ...and its gateway session cannot start a new one
    assert again.json()["error"]["data"]["reason_code"] == "session_ended"


async def test_gateway_shutdown_ends_upstream_sessions(tmp_path):
    async with running_upstreams() as (transport, log):
        async with running_gateway(tmp_path, transport=transport) as gateway:
            stack = MCPStack(gateway, transport, log)
            sales = await connect(stack, ANNA, "sales_db")
            await sales.call("query", sql=COUNT_CUSTOMERS)
        assert [r for r in transport.sent("mcp-postgres") if r.method == "DELETE"]


# ------------------------------------------------------------------ upstream failures


async def test_upstream_failure_is_a_generic_tool_error(stack: MCPStack):
    sales = await connect(stack, ANNA, "sales_db")
    await sales.tools()  # the session and its schemas exist; now tools/call fails
    stack.transport.fail_tool_calls.add("mcp-postgres")
    response = await sales.request(
        "tools/call", {"name": "query", "arguments": {"sql": COUNT_CUSTOMERS}}
    )
    assert RoutingTransport.LEAK not in response.text
    assert error_text(response.json()["result"]) == "upstream_error"
    assert stack.gateway.audit_entries()[-1]["reason_code"] == "upstream_error"


async def test_unreachable_upstream_is_a_generic_error(stack: MCPStack, tmp_path):
    policy = stack.gateway.policy_path
    policy.write_text(policy.read_text().replace("mcp-fetch:8000", "mcp-gone:8000"))
    assert (
        await stack.gateway.operator.post(
            "/admin/reload", headers=bearer(await stack.gateway.operator_token("root@demo"))
        )
    ).status_code == 200
    web = await connect(stack, ANNA, "web")
    assert error_text(await web.call("fetch", url="https://example.com")) == "upstream_unreachable"
    listing = await web.request("tools/list")
    assert listing.json()["error"]["data"]["reason_code"] == "upstream_unreachable"


async def test_untrusted_upstream_error_still_taints(stack: MCPStack):
    """The request reached the untrusted server; its answer (here an error) reached us."""
    web, reports = await connect_all(stack, ANNA, "web", "reports")
    stack.transport.rpc_error_tool_calls.add("mcp-fetch")
    result = await web.call("fetch", url="https://example.com/outlook")
    assert error_text(result) == "upstream_rpc_error"
    assert RoutingTransport.LEAK not in json.dumps(result)
    session_id = stack.gateway.audit_entries()[-1]["session_id"]
    session = await stack.gateway.container.sessions.get(session_id)
    assert session is not None
    assert session.taint

    denied = await reports.call("write_report", name="after.md", content="x")
    assert error_text(denied) == "action_removed_by_session_risk"
    assert stack.log.of("write_report") == []


async def test_trusted_upstream_error_does_not_taint(stack: MCPStack):
    sales, reports = await connect_all(stack, ANNA, "sales_db", "reports")
    stack.transport.rpc_error_tool_calls.add("mcp-postgres")
    assert error_text(await sales.call("query", sql=COUNT_CUSTOMERS)) == "upstream_rpc_error"
    assert (await reports.call("write_report", name="ok.md", content="x"))["isError"] is False


async def test_fetch_receives_the_canonical_url_it_was_authorized_for(stack: MCPStack):
    web = await connect(stack, ANNA, "web")
    await web.call("fetch", url="https://faß.de/seite")
    [call] = stack.log.of("fetch")
    assert call.arguments["url"] == "https://xn--fa-hia.de/seite"
    entry = stack.gateway.audit_entries()[-1]
    assert entry["resource"] == "web:xn--fa-hia.de"


async def frozen_session(stack: MCPStack, update: SessionUpdate) -> MCPClient:
    """anna on sales_db in session ``s-frozen``, with ``update`` applied to that session."""
    token = sign(claims(stack.gateway.clock, session_id="s-frozen"))
    sales = MCPClient(stack.gateway.agent, token, "sales_db")
    assert (await sales.initialize()).status_code == 200
    await stack.gateway.container.sessions.apply("s-frozen", update, half_life_s=600)
    return sales


async def test_no_tools_listed_while_the_freeze_threshold_holds(stack: MCPStack):
    sales = await frozen_session(stack, SessionUpdate(risk_delta=0.9))  # risk > 0.8
    assert await sales.tools() == []
    session = await stack.gateway.container.sessions.get("s-frozen")
    assert session is not None
    assert session.freeze_until is None  # listing never starts the freeze timer


async def test_no_tools_listed_while_a_freeze_runs(stack: MCPStack):
    until = stack.gateway.clock() + timedelta(seconds=120)
    sales = await frozen_session(stack, SessionUpdate(freeze_until=until))  # risk stays 0
    assert await sales.tools() == []
    stack.gateway.clock.advance(121)
    assert await sales.tools() == ["query"]
