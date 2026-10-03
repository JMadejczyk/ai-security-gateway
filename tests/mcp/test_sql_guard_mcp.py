"""``sql_guard`` through ``/mcp/sales_db`` against a real in-process MCP upstream.

The upstream double (`upstreams.sales_db_server`) records every ``query`` and ``explain`` it
runs with the headers it got, so these tests check what actually reached the SQL server: the
exact statement that executes, the statement that was priced, the principal and the signed
execution limits of both, and that a refused statement never executes.
"""

import json

import jwt
import pytest
from gateway_testkit import INTERNAL_KEY, bearer
from mcp_harness import MCPStack, connect, error_text

from gateway.proxies.mcp.explain import EXPLAIN_TOOL

ANNA, BARTEK = "anna@demo", "bartek@demo"
ALL_ORDERS = "SELECT * FROM sales.orders"
LIMITED_ORDERS = "SELECT * FROM sales.orders LIMIT 500"
HEAVY = (
    "SELECT COUNT(*) FROM sales.customers c CROSS JOIN sales.orders o CROSS JOIN sales.payments p"
)
POLICY_LIMITS = {"stmt_timeout_ms": 3000, "max_rows": 500, "max_result_bytes": 1_048_576}


def assertion(headers: dict[str, str]) -> dict:
    return jwt.decode(
        headers["x-acl-principal"],
        INTERNAL_KEY,
        algorithms=["HS256"],
        audience="mcp-postgres",
        options={"verify_exp": False, "verify_iat": False},
    )


def sql_guard_verdicts(stack: MCPStack) -> list[dict]:
    return [
        verdict
        for entry in stack.gateway.audit_entries()
        for verdict in entry.get("verdicts", [])
        if verdict["control"] == "sql_guard"
    ]


async def test_the_limited_rewrite_is_priced_then_executed_exactly(stack: MCPStack):
    sales = await connect(stack, ANNA, "sales_db")
    result = await sales.call("query", sql=ALL_ORDERS)
    assert result["isError"] is False

    [explain] = stack.log.of("explain")
    [query] = stack.log.of("query")
    assert explain.arguments == {"sql": LIMITED_ORDERS}
    assert query.arguments == {"sql": LIMITED_ORDERS}  # never the agent's unlimited text
    assert stack.log.calls.index(explain) < stack.log.calls.index(query)
    # Same principal and signed limits for both; the plan runs in a session of its own.
    for call in (explain, query):
        claims = assertion(call.headers)
        assert (claims["sub"], claims["limits"]) == (ANNA, POLICY_LIMITS)
    assert explain.headers["mcp-session-id"] != query.headers["mcp-session-id"]
    [verdict] = sql_guard_verdicts(stack)
    assert (verdict["decision"], verdict["reason_code"]) == ("allow", "sql_allowed")


async def test_a_join_is_priced_per_table_and_executed_once(stack: MCPStack):
    sales = await connect(stack, ANNA, "sales_db")
    sql = "SELECT c.name FROM sales.customers c JOIN sales.orders o ON o.customer_id = c.id"
    assert (await sales.call("query", sql=sql))["isError"] is False
    rewritten = (
        "SELECT c.name FROM sales.customers AS c JOIN sales.orders AS o "
        "ON o.customer_id = c.id LIMIT 500"
    )
    assert [c.arguments["sql"] for c in stack.log.of("explain")] == [rewritten, rewritten]
    assert [c.arguments["sql"] for c in stack.log.of("query")] == [rewritten]


async def test_cost_over_the_threshold_never_executes(stack: MCPStack):
    stack.log.plan_cost = 70_495.2  # the demo database's estimate for this cross join
    sales = await connect(stack, ANNA, "sales_db")
    assert error_text(await sales.call("query", sql=HEAVY)) == "sql_cost_exceeded"
    assert len(stack.log.of("explain")) == 3  # one per table, each refused
    assert stack.log.of("query") == []
    assert stack.transport.tool_calls("mcp-postgres", "query") == []
    entries = stack.gateway.audit_entries()
    assert {e["reason_code"] for e in entries[-3:]} == {"sql_cost_exceeded"}
    assert "CROSS JOIN" not in json.dumps(entries)  # no SQL text in the audit


async def test_raising_max_cost_on_reload_lets_the_same_query_run(stack: MCPStack):
    """Demo step 7: same request, new policy revision, different verdict."""
    stack.log.plan_cost = 70_495.2
    sales = await connect(stack, ANNA, "sales_db")
    assert error_text(await sales.call("query", sql=HEAVY)) == "sql_cost_exceeded"

    policy = stack.gateway.policy_path
    policy.write_text(policy.read_text().replace("max_cost: 10000", "max_cost: 100000"))
    reload = await stack.gateway.operator.post(
        "/admin/reload", headers=bearer(await stack.gateway.token("root@demo"))
    )
    assert reload.status_code == 200
    assert (await sales.call("query", sql=HEAVY))["isError"] is False
    [query] = stack.log.of("query")
    assert query.arguments["sql"].endswith("LIMIT 500")


async def test_no_plan_means_no_execution(stack: MCPStack):
    stack.log.explain_fails = True
    sales = await connect(stack, ANNA, "sales_db")
    assert error_text(await sales.call("query", sql=ALL_ORDERS)) == "sql_plan_unavailable"
    assert stack.log.of("query") == []


async def test_an_unreachable_planner_fails_closed(stack: MCPStack):
    sales = await connect(stack, ANNA, "sales_db")
    await sales.tools()  # the agent's own session is up; now every plan request fails
    stack.transport.fail_explain = True
    assert error_text(await sales.call("query", sql=ALL_ORDERS)) == "sql_plan_unavailable"
    assert stack.log.of("query") == []


async def test_set_config_never_reaches_the_planner_or_the_database(stack: MCPStack):
    sales = await connect(stack, BARTEK, "sales_db")
    sql = "SELECT set_config('app.user_id', 'root@demo', true) FROM sales.customers"
    assert error_text(await sales.call("query", sql=sql)) == "unsupported_sql"
    assert stack.log.calls == []


async def test_agents_never_see_or_call_explain(stack: MCPStack):
    sales = await connect(stack, ANNA, "sales_db")
    assert await sales.tools() == ["query"]  # the upstream advertises explain too
    result = await sales.call(EXPLAIN_TOOL, sql=LIMITED_ORDERS)
    assert error_text(result) == "tool_not_mapped"
    assert stack.log.of("explain") == []
    assert stack.transport.tool_calls("mcp-postgres", EXPLAIN_TOOL) == []


@pytest.mark.parametrize("principal", [ANNA, BARTEK])
async def test_each_principal_prices_and_runs_as_itself(stack: MCPStack, principal):
    sales = await connect(stack, principal, "sales_db")
    await sales.call("query", sql="SELECT COUNT(*) FROM sales.customers")
    assert [assertion(c.headers)["sub"] for c in stack.log.calls] == [principal, principal]
