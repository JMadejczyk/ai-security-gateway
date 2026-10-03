"""``sql_guard`` through ``/mcp/sales_db`` against a real in-process MCP upstream.

The upstream double (`upstreams.sales_db_server`) records every ``query`` and ``explain`` it
runs with the headers it got, so these tests check what actually reached the SQL server: the
exact statement that executes, the statement that was priced, the principal and the signed
execution limits of both, and that a refused statement never executes.
"""

import json
from typing import ClassVar

import jwt
import pytest
from gateway_testkit import INTERNAL_KEY, bearer
from mcp_harness import MCPStack, connect, error_text
from pin_kit import pin_with_schema

from gateway.core.envelope import Span, Verdict
from gateway.core.interfaces import Control
from gateway.core.types import ControlKind, Decision, Stage
from gateway.proxies.mcp.explain import EXPLAIN_TOOL

SQL_ALLOW = pytest.mark.control("sql_guard", "allow")
SQL_DENY = pytest.mark.control("sql_guard", "deny")

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


@SQL_ALLOW
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


@SQL_DENY
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


@SQL_ALLOW
@SQL_DENY
async def test_raising_max_cost_on_reload_lets_the_same_query_run(stack: MCPStack):
    """Demo step 7: same request, new policy revision, different verdict."""
    stack.log.plan_cost = 70_495.2
    sales = await connect(stack, ANNA, "sales_db")
    assert error_text(await sales.call("query", sql=HEAVY)) == "sql_cost_exceeded"

    policy = stack.gateway.policy_path
    policy.write_text(policy.read_text().replace("max_cost: 10000", "max_cost: 100000"))
    reload = await stack.gateway.operator.post(
        "/admin/reload", headers=bearer(await stack.gateway.operator_token("root@demo"))
    )
    assert reload.status_code == 200
    assert (await sales.call("query", sql=HEAVY))["isError"] is False
    [query] = stack.log.of("query")
    assert query.arguments["sql"].endswith("LIMIT 500")


@SQL_DENY
async def test_no_plan_means_no_execution(stack: MCPStack):
    stack.log.explain_fails = True
    sales = await connect(stack, ANNA, "sales_db")
    assert error_text(await sales.call("query", sql=ALL_ORDERS)) == "sql_plan_unavailable"
    assert stack.log.of("query") == []


@SQL_DENY
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


# ------------------------------------------------- sql_guard prices the final statement

EXPLOIT = HEAVY + " WHERE 'alice@example.com' = '[REDACTED:EMAIL_ADDRESS]'"


def planner_cost(sql: str) -> float:
    """Like Postgres: a constant-false WHERE costs next to nothing; true, the full cross join."""
    if "'alice@example.com' = '[REDACTED:EMAIL_ADDRESS]'" in sql:
        return 0.02  # One-Time Filter: false
    return 70_495.2


@SQL_DENY
async def test_pii_redaction_cannot_turn_a_cheap_plan_into_a_heavy_query(stack: MCPStack):
    """Codex P1: priced as constant-false, then made true by the pii redaction."""
    stack.log.plan_cost_for = planner_cost
    sales = await connect(stack, ANNA, "sales_db")
    assert error_text(await sales.call("query", sql=EXPLOIT)) == "sql_cost_exceeded"
    assert stack.log.of("query") == []
    priced = {c.arguments["sql"] for c in stack.log.of("explain")}
    assert len(priced) == 1  # every table interaction priced the one final statement...
    [statement] = priced
    assert "alice@example.com" not in statement  # ...the redacted one, as it would execute
    assert statement.count("[REDACTED:EMAIL_ADDRESS]") == 2


async def test_the_priced_statement_is_the_executed_statement(stack: MCPStack):
    stack.log.plan_cost = 12.5  # cheap whatever the statement: it executes
    sales = await connect(stack, ANNA, "sales_db")
    sql = "SELECT COUNT(*) FROM sales.customers WHERE email = 'alice@example.com'"
    assert (await sales.call("query", sql=sql))["isError"] is False
    [explain] = stack.log.of("explain")
    [query] = stack.log.of("query")
    assert explain.arguments == query.arguments
    assert query.arguments["sql"] == (
        "SELECT COUNT(*) FROM sales.customers WHERE email = '[REDACTED:EMAIL_ADDRESS]' LIMIT 500"
    )


class RedactSqlKeyword(Control):
    """A pre control whose redaction breaks the statement (it masks the ``FROM`` keyword)."""

    # Borrows the id of egress, which never applies to sales_db (sql adapter): the test
    # swaps the real one out (each catalog id registers once).
    id: ClassVar[str] = "egress"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE})
    kind: ClassVar[ControlKind] = ControlKind.DETERMINISTIC

    async def evaluate(self, interaction, stage, cfg):
        sql = interaction.payload["arguments"]["sql"]
        start = sql.index("FROM")
        span = Span(path="/arguments/sql", start=start, end=start + 4, label="X")
        return Verdict(
            decision=Decision.REDACT, control_id=self.id, reason_code="x", redactions=(span,)
        )


async def test_redaction_that_breaks_the_statement_fails_closed(stack: MCPStack):
    stack.gateway.container.pipeline.controls.remove("egress")
    stack.gateway.container.pipeline.controls.register(RedactSqlKeyword())
    sales = await connect(stack, ANNA, "sales_db")
    result = await sales.call("query", sql="SELECT COUNT(*) FROM sales.customers")
    assert error_text(result) == "unsupported_sql"
    assert stack.log.calls == []


@SQL_DENY
async def test_a_statement_changed_after_sql_guard_is_never_dispatched(stack, monkeypatch):
    """Anything between sql_guard and dispatch (here the budget hold) must not alter the SQL."""
    ledger = stack.gateway.container.pipeline._budgets
    reserve = ledger.reserve

    async def tampering(call, snapshot):
        reservation = await reserve(call, snapshot)
        payload = {**reservation.payload, "arguments": {"sql": HEAVY}}
        return reservation.model_copy(update={"payload": payload})

    monkeypatch.setattr(ledger, "reserve", tampering)
    sales = await connect(stack, ANNA, "sales_db")
    result = await sales.call("query", sql="SELECT COUNT(*) FROM sales.customers")
    assert error_text(result) == "sql_changed_after_guard"
    assert stack.log.of("query") == []


def tamper_budget_hold(stack: MCPStack, monkeypatch, change) -> None:
    """Make the budget hold (the step between sql_guard and dispatch) apply ``change``."""
    ledger = stack.gateway.container.pipeline._budgets
    assert ledger is not None
    reserve = ledger.reserve

    async def tampering(*args, **kwargs):
        return change(await reserve(*args, **kwargs))

    monkeypatch.setattr(ledger, "reserve", tampering)


async def test_an_in_place_change_after_sql_guard_is_never_dispatched(stack, monkeypatch):
    """Codex: mutating the dict sql_guard sealed must not move the seal's baseline with it."""

    def in_place(reservation):
        reservation.payload["arguments"]["sql"] = HEAVY  # a cross join without LIMIT
        return reservation

    tamper_budget_hold(stack, monkeypatch, in_place)
    sales = await connect(stack, ANNA, "sales_db")
    result = await sales.call("query", sql="SELECT COUNT(*) FROM sales.customers")
    assert error_text(result) == "sql_changed_after_guard"
    assert stack.log.of("query") == []
    assert stack.transport.tool_calls("mcp-postgres", "query") == []


async def test_an_equal_but_different_json_value_is_a_change(stack, monkeypatch, tmp_path):
    """``True == 1`` in Python, not on the wire: the seal compares serialized bytes."""
    pinned = {
        "type": "object",
        "properties": {"sql": {"type": "string"}, "dry_run": {"type": ["boolean", "integer"]}},
    }
    # A pinned schema the upstream does not advertise: take tool_pinning out of the way, so
    # the call reaches the seal with the pinned schema as its argument schema.
    pin_with_schema(tmp_path / "pins", "sales_db", "query", pinned)
    stack.gateway.container.pipeline.controls.remove("tool_pinning")

    def true_to_one(reservation):
        arguments = {**reservation.payload["arguments"], "dry_run": 1}
        return reservation.model_copy(
            update={"payload": {**reservation.payload, "arguments": arguments}}
        )

    tamper_budget_hold(stack, monkeypatch, true_to_one)
    sales = await connect(stack, ANNA, "sales_db")
    result = await sales.call("query", sql="SELECT COUNT(*) FROM sales.customers", dry_run=True)
    assert error_text(result) == "sql_changed_after_guard"
    assert stack.transport.tool_calls("mcp-postgres", "query") == []


# ----------------------------------------------- the plan uses the admitted snapshot


class ReloadMidCall(Control):
    """A pre control that swaps in a new policy while the call is between admission and
    sql_guard: another sales_db endpoint and other execution limits."""

    id: ClassVar[str] = "egress"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE})
    kind: ClassVar[ControlKind] = ControlKind.DETERMINISTIC

    def __init__(self, stack: MCPStack) -> None:
        self.stack = stack

    async def evaluate(self, interaction, stage, cfg):
        policy = self.stack.gateway.policy_path
        text = policy.read_text().replace("mcp-postgres:8000", "mcp-gone:8000")
        policy.write_text(text.replace("timeout_ms: 3000", "timeout_ms: 999"))
        assert self.stack.gateway.container.policy_store.reload().result == "ok"
        return Verdict(decision=Decision.ALLOW, control_id=self.id, reason_code="ok")


async def test_explain_runs_under_the_admitted_snapshot_across_a_reload(stack: MCPStack):
    """Codex P2: the plan goes to the endpoint, with the limits, the statement runs under."""
    sales = await connect(stack, ANNA, "sales_db")
    await sales.tools()
    stack.gateway.container.pipeline.controls.remove("egress")  # see RedactSqlKeyword
    stack.gateway.container.pipeline.controls.register(ReloadMidCall(stack))
    assert (await sales.call("query", sql=ALL_ORDERS))["isError"] is False
    [explain] = stack.log.of("explain")
    [query] = stack.log.of("query")
    assert assertion(explain.headers)["limits"] == POLICY_LIMITS  # not timeout 999
    assert assertion(query.headers)["limits"] == POLICY_LIMITS
    assert stack.transport.sent("mcp-gone") == []
