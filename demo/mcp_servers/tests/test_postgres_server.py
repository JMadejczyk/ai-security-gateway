"""`mcp-postgres`: principal and execution limits required and applied; `explain` plans only.

The first group needs no database: every refusal happens before a connection is taken. The
second runs against a real Postgres seeded from demo/db (`demo_postgres`, docker, opt-in).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, cast

import jwt
import pytest
from mcp.server.mcpserver import Context
from pydantic import SecretStr

from acl_demo_mcp.postgres_server import (
    PlanCost,
    PostgresSettings,
    PrincipalRejectedError,
    QueryRunner,
    ResultLimitError,
    StatementFailedError,
    StatementRejectedError,
    StatementTimeoutError,
    build_postgres_server,
)
from acl_demo_mcp.principal import PRINCIPAL_HEADER, PrincipalVerifier

KEY = b"k" * 32
VERIFIER = PrincipalVerifier(key=KEY, audience="mcp-postgres", issuer="ai-control-layer")
LIMITS = {"stmt_timeout_ms": 3000, "max_rows": 500, "max_result_bytes": 1_048_576}
# Nothing listens here: a test that reached the database would fail on the connection.
NOWHERE = PostgresSettings(host="127.0.0.1", port=9, password=SecretStr("unused"))

COUNT_CUSTOMERS = "SELECT COUNT(*) AS count FROM sales.customers LIMIT 500"
HEAVY = (
    "SELECT COUNT(*) FROM sales.customers AS c CROSS JOIN sales.orders AS o "
    "CROSS JOIN sales.payments AS p LIMIT 500"
)
MAX_COST = 10_000  # policy.yaml controls.sql_guard.max_cost


@dataclass(frozen=True)
class FakeContext:
    """The one thing the runner reads from the MCP request context."""

    headers: dict[str, str] | None


def _context(sub: str | None = "anna@demo", **limits: Any) -> Context:
    if sub is None:
        return cast(Context, FakeContext({}))
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": "ai-control-layer",
        "aud": "mcp-postgres",
        "sub": sub,
        "iat": now,
        "exp": now + 30,
    }
    if limits.get("limits", True) is not None:
        claims["limits"] = {**LIMITS, **{k: v for k, v in limits.items() if k != "limits"}}
    return cast(Context, FakeContext({PRINCIPAL_HEADER: jwt.encode(claims, KEY)}))


def _query(settings: PostgresSettings, sql: str, ctx: Context) -> list[dict[str, Any]]:
    async def run() -> list[dict[str, Any]]:
        runner = QueryRunner(settings, VERIFIER)
        try:
            return await runner.query(sql, ctx)
        finally:
            await runner.aclose()

    return asyncio.run(run())


def _explain(settings: PostgresSettings, sql: str, ctx: Context) -> PlanCost:
    async def run() -> PlanCost:
        runner = QueryRunner(settings, VERIFIER)
        try:
            return await runner.explain(sql, ctx)
        finally:
            await runner.aclose()

    return asyncio.run(run())


# ------------------------------------------------------------------- no database


def test_both_tools_are_served_and_read_only() -> None:
    tools = asyncio.run(build_postgres_server(NOWHERE, VERIFIER).list_tools())
    assert sorted(tool.name for tool in tools) == ["explain", "query"]
    assert all(tool.annotations and tool.annotations.read_only_hint for tool in tools)


@pytest.mark.parametrize(
    "ctx",
    [_context(sub=None), _context(limits=None)],
    ids=["no-principal", "no-limits"],
)
def test_principal_and_limits_are_required_by_both_tools(ctx: Context) -> None:
    with pytest.raises(PrincipalRejectedError):
        _query(NOWHERE, COUNT_CUSTOMERS, ctx)
    with pytest.raises(PrincipalRejectedError):
        _explain(NOWHERE, COUNT_CUSTOMERS, ctx)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 FROM sales.orders; SELECT 2 FROM sales.orders",
        "ANALYZE SELECT * FROM sales.orders",
        "EXPLAIN ANALYZE SELECT * FROM sales.orders",
        "DELETE FROM sales.orders",
        "SET statement_timeout = 0",
        "WITH d AS (DELETE FROM sales.orders RETURNING *) SELECT * FROM d",
    ],
)
def test_explain_and_query_refuse_anything_but_one_select(sql: str) -> None:
    with pytest.raises(StatementRejectedError):
        _explain(NOWHERE, sql, _context())
    with pytest.raises(StatementRejectedError):
        _query(NOWHERE, sql, _context())


# -------------------------------------------------------------- real database


def test_rows_are_filtered_per_principal(demo_postgres: PostgresSettings) -> None:
    anna = _query(demo_postgres, COUNT_CUSTOMERS, _context("anna@demo"))
    bartek = _query(demo_postgres, COUNT_CUSTOMERS, _context("bartek@demo"))
    assert (anna, bartek) == ([{"count": 40}], [{"count": 7}])


def test_row_cap_from_the_claim_refuses_larger_results(demo_postgres: PostgresSettings) -> None:
    sql = "SELECT id FROM sales.orders ORDER BY id LIMIT 20"
    assert len(_query(demo_postgres, sql, _context(max_rows=20))) == 20
    with pytest.raises(ResultLimitError, match="more than 19 rows"):
        _query(demo_postgres, sql, _context(max_rows=19))


def test_byte_cap_from_the_claim_refuses_larger_results(demo_postgres: PostgresSettings) -> None:
    sql = "SELECT id, ordered_at, amount FROM sales.orders ORDER BY id LIMIT 50"
    assert len(_query(demo_postgres, sql, _context())) == 50
    with pytest.raises(ResultLimitError, match="more than 200 bytes"):
        _query(demo_postgres, sql, _context(max_result_bytes=200))


def test_statement_timeout_from_the_claim_cancels(demo_postgres: PostgresSettings) -> None:
    with pytest.raises(StatementTimeoutError):
        _query(demo_postgres, HEAVY, _context(stmt_timeout_ms=50))


def test_explain_prices_without_executing(demo_postgres: PostgresSettings) -> None:
    cheap = _explain(demo_postgres, COUNT_CUSTOMERS, _context()).total_cost
    started = time.monotonic()
    # 50 ms would cancel the heavy statement if EXPLAIN ran it (it takes ~0.5 s).
    heavy = _explain(demo_postgres, HEAVY, _context(stmt_timeout_ms=50)).total_cost
    assert time.monotonic() - started < 5
    assert cheap < 100 < MAX_COST < heavy


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT set_config('app.user_id', 'root@demo', true) FROM sales.customers LIMIT 1",
        "SELECT set_config('statement_timeout', '0', true) FROM sales.customers LIMIT 1",
        "SELECT acl.set_principal('root@demo') FROM sales.customers LIMIT 1",
    ],
    ids=["set-config-principal", "set-config-timeout", "second-set-principal"],
)
def test_a_statement_can_never_change_the_principal_or_settings(
    demo_postgres: PostgresSettings, sql: str
) -> None:
    """Defense in depth behind sql_guard, which refuses these before they get here."""
    with pytest.raises(StatementFailedError):
        _query(demo_postgres, sql, _context("bartek@demo"))
