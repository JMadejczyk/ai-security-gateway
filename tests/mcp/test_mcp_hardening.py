"""Review findings on the shared pipeline, exercised through ``/mcp/{server}``."""

from typing import ClassVar

import jwt
from mcp_harness import MCPClient, MCPStack, connect, error_text

from gateway.core.envelope import Verdict
from gateway.core.interfaces import Control
from gateway.core.types import ControlKind, Decision, Stage
from gateway.sessions import SessionUpdate

BARTEK, ETL = "bartek@demo", "svc:nightly_etl"
COUNT_CUSTOMERS = "SELECT COUNT(*) FROM sales.customers"


class RewriteSql(Control):
    """A pre control that rewrites every ``query`` to read another table."""

    id: ClassVar[str] = "pii"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE})
    kind: ClassVar[ControlKind] = ControlKind.DETERMINISTIC

    def __init__(self, sql: str) -> None:
        self.sql = sql

    async def evaluate(self, interaction, stage, cfg):
        payload = {**interaction.payload, "arguments": {"sql": self.sql}}
        return Verdict(
            decision=Decision.ALLOW, control_id="pii", reason_code="rewritten", rewrite=payload
        )


async def test_rewrite_to_a_forbidden_table_is_blocked(stack: MCPStack):
    stack.gateway.container.pipeline.controls.clear()  # scripted below, replacing the real controls
    stack.gateway.container.pipeline.controls.register(
        RewriteSql("SELECT SUM(amount) FROM sales.payments")
    )
    bartek = await connect(stack, BARTEK, "sales_db")
    result = await bartek.call("query", sql=COUNT_CUSTOMERS)
    assert error_text(result) == "rewrite_unauthorized"
    assert stack.log.of("query") == []
    entry = stack.gateway.audit_entries()[-1]
    assert (entry["channel"], entry["resource"], entry["decision"]) == (
        "mcp",
        "db:sales.payments",
        "block",
    )


async def test_rewrite_within_the_grant_still_runs(stack: MCPStack):
    stack.gateway.container.pipeline.controls.clear()  # scripted below, replacing the real controls
    stack.gateway.container.pipeline.controls.register(
        RewriteSql("SELECT COUNT(*) FROM sales.orders")
    )
    bartek = await connect(stack, BARTEK, "sales_db")
    result = await bartek.call("query", sql=COUNT_CUSTOMERS)
    assert result["isError"] is False
    [call] = stack.log.of("query")
    assert call.arguments == {"sql": "SELECT COUNT(*) FROM sales.orders"}


async def test_autonomous_tool_calls_are_throttled_with_retry_after(stack: MCPStack):
    etl = await connect(stack, ETL, "sales_db")  # initialize opened the gateway session
    assert etl.token is not None
    session_id = jwt.decode(etl.token, options={"verify_signature": False})["session_id"]
    await stack.gateway.container.sessions.apply(
        session_id, SessionUpdate(risk_delta=0.6), half_life_s=600
    )
    params = {"name": "query", "arguments": {"sql": COUNT_CUSTOMERS}}
    responses = [await etl.request("tools/call", params) for _ in range(3)]
    assert [r.status_code for r in responses] == [200, 429, 429]
    assert [r.headers.get("retry-after") for r in responses] == [None, "5", "10"]
    assert responses[1].json()["error"]["data"]["reason_code"] == "throttled"
    assert len(stack.log.of("query")) == 1


async def test_oversize_mcp_request_is_413_and_audited(stack: MCPStack):
    bartek = await connect(stack, BARTEK, "sales_db")
    params = {"name": "query", "arguments": {"sql": "SELECT 1 -- " + "x" * 1_100_000}}
    response = await bartek.request("tools/call", params)
    assert response.status_code == 413
    assert response.json()["error"]["data"]["reason_code"] == "request_too_large"
    entry = stack.gateway.audit_entries()[-1]
    assert (entry["channel"], entry["reason_code"], entry["principal"]) == (
        "mcp",
        "request_too_large",
        BARTEK,
    )
    assert stack.log.of("query") == []


async def test_mcp_admission_refusals_are_audited(stack: MCPStack):
    anonymous = MCPClient(stack.gateway.agent, None, "sales_db")
    assert (await anonymous.initialize()).status_code == 401
    entry = stack.gateway.audit_entries()[-1]
    assert (entry["channel"], entry["reason_code"], entry["status"], entry["principal"]) == (
        "mcp",
        "token_missing",
        401,
        None,
    )
