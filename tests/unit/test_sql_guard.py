"""`SqlGuardControl` on its own, with a fake planner: rewrite, cost threshold, fail closed."""

import asyncio
from dataclasses import dataclass, field

import pytest

from gateway.controls.registry import ControlRegistry
from gateway.controls.sql_guard import PlanRequest, PlanUnavailableError, SqlGuardControl
from gateway.core.catalog import control_spec
from gateway.core.envelope import Interaction
from gateway.core.interfaces import ControlConfig
from gateway.core.types import Action, Channel, ControlMode, Decision, Stage
from gateway.policy.schema import SqlGuardConfig

CFG = SqlGuardConfig(mode=ControlMode.BLOCK, max_cost=10_000, force_limit=500, timeout_ms=100)
COUNT = "SELECT COUNT(*) FROM sales.customers"


@dataclass
class FakePlanner:
    cost: float = 12.5
    error: Exception | None = None
    delay_s: float = 0.0
    requests: list[PlanRequest] = field(default_factory=list)

    async def __call__(self, request: PlanRequest) -> float:
        self.requests.append(request)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        return self.cost


@pytest.fixture
def interaction(make_ctx):
    def build(sql: object = COUNT, *, channel=Channel.MCP, server="sales_db", **arguments):
        payload = {"name": "query", "arguments": {"sql": sql, **arguments}}
        return Interaction(
            session_id="s-1",
            principal="anna@demo",
            actor="databot",
            mode=make_ctx().mode,
            channel=channel,
            action=Action.READ,
            resource="db:sales.customers",
            payload=payload,
            context=make_ctx(),
            server=server,
        )

    return build


async def test_cheap_query_is_allowed_with_the_limited_rewrite(interaction):
    planner = FakePlanner(cost=12.5)
    verdict = await SqlGuardControl(planner).evaluate(interaction(), Stage.PRE, CFG)
    assert (verdict.decision, verdict.reason_code) == (Decision.ALLOW, "sql_allowed")
    limited = f"{COUNT} LIMIT 500"
    assert verdict.rewrite == {"name": "query", "arguments": {"sql": limited}}
    # The planner prices exactly the statement that will execute, as the caller.
    assert planner.requests == [PlanRequest(server="sales_db", principal="anna@demo", sql=limited)]


async def test_rewrite_keeps_the_other_arguments(interaction):
    verdict = await SqlGuardControl(FakePlanner()).evaluate(
        interaction(COUNT, note="kept"), Stage.PRE, CFG
    )
    assert verdict.rewrite["arguments"] == {"sql": f"{COUNT} LIMIT 500", "note": "kept"}


async def test_already_canonical_and_capped_query_needs_no_rewrite(interaction):
    sql = "SELECT * FROM sales.customers LIMIT 10"
    planner = FakePlanner()
    verdict = await SqlGuardControl(planner).evaluate(interaction(sql), Stage.PRE, CFG)
    assert (verdict.decision, verdict.rewrite) == (Decision.ALLOW, None)
    assert [r.sql for r in planner.requests] == [sql]


@pytest.mark.parametrize("cost", [0.0, 9_999.9, 10_000.0])
async def test_cost_at_or_under_the_threshold_is_allowed(interaction, cost):
    verdict = await SqlGuardControl(FakePlanner(cost=cost)).evaluate(interaction(), Stage.PRE, CFG)
    assert verdict.decision is Decision.ALLOW


async def test_cost_over_the_threshold_blocks_with_numbers_only(interaction):
    sql = "SELECT COUNT(*) FROM sales.customers c CROSS JOIN sales.orders o"
    verdict = await SqlGuardControl(FakePlanner(cost=123_456.78)).evaluate(
        interaction(sql), Stage.PRE, CFG
    )
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "sql_cost_exceeded")
    assert verdict.reason == "planner cost 123456.8 exceeds max_cost 10000.0"
    assert verdict.rewrite is None
    assert "customers" not in verdict.reason


@pytest.mark.parametrize(
    "planner",
    [
        FakePlanner(error=PlanUnavailableError()),
        FakePlanner(delay_s=5.0),  # longer than timeout_ms + the planner overhead
    ],
    ids=["planner-error", "planner-timeout"],
)
async def test_no_plan_means_no_execution(interaction, planner, monkeypatch):
    monkeypatch.setattr("gateway.controls.sql_guard.PLANNER_OVERHEAD_S", 0.05)
    verdict = await SqlGuardControl(planner).evaluate(interaction(), Stage.PRE, CFG)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "sql_plan_unavailable")
    assert verdict.rewrite is None


async def test_an_unexpected_planner_failure_propagates_to_the_pipeline(interaction):
    """The pipeline turns any control exception into a ``control_error`` block."""
    with pytest.raises(RuntimeError):
        await SqlGuardControl(FakePlanner(error=RuntimeError("boom"))).evaluate(
            interaction(), Stage.PRE, CFG
        )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT set_config('app.user_id', 'root@demo', true) FROM sales.customers",
        "DELETE FROM sales.customers",
        "SELECT * FROM sales.customers LIMIT 1 + 1",
        42,
    ],
)
async def test_statements_outside_the_subset_are_blocked_before_planning(interaction, sql):
    planner = FakePlanner()
    verdict = await SqlGuardControl(planner).evaluate(interaction(sql), Stage.PRE, CFG)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "unsupported_sql")
    assert planner.requests == []


async def test_a_sql_call_without_a_server_cannot_be_priced(interaction):
    verdict = await SqlGuardControl(FakePlanner()).evaluate(
        interaction(server=None), Stage.PRE, CFG
    )
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "sql_plan_unavailable")


async def test_calls_without_sql_are_not_its_business(interaction, make_ctx):
    planner = FakePlanner()
    control = SqlGuardControl(planner)
    fetch = Interaction(
        session_id="s-1",
        principal="anna@demo",
        actor="databot",
        mode=make_ctx().mode,
        channel=Channel.MCP,
        action=Action.READ,
        resource="web:example.com",
        payload={"name": "fetch", "arguments": {"url": "https://example.com"}},
        context=make_ctx(),
        server="web",
    )
    llm = interaction(channel=Channel.LLM)
    for other in (fetch, llm):
        verdict = await control.evaluate(other, Stage.PRE, CFG)
        assert (verdict.decision, verdict.reason_code, verdict.rewrite) == (
            Decision.ALLOW,
            "not_sql",
            None,
        )
    assert planner.requests == []


async def test_defaults_apply_to_a_plain_config(interaction):
    verdict = await SqlGuardControl(FakePlanner(cost=10_001)).evaluate(
        interaction(), Stage.PRE, ControlConfig()
    )
    assert verdict.reason_code == "sql_cost_exceeded"  # max_cost defaults to 10000


def test_mandatory_pre_mcp_control_that_matches_its_catalog_entry():
    control = SqlGuardControl(FakePlanner())
    spec = control_spec("sql_guard")
    assert control.mandatory
    assert spec.mandatory
    assert spec.modes == (ControlMode.BLOCK,)
    assert control.stages == frozenset({Stage.PRE})
    registry = ControlRegistry([control])
    assert registry.for_stage(Stage.PRE, Channel.MCP) == (control,)
    assert registry.for_stage(Stage.PRE, Channel.LLM) == ()
    assert registry.for_stage(Stage.POST, Channel.MCP) == ()


def test_policy_rejects_log_only_for_sql_guard(policy_doc, snapshot_from):
    policy_doc["controls"]["sql_guard"]["mode"] = "log_only"
    with pytest.raises(Exception, match="mandatory"):
        snapshot_from(policy_doc)
