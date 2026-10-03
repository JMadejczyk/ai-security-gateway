"""The per-agent kill switch: admin-only, persisted, checked at admission, right before
dispatch and before a result is released; fail closed when its state is unknown."""

from typing import Any

import httpx
import pytest
from approvals_kit import mcp_harness
from gateway_testkit import T0, bearer, chat, completion
from redis.asyncio import Redis

from gateway.approvals.kill_switch import (
    InMemoryKillSwitchStore,
    KillRecord,
    KillSwitch,
    KillSwitchUnavailableError,
    RedisKillSwitchStore,
)
from gateway.approvals.model import ApprovalState
from gateway.core.types import Channel, Stage
from gateway.telemetry import REGISTRY

connect_all = mcp_harness.connect_all
error_text = mcp_harness.error_text

ETL, OLGA, ROOT = "svc:nightly_etl", "olga@demo", "root@demo"
META = "ai-control-layer/approval_id"
REPORT = {"name": "nightly.md", "content": "numbers"}


async def admin(stack_or_gateway: Any, path: str, sub: str = ROOT, **body: Any) -> httpx.Response:
    harness = getattr(stack_or_gateway, "gateway", stack_or_gateway)
    return await harness.operator.post(path, json=body, headers=bearer(await harness.token(sub)))


def killed_gauge(agent: str) -> float:
    return REGISTRY.get_sample_value("acl_kill_switch_active", {"agent": agent}) or 0.0


async def test_kill_blocks_the_next_call_and_unkill_restores(stack: Any):
    (reports,) = await connect_all(stack, ETL, "reports")
    assert (await reports.call("write_report", **REPORT))["isError"] is False

    killed = await admin(stack, "/admin/kill", agent="nightly_etl", reason="runaway job")
    assert killed.status_code == 200, killed.text
    assert (killed.json()["killed"], killed.json()["killed_by"]) == (True, ROOT)
    assert killed_gauge("nightly_etl") == 1
    blocked = await reports.call("write_report", name="again.md", content="x")
    assert error_text(blocked) == "agent_killed"
    assert len(stack.log.of("write_report")) == 1
    entry = stack.gateway.audit_entries()[-1]
    assert any(v["control"] == "kill_switch" for v in entry["verdicts"])

    listed = await stack.gateway.operator.get(
        "/admin/kill", headers=bearer(await stack.gateway.token(ROOT))
    )
    assert [k["agent"] for k in listed.json()["kills"]] == ["nightly_etl"]

    assert (await admin(stack, "/admin/unkill", agent="nightly_etl")).status_code == 200
    assert killed_gauge("nightly_etl") == 0
    assert (await reports.call("write_report", name="again.md", content="x"))["isError"] is False


async def test_a_killed_agent_cannot_use_the_llm_either(gateway, llm_upstream):
    upstream = llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion())
    )
    assert (await admin(gateway, "/admin/kill", agent="nightly_etl")).status_code == 200
    token = await gateway.token(ETL)
    response = await gateway.agent.post("/v1/chat/completions", json=chat(), headers=bearer(token))
    assert (response.status_code, response.json()["error"]["code"]) == (403, "agent_killed")
    assert not upstream.called
    # Other agents are untouched.
    anna = await gateway.token("anna@demo")
    ok = await gateway.agent.post("/v1/chat/completions", json=chat(), headers=bearer(anna))
    assert ok.status_code == 200


async def test_a_kill_landing_between_pre_controls_and_dispatch_stops_the_call(
    stack: Any, monkeypatch: pytest.MonkeyPatch
):
    """In-flight: the kill arrives while pre controls run; the call is never dispatched."""
    (reports,) = await connect_all(stack, ETL, "reports")
    container = stack.gateway.container
    loop_detect = next(
        c for c in container.pipeline.controls.for_stage(Stage.PRE, Channel.MCP)
        if c.id == "loop_detect"
    )  # fmt: skip
    evaluate = loop_detect.evaluate

    async def kill_meanwhile(interaction, stage, cfg):
        record = KillRecord(agent="nightly_etl", killed_by=ROOT, killed_at=T0)
        await container.oversight.kill_switch.kill(record)
        return await evaluate(interaction, stage, cfg)

    monkeypatch.setattr(loop_detect, "evaluate", kill_meanwhile)
    blocked = await reports.call("write_report", **REPORT)
    assert error_text(blocked) == "agent_killed"
    assert stack.log.of("write_report") == []
    assert stack.transport.tool_calls("mcp-files", "write_report") == []


async def test_a_kill_landing_during_the_upstream_call_withholds_the_result(
    stack: Any, monkeypatch: pytest.MonkeyPatch
):
    (reports,) = await connect_all(stack, ETL, "reports")
    container = stack.gateway.container
    original = stack.transport.handle_async_request

    async def kill_while_running(request: httpx.Request) -> httpx.Response:
        response = await original(request)
        if b'"write_report"' in request.content:
            record = KillRecord(agent="nightly_etl", killed_by=ROOT, killed_at=T0)
            await container.oversight.kill_switch.kill(record)
        return response

    monkeypatch.setattr(stack.transport, "handle_async_request", kill_while_running)
    blocked = await reports.call("write_report", **REPORT)
    assert error_text(blocked) == "agent_killed"
    assert "wrote" not in str(blocked)  # the upstream's answer never reaches the agent
    assert len(stack.log.of("write_report")) == 1  # it ran: a sent request cannot be recalled


async def test_kill_revokes_the_agents_unused_approvals(stack: Any):
    web, reports = await connect_all(stack, ETL, "web", "reports")
    await web.call("fetch", url="https://example.com/outlook")
    held = await reports.call("write_report", **REPORT)
    approval_id = held["_meta"][META]

    killed = await admin(stack, "/admin/kill", agent="nightly_etl", reason="incident")
    assert killed.json()["revoked_approvals"] == 1
    record = await stack.gateway.container.oversight.approvals.get(approval_id)
    assert (record.state, record.outcome) == (ApprovalState.DENIED, "agent_killed")
    olga = bearer(await stack.gateway.token(OLGA))
    path = f"/admin/approvals/{approval_id}/approve"
    late = await stack.gateway.operator.post(path, headers=olga)
    assert late.status_code == 409

    assert (await admin(stack, "/admin/unkill", agent="nightly_etl")).status_code == 200
    params = {"name": "write_report", "arguments": REPORT, "_meta": {META: approval_id}}
    retry = (await reports.request("tools/call", params)).json()["result"]
    assert error_text(retry) == "approval_denied"  # unusable for good
    assert stack.log.of("write_report") == []


async def test_redis_down_fails_closed(stack: Any, monkeypatch: pytest.MonkeyPatch):
    (reports,) = await connect_all(stack, ETL, "reports")
    dead = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2)  # pyright: ignore[reportUnknownMemberType]
    oversight = stack.gateway.container.oversight
    monkeypatch.setattr(oversight, "_kill_switch", KillSwitch(RedisKillSwitchStore(dead)))
    response = await reports.request("tools/call", {"name": "write_report", "arguments": REPORT})
    assert response.status_code == 503
    assert response.json()["error"]["data"]["reason_code"] == "kill_switch_unavailable"
    assert stack.log.of("write_report") == []
    await dead.aclose()


async def test_kill_and_unkill_are_admin_only(stack: Any):
    for path in ("/admin/kill", "/admin/unkill"):
        refused = await admin(stack, path, sub=OLGA, agent="nightly_etl")
        assert (refused.status_code, refused.json()["error"]["code"]) == (403, "admin_required")
        anonymous = await stack.gateway.operator.post(path, json={"agent": "nightly_etl"})
        assert anonymous.status_code == 401
        # Not on the agent listener at all.
        on_agent_api = await stack.gateway.agent.post(path, json={"agent": "nightly_etl"})
        assert on_agent_api.status_code == 404
    unknown = await admin(stack, "/admin/kill", agent="ghost")
    assert (unknown.status_code, unknown.json()["error"]["code"]) == (404, "unknown_agent")


# ------------------------------------------------------------------ the cache bound


class FakeMonotonic:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_another_process_sees_a_kill_within_the_cache_ttl():
    store = InMemoryKillSwitchStore()  # the shared Redis, as two gateways see it
    clock = FakeMonotonic()
    here = KillSwitch(store, cache_ttl_s=1.0, monotonic=clock)
    there = KillSwitch(store, cache_ttl_s=1.0, monotonic=clock)
    assert await there.check("nightly_etl") is None  # cached "not killed"

    await here.kill(KillRecord(agent="nightly_etl", killed_by=ROOT, killed_at=T0))
    assert await here.check("nightly_etl") is not None  # this process: immediately
    clock.now = 0.99
    assert await there.check("nightly_etl") is None  # the other: still within its TTL
    clock.now = 1.0
    assert await there.check("nightly_etl") is not None  # bounded by cache_ttl_s


class FlakyStore(InMemoryKillSwitchStore):
    def __init__(self) -> None:
        super().__init__()
        self.down = False

    async def killed(self) -> dict[str, KillRecord]:
        if self.down:
            raise KillSwitchUnavailableError
        return await super().killed()


async def test_a_stale_cache_is_never_served_when_the_store_is_down():
    store, clock = FlakyStore(), FakeMonotonic()
    switch = KillSwitch(store, cache_ttl_s=1.0, monotonic=clock)
    assert await switch.check("nightly_etl") is None
    store.down = True
    clock.now = 0.5
    assert await switch.check("nightly_etl") is None  # fresh cache: no round trip needed
    clock.now = 1.5
    with pytest.raises(KillSwitchUnavailableError):
        await switch.check("nightly_etl")
    store.down = False
    assert await switch.check("nightly_etl") is None
