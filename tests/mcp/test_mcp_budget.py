"""Budgets on the MCP surface: per-session tool calls, and failing closed without Redis."""

import json
from pathlib import Path

import pytest
import yaml
from gateway_testkit import running_gateway
from mcp_harness import MCPStack, connect
from pin_kit import capture_pins, write_pins
from upstreams import running_upstreams

from gateway.telemetry import ReloadResult

ALLOW = pytest.mark.control("budget", "allow")
DENY = pytest.mark.control("budget", "deny")

DEAD_REDIS = "redis://127.0.0.1:1/0"  # nothing listens on port 1


def set_budgets(stack: MCPStack, budgets: dict[str, object]) -> None:
    document = yaml.safe_load(stack.gateway.policy_path.read_text())
    document["budgets"] = budgets
    stack.gateway.policy_path.write_text(yaml.safe_dump(document))
    outcome = stack.gateway.container.policy_store.reload()
    assert outcome.result is ReloadResult.OK, outcome.error


@ALLOW
@DENY
async def test_the_51st_tool_call_of_a_session_is_blocked(stack: MCPStack):
    set_budgets(stack, {"per_session": {"tool_calls": 50}})
    reports = await connect(stack, "anna@demo", "reports")
    for index in range(50):
        result = await reports.call("write_report", name=f"r{index}.md", content="ok")
        assert result["isError"] is False, result
    blocked = await reports.call("write_report", name="r50.md", content="ok")
    assert blocked["isError"] is True
    assert "budget_exceeded" in json.dumps(blocked)
    assert len(stack.log.of("write_report")) == 50


@DENY
async def test_mcp_calls_fail_closed_with_503_when_redis_is_down(tmp_path: Path):
    async with running_upstreams() as (transport, log):
        write_pins(tmp_path / "pins", await capture_pins(transport))
        gateway_ctx = running_gateway(
            tmp_path,
            transport=transport,
            budget_store="redis",
            session_store="memory",  # the budget store alone is down
            redis_url=DEAD_REDIS,
        )
        async with gateway_ctx as gateway:
            reports = await connect(MCPStack(gateway, transport, log), "anna@demo", "reports")
            response = await reports.request(
                "tools/call",
                {"name": "write_report", "arguments": {"name": "r.md", "content": "ok"}},
            )
            assert response.status_code == 503
            assert response.json()["error"]["data"]["reason_code"] == "budget_store_unavailable"
            assert log.of("write_report") == []
