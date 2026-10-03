"""Budgets through the LLM surface: reserve before dispatch, settle on every exit.

Calls go to `ScriptedLLM`. Budgets and pricing are set per test by rewriting the gateway's
policy copy and reloading it. The MCP side (tool calls, 503) is in tests/mcp/test_mcp_budget.py.
"""

import asyncio
import contextlib
import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
import yaml
from budget_kit import ScriptedLLM, scope
from gateway_testkit import Harness, bearer, chat, running_gateway

from gateway.budget.model import ScopeKind, Spend
from gateway.telemetry import REGISTRY, ReloadResult

CHAT = "/v1/chat/completions"
DEAD_REDIS = "redis://127.0.0.1:1/0"  # nothing listens on port 1


def with_budgets(harness: Harness, budgets: dict[str, Any], pricing: dict[str, Any] | None = None):
    document = yaml.safe_load(harness.policy_path.read_text())
    document["budgets"] = budgets
    if pricing is not None:
        document["pricing"] = pricing
    harness.policy_path.write_text(yaml.safe_dump(document))
    outcome = harness.container.policy_store.reload()
    assert outcome.result is ReloadResult.OK, outcome.error


@asynccontextmanager
async def llm_gateway(
    tmp_path: Path, llm: ScriptedLLM, budgets: dict[str, Any] | None = None, **settings: Any
) -> AsyncIterator[Harness]:
    async with running_gateway(tmp_path, transport=llm, **settings) as harness:
        if budgets is not None:
            with_budgets(harness, budgets)
        yield harness


async def ask(harness: Harness, token: str, **body: Any) -> httpx.Response:
    return await harness.agent.post(CHAT, json=chat(**body), headers=bearer(token))


async def usage(harness: Harness, kind: ScopeKind = ScopeKind.USER, subject: str = "anna@demo"):
    window = "session" if kind is ScopeKind.SESSION else "2026-10-04"  # the test clock's day
    return await harness.container.budgets.store.usage(scope(kind, subject, window=window))


def refusal(response: httpx.Response) -> tuple[int, str, str]:
    error = response.json()["error"]
    return response.status_code, error["code"], error["message"]


def budget_verdicts(harness: Harness) -> list[dict[str, Any]]:
    return [
        verdict
        for entry in harness.audit_entries()
        for verdict in entry["verdicts"]
        if verdict["control"] == "budget"
    ]


def metric(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


# ------------------------------------------------------------------------------ tokens


async def test_daily_token_budget_blocks_before_the_upstream(tmp_path):
    llm = ScriptedLLM()
    async with llm_gateway(tmp_path, llm, {"per_user": {"daily_tokens": 30}}) as gateway:
        token = await gateway.token("anna@demo")
        first = await ask(gateway, token, max_tokens=10)  # holds 5 + 10, settles at 19
        assert first.status_code == 200, first.text
        assert await usage(gateway) == Spend(tokens=19)

        second = await ask(gateway, token, max_tokens=10)  # 19 + 15 > 30
        assert refusal(second) == (403, "budget_exceeded", "budget exceeded: per_user.daily_tokens")
        assert llm.calls == 1
        assert [v["reason_code"] for v in budget_verdicts(gateway)] == [
            "within_budget",
            "budget_exceeded",
        ]


async def test_a_request_without_a_cap_is_sent_with_the_default_one(tmp_path):
    llm = ScriptedLLM()
    async with llm_gateway(tmp_path, llm, {"per_user": {"daily_tokens": 10_000}}) as gateway:
        response = await ask(gateway, await gateway.token("anna@demo"))
        assert response.status_code == 200, response.text
        assert llm.bodies[0]["max_tokens"] == 4096  # limits.default_max_tokens


async def test_an_autonomous_agent_is_its_own_user(tmp_path):
    llm = ScriptedLLM()
    async with llm_gateway(tmp_path, llm, {"per_user": {"daily_tokens": 30}}) as gateway:
        etl = await gateway.token("svc:nightly_etl")
        assert (await ask(gateway, etl, max_tokens=10)).status_code == 200
        assert (await ask(gateway, etl, max_tokens=10)).status_code == 403
        anna = await gateway.token("anna@demo")
        assert (await ask(gateway, anna, max_tokens=10)).status_code == 200
        assert (await usage(gateway, subject="svc:nightly_etl")).tokens == 19


# -------------------------------------------------------------------------------- cost


async def test_daily_cost_budget_uses_the_pricing_table(tmp_path):
    llm = ScriptedLLM()
    pricing = {"qwen3:8b": {"prompt_per_1k": 1.0, "completion_per_1k": 1.0}}
    async with llm_gateway(tmp_path, llm) as gateway:
        with_budgets(gateway, {"per_user": {"daily_cost_usd": 0.05}}, pricing)
        token = await gateway.token("anna@demo")
        before = metric("acl_cost_usd_total", user="anna@demo", agent="databot", model="qwen3:8b")
        for _ in range(2):  # each holds 15 tokens ($0.015) and settles at 19 ($0.019)
            assert (await ask(gateway, token, max_tokens=10)).status_code == 200
        third = await ask(gateway, token, max_tokens=10)  # $0.038 + $0.015 > $0.05
        assert refusal(third) == (
            403,
            "budget_exceeded",
            "budget exceeded: per_user.daily_cost_usd",
        )
        after = metric("acl_cost_usd_total", user="anna@demo", agent="databot", model="qwen3:8b")
        assert after - before == pytest.approx(0.038)
        assert (await usage(gateway)).cost_nano_usd == 38_000_000


async def test_tokens_metric_matches_the_settled_usage(tmp_path):
    llm = ScriptedLLM(usage={"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 1})
    async with llm_gateway(tmp_path, llm, {"per_user": {"daily_tokens": 10_000}}) as gateway:
        labels = {"user": "anna@demo", "agent": "databot", "model": "qwen3:8b"}
        before = metric("acl_tokens_total", **labels)
        assert (await ask(gateway, await gateway.token("anna@demo"))).status_code == 200
        assert metric("acl_tokens_total", **labels) - before == 42
        assert (await usage(gateway)).tokens == 42


# ----------------------------------------------------------------------- GPU wall time


async def test_gpu_seconds_are_charged_from_upstream_wall_time(tmp_path):
    llm = ScriptedLLM(delay_s=0.15)
    async with llm_gateway(tmp_path, llm, {"per_session": {"gpu_seconds": 0.1}}) as gateway:
        token = await gateway.token("anna@demo")
        labels = {"user": "anna@demo", "agent": "databot", "model": "qwen3:8b"}
        before = metric("acl_cost_usd_total", **labels)
        assert (await ask(gateway, token)).status_code == 200  # admitted: headroom left
        session = await usage(gateway, ScopeKind.SESSION, session_of(token))
        assert 150 <= session.gpu_ms < 1500
        second = await ask(gateway, token)
        assert refusal(second) == (
            403,
            "budget_exceeded",
            "budget exceeded: per_session.gpu_seconds",
        )
        assert llm.calls == 1
        # qwen3:8b costs $0.0005 per GPU second in policy.yaml
        cost = metric("acl_cost_usd_total", **labels) - before
        assert cost == pytest.approx(session.gpu_ms / 1000 * 0.0005)


def session_of(token: str) -> str:
    session: str = jwt.decode(token, options={"verify_signature": False})["session_id"]
    return session


# ------------------------------------------------------------------ failure and cancel


async def test_an_upstream_failure_releases_the_hold(tmp_path):
    llm = ScriptedLLM(status=500)
    async with llm_gateway(tmp_path, llm, {"per_user": {"daily_tokens": 100}}) as gateway:
        token = await gateway.token("anna@demo")
        failed = await ask(gateway, token, max_tokens=95)  # holds 5 + 95 = the whole budget
        assert failed.status_code == 502
        assert (await usage(gateway)).tokens == 0
        llm.status = 200
        assert (await ask(gateway, token, max_tokens=95)).status_code == 200


async def test_a_client_disconnect_mid_call_releases_the_hold(tmp_path):
    llm = ScriptedLLM(gate=asyncio.Event())
    budgets = {"per_user": {"daily_tokens": 10_000}, "per_session": {"gpu_seconds": 100}}
    async with llm_gateway(tmp_path, llm, budgets) as gateway:
        token = await gateway.token("anna@demo")
        call = asyncio.create_task(ask(gateway, token))
        await llm.entered.wait()
        assert (await usage(gateway)).tokens == 5 + 4096  # held while the upstream works
        await asyncio.sleep(0.05)
        call.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await call
        await gateway.container.budgets.drain()
        assert (await usage(gateway)).tokens == 0
        session = await usage(gateway, ScopeKind.SESSION, session_of(token))
        assert session.gpu_ms >= 50  # the GPU time spent before the disconnect still counts
        llm.gate.set()  # pyright: ignore[reportOptionalMemberAccess]
        assert (await ask(gateway, token)).status_code == 200  # the session lock was released


# ---------------------------------------------------------------------------- soft limit


async def test_soft_limit_warns_once_and_tracks_the_usage_ratio(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="gateway.alerts")
    llm = ScriptedLLM()
    budgets = {"per_user": {"daily_tokens": 100}, "soft_limit_pct": 50}
    async with llm_gateway(tmp_path, llm, budgets) as gateway:
        token = await gateway.token("anna@demo")
        for _ in range(5):  # 19, 38, 57, 76, 95 tokens: three settlements above 50 %
            assert (await ask(gateway, token, max_tokens=1)).status_code == 200
        assert (await ask(gateway, token, max_tokens=1)).status_code == 403  # 95 + 6 > 100
        warnings = [
            json.loads(record.getMessage())
            for record in caplog.records
            if "budget_soft_limit" in record.getMessage()
        ]
        assert len(warnings) == 1
        assert warnings[0]["limit"] == "per_user.daily_tokens"
        assert warnings[0]["usage_ratio"] == pytest.approx(0.57)
        ratio = metric("acl_budget_usage_ratio", scope="per_user.daily_tokens", id="anna@demo")
        assert ratio == pytest.approx(0.95)


# ------------------------------------------------------------------------ store failure


async def test_redis_down_fails_closed_with_503(tmp_path):
    llm = ScriptedLLM()
    async with llm_gateway(tmp_path, llm, budget_store="redis", redis_url=DEAD_REDIS) as gateway:
        response = await ask(gateway, await gateway.token("anna@demo"))
        assert refusal(response) == (
            503,
            "budget_store_unavailable",
            "budget store unavailable; try again later",
        )
        assert llm.calls == 0
        assert budget_verdicts(gateway)[-1]["reason_code"] == "budget_store_unavailable"
        health = await gateway.operator.get("/healthz")
        assert health.status_code == 200
        assert (health.json()["status"], health.json()["budget_store"]) == ("degraded", "down")
        assert metric("acl_budget_store_up") == 0


# -------------------------------------------------------------------------------- privacy


async def test_refusals_name_the_limit_never_anyone_s_usage(tmp_path):
    llm = ScriptedLLM()
    async with llm_gateway(tmp_path, llm, {"per_agent": {"daily_tokens": 30}}) as gateway:
        anna = await gateway.token("anna@demo")
        assert (await ask(gateway, anna, max_tokens=10)).status_code == 200
        response = await ask(gateway, await gateway.token("bartek@demo"), max_tokens=10)
        status, code, message = refusal(response)
        assert (status, code, message) == (
            403,
            "budget_exceeded",
            "budget exceeded: per_agent.daily_tokens",
        )
        assert not re.search(r"\d", message)
        assert "anna" not in response.text
        refused = gateway.audit_entries()[-1]
        assert refused["principal"] == "bartek@demo"
        assert "anna" not in json.dumps(refused)
