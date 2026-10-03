"""Regression tests for the budget review findings (each failed before its fix).

1. GPU time is reserved as an enforced allowance, so concurrent sessions cannot overspend.
2. ``n`` / ``best_of`` cannot multiply completions past the hold; caps are sent consistently.
3. A response without ``usage`` keeps its estimated token cost, not only its tokens.
4. A failed settlement is retried, and blocks its scopes until it lands.
5. A reservation whose reply was lost is released by operation id.
6. Absurd amounts are bounded before anything is written.
7. Session counters outlive a session whose lifetime a reload extended.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import fakeredis
import httpx
import pytest
import yaml
from budget_kit import FlakyStore, ScriptedLLM, StoreUnderTest, scope
from gateway_testkit import T0, Harness, MutableClock, bearer, chat, running_gateway
from test_budget_pipeline import session_of

import gateway.container
from gateway.budget.ledger import SESSION_TTL_SLACK_S, BudgetLedger
from gateway.budget.model import BudgetedCall, BudgetStoreUnavailableError, ScopeKind
from gateway.budget.redis_store import RedisBudgetStore
from gateway.budget.store import BudgetStore, InMemoryBudgetStore
from gateway.core.types import Channel
from gateway.telemetry import ReloadResult

ALLOW = pytest.mark.control("budget", "allow")
DENY = pytest.mark.control("budget", "deny")

CHAT = "/v1/chat/completions"
GPU_PRICED = {"qwen3:8b": {"gpu_second": 1.0}}  # $1 per GPU second
TOKEN_PRICED = {"qwen3:8b": {"prompt_per_1k": 1.0, "completion_per_1k": 1.0}}


def configure(harness: Harness, **sections: Any) -> None:
    document = yaml.safe_load(harness.policy_path.read_text())
    document.update(sections)
    harness.policy_path.write_text(yaml.safe_dump(document))
    outcome = harness.container.policy_store.reload()
    assert outcome.result is ReloadResult.OK, outcome.error


@asynccontextmanager
async def gateway_with(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    llm: ScriptedLLM,
    store: Callable[[], Any] | None = None,
    **sections: Any,
) -> AsyncIterator[Harness]:
    if store is not None:
        made = store()
        monkeypatch.setattr(gateway.container, "budget_store_from_settings", lambda *_a, **_k: made)
    async with running_gateway(tmp_path, transport=llm) as harness:
        configure(harness, **sections)
        yield harness


async def ask(harness: Harness, token: str, **body: Any) -> httpx.Response:
    return await harness.agent.post(CHAT, json=chat(**body), headers=bearer(token))


async def usage(harness: Harness, kind: ScopeKind = ScopeKind.USER, subject: str = "anna@demo"):
    window = "session" if kind is ScopeKind.SESSION else "2026-10-04"
    store: BudgetStore = harness.container.budgets.store
    inner = getattr(store, "inner", store)
    return await inner.usage(scope(kind, subject, window=window))


def code(response: httpx.Response) -> tuple[int, str]:
    if response.status_code == 200:
        return 200, "ok"
    return response.status_code, response.json()["error"]["code"]


# ------------------------------------------------------------ 1. GPU allowance (P1)


@ALLOW
@DENY
@pytest.mark.parametrize(
    ("budgets", "pricing", "meter", "limit"),
    [
        ({"per_user": {"daily_gpu_seconds": 1}}, None, "gpu_ms", 1000),
        ({"per_user": {"daily_cost_usd": 1.0}}, GPU_PRICED, "cost_nano_usd", 1_000_000_000),
    ],
    ids=["gpu-seconds", "gpu-cost"],
)
async def test_concurrent_sessions_cannot_overspend_gpu_time(
    tmp_path, monkeypatch, budgets, pricing, meter, limit
):
    llm = ScriptedLLM(delay_s=0.3)
    sections: dict[str, Any] = {"budgets": budgets}
    if pricing is not None:
        sections["pricing"] = pricing
    async with gateway_with(tmp_path, monkeypatch, llm, **sections) as gateway:
        tokens = [await gateway.token("anna@demo") for _ in range(10)]  # ten sessions
        responses = await asyncio.gather(*(ask(gateway, token) for token in tokens))
        statuses = [code(response) for response in responses]
        assert statuses.count((200, "ok")) >= 1
        assert (403, "budget_exceeded") in statuses
        spent = (await usage(gateway)).model_dump()[meter]
        assert spent <= limit
        assert spent >= limit * 0.25  # the admitted call's 0.3 s were charged, the rest refunded


@DENY
async def test_the_gpu_allowance_bounds_the_upstream_call(tmp_path, monkeypatch):
    llm = ScriptedLLM(delay_s=2.0)
    async with gateway_with(
        tmp_path, monkeypatch, llm, budgets={"per_session": {"gpu_seconds": 0.5}}
    ) as gateway:
        token = await gateway.token("anna@demo")
        started = time.perf_counter()
        response = await ask(gateway, token)
        assert time.perf_counter() - started < 1.5
        assert code(response) == (502, "upstream_timeout")
        assert code(await ask(gateway, token)) == (403, "budget_exceeded")
        assert llm.calls == 1


# --------------------------------------------------------------- 2. n / best_of (P1)


@pytest.mark.parametrize("extra", [{"n": 2}, {"n": 0}, {"best_of": 3}], ids=["n2", "n0", "best_of"])
async def test_multiple_completions_are_refused(tmp_path, monkeypatch, extra):
    llm = ScriptedLLM()
    async with gateway_with(
        tmp_path, monkeypatch, llm, budgets={"per_user": {"daily_tokens": 10_000}}
    ) as gateway:
        response = await ask(gateway, await gateway.token("anna@demo"), max_tokens=10, **extra)
        assert code(response) == (400, "unsupported_parameter")
        assert llm.calls == 0


@ALLOW
async def test_n_of_one_is_fine_and_both_caps_carry_the_held_value(tmp_path, monkeypatch):
    llm = ScriptedLLM()
    async with gateway_with(
        tmp_path, monkeypatch, llm, budgets={"per_user": {"daily_tokens": 10_000}}
    ) as gateway:
        response = await ask(
            gateway,
            await gateway.token("anna@demo"),
            n=1,
            max_tokens=10,
            max_completion_tokens=5000,
        )
        assert response.status_code == 200, response.text
        sent = llm.bodies[0]
        assert (sent["max_tokens"], sent["max_completion_tokens"]) == (10, 10)


# --------------------------------------------------------- 3. no usage keeps cost (P1)


async def test_an_answer_without_usage_keeps_the_estimated_token_cost(tmp_path, monkeypatch):
    llm = ScriptedLLM(omit_usage=True)
    async with gateway_with(
        tmp_path,
        monkeypatch,
        llm,
        budgets={"per_user": {"daily_cost_usd": 1.0}},
        pricing=TOKEN_PRICED,
    ) as gateway:
        response = await ask(gateway, await gateway.token("anna@demo"), max_tokens=10)
        assert response.status_code == 200, response.text
        spent = await usage(gateway)
        assert spent.tokens == 15  # 5 prompt (estimated) + 10 completion (the cap)
        assert spent.cost_nano_usd == 15_000_000  # 15 tokens at $1 per 1k


# ------------------------------------------------------- 4. failed settlement (P1)


@DENY
async def test_a_failed_token_settlement_blocks_then_lands(tmp_path, monkeypatch):
    llm = ScriptedLLM(usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25})
    flaky = FlakyStore(InMemoryBudgetStore())
    async with gateway_with(
        tmp_path, monkeypatch, llm, lambda: flaky, budgets={"per_user": {"daily_tokens": 30}}
    ) as gateway:
        token = await gateway.token("anna@demo")
        flaky.fail_settlements = True
        assert code(await ask(gateway, token, max_tokens=1)) == (200, "ok")  # held 6, used 25
        assert code(await ask(gateway, token, max_tokens=1)) == (503, "budget_settlement_pending")
        flaky.fail_settlements = False
        await gateway.container.budgets.drain()  # the retry lands
        assert (await usage(gateway)).tokens == 25
        assert code(await ask(gateway, token, max_tokens=1)) == (403, "budget_exceeded")
        assert llm.calls == 1


@DENY
async def test_a_long_call_whose_settlement_failed_cannot_be_followed(tmp_path, monkeypatch):
    """The codex reproduction: a 2 s call under a 1 s GPU budget, then a settlement error."""
    llm = ScriptedLLM(delay_s=2.0)
    flaky = FlakyStore(InMemoryBudgetStore())
    async with gateway_with(
        tmp_path, monkeypatch, llm, lambda: flaky, budgets={"per_session": {"gpu_seconds": 1}}
    ) as gateway:
        token = await gateway.token("anna@demo")
        flaky.fail_settlements = True
        await ask(gateway, token)
        assert code(await ask(gateway, token))[0] in {403, 503}  # refused while unsettled
        flaky.fail_settlements = False
        await gateway.container.budgets.drain()
        assert code(await ask(gateway, token)) == (403, "budget_exceeded")
        assert llm.calls == 1


# -------------------------------------------------------- 5. lost reserve reply (P2)


def llm_call() -> BudgetedCall:
    return BudgetedCall(
        session_id="s-1",
        principal="anna@demo",
        agent="databot",
        channel=Channel.LLM,
        model="qwen3:8b",
        payload={"model": "qwen3:8b", "messages": [{"role": "user", "content": "hi"}]},
        user_label="anna@demo",
        agent_label="databot",
    )


async def test_a_reservation_whose_reply_was_lost_is_released(snapshot_from, policy_doc):
    policy_doc["budgets"] = {"per_user": {"daily_tokens": 10_000}}
    flaky = FlakyStore(InMemoryBudgetStore())
    ledger = BudgetLedger(flaky, clock=MutableClock(T0))  # type: ignore[arg-type]
    flaky.lose_reserve_reply = True
    with pytest.raises(BudgetStoreUnavailableError):
        await ledger.reserve(llm_call(), snapshot_from(policy_doc))
    flaky.lose_reserve_reply = False
    await ledger.drain()
    assert (await flaky.inner.usage(scope())).tokens == 0


# ------------------------------------------------------------ 6. overflow (P2)


async def test_an_absurd_max_tokens_is_capped_and_writes_stay_whole(tmp_path, monkeypatch):
    llm = ScriptedLLM()
    client = fakeredis.FakeAsyncRedis()
    async with gateway_with(
        tmp_path,
        monkeypatch,
        llm,
        lambda: RedisBudgetStore(client),
        budgets={"per_session": {"gpu_seconds": 100}},  # nothing limits tokens or cost
        pricing=TOKEN_PRICED,
    ) as gateway:
        response = await ask(gateway, await gateway.token("anna@demo"), max_tokens=10**18)
        assert response.status_code == 200, response.text
        assert llm.bodies[0]["max_tokens"] == 32768  # limits.max_completion_tokens
        for key in await client.keys("acl:budget:*"):
            assert await client.ttl(key) > 0, key  # nothing written without an expiry


# ------------------------------------------------------- 7. session TTL (P2)


async def test_session_counters_follow_an_extended_lifetime(
    store_under_test: StoreUnderTest, snapshot_from, policy_doc
):
    policy_doc["budgets"] = {"per_session": {"gpu_seconds": 100}}
    policy_doc["sessions"] = {"idle_ttl_s": 3600, "max_lifetime_s": 3600}
    short = snapshot_from(policy_doc)
    policy_doc["sessions"] = {"idle_ttl_s": 3600, "max_lifetime_s": 7200}
    extended = snapshot_from(policy_doc)
    ledger = BudgetLedger(store_under_test.store, clock=store_under_test.clock)
    session = scope(ScopeKind.SESSION, "s-1", window="session")

    await ledger.settle(await ledger.reserve(llm_call(), short), None)
    await ledger.settle(await ledger.reserve(llm_call(), extended), None)  # after the reload
    ttl = await store_under_test.ttl(session.key)
    assert ttl is not None
    assert ttl >= 7200 + SESSION_TTL_SLACK_S - 10  # the new lifetime plus slack


# ------------------------------------------- 1b. every LLM call has a total deadline (P1)


def with_upstream_timeout(harness: Harness, seconds: float) -> None:
    document = yaml.safe_load(harness.policy_path.read_text())
    document["limits"]["upstream_timeout_s"] = seconds
    configure(harness, limits=document["limits"])


@pytest.mark.parametrize(
    "budgets",
    [
        {"per_user": {"daily_tokens": 10_000}},  # no GPU budget at all
        {"per_session": {"gpu_seconds": 100}},  # a GPU budget far above the timeout
    ],
    ids=["no-gpu-budget", "gpu-budget-above-timeout"],
)
async def test_a_trickling_upstream_cannot_outlast_the_upstream_timeout(
    tmp_path, monkeypatch, budgets
):
    """The codex reproduction: ~0.12 s of chunks 20 ms apart against a 50 ms timeout. httpx
    times each read, so only a total deadline stops it."""
    llm = ScriptedLLM(trickle_s=0.02)
    async with gateway_with(tmp_path, monkeypatch, llm, budgets=budgets) as gateway:
        with_upstream_timeout(gateway, 0.05)
        token = await gateway.token("anna@demo")
        started = time.perf_counter()
        response = await ask(gateway, token)
        assert code(response) == (502, "upstream_timeout")
        assert time.perf_counter() - started < 0.1
        if "per_session" in budgets:
            session = await usage(gateway, ScopeKind.SESSION, session_of(token))
            assert session.gpu_ms <= 50  # charged at most the 50 ms it was allowed
