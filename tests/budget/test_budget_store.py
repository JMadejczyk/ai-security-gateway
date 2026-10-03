"""The `BudgetStore` contract, run against memory, fakeredis (Lua) and a real Redis 7.2.

(Not ``test_store.py``: tests/reload already has one, and test modules here share a namespace.)
"""

import asyncio
import secrets
from datetime import UTC, datetime

import pytest
from budget_kit import StoreUnderTest, scope
from gateway_testkit import T0, MutableClock

from gateway.budget.ledger import BudgetLedger
from gateway.budget.model import (
    BudgetedCall,
    BudgetExceededError,
    BudgetStoreUnavailableError,
    Meter,
    ScopeKind,
    Spend,
)
from gateway.budget.redis_store import RedisBudgetStore
from gateway.core.types import Channel

TOKENS = frozenset({Meter.TOKENS})


async def test_reserve_holds_on_every_scope(store_under_test: StoreUnderTest):
    store = store_under_test.store
    user, agent = scope(tokens=100), scope(ScopeKind.AGENT, "databot", tokens=1000)
    outcome = await store.reserve([user, agent], Spend(tokens=30, cost_nano_usd=5), TOKENS)
    assert outcome.granted
    assert outcome.usage == (Spend(tokens=30, cost_nano_usd=5),) * 2
    assert await store.usage(agent) == Spend(tokens=30, cost_nano_usd=5)


async def test_a_refusal_holds_nothing_anywhere(store_under_test: StoreUnderTest):
    store = store_under_test.store
    roomy, tight = scope(tokens=1000), scope(ScopeKind.AGENT, "databot", tokens=10)
    outcome = await store.reserve([roomy, tight], Spend(tokens=11), TOKENS)
    assert not outcome.granted
    assert (outcome.exceeded_scope, outcome.exceeded_meter) == (1, Meter.TOKENS)
    assert outcome.exceeded_limit([roomy, tight]) == "per_agent.daily_tokens"
    assert await store.usage(roomy) == Spend()


async def test_exactly_reaching_the_limit_is_allowed_then_nothing_more(
    store_under_test: StoreUnderTest,
):
    store, user = store_under_test.store, scope(tokens=10)
    assert (await store.reserve([user], Spend(tokens=10), TOKENS)).granted
    assert not (await store.reserve([user], Spend(tokens=1), TOKENS)).granted


async def test_a_post_paid_meter_admits_only_while_headroom_remains(
    store_under_test: StoreUnderTest,
):
    """GPU time is held at zero; the call is admitted while usage is below the limit."""
    store, session = (
        store_under_test.store,
        scope(ScopeKind.SESSION, "s-1", window="session", gpu_ms=1000),
    )
    gpu = frozenset({Meter.GPU})
    assert (await store.reserve([session], Spend(), gpu)).granted
    await store.adjust([session], {Meter.GPU: 1500})  # the call overran the remaining headroom
    refused = await store.reserve([session], Spend(), gpu)
    assert not refused.granted
    assert refused.exceeded_limit([session]) == "per_session.gpu_seconds"


async def test_limits_of_meters_the_call_does_not_draw_on_are_ignored(
    store_under_test: StoreUnderTest,
):
    store, session = (
        store_under_test.store,
        scope(ScopeKind.SESSION, "s-1", window="session", tool_calls=1, gpu_ms=1),
    )
    await store.adjust([session], {Meter.GPU: 5})  # GPU exhausted by an LLM call
    assert (
        await store.reserve([session], Spend(tool_calls=1), frozenset({Meter.TOOL_CALLS}))
    ).granted


async def test_concurrent_reservations_never_exceed_the_limit(store_under_test: StoreUnderTest):
    store, user = store_under_test.store, scope(tokens=50)
    outcomes = await asyncio.gather(
        *(store.reserve([user], Spend(tokens=1), TOKENS) for _ in range(100))
    )
    assert sum(outcome.granted for outcome in outcomes) == 50
    assert await store.usage(user) == Spend(tokens=50)


async def test_adjust_reconciles_up_and_down_and_never_below_zero(
    store_under_test: StoreUnderTest,
):
    store, user = store_under_test.store, scope(tokens=100)
    await store.reserve([user], Spend(tokens=40, cost_nano_usd=10), TOKENS)
    assert await store.adjust([user], {Meter.TOKENS: -25, Meter.COST: 3}) == (
        Spend(tokens=15, cost_nano_usd=13),
    )
    assert await store.adjust([user], {Meter.TOKENS: 30}) == (Spend(tokens=45, cost_nano_usd=13),)
    assert await store.adjust([user], {Meter.TOKENS: -1000}) == (Spend(cost_nano_usd=13),)


async def test_release_on_failure_returns_the_whole_hold(store_under_test: StoreUnderTest):
    store, user = store_under_test.store, scope(tokens=100)
    await store.reserve([user], Spend(tokens=100), TOKENS)
    await store.adjust([user], {Meter.TOKENS: -100})
    assert (await store.reserve([user], Spend(tokens=100), TOKENS)).granted


async def test_ttl_is_set_once_on_creation_and_never_extended(store_under_test: StoreUnderTest):
    store, session = (
        store_under_test.store,
        scope(ScopeKind.SESSION, "s-1", window="session", ttl_s=86400, tool_calls=50),
    )
    await store.reserve([session], Spend(tool_calls=1), frozenset({Meter.TOOL_CALLS}))
    first = await store_under_test.ttl(session.key)
    assert first is not None
    assert 86390 <= first <= 86400
    store_under_test.clock.advance(100)
    if store_under_test.redis is not None:  # Redis TTLs follow the server clock: move it instead
        await store_under_test.redis.expire(session.key, 86300)
    await store.reserve([session], Spend(tool_calls=1), frozenset({Meter.TOOL_CALLS}))
    after = await store_under_test.ttl(session.key)
    assert after is not None
    assert after <= 86300


async def test_counters_expire_with_their_ttl(store_under_test: StoreUnderTest):
    store, session = (
        store_under_test.store,
        scope(ScopeKind.SESSION, "s-1", window="session", ttl_s=60, tool_calls=1),
    )
    await store.reserve([session], Spend(tool_calls=1), frozenset({Meter.TOOL_CALLS}))
    if store_under_test.redis is not None:
        await store_under_test.redis.pexpire(session.key, 1)
        await asyncio.sleep(0.05)
    else:
        store_under_test.clock.advance(61)
    assert await store.usage(session) == Spend()


async def test_a_subject_with_colons_cannot_collide_with_another_scope(
    store_under_test: StoreUnderTest,
):
    store = store_under_test.store
    service = scope(ScopeKind.USER, "svc:nightly_etl", tokens=10)
    lookalike = scope(ScopeKind.USER, "svc", window="2026-10-04", tokens=10)
    await store.reserve([service], Spend(tokens=7), TOKENS)
    assert await store.usage(lookalike) == Spend()


# --------------------------------------------------------------- daily rollover via the ledger


def llm_call(principal: str = "anna@demo", session_id: str = "s-1") -> BudgetedCall:
    return BudgetedCall(
        session_id=session_id,
        principal=principal,
        agent="databot",
        channel=Channel.LLM,
        model="qwen3:8b",
        payload={
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 9,
        },
        user_label=principal,
        agent_label="databot",
    )


async def test_daily_budgets_roll_over_at_midnight_utc(store_under_test, snapshot_from, policy_doc):
    policy_doc["budgets"] = {"per_user": {"daily_tokens": 10}}  # hi -> 1 token + max_tokens 9
    snapshot = snapshot_from(policy_doc)
    clock = store_under_test.clock
    clock.now = datetime(2026, 10, 4, 23, 59, 59, tzinfo=UTC)
    ledger = BudgetLedger(store_under_test.store, clock=clock)

    late = await ledger.reserve(llm_call(), snapshot)
    with pytest.raises(BudgetExceededError, match=r"per_user\.daily_tokens"):
        await ledger.reserve(llm_call(), snapshot)

    clock.now = datetime(2026, 10, 5, 0, 0, 0, tzinfo=UTC)
    await ledger.reserve(llm_call(), snapshot)  # a new day starts from zero
    await ledger.settle(late, None)  # settles against the day it was reserved on
    yesterday = scope(window="2026-10-04")
    assert (await store_under_test.store.usage(yesterday)).tokens == 0
    assert (await store_under_test.store.usage(scope(window="2026-10-05"))).tokens == 10


async def test_without_limits_on_its_meters_a_call_never_touches_the_store(
    snapshot_from, policy_doc
):
    policy_doc["budgets"] = {"per_session": {"tool_calls": 5}}  # nothing an LLM call draws on
    unreachable = RedisBudgetStore.from_url("redis://127.0.0.1:1/0", password=None, timeout_s=0.2)
    ledger = BudgetLedger(unreachable, clock=MutableClock(T0))
    reservation = await ledger.reserve(llm_call(), snapshot_from(policy_doc))
    assert reservation.scopes == ()
    await ledger.settle(reservation, None)


# ------------------------------------------------------------------------- unavailability


async def test_an_unreachable_redis_fails_closed(snapshot_from, policy_doc):
    store = RedisBudgetStore.from_url("redis://127.0.0.1:1/0", password=None, timeout_s=0.2)
    assert not await store.healthy()
    with pytest.raises(BudgetStoreUnavailableError):
        await store.reserve([scope(tokens=1)], Spend(tokens=1), TOKENS)
    with pytest.raises(BudgetStoreUnavailableError):
        await store.adjust([scope(tokens=1)], {Meter.TOKENS: -1})
    ledger = BudgetLedger(store, clock=MutableClock(T0))
    with pytest.raises(BudgetStoreUnavailableError) as refused:
        await ledger.reserve(llm_call(), snapshot_from(policy_doc))
    assert refused.value.status_code == 503
    assert await ledger.status() == "down"


@pytest.mark.redis
async def test_a_wrong_redis_password_fails_closed(real_redis):
    store = RedisBudgetStore.from_url(
        real_redis.url, password=secrets.token_urlsafe(16), timeout_s=1
    )
    assert not await store.healthy()
    with pytest.raises(BudgetStoreUnavailableError):
        await store.reserve([scope(tokens=1)], Spend(tokens=1), TOKENS)
    await store.aclose()


async def test_counters_first_created_by_a_settlement_get_their_ttl(
    store_under_test: StoreUnderTest,
):
    session = scope(ScopeKind.SESSION, "s-2", window="session", ttl_s=600, gpu_ms=10)
    await store_under_test.store.adjust([session], {Meter.GPU: 5})
    ttl = await store_under_test.ttl(session.key)
    assert ttl is not None
    assert 590 <= ttl <= 600
