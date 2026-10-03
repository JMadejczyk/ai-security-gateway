"""The `BudgetStore` contract, run against memory, fakeredis (Lua) and a real Redis 7.2.

(Not ``test_store.py``: tests/reload already has one, and test modules here share a namespace.)
"""

import asyncio
import secrets
import uuid
from datetime import UTC, datetime

import pytest
from budget_kit import StoreUnderTest, scope
from gateway_testkit import T0, MutableClock

from gateway.budget.ledger import BudgetLedger
from gateway.budget.model import (
    MAX_COUNTER,
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
GPU = frozenset({Meter.GPU})
CALLS = frozenset({Meter.TOOL_CALLS})


def op() -> str:
    return uuid.uuid4().hex


async def test_reserve_holds_on_every_scope(store_under_test: StoreUnderTest):
    store = store_under_test.store
    user, agent = scope(tokens=100), scope(ScopeKind.AGENT, "databot", tokens=1000)
    outcome = await store.reserve(op(), [user, agent], Spend(tokens=30, cost_nano_usd=5), TOKENS)
    assert outcome.granted
    assert outcome.usage == (Spend(tokens=30, cost_nano_usd=5),) * 2
    assert await store.usage(agent) == Spend(tokens=30, cost_nano_usd=5)


async def test_a_refusal_holds_nothing_anywhere(store_under_test: StoreUnderTest):
    store, op_id = store_under_test.store, op()
    roomy, tight = scope(tokens=1000), scope(ScopeKind.AGENT, "databot", tokens=10)
    outcome = await store.reserve(op_id, [roomy, tight], Spend(tokens=11), TOKENS)
    assert not outcome.granted
    assert (outcome.exceeded_scope, outcome.exceeded_meter) == (1, Meter.TOKENS)
    assert outcome.exceeded_limit([roomy, tight]) == "per_agent.daily_tokens"
    assert await store.usage(roomy) == Spend()
    assert (await store.reserve(op_id, [roomy], Spend(tokens=11), TOKENS)).granted  # id unused


async def test_exactly_reaching_the_limit_is_allowed_then_nothing_more(
    store_under_test: StoreUnderTest,
):
    store, user = store_under_test.store, scope(tokens=10)
    assert (await store.reserve(op(), [user], Spend(tokens=10), TOKENS)).granted
    assert not (await store.reserve(op(), [user], Spend(tokens=1), TOKENS)).granted


async def test_a_scope_at_its_limit_refuses_even_a_zero_hold(store_under_test: StoreUnderTest):
    store = store_under_test.store
    session = scope(ScopeKind.SESSION, "s-1", window="session", gpu_ms=1000)
    first = op()
    assert (await store.reserve(first, [session], Spend(gpu_ms=500), GPU)).granted
    await store.settle(first, [session], Spend(gpu_ms=1500))  # an overrun beyond the hold
    refused = await store.reserve(op(), [session], Spend(), GPU)
    assert not refused.granted
    assert refused.exceeded_limit([session]) == "per_session.gpu_seconds"


async def test_limits_of_meters_the_call_does_not_draw_on_are_ignored(
    store_under_test: StoreUnderTest,
):
    store = store_under_test.store
    session = scope(ScopeKind.SESSION, "s-1", window="session", tool_calls=1, gpu_ms=1)
    llm = op()
    await store.reserve(llm, [session], Spend(), frozenset())
    await store.settle(llm, [session], Spend(gpu_ms=5))  # GPU exhausted by an LLM call
    assert (await store.reserve(op(), [session], Spend(tool_calls=1), CALLS)).granted


async def test_concurrent_reservations_never_exceed_the_limit(store_under_test: StoreUnderTest):
    store, user = store_under_test.store, scope(tokens=50)
    outcomes = await asyncio.gather(
        *(store.reserve(op(), [user], Spend(tokens=1), TOKENS) for _ in range(100))
    )
    assert sum(outcome.granted for outcome in outcomes) == 50
    assert await store.usage(user) == Spend(tokens=50)


async def test_settle_reconciles_up_and_down_and_never_below_zero(
    store_under_test: StoreUnderTest,
):
    store, user = store_under_test.store, scope(tokens=100)
    up, down = op(), op()
    await store.reserve(up, [user], Spend(tokens=40, cost_nano_usd=10), TOKENS)
    assert await store.settle(up, [user], Spend(tokens=55, cost_nano_usd=13)) == (
        Spend(tokens=55, cost_nano_usd=13),
    )
    await store.reserve(down, [user], Spend(tokens=40), TOKENS)
    assert await store.settle(down, [user], Spend(tokens=5)) == (
        Spend(tokens=60, cost_nano_usd=13),
    )


async def test_settlement_is_idempotent_by_operation_id(store_under_test: StoreUnderTest):
    store, user, op_id = store_under_test.store, scope(tokens=100), op()
    await store.reserve(op_id, [user], Spend(tokens=10), TOKENS)
    await store.settle(op_id, [user], Spend(tokens=30))
    assert await store.settle(op_id, [user], Spend(tokens=30)) == (Spend(tokens=30),)  # a retry
    assert await store.usage(user) == Spend(tokens=30)


async def test_a_release_returns_the_whole_hold(store_under_test: StoreUnderTest):
    store, user, op_id = store_under_test.store, scope(tokens=100), op()
    await store.reserve(op_id, [user], Spend(tokens=100), TOKENS)
    await store.settle(op_id, [user], Spend())
    assert (await store.reserve(op(), [user], Spend(tokens=100), TOKENS)).granted


async def test_a_release_before_a_late_reservation_makes_it_refuse(
    store_under_test: StoreUnderTest,
):
    """A lost or slow reservation reply: the release's tombstone wins over a late arrival."""
    store, user, op_id = store_under_test.store, scope(tokens=100), op()
    await store.settle(op_id, [user], Spend())  # released before the reservation landed
    late = await store.reserve(op_id, [user], Spend(tokens=10), TOKENS)
    assert (late.granted, late.duplicate) == (False, True)
    assert await store.usage(user) == Spend()


async def test_an_operation_id_reserves_once(store_under_test: StoreUnderTest):
    store, user, op_id = store_under_test.store, scope(tokens=100), op()
    assert (await store.reserve(op_id, [user], Spend(tokens=10), TOKENS)).granted
    assert (await store.reserve(op_id, [user], Spend(tokens=10), TOKENS)).duplicate
    assert await store.usage(user) == Spend(tokens=10)


async def test_ttls_are_extended_by_every_write_and_never_shortened(
    store_under_test: StoreUnderTest,
):
    store = store_under_test.store
    short = scope(ScopeKind.SESSION, "s-1", window="session", ttl_s=3600, tool_calls=50)
    longer = short.model_copy(update={"ttl_s": 7200})  # after a reload lengthened sessions
    await store.reserve(op(), [short], Spend(tool_calls=1), CALLS)
    first = await store_under_test.ttl(short.key)
    assert first is not None
    assert 3590 <= first <= 3600
    await store.reserve(op(), [longer], Spend(tool_calls=1), CALLS)
    extended = await store_under_test.ttl(short.key)
    assert extended is not None
    assert extended >= 7190
    await store.reserve(op(), [short], Spend(tool_calls=1), CALLS)
    kept = await store_under_test.ttl(short.key)
    assert kept is not None
    assert kept >= 7180  # a shorter TTL never shortens a live counter


async def test_counters_expire_with_their_ttl(store_under_test: StoreUnderTest):
    store = store_under_test.store
    session = scope(ScopeKind.SESSION, "s-1", window="session", ttl_s=60, tool_calls=1)
    await store.reserve(op(), [session], Spend(tool_calls=1), CALLS)
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
    await store.reserve(op(), [service], Spend(tokens=7), TOKENS)
    assert await store.usage(lookalike) == Spend()


async def test_a_reservation_past_the_counter_bound_writes_nothing(
    store_under_test: StoreUnderTest,
):
    """No limit on tokens or cost here: only the overflow preflight stands in the way."""
    store = store_under_test.store
    session = scope(ScopeKind.SESSION, "s-1", window="session", gpu_ms=10**9)
    first = op()
    assert (await store.reserve(first, [session], Spend(tokens=1, gpu_ms=1), GPU)).granted
    huge = Spend(tokens=MAX_COUNTER, cost_nano_usd=MAX_COUNTER, gpu_ms=1)
    refused = await store.reserve(op(), [session], huge, GPU)
    assert (refused.granted, refused.exceeded_meter) == (False, Meter.TOKENS)
    assert await store.usage(session) == Spend(tokens=1, gpu_ms=1)


async def test_settlement_clamps_at_the_counter_bound(store_under_test: StoreUnderTest):
    store = store_under_test.store
    session = scope(ScopeKind.SESSION, "s-1", window="session", gpu_ms=10)
    earlier, op_id = op(), op()
    await store.reserve(earlier, [session], Spend(tokens=10), GPU)
    await store.settle(earlier, [session], Spend(tokens=10))
    await store.reserve(op_id, [session], Spend(tokens=5), GPU)
    after = await store.settle(op_id, [session], Spend(tokens=MAX_COUNTER, cost_nano_usd=7))
    assert after == (Spend(tokens=MAX_COUNTER, cost_nano_usd=7),)  # 10 + MAX - 5, clamped
    assert await store.usage(session) == after[0]


async def test_every_key_written_carries_a_ttl(store_under_test: StoreUnderTest):
    if store_under_test.redis is None:
        pytest.skip("Redis keys only")
    store, user, op_id = store_under_test.store, scope(tokens=100), op()
    await store.reserve(op_id, [user], Spend(tokens=10), TOKENS)
    await store.settle(op_id, [user], Spend(tokens=12))
    await store.settle(op(), [user], Spend())  # a release of an id never reserved
    keys = await store_under_test.redis.keys("acl:budget:*")
    assert len(keys) == 3
    for key in keys:
        assert await store_under_test.redis.ttl(key) > 0, key


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
        await store.reserve(op(), [scope(tokens=1)], Spend(tokens=1), TOKENS)
    with pytest.raises(BudgetStoreUnavailableError):
        await store.settle(op(), [scope(tokens=1)], Spend())
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
        await store.reserve(op(), [scope(tokens=1)], Spend(tokens=1), TOKENS)
    await store.aclose()
