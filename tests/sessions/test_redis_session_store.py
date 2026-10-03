"""What sharing session state through Redis adds to the `SessionStore` contract.

Two `RedisSessionStore` instances over one Redis are two gateways (or one gateway before and
after a restart): calls of one session serialize across them, a lock whose holder died
expires, taint and tombstones survive the instance that wrote them, concurrent writes never
lose an update, every field of `SessionContext` round-trips, and a Redis that is down or
holds an unreadable record refuses the call (fail closed).
"""

import asyncio
from pathlib import Path

import pytest
from gateway_testkit import bearer, chat, claims, running_gateway, sign
from redis.asyncio import Redis

from gateway.core.envelope import FlaggedToolCall
from gateway.core.types import SessionMode
from gateway.policy.schema import Sessions
from gateway.redis_sessions import (
    LOCK_TTL_S,
    RedisSessionLock,
    RedisSessionStore,
    SessionBusyError,
    SessionStoreUnavailableError,
    _LocalLocks,
    tombstone_ttl_s,
)
from gateway.sessions import SessionBinding, SessionError, SessionUpdate

ANNA = SessionBinding(principal="anna@demo", actor="databot", mode=SessionMode.INTERACTIVE)
LIMITS = Sessions(idle_ttl_s=3600, max_lifetime_s=86400)
HALF_LIFE = 600.0
DEAD_REDIS = "redis://127.0.0.1:1/0"  # nothing listens on port 1


@pytest.fixture
def shared(backend):
    if not backend.shared:
        pytest.skip("the in-memory store serves one process")
    return backend


async def test_calls_of_one_session_serialize_across_gateways(shared):
    one, two = shared.store(), shared.store()
    events: list[str] = []
    one_inside = asyncio.Event()

    async def call(store, name: str, signal: asyncio.Event | None = None) -> None:
        async with store.lock("s-1"):
            events.append(f"{name}:enter")
            if signal is not None:
                signal.set()
            await asyncio.sleep(0.05)
            events.append(f"{name}:exit")

    first = asyncio.create_task(call(one, "gw1", one_inside))
    await one_inside.wait()
    await asyncio.gather(first, call(two, "gw2"))
    assert events == ["gw1:enter", "gw1:exit", "gw2:enter", "gw2:exit"]


async def test_a_busy_session_is_refused_after_the_bounded_wait(shared):
    holder, waiter = shared.store(), shared.store(lock_wait_s=0.2)
    async with holder.lock("s-1"):
        with pytest.raises(SessionBusyError) as caught:
            async with waiter.lock("s-1"):
                pytest.fail("entered a lock another gateway holds")
    assert (caught.value.status_code, caught.value.reason_code) == (429, "session_busy")
    async with waiter.lock("s-1"):  # free again once released
        pass


async def test_a_held_lock_is_renewed_past_its_ttl(shared):
    holder = shared.store(lock_ttl_s=0.3)
    waiter = shared.store(lock_wait_s=0.1)
    async with holder.lock("s-1"):
        await asyncio.sleep(1.0)  # > 3 TTLs: only renewal keeps it
        with pytest.raises(SessionBusyError):
            async with waiter.lock("s-1"):
                pass


async def test_a_crashed_holder_s_lock_expires(shared):
    """A gateway that dies mid-call never releases; the TTL frees the session."""
    lock = RedisSessionLock(
        shared.client, "acl:session:s-1:lock", _LocalLocks(), wait_s=1, ttl_s=0.3
    )
    await lock.__aenter__()
    assert lock._renewer is not None
    lock._renewer.cancel()  # the process is gone: no renewal, no release
    waiter = shared.store(lock_wait_s=2.0)
    started = asyncio.get_running_loop().time()
    async with waiter.lock("s-1"):
        waited = asyncio.get_running_loop().time() - started
    assert 0.1 < waited < 1.5


async def test_release_never_deletes_another_holder_s_lock(shared):
    """A holder whose lock expired must not free the lock its successor now holds."""
    stale = RedisSessionLock(
        shared.client, "acl:session:s-1:lock", _LocalLocks(), wait_s=1, ttl_s=0.2
    )
    await stale.__aenter__()
    assert stale._renewer is not None
    stale._renewer.cancel()
    await asyncio.sleep(0.3)  # expired
    successor = shared.store(lock_wait_s=0.1)
    async with successor.lock("s-1"):
        await stale.__aexit__(None, None, None)  # late release by the stale holder
        with pytest.raises(SessionBusyError):
            async with shared.store(lock_wait_s=0.1).lock("s-1"):
                pass


async def test_taint_survives_a_gateway_restart(shared):
    before = shared.store()
    async with before.lock("s-1"):
        await before.open("s-1", ANNA, LIMITS)
        await before.apply("s-1", SessionUpdate(risk_delta=0.4, taint=True), half_life_s=HALF_LIFE)
    after = shared.store()  # a new process: nothing in memory
    async with after.lock("s-1"):
        state = await after.open("s-1", ANNA, LIMITS)
    assert (state.taint, state.risk) == (True, pytest.approx(0.4))
    assert await after.tainted_count() == 1


async def test_an_ended_session_never_revives_on_another_gateway(shared):
    one, two = shared.store(), shared.store()
    await one.open("s-1", ANNA, LIMITS)
    await one.apply("s-1", SessionUpdate(taint=True), half_life_s=HALF_LIFE)
    await one.end("s-1")
    assert await two.is_retired("s-1")
    with pytest.raises(SessionError, match="session_ended"):
        await two.open("s-1", ANNA, LIMITS)
    with pytest.raises(SessionError, match="session_ended"):  # nor through a late write
        await two.apply("s-1", SessionUpdate(risk_delta=0.1), half_life_s=HALF_LIFE)
    assert await two.get("s-1") is None


async def test_tombstones_outlive_every_session_and_token(shared):
    store = shared.store()
    await store.open("s-1", ANNA, LIMITS)
    await store.end("s-1")
    ttl_ms = await shared.client.pttl("acl:session:s-1:ended")
    assert ttl_ms >= (LIMITS.max_lifetime_s + 3600) * 1000
    assert tombstone_ttl_s(LIMITS) * 1000 >= ttl_ms > 0


async def test_concurrent_writes_never_lose_an_update(shared):
    """Without the lock (it expired, or a bug), compare-and-set still merges every delta."""
    stores = [shared.store() for _ in range(4)]
    await stores[0].open("s-1", ANNA, LIMITS)
    update = SessionUpdate(risk_delta=0.05)
    await asyncio.gather(
        *(s.apply("s-1", update, half_life_s=1e9) for s in stores for _ in range(5))
    )
    state = await stores[0].get("s-1")
    assert state is not None
    assert state.risk == pytest.approx(1.0)  # 20 x 0.05, clamped at 1


async def test_every_session_field_round_trips(shared):
    store = shared.store()
    await store.open("s-1", ANNA, LIMITS)
    flag = FlaggedToolCall(tool="write_report", args_digest="ab" * 32)
    await store.apply(
        "s-1",
        SessionUpdate(goal="count customers", flagged_tool_calls=(flag,)),
        half_life_s=HALF_LIFE,
    )
    state = await shared.store().get("s-1")
    assert state is not None
    assert (state.goal, state.flagged_tool_calls) == ("count customers", (flag,))


async def test_an_unreadable_record_fails_closed(shared):
    store = shared.store()
    await store.open("s-1", ANNA, LIMITS)
    await shared.client.hset("acl:session:s-1", "doc", "{not json")
    with pytest.raises(SessionStoreUnavailableError):
        await store.open("s-1", ANNA, LIMITS)


async def test_redis_down_fails_closed():
    client = Redis.from_url(DEAD_REDIS, socket_connect_timeout=0.2, socket_timeout=0.2)  # pyright: ignore[reportUnknownMemberType]
    store = RedisSessionStore(client)
    with pytest.raises(SessionStoreUnavailableError) as caught:
        async with store.lock("s-1"):
            pass
    assert (caught.value.status_code, caught.value.reason_code) == (
        503,
        "session_store_unavailable",
    )
    for check in (store.is_retired("s-1"), store.get("s-1"), store.tainted_count()):
        with pytest.raises(SessionStoreUnavailableError):
            await check
    await client.aclose()


async def test_gateway_answers_503_when_the_session_store_is_down(tmp_path: Path):
    async with running_gateway(tmp_path, session_store="redis", redis_url=DEAD_REDIS) as gateway:
        # The demo issuer checks retirement in the store too: sign a token directly.
        token = sign(claims(gateway.clock))
        response = await gateway.agent.post(
            "/v1/chat/completions", json=chat(), headers=bearer(token)
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "session_store_unavailable"


def test_lock_ttl_is_long_enough_to_renew():
    assert LOCK_TTL_S >= 3  # renewed every TTL/3: a slow event loop must not lose it
