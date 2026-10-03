"""The `SessionStore` contract: binding, refresh, decay, clamp, sticky taint, lifetimes and
serialization, against every store (``backend`` in conftest: memory, fakeredis, real Redis)."""

import asyncio
from datetime import timedelta

import pytest

from gateway.core.envelope import Cooldown
from gateway.core.types import SessionMode
from gateway.policy.schema import Sessions
from gateway.sessions import (
    SessionBinding,
    SessionError,
    SessionReason,
    SessionUpdate,
)

AUTHN_DENY = pytest.mark.control("authn", "deny")

ANNA = SessionBinding(principal="anna@demo", actor="databot", mode=SessionMode.INTERACTIVE)
LIMITS = Sessions(idle_ttl_s=3600, max_lifetime_s=86400)
HALF_LIFE = 600.0


async def open_anna(store, session_id="s-1", limits=LIMITS):
    return await store.open(session_id, ANNA, limits)


async def bump(store, delta: float = 0.0, *, taint: bool = False, session_id: str = "s-1"):
    update = SessionUpdate(risk_delta=delta, taint=taint)
    return await store.apply(session_id, update, half_life_s=HALF_LIFE)


@AUTHN_DENY
@pytest.mark.parametrize(
    "other",
    [
        SessionBinding(principal="bartek@demo", actor="databot", mode=SessionMode.INTERACTIVE),
        SessionBinding(principal="anna@demo", actor="reportbot", mode=SessionMode.INTERACTIVE),
        SessionBinding(principal="anna@demo", actor="databot", mode=SessionMode.AUTONOMOUS),
    ],
    ids=["principal", "agent", "mode"],
)
async def test_binding_mismatch_rejected(store, other):
    await open_anna(store)
    with pytest.raises(SessionError) as caught:
        await store.open("s-1", other, LIMITS)
    assert (caught.value.reason, caught.value.status_code) == (SessionReason.BINDING_MISMATCH, 403)


async def test_refresh_keeps_taint_and_risk(store):
    await open_anna(store)
    await bump(store, 0.4, taint=True)
    # A new token with the same binding reopens the same session.
    reopened = await open_anna(store)
    assert reopened.taint
    assert reopened.risk == pytest.approx(0.4)


async def test_risk_decays_by_half_life(store, clock):
    await open_anna(store)
    await bump(store, 0.8)
    clock.advance(HALF_LIFE)
    after = await bump(store, 0.0)
    assert after.risk == pytest.approx(0.4)
    clock.advance(2 * HALF_LIFE)
    assert (await bump(store, 0.0)).risk == pytest.approx(0.1)


async def test_decayed_risk_plus_delta(store, clock):
    await open_anna(store)
    await bump(store, 0.6)
    clock.advance(HALF_LIFE)
    assert (await bump(store, 0.1)).risk == pytest.approx(0.4)


async def test_risk_is_clamped_to_one(store):
    await open_anna(store)
    await bump(store, 0.7)
    assert (await bump(store, 0.9)).risk == 1.0


async def test_taint_survives_decay(store, clock):
    await open_anna(store)
    await bump(store, 0.6, taint=True)
    clock.advance(100 * HALF_LIFE)
    state = await bump(store, 0.0)
    assert state.risk == pytest.approx(0.0, abs=1e-9)
    assert state.taint


async def test_timers_are_kept_until_they_expire(store, clock):
    await open_anna(store)
    until = clock() + timedelta(seconds=300)
    cooldown = Cooldown(key="write:fs:reports/q3.md", until=until)
    update = SessionUpdate(freeze_until=until, cooldowns=(cooldown,))
    state = await store.apply("s-1", update, half_life_s=HALF_LIFE)
    assert (state.freeze_until, state.cooldowns) == (until, (cooldown,))
    clock.advance(301)
    state = await bump(store)
    assert (state.freeze_until, state.cooldowns) == (None, ())


@AUTHN_DENY
async def test_idle_ttl_ends_the_session(store, clock):
    limits = Sessions(idle_ttl_s=60, max_lifetime_s=86400)
    await open_anna(store, limits=limits)
    clock.advance(59)
    await open_anna(store, limits=limits)  # activity resets the idle timer
    clock.advance(60)
    with pytest.raises(SessionError) as caught:
        await open_anna(store, limits=limits)
    assert (caught.value.reason, caught.value.status_code) == (SessionReason.ENDED, 401)


@AUTHN_DENY
async def test_max_lifetime_ends_an_active_session(store, clock):
    limits = Sessions(idle_ttl_s=60, max_lifetime_s=120)
    await open_anna(store, limits=limits)
    for _ in range(3):
        clock.advance(39)
        await open_anna(store, limits=limits)  # never idle, 117 s old
    clock.advance(3)
    with pytest.raises(SessionError, match="session_ended"):
        await open_anna(store, limits=limits)


@AUTHN_DENY
async def test_ended_session_is_never_revived(store):
    await open_anna(store)
    await bump(store, 0.5, taint=True)
    await store.end("s-1")
    assert await store.get("s-1") is None
    with pytest.raises(SessionError, match="session_ended"):
        await open_anna(store)
    assert (await open_anna(store, "s-2")).risk == 0.0


async def test_tainted_count(store):
    await open_anna(store, "s-1")
    await open_anna(store, "s-2")
    await bump(store, taint=True, session_id="s-1")
    assert await store.tainted_count() == 1
    await store.end("s-1")
    assert await store.tainted_count() == 0


async def test_lock_serializes_calls_in_one_session(store):
    events: list[str] = []
    first_inside = asyncio.Event()

    async def call(name: str, signal: asyncio.Event | None = None) -> None:
        async with store.lock("s-1"):
            events.append(f"{name}:enter")
            if signal is not None:
                signal.set()
            await asyncio.sleep(0.01)  # yield while holding the lock
            events.append(f"{name}:exit")

    first = asyncio.create_task(call("a", first_inside))
    await first_inside.wait()
    second = asyncio.create_task(call("b"))
    await asyncio.gather(first, second)
    assert events == ["a:enter", "a:exit", "b:enter", "b:exit"]


async def test_lock_does_not_serialize_different_sessions(store):
    events: list[str] = []
    a_inside = asyncio.Event()
    release_a = asyncio.Event()

    async def hold_a() -> None:
        async with store.lock("s-a"):
            a_inside.set()
            await release_a.wait()

    task = asyncio.create_task(hold_a())
    await a_inside.wait()
    async with store.lock("s-b"):
        events.append("b ran while a held its lock")
    release_a.set()
    await task
    assert events == ["b ran while a held its lock"]


@AUTHN_DENY
async def test_retired_ids_stay_retired_past_every_lifetime(store, clock):
    await open_anna(store)
    await bump(store, 0.5, taint=True)
    await store.end("s-1")
    clock.advance(LIMITS.max_lifetime_s + 3600 + 1)  # past the session and token lifetimes
    await open_anna(store, "s-other")  # any activity that would prune old tombstones
    assert await store.is_retired("s-1")
    with pytest.raises(SessionError, match="session_ended"):
        await open_anna(store)


async def test_expired_ids_are_retired_too(store, clock):
    limits = Sessions(idle_ttl_s=60, max_lifetime_s=120)
    await open_anna(store, limits=limits)
    assert not await store.is_retired("s-1")
    clock.advance(10_000)
    assert await store.is_retired("s-1")
    with pytest.raises(SessionError, match="session_ended"):
        await open_anna(store, limits=limits)


async def test_a_session_in_use_is_not_retired_under_its_call(store, clock):
    limits = Sessions(idle_ttl_s=60, max_lifetime_s=86400)
    await open_anna(store, limits=limits)
    async with store.lock("s-1"):  # a call of s-1 is in flight
        clock.advance(61)
        await open_anna(store, "s-2", limits=limits)  # another session's call retires stale ones
        state = await bump(store, 0.3, taint=True)  # the in-flight call persists its state
    assert state.taint
    reopened = await open_anna(store, limits=limits)
    assert (reopened.taint, reopened.risk) == (True, pytest.approx(0.3))


async def test_an_idle_session_nobody_uses_is_still_retired(store, clock):
    limits = Sessions(idle_ttl_s=60, max_lifetime_s=86400)
    await open_anna(store, limits=limits)
    clock.advance(61)
    await open_anna(store, "s-2", limits=limits)
    assert await store.get("s-1") is None
