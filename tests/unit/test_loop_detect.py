"""``loop_detect`` on its own: fingerprints, the sliding window, approval retries, modes.

The window tests run against every `CallCounter`: in-memory, Redis over fakeredis, and a real
Redis (marker ``redis``); a counter that cannot count fails closed."""

from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import fakeredis
import pytest
from gateway_testkit import MutableClock
from redis.asyncio import Redis

from gateway.controls.loop_counter_redis import RedisCallCounter
from gateway.controls.loop_detect import (
    STORE_UNAVAILABLE,
    SWEEP_EVERY,
    CallCounter,
    CallFingerprint,
    InMemoryCallCounter,
    LoopDetectControl,
)
from gateway.controls.scope import CallScope, call_scope
from gateway.core.envelope import Interaction
from gateway.core.types import Action, Channel, ControlMode, Decision, SessionMode, Stage
from gateway.policy.evaluator import PrincipalContext
from gateway.policy.schema import LoopDetectConfig

BLOCK = LoopDetectConfig(mode=ControlMode.BLOCK, max_repeats=5, window_s=60)
LOG_ONLY = LoopDetectConfig(mode=ControlMode.LOG_ONLY, max_repeats=5, window_s=60)
DEAD_REDIS = "redis://127.0.0.1:1/0"  # nothing listens on port 1
ALLOWED = pytest.mark.control("loop_detect", "allow")
DENIED = pytest.mark.control("loop_detect", "deny")


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def memory_counter() -> InMemoryCallCounter:
    return InMemoryCallCounter()


@pytest.fixture(params=["memory", "fakeredis", pytest.param("redis", marks=pytest.mark.redis)])
async def counter(request, memory_counter) -> AsyncIterator[CallCounter]:
    match request.param:
        case "memory":
            yield memory_counter
        case "fakeredis":
            client = fakeredis.FakeAsyncRedis()
            yield RedisCallCounter(client)
            await client.aclose()
        case _:
            server = request.getfixturevalue("real_redis")
            client = Redis.from_url(server.url.rsplit("/", 1)[0] + "/4", password=server.password)
            await client.flushdb()
            yield RedisCallCounter(client)
            await client.flushdb()
            await client.aclose()


@pytest.fixture
def control(counter, clock) -> LoopDetectControl:
    return LoopDetectControl(counter, clock=clock)


@pytest.fixture
def interaction(make_ctx):
    def build(
        payload: Any = None,
        *,
        session_id: str = "s-test",
        channel: Channel = Channel.MCP,
        resource: str = "fs:reports/q3.md",
        server: str | None = "reports",
    ) -> Interaction:
        return Interaction(
            session_id=session_id,
            principal="anna@demo",
            actor="databot",
            mode=SessionMode.INTERACTIVE,
            channel=channel,
            action=Action.WRITE,
            resource=resource,
            payload={"name": "write_report", "arguments": {"name": "q3.md"}}
            if payload is None
            else payload,
            context=make_ctx(session_id=session_id),
            server=server,
        )

    return build


async def decisions(control, interaction: Interaction, times: int, cfg=BLOCK) -> list[str]:
    return [(await control.evaluate(interaction, Stage.PRE, cfg)).reason_code for _ in range(times)]


# ------------------------------------------------------------------------- fingerprint


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        pytest.param({"x": 1, "y": 2}, {"y": 2, "x": 1}, True, id="key-order"),
        pytest.param(
            {"model": "m", "messages": [], "stream": True},
            {"model": "m", "messages": [], "stream": False, "user": "u"},
            True,
            id="volatile-fields",
        ),
        pytest.param({"arguments": {"name": "a"}}, {"arguments": {"name": "b"}}, False, id="args"),
        pytest.param({"arguments": {"n": 1}}, {"arguments": {"n": "1"}}, False, id="types"),
    ],
)
def test_fingerprint_canonicalizes_payloads(interaction, a, b, same):
    assert (CallFingerprint.of(interaction(a)) == CallFingerprint.of(interaction(b))) is same


def test_fingerprint_covers_route_and_resource(interaction):
    base = CallFingerprint.of(interaction())
    assert base != CallFingerprint.of(interaction(resource="fs:reports/q4.md"))
    assert base != CallFingerprint.of(interaction(server="other"))
    assert len(base.digest) == 64


# ----------------------------------------------------------------------------- control


@ALLOWED
@pytest.mark.parametrize(
    ("cfg", "enforced"),
    [pytest.param(BLOCK, True, marks=DENIED), (LOG_ONLY, False)],
    ids=["block", "log_only"],
)
async def test_more_than_max_repeats_is_a_loop(control, interaction, cfg, enforced):
    call = interaction()
    assert await decisions(control, call, 5, cfg) == ["no_loop"] * 5
    verdict = await control.evaluate(call, Stage.PRE, cfg)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "loop_detected")
    assert verdict.enforced is enforced
    assert verdict.reason == "6 identical calls within 60s"


@ALLOWED
@DENIED
async def test_a_different_call_has_its_own_count(control, interaction):
    await decisions(control, interaction(), 6)
    other = interaction({"name": "write_report", "arguments": {"name": "q4.md"}})
    assert await decisions(control, other, 1) == ["no_loop"]
    assert await decisions(control, interaction(), 1) == ["loop_detected"]


@ALLOWED
async def test_sessions_are_counted_separately(control, interaction):
    await decisions(control, interaction(), 6)
    assert await decisions(control, interaction(session_id="s-other"), 1) == ["no_loop"]


@ALLOWED
@DENIED
async def test_the_window_slides(control, interaction, clock):
    call = interaction()
    await decisions(control, call, 5)
    clock.advance(30)
    assert await decisions(control, call, 1) == ["loop_detected"]  # 6 within 60 s
    clock.advance(30)  # the first five left the window; the 6th stays until t=90
    assert await decisions(control, call, 4) == ["no_loop"] * 4
    assert await decisions(control, call, 1) == ["loop_detected"]


@ALLOWED
async def test_approval_retries_are_neither_counted_nor_blocked(
    control, interaction, snapshot, counter
):
    call = interaction()
    await decisions(control, call, 6)
    principal = PrincipalContext(
        principal="anna@demo", roles=("analyst",), agent="databot", mode=SessionMode.INTERACTIVE
    )
    with call_scope(CallScope(snapshot=snapshot, principal=principal, approval_id="apr-1")):
        assert await decisions(control, call, 3) == ["approval_retry"] * 3
    count = await counter.hit("s-test", CallFingerprint.of(call).key, MutableClock()(), 60)
    assert count == 7  # the three retries were not recorded


@DENIED
async def test_redis_down_fails_closed(interaction):
    client = Redis.from_url(DEAD_REDIS, socket_connect_timeout=0.2, socket_timeout=0.2)
    control = LoopDetectControl(RedisCallCounter(client))
    verdict = await control.evaluate(interaction(), Stage.PRE, BLOCK)
    assert (verdict.decision, verdict.reason_code, verdict.enforced) == (
        Decision.BLOCK,
        STORE_UNAVAILABLE,
        True,
    )
    await client.aclose()


async def test_counter_sweeps_expired_sessions(memory_counter, clock):
    counter = memory_counter
    for i in range(SWEEP_EVERY - 1):
        await counter.hit(f"s-{i}", "k", clock(), 60)
    assert len(counter) == SWEEP_EVERY - 1
    clock.now += timedelta(seconds=61)
    await counter.hit("s-new", "k", clock(), 60)  # the SWEEP_EVERY-th hit sweeps
    assert len(counter) == 1


async def test_memory_is_bounded_by_the_window(memory_counter, clock):
    """A session making 2048 distinct calls, one a second, holds only the last window's."""
    counter = memory_counter
    for i in range(2 * SWEEP_EVERY):
        await counter.hit("s-busy", f"call-{i}", clock(), 60)
        clock.advance(1)
        fingerprints, queued = counter.held("s-busy")
        assert fingerprints <= 61
        assert queued <= 61
    clock.advance(61)
    await counter.hit("s-other", "k", clock(), 60)
    for _ in range(SWEEP_EVERY):  # reach the next sweep without touching s-busy
        await counter.hit("s-other", "k", clock(), 60)
    assert counter.held("s-busy") == (0, 0)
