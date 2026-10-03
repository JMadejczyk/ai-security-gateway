"""The tool quarantine store: one contract for memory, Redis over fakeredis and real Redis."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import fakeredis
import pytest
from redis.asyncio import Redis

from gateway.controls.tool_quarantine import (
    InMemoryToolQuarantine,
    QuarantineEntry,
    RedisToolQuarantine,
    ToolQuarantine,
    ToolQuarantineUnavailableError,
)

DEAD_REDIS = "redis://127.0.0.1:1/0"  # nothing listens on port 1
T0 = datetime(2026, 10, 4, 10, 0, tzinfo=UTC)


def entry(tool: str = "write_report", advertised: str = "b" * 64, pin: str = "a" * 64):
    return QuarantineEntry(
        tool=tool,
        pin_digest=pin,
        advertised_digest=advertised,
        reason="tool_pin_mismatch",
        detected_at=T0,
    )


@pytest.fixture(params=["memory", "fakeredis", pytest.param("redis", marks=pytest.mark.redis)])
async def quarantine(request) -> AsyncIterator[ToolQuarantine]:
    match request.param:
        case "memory":
            yield InMemoryToolQuarantine()
        case "fakeredis":
            client = fakeredis.FakeAsyncRedis()
            yield RedisToolQuarantine(client)
            await client.aclose()
        case _:
            server = request.getfixturevalue("real_redis")
            client = Redis.from_url(server.url.rsplit("/", 1)[0] + "/5", password=server.password)
            await client.flushdb()
            yield RedisToolQuarantine(client)
            await client.flushdb()
            await client.aclose()


async def test_record_keeps_the_first_detection_and_clear_lifts_it(quarantine):
    assert await quarantine.entries("reports") == {}
    await quarantine.record("reports", entry())
    await quarantine.record("reports", entry(advertised="c" * 64))  # a later drift, same baseline
    assert await quarantine.entries("reports") == {"write_report": entry()}
    assert await quarantine.entries("web") == {}  # per server
    assert await quarantine.clear("reports", "write_report")
    assert not await quarantine.clear("reports", "write_report")
    assert await quarantine.entries("reports") == {}


async def test_record_replaces_an_entry_of_an_obsolete_baseline(quarantine):
    """Codex P1 (regression): a drift against the new baseline B must replace the stale
    entry for A atomically, not be dropped while A is kept and then cleaned up."""
    await quarantine.record("reports", entry(pin="a" * 64))
    against_b = entry(pin="d" * 64, advertised="c" * 64)
    await quarantine.record("reports", against_b)
    assert await quarantine.entries("reports") == {"write_report": against_b}


async def test_discard_deletes_only_the_exact_entry_inspected(quarantine):
    stale = entry(pin="a" * 64)
    await quarantine.record("reports", stale)
    [inspected] = (await quarantine.entries("reports")).values()
    assert inspected == stale
    fresh = entry(pin="d" * 64)
    await quarantine.record("reports", fresh)  # another gateway records meanwhile
    assert not await quarantine.discard("reports", inspected)  # the fresh one survives
    assert await quarantine.entries("reports") == {"write_report": fresh}
    assert await quarantine.discard("reports", fresh)
    assert await quarantine.entries("reports") == {}


async def test_redis_down_fails_closed():
    client = Redis.from_url(DEAD_REDIS, socket_connect_timeout=0.2, socket_timeout=0.2)
    store = RedisToolQuarantine(client)
    for operation in (
        store.entries("reports"),
        store.record("reports", entry()),
        store.discard("reports", entry()),
        store.clear("reports", "write_report"),
    ):
        with pytest.raises(ToolQuarantineUnavailableError):
            await operation
    await client.aclose()
