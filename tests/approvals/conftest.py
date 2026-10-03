"""Fixtures for the approval suites: every approval store behind one contract, and the MCP
stack (real in-process MCP servers) the demo-step-3 flow runs against."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import fakeredis
import pytest
from approvals_kit import StoreUnderTest, mcp_harness, pin_kit, upstreams
from gateway_testkit import MutableClock, running_gateway
from redis.asyncio import Redis
from redis_kit import RealRedis

from gateway.approvals.redis_store import RedisApprovalStore
from gateway.approvals.store import InMemoryApprovalStore


@pytest.fixture(params=["memory", "fakeredis", pytest.param("redis", marks=pytest.mark.redis)])
async def store_under_test(request: pytest.FixtureRequest) -> AsyncIterator[StoreUnderTest]:
    clock = MutableClock()
    match request.param:
        case "memory":
            yield StoreUnderTest("memory", InMemoryApprovalStore(), clock)
        case "fakeredis":
            client = fakeredis.FakeAsyncRedis()
            yield StoreUnderTest("fakeredis", RedisApprovalStore(client), clock)
            await client.aclose()
        case _:
            server: RealRedis = request.getfixturevalue("real_redis")
            client = Redis.from_url(server.url, password=server.password)  # pyright: ignore[reportUnknownMemberType]
            store = RedisApprovalStore(client)
            for _ in range(100):
                if await store.healthy():
                    break
                await asyncio.sleep(0.1)
            else:
                pytest.fail("the redis container never answered")
            await client.flushdb()
            yield StoreUnderTest("redis", store, clock)
            await client.aclose()


@pytest.fixture
async def stack(tmp_path: Path) -> AsyncIterator[Any]:
    """As in tests/mcp: real upstreams, their tools pinned (``require_pin``), one gateway."""
    async with upstreams.running_upstreams() as (transport, log):
        pin_kit.write_pins(tmp_path / "pins", await pin_kit.capture_pins(transport))
        async with running_gateway(tmp_path, transport=transport) as gateway:
            yield mcp_harness.MCPStack(gateway, transport, log)
