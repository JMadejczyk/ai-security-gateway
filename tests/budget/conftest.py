"""Fixtures for the budget suites: every store behind one contract, a real Redis, a scripted LLM.

Store backends:

- ``memory``: `InMemoryBudgetStore` on the movable test clock.
- ``fakeredis``: `RedisBudgetStore` over fakeredis with Lua (lupa), so the Lua scripts run in
  every ``make test``.
- ``redis``: `RedisBudgetStore` over a real ``redis:7.2.16`` (marker ``redis``). The session
  starts a throwaway container on a random loopback port when docker is usable (or uses
  ``ACL_TEST_REDIS_URL``), and the tests skip when neither is available.
"""

import asyncio
import contextlib
import os
import secrets
import shutil
import subprocess
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import fakeredis
import pytest
from budget_kit import StoreUnderTest
from gateway_testkit import MutableClock
from redis.asyncio import Redis

from gateway.budget.redis_store import RedisBudgetStore
from gateway.budget.store import InMemoryBudgetStore

REDIS_IMAGE = "redis:7.2.16"
REDIS_URL_ENV = "ACL_TEST_REDIS_URL"
REDIS_AUTH_ENV = "ACL_TEST_REDIS_PASSWORD"


@dataclass(frozen=True)
class RealRedis:
    url: str
    password: str | None


@contextlib.contextmanager
def _docker_redis() -> Iterator[RealRedis | None]:
    """A throwaway Redis on a random loopback port, removed afterwards; None without docker."""
    docker = shutil.which("docker")
    if docker is None:
        yield None
        return
    password = secrets.token_urlsafe(24)
    run = subprocess.run(  # noqa: S603 -- fixed argv, docker resolved from PATH
        [
            docker, "run", "--rm", "-d", "-p", "127.0.0.1::6379", REDIS_IMAGE,
            "redis-server", "--save", "", "--appendonly", "no", "--requirepass", password,
        ],
        capture_output=True, text=True, check=False, timeout=120,
    )  # fmt: skip
    if run.returncode != 0:
        yield None
        return
    container = run.stdout.strip()
    try:
        port = subprocess.run(  # noqa: S603 -- fixed argv
            [docker, "port", container, "6379/tcp"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.split()[0].rsplit(":", 1)[1]  # fmt: skip
        yield RealRedis(url=f"redis://127.0.0.1:{port}/0", password=password)
    finally:
        subprocess.run(  # noqa: S603 -- fixed argv
            [docker, "rm", "-f", container], capture_output=True, check=False, timeout=60
        )


@pytest.fixture(scope="session")
def real_redis() -> Iterator[RealRedis]:
    if url := os.environ.get(REDIS_URL_ENV):
        yield RealRedis(url=url, password=os.environ.get(REDIS_AUTH_ENV))
        return
    with _docker_redis() as server:
        if server is None:
            pytest.skip(f"no docker to start {REDIS_IMAGE} and {REDIS_URL_ENV} is not set")
        yield server


async def _wait_until_up(store: RedisBudgetStore) -> None:
    for _ in range(100):
        if await store.healthy():
            return
        await asyncio.sleep(0.1)
    pytest.fail("the redis container never answered")


@pytest.fixture(params=["memory", "fakeredis", pytest.param("redis", marks=pytest.mark.redis)])
async def store_under_test(request: pytest.FixtureRequest) -> AsyncIterator[StoreUnderTest]:
    clock = MutableClock()
    match request.param:
        case "memory":
            yield StoreUnderTest("memory", InMemoryBudgetStore(clock=clock), clock)
        case "fakeredis":
            client = fakeredis.FakeAsyncRedis()
            yield StoreUnderTest("fakeredis", RedisBudgetStore(client), clock, client)
            await client.aclose()
        case _:
            server: RealRedis = request.getfixturevalue("real_redis")
            store = RedisBudgetStore.from_url(server.url, password=server.password)
            await _wait_until_up(store)
            client = Redis.from_url(server.url, password=server.password)  # pyright: ignore[reportUnknownMemberType]
            await client.flushdb()
            yield StoreUnderTest("redis", store, clock, client)
            await client.aclose()
            await store.aclose()
