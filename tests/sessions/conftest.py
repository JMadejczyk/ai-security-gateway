"""Every `SessionStore` behind one contract: in-memory, Redis over fakeredis, real Redis.

``backend.store()`` builds a new store instance over the same state, which is what a second
gateway (or a restarted one) is for the Redis stores. The in-memory store cannot share state,
so it hands back its one instance.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import fakeredis
import pytest
from gateway_testkit import MutableClock
from redis.asyncio import Redis
from redis.exceptions import RedisError
from redis_kit import RealRedis

from gateway.redis_sessions import RedisSessionStore
from gateway.sessions import InMemorySessionStore, SessionStore

SESSIONS_DB = 3  # its own database on the shared test Redis


@dataclass
class Backend:
    kind: str
    clock: MutableClock
    build: Callable[..., SessionStore]
    client: Redis | None = None
    _first: SessionStore | None = field(default=None, init=False)

    @property
    def shared(self) -> bool:
        return self.client is not None

    def store(self, **options: float) -> SessionStore:
        if not self.shared:
            self._first = self._first or self.build()
            return self._first
        return self.build(**options)


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


def _redis_url(server: RealRedis) -> str:
    return server.url.rsplit("/", 1)[0] + f"/{SESSIONS_DB}"


async def real_client(server: RealRedis) -> Redis:
    client = Redis.from_url(_redis_url(server), password=server.password)  # pyright: ignore[reportUnknownMemberType]
    for _ in range(100):
        try:
            await client.ping()  # pyright: ignore[reportUnknownMemberType]
        except (RedisError, OSError):  # the container is still booting
            await asyncio.sleep(0.1)
        else:
            return client
    pytest.fail("the redis container never answered")


@pytest.fixture(params=["memory", "fakeredis", pytest.param("redis", marks=pytest.mark.redis)])
async def backend(request: pytest.FixtureRequest, clock: MutableClock) -> AsyncIterator[Backend]:
    match request.param:
        case "memory":
            yield Backend("memory", clock, lambda: InMemorySessionStore(clock=clock))
        case "fakeredis":
            client = fakeredis.FakeAsyncRedis()
            yield Backend(
                "fakeredis",
                clock,
                lambda **o: RedisSessionStore(client, clock=clock, **o),
                client,
            )
            await client.aclose()
        case _:
            client = await real_client(request.getfixturevalue("real_redis"))
            await client.flushdb()
            yield Backend(
                "redis", clock, lambda **o: RedisSessionStore(client, clock=clock, **o), client
            )
            await client.flushdb()
            await client.aclose()


@pytest.fixture
def store(backend: Backend) -> SessionStore:
    return backend.store()
