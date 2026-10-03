"""Session state and ``loop_detect`` counters, built from the settings (`state_store_kind`).

``redis``: `RedisSessionStore` + `RedisCallCounter` over one client on the ``state`` network;
both fail closed while Redis is down. ``memory``: the single-process stores, for tests and
local development only, and only when the settings say so.
"""

from dataclasses import dataclass
from typing import Self

from redis.asyncio import Redis

from gateway.clock import Clock, utc_now
from gateway.controls.loop_counter_redis import RedisCallCounter
from gateway.controls.loop_detect import CallCounter, InMemoryCallCounter
from gateway.redis_sessions import RedisSessionStore
from gateway.sessions import InMemorySessionStore, SessionStore
from gateway.settings import Settings
from gateway.state_redis import redis_client_from_settings, state_store_kind


@dataclass(frozen=True, slots=True, kw_only=True)
class StateStores:
    sessions: SessionStore
    calls: CallCounter
    client: Redis | None = None  # owned here when the stores are Redis-backed

    @classmethod
    def from_settings(cls, settings: Settings, *, clock: Clock = utc_now) -> Self:
        match state_store_kind(settings):
            case "memory":
                return cls(sessions=InMemorySessionStore(clock=clock), calls=InMemoryCallCounter())
            case "redis":
                client = redis_client_from_settings(settings)
                return cls(
                    sessions=RedisSessionStore(
                        client, clock=clock, lock_wait_s=settings.session_lock_wait_s
                    ),
                    calls=RedisCallCounter(client),
                    client=client,
                )

    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.aclose()
