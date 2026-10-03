"""The Redis on the ``state`` network, shared by session state, loop counters and approvals.

``redis:7.2`` (BSD-3), reachable only from the gateway. Clients fail fast: a dead Redis must
refuse calls (fail closed), never stall them.
"""

from typing import Final, Literal

from redis.asyncio import Redis

from gateway.settings import Settings

TIMEOUT_S: Final = 1.0

type StoreKind = Literal["redis", "memory"]


def state_store_kind(settings: Settings) -> StoreKind:
    """``ACL_SESSION_STORE``, or the budget store's kind when it is not set."""
    return settings.session_store or settings.budget_store


def redis_client_from_settings(settings: Settings) -> Redis:
    """A client for ``ACL_REDIS_URL``; it connects lazily, so the gateway starts while Redis
    is down (and refuses what needs it with 503)."""
    password = settings.redis_password
    return Redis.from_url(  # pyright: ignore[reportUnknownMemberType] -- untyped **kwargs in redis-py
        settings.redis_url,
        password=password.get_secret_value() if password is not None else None,
        socket_connect_timeout=TIMEOUT_S,
        socket_timeout=TIMEOUT_S,
        health_check_interval=30,
    )
