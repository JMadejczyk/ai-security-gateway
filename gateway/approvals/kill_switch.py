"""The per-agent kill switch (SPEC "Operator levers": ``POST /admin/kill`` / ``/admin/unkill``).

Killed agents live in one Redis hash (``acl:kill``: agent → JSON `KillRecord`), next to
budgets and approvals. The pipeline asks at every step that can still stop a call: at
admission, right before upstream dispatch, and before a result is released.

`KillSwitch` puts a short in-process cache in front of the store so those checks stay cheap.
The bound on how late a kill takes effect:

- in the process that served ``POST /admin/kill`` (the demo's single gateway): immediately,
  the cache is updated with the write;
- in any other gateway process sharing the Redis: within ``cache_ttl_s`` (default 1 s), at
  the in-flight call's next pipeline step. A request already sent upstream cannot be recalled;
  its result is withheld at the release check.

Fail closed: when the cache is stale and the store cannot be read, the check raises
`KillSwitchUnavailableError` (503) instead of guessing "not killed". Budgets already fail
closed on the same Redis, so a Redis outage stops budget-limited traffic anyway; letting
calls through here would make a kill unenforceable exactly when it cannot be observed.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import ClassVar, Final, cast

from pydantic import AwareDatetime, Field
from redis.asyncio import Redis
from redis.exceptions import RedisError

from gateway.core.envelope import FrozenModel
from gateway.errors import RejectionError

KILL_KEY: Final = "acl:kill"
DEFAULT_CACHE_TTL_S: Final = 1.0
MAX_REASON_CHARS: Final = 200
MAX_REFRESH_ATTEMPTS: Final = 3


class KillRecord(FrozenModel):
    """Why and by whom an agent was killed."""

    agent: str
    reason: str = Field(default="", max_length=MAX_REASON_CHARS)
    killed_by: str
    killed_at: AwareDatetime


class KillSwitchUnavailableError(RejectionError):
    status_code = 503

    def __init__(self) -> None:
        super().__init__(
            "kill_switch_unavailable", "the kill switch state is unavailable; retry later"
        )


class KillSwitchStore(ABC):
    """Persisted kill records. Raises `KillSwitchUnavailableError` when unreachable."""

    kind: ClassVar[str]

    @abstractmethod
    async def kill(self, record: KillRecord) -> None: ...

    @abstractmethod
    async def unkill(self, agent: str) -> bool:
        """True if the agent was killed."""

    @abstractmethod
    async def killed(self) -> dict[str, KillRecord]: ...

    async def aclose(self) -> None:  # noqa: B027 -- optional hook, the memory store holds nothing
        """Release connections."""


class InMemoryKillSwitchStore(KillSwitchStore):
    kind: ClassVar[str] = "memory"

    def __init__(self) -> None:
        self._killed: dict[str, KillRecord] = {}

    async def kill(self, record: KillRecord) -> None:
        self._killed[record.agent] = record

    async def unkill(self, agent: str) -> bool:
        return self._killed.pop(agent, None) is not None

    async def killed(self) -> dict[str, KillRecord]:
        return dict(self._killed)


class RedisKillSwitchStore(KillSwitchStore):
    kind: ClassVar[str] = "redis"

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def kill(self, record: KillRecord) -> None:
        try:
            await self._client.hset(KILL_KEY, record.agent, record.model_dump_json())  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues] -- untyped in redis-py
        except (RedisError, OSError) as exc:
            raise KillSwitchUnavailableError from exc

    async def unkill(self, agent: str) -> bool:
        try:
            return bool(await self._client.hdel(KILL_KEY, agent))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportGeneralTypeIssues] -- untyped in redis-py
        except (RedisError, OSError) as exc:
            raise KillSwitchUnavailableError from exc

    async def killed(self) -> dict[str, KillRecord]:
        try:
            raw = await self._client.hgetall(KILL_KEY)  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues] -- untyped in redis-py
        except (RedisError, OSError) as exc:
            raise KillSwitchUnavailableError from exc
        records = (
            KillRecord.model_validate(json.loads(v))
            for v in cast("dict[bytes, bytes]", raw).values()
        )
        return {record.agent: record for record in records}


class KillSwitch:
    """Cached reads, write-through updates; see the module docstring for the delay bound."""

    def __init__(
        self,
        store: KillSwitchStore,
        *,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        monotonic: Callable[[], float] = time.monotonic,
        on_change: Callable[[Mapping[str, KillRecord]], None] | None = None,
    ) -> None:
        self._store = store
        self._ttl = cache_ttl_s
        self._monotonic = monotonic
        self._on_change = on_change
        self._cache: dict[str, KillRecord] | None = None
        self._fetched_at = 0.0
        self._refresh_lock = asyncio.Lock()
        # Bumped by every kill/unkill through this instance: a refresh that read the store
        # before such a write must not overwrite the cache the write just updated.
        self._version = 0

    @property
    def store(self) -> KillSwitchStore:
        return self._store

    async def check(self, agent: str) -> KillRecord | None:
        """The agent's kill record, or None. Raises `KillSwitchUnavailableError` when the
        cache is stale and the store cannot be read."""
        return (await self.active()).get(agent)

    async def active(self) -> Mapping[str, KillRecord]:
        cache = self._cache
        if cache is not None and self._monotonic() - self._fetched_at < self._ttl:
            return cache
        async with self._refresh_lock:  # one refresh at a time; others reuse its answer
            if self._cache is not None and self._monotonic() - self._fetched_at < self._ttl:
                return self._cache
            for _ in range(MAX_REFRESH_ATTEMPTS):
                version, started = self._version, self._monotonic()
                try:
                    fresh = await self._store.killed()
                except KillSwitchUnavailableError:
                    self._cache = None  # never serve a stale answer after a failed read
                    raise
                if version == self._version:  # no kill/unkill landed while reading
                    self._remember(fresh, started)
                    return fresh
            raise KillSwitchUnavailableError

    async def kill(self, record: KillRecord) -> None:
        await self._store.kill(record)
        self._version += 1
        if self._cache is not None:
            self._remember({**self._cache, record.agent: record}, self._fetched_at)
        else:
            await self.active()

    async def unkill(self, agent: str) -> bool:
        was_killed = await self._store.unkill(agent)
        self._version += 1
        if self._cache is not None:
            remaining = {a: r for a, r in self._cache.items() if a != agent}
            self._remember(remaining, self._fetched_at)
        return was_killed

    def _remember(self, killed: dict[str, KillRecord], fetched_at: float) -> None:
        self._cache, self._fetched_at = killed, fetched_at
        if self._on_change is not None:
            self._on_change(killed)
