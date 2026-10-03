"""Session state in Redis (``state`` network): shared by every gateway, kept across restarts.

`RedisSessionStore` implements `SessionStore` exactly as `InMemorySessionStore` does (binding,
lifetimes, sticky taint, decay, retirement), with three differences forced by sharing:

- **Records.** One hash per session (``acl:session:<id>``): ``v``, a version that every write
  bumps, and ``doc``, the whole `StoredSession` as JSON: the `SessionContext` (every field,
  so state added later round-trips unchanged) plus the ``sessions`` limits it was admitted
  under. Writes are compare-and-set on ``v`` in one Lua script that also refuses a session with
  a tombstone, so a write can neither lose another's update nor revive an ended session.
  `apply` re-reads and re-applies its `SessionUpdate` on a conflict: updates are deltas.
- **Retirement.** Expiry is decided from the record (``last_seen`` + idle TTL, ``created_at``
  + lifetime) on the store's clock, lazily, and never while a call holds the session's lock.
  An ended or expired session leaves a tombstone (``acl:session:<id>:ended``); records and
  tombstones carry a Redis TTL of at least ``max(max_lifetime_s, token lifetime) + 1 day``
  past retirement, so a retired id is refused for at least that long (a record past its
  logical expiry is itself proof of retirement until it is collected).
- **Serialization across processes.** `RedisSessionLock`: a lock key set with ``NX PX`` and a
  random token, renewed while held, released only by its owner (compare-and-delete), expiring
  on its own if the holder dies. A call waits at most ``session_lock_wait_s`` and is then
  refused with 429 ``session_busy`` instead of queueing without bound.

Any Redis or connection error fails closed: `SessionStoreUnavailableError` (503).
"""

import asyncio
import contextlib
import logging
import random
import secrets
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import TracebackType
from typing import Final, cast, override

from pydantic import ValidationError
from redis.asyncio import Redis
from redis.commands.core import AsyncScript
from redis.exceptions import RedisError

from gateway.clock import Clock, utc_now
from gateway.core.envelope import FrozenModel, SessionContext
from gateway.errors import RejectionError
from gateway.identity import MAX_TOKEN_LIFETIME_S
from gateway.policy.schema import Sessions
from gateway.sessions import (
    SessionBinding,
    SessionError,
    SessionReason,
    SessionStore,
    SessionUpdate,
    apply_update,
)

logger = logging.getLogger(__name__)

KEY_PREFIX: Final = "acl:session"
TAINTED_KEY: Final = "acl:sessions:tainted"  # zset: tainted session id -> expiry (epoch ms)
TOMBSTONE_SLACK_S: Final = 86_400
LOCK_TTL_S: Final = 15.0
MAX_WRITE_ATTEMPTS: Final = 32  # conflicts are rare: the session lock serializes writers
_BACKOFF_MIN_S: Final = 0.005
_BACKOFF_MAX_S: Final = 0.2
_ENDED: Final = -1
_CONFLICT: Final = 0

# KEYS: record, tombstone, tainted zset. ARGV: expected version, doc, record TTL (ms),
# tainted (1/0), tainted score, session id. Returns the new version, 0 on a version conflict,
# -1 when the session has a tombstone.
_SAVE: Final = """
if redis.call('EXISTS', KEYS[2]) == 1 then return -1 end
local version = tonumber(redis.call('HGET', KEYS[1], 'v') or '0')
if version ~= tonumber(ARGV[1]) then return 0 end
redis.call('HSET', KEYS[1], 'v', version + 1, 'doc', ARGV[2])
redis.call('PEXPIRE', KEYS[1], ARGV[3])
if ARGV[4] == '1' then
  redis.call('ZADD', KEYS[3], ARGV[5], ARGV[6])
else
  redis.call('ZREM', KEYS[3], ARGV[6])
end
return version + 1
"""

# KEYS: record, tombstone, tainted zset. ARGV: retired-at (ISO), tombstone TTL (ms), id.
# An existing tombstone keeps its original time; its TTL is only ever extended.
_RETIRE: Final = """
redis.call('DEL', KEYS[1])
redis.call('ZREM', KEYS[3], ARGV[3])
redis.call('SET', KEYS[2], ARGV[1], 'NX')
if redis.call('PTTL', KEYS[2]) < tonumber(ARGV[2]) then
  redis.call('PEXPIRE', KEYS[2], ARGV[2])
end
return 1
"""

_RENEW: Final = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""

_RELEASE: Final = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""


class SessionStoreUnavailableError(RejectionError):
    """The session store cannot be reached or read: no call is admitted without its state."""

    status_code = 503

    def __init__(self) -> None:
        super().__init__("session_store_unavailable", "session state is unavailable; retry later")


class SessionBusyError(RejectionError):
    """Another call of the same session held its lock for longer than the caller may wait."""

    status_code = 429

    def __init__(self) -> None:
        super().__init__("session_busy", "another call of this session is still running")


class StoredSession(FrozenModel):
    """What one session record holds: its state and the limits it was last admitted under."""

    context: SessionContext
    limits: Sessions

    def expires_at(self) -> datetime:
        idle = self.context.last_seen + timedelta(seconds=self.limits.idle_ttl_s)
        old = self.context.created_at + timedelta(seconds=self.limits.max_lifetime_s)
        return min(idle, old)

    def expired(self, now: datetime) -> bool:
        return now >= self.expires_at()


def tombstone_ttl_s(limits: Sessions) -> float:
    """How long a retired id stays refused, at least: longer than any session or token that
    could still name it, plus a day."""
    return max(limits.max_lifetime_s, MAX_TOKEN_LIFETIME_S) + TOMBSTONE_SLACK_S


async def _conflict_backoff() -> None:
    await asyncio.sleep(_BACKOFF_MIN_S * (1 + random.random()))  # noqa: S311 -- jitter only


def _ms(seconds: float) -> int:
    return max(int(seconds * 1000), 1)


def _epoch_ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


@dataclass(slots=True)
class _Local:
    """In-process queue in front of the Redis lock: one Redis contender per process."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class RedisSessionLock(AbstractAsyncContextManager[None]):
    """A per-session lock shared by every gateway (one use: enter once, exit once).

    ``SET key token NX PX ttl`` acquires; a background task renews the TTL every third of it
    while the lock is held; release deletes the key only if it still holds this token. A holder
    that dies stops renewing, and the lock expires after ``ttl_s``. Should renewal ever fail
    (Redis down for longer than the TTL), ``lost`` is set: the version check on every session
    write still keeps a second holder from overwriting this one's state.
    """

    def __init__(
        self,
        client: Redis,
        key: str,
        local: "_LocalLocks",
        *,
        wait_s: float,
        ttl_s: float = LOCK_TTL_S,
    ) -> None:
        self._client = client
        self._renew_script: AsyncScript = client.register_script(_RENEW)
        self._release_script: AsyncScript = client.register_script(_RELEASE)
        self._key = key
        self._local = local
        self._wait_s = wait_s
        self._ttl_ms = _ms(ttl_s)
        self._token = secrets.token_hex(16)
        self._renewer: asyncio.Task[None] | None = None
        self._local_held = False
        self.lost = False

    @override
    async def __aenter__(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._wait_s
        try:
            async with asyncio.timeout(self._wait_s):
                await self._local.acquire(self._key)
        except TimeoutError:
            raise SessionBusyError from None
        self._local_held = True
        try:
            await self._acquire(deadline)
        except BaseException:
            self._release_local()
            raise
        self._renewer = asyncio.create_task(self._renew())

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._renewer is not None:
            self._renewer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._renewer
        try:
            await self._release_script(keys=[self._key], args=[self._token])
        except (RedisError, OSError) as error:  # it expires on its own
            logger.warning("session lock release failed: %s", type(error).__name__)
        finally:
            self._release_local()

    async def _acquire(self, deadline: float) -> None:
        loop = asyncio.get_running_loop()
        backoff = _BACKOFF_MIN_S
        while True:
            try:
                acquired = await self._client.set(self._key, self._token, nx=True, px=self._ttl_ms)
            except (RedisError, OSError) as error:
                raise SessionStoreUnavailableError from error
            if acquired:
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise SessionBusyError
            await asyncio.sleep(min(remaining, backoff * (1 + random.random())))  # noqa: S311 -- jitter, not security
            backoff = min(backoff * 2, _BACKOFF_MAX_S)

    async def _renew(self) -> None:
        interval = self._ttl_ms / 3000
        while True:
            await asyncio.sleep(interval)
            try:
                renewed: object = await self._renew_script(
                    keys=[self._key], args=[self._token, self._ttl_ms]
                )
            except (RedisError, OSError) as error:
                logger.warning("session lock renewal failed: %s", type(error).__name__)
                continue
            if not renewed:
                self.lost = True
                logger.error("a session lock expired while held; writes stay version-checked")
                return

    def _release_local(self) -> None:
        if self._local_held:
            self._local_held = False
            self._local.release(self._key)


class _LocalLocks:
    def __init__(self) -> None:
        self._entries: dict[str, _Local] = {}

    async def acquire(self, key: str) -> None:
        entry = self._entries.setdefault(key, _Local())
        entry.users += 1
        try:
            await entry.lock.acquire()
        except BaseException:
            self._drop(key, entry)
            raise

    def release(self, key: str) -> None:
        entry = self._entries[key]
        entry.lock.release()
        self._drop(key, entry)

    def _drop(self, key: str, entry: _Local) -> None:
        entry.users -= 1
        if entry.users == 0:
            del self._entries[key]


@dataclass(frozen=True, slots=True)
class _Loaded:
    stored: StoredSession | None
    version: int
    ended: bool
    locked: bool


class RedisSessionStore(SessionStore):
    """`SessionStore` over Redis; see the module docstring."""

    def __init__(
        self,
        client: Redis,
        *,
        clock: Clock = utc_now,
        lock_wait_s: float = 30.0,
        lock_ttl_s: float = LOCK_TTL_S,
    ) -> None:
        self._client = client
        self._clock = clock
        self._lock_wait_s = lock_wait_s
        self._lock_ttl_s = lock_ttl_s
        self._local = _LocalLocks()
        self._limits = Sessions()  # the latest admission's limits, for ids without a record
        self._save = client.register_script(_SAVE)
        self._retire = client.register_script(_RETIRE)

    # ------------------------------------------------------------------- SessionStore

    @override
    def lock(self, session_id: str) -> AbstractAsyncContextManager[None]:
        return RedisSessionLock(
            self._client,
            self._key(session_id, "lock"),
            self._local,
            wait_s=self._lock_wait_s,
            ttl_s=self._lock_ttl_s,
        )

    @override
    async def open(
        self, session_id: str, binding: SessionBinding, limits: Sessions
    ) -> SessionContext:
        self._limits = limits
        for _ in range(MAX_WRITE_ATTEMPTS):
            now = self._clock()
            loaded = await self._load(session_id)
            if loaded.ended:
                raise SessionError(SessionReason.ENDED)
            stored = loaded.stored
            if stored is not None and stored.expired(now):
                # The caller holds this session's lock to admit a new call: nothing in flight.
                await self._retire_id(session_id, stored.limits, now)
                raise SessionError(SessionReason.ENDED)
            if stored is None:
                ctx = SessionContext(
                    session_id=session_id,
                    principal=binding.principal,
                    actor=binding.actor,
                    mode=binding.mode,
                    risk_updated_at=now,
                    created_at=now,
                    last_seen=now,
                )
            elif not binding.matches(stored.context):
                raise SessionError(SessionReason.BINDING_MISMATCH)
            else:
                ctx = stored.context.model_copy(update={"last_seen": now})
            record = StoredSession(context=ctx, limits=limits)
            if await self._write(session_id, record, loaded.version, now):
                return ctx
            await _conflict_backoff()
        raise SessionStoreUnavailableError

    @override
    async def get(self, session_id: str) -> SessionContext | None:
        loaded = await self._load(session_id)
        if loaded.ended or loaded.stored is None:
            return None
        if loaded.stored.expired(self._clock()) and not loaded.locked:
            return None
        return loaded.stored.context

    @override
    async def apply(
        self, session_id: str, update: SessionUpdate, *, half_life_s: float
    ) -> SessionContext:
        for _ in range(MAX_WRITE_ATTEMPTS):
            now = self._clock()
            loaded = await self._load(session_id)
            if loaded.ended or loaded.stored is None:
                raise SessionError(SessionReason.ENDED)
            ctx = apply_update(loaded.stored.context, update, now=now, half_life_s=half_life_s)
            record = loaded.stored.model_copy(update={"context": ctx})
            if await self._write(session_id, record, loaded.version, now):
                return ctx
            await _conflict_backoff()
        raise SessionStoreUnavailableError

    @override
    async def end(self, session_id: str) -> None:
        loaded = await self._load(session_id)
        limits = loaded.stored.limits if loaded.stored is not None else self._limits
        await self._retire_id(session_id, limits, self._clock())

    @override
    async def is_retired(self, session_id: str) -> bool:
        loaded = await self._load(session_id)
        if loaded.ended:
            return True
        stored = loaded.stored
        return stored is not None and stored.expired(self._clock()) and not loaded.locked

    @override
    async def tainted_count(self) -> int:
        now_ms = _epoch_ms(self._clock())
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(TAINTED_KEY, "-inf", now_ms)
                pipe.zcard(TAINTED_KEY)
                _, count = cast("tuple[int, int]", tuple(await pipe.execute()))
        except (RedisError, OSError) as error:
            raise SessionStoreUnavailableError from error
        return int(count)

    async def healthy(self) -> bool:
        try:
            return bool(await self._client.ping())  # pyright: ignore[reportUnknownMemberType] -- untyped in redis-py
        except (RedisError, OSError):
            return False

    # ------------------------------------------------------------------------ helpers

    @staticmethod
    def _key(session_id: str, suffix: str | None = None) -> str:
        key = f"{KEY_PREFIX}:{session_id}"
        return f"{key}:{suffix}" if suffix else key

    async def _load(self, session_id: str) -> _Loaded:
        try:
            async with self._client.pipeline(transaction=False) as pipe:
                pipe.hmget(self._key(session_id), ["v", "doc"])
                pipe.exists(self._key(session_id, "ended"))
                pipe.exists(self._key(session_id, "lock"))
                replies = cast("list[object]", await pipe.execute())
        except (RedisError, OSError) as error:
            raise SessionStoreUnavailableError from error
        (version, doc), ended, locked = cast("tuple[list[bytes | None], int, int]", tuple(replies))
        stored: StoredSession | None = None
        if doc is not None:
            try:
                stored = StoredSession.model_validate_json(doc)
            except ValidationError:
                # No traceback: a validation error quotes the record (the raw goal text).
                logger.error("unreadable session record in Redis")  # noqa: TRY400
                raise SessionStoreUnavailableError from None
        return _Loaded(
            stored=stored,
            version=int(version) if version is not None else 0,
            ended=bool(ended),
            locked=bool(locked),
        )

    async def _write(
        self, session_id: str, record: StoredSession, version: int, now: datetime
    ) -> bool:
        """Compare-and-set; False on a version conflict. Raises `SessionError` when the
        session was ended meanwhile."""
        expires_at = record.expires_at()
        ttl_s = max((expires_at - now).total_seconds(), 0.0) + tombstone_ttl_s(record.limits)
        keys = [self._key(session_id), self._key(session_id, "ended"), TAINTED_KEY]
        args: list[str | int] = [
            version,
            record.model_dump_json(),
            _ms(ttl_s),
            int(record.context.taint),
            _epoch_ms(expires_at),
            session_id,
        ]
        try:
            reply = int(cast("int", await self._save(keys=keys, args=args)))
        except (RedisError, OSError) as error:
            raise SessionStoreUnavailableError from error
        if reply == _ENDED:
            raise SessionError(SessionReason.ENDED)
        return reply != _CONFLICT

    async def _retire_id(self, session_id: str, limits: Sessions, now: datetime) -> None:
        keys = [self._key(session_id), self._key(session_id, "ended"), TAINTED_KEY]
        args = [now.isoformat(), _ms(tombstone_ttl_s(limits)), session_id]
        try:
            await self._retire(keys=keys, args=args)
        except (RedisError, OSError) as error:
            raise SessionStoreUnavailableError from error
