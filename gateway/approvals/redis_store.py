"""Approval records in Redis (``redis:7.2``, BSD-3), next to budgets on the ``state`` network.

Keys (one Redis, never a cluster: the scripts touch record keys they derive from ids):

- ``acl:approval:rec:<id>``: hash. ``doc`` is the immutable draft (JSON); ``state``,
  ``expires_at``/``updated_at``/``decided_at`` (epoch seconds), ``decided_by``, ``note`` and
  ``outcome`` are the mutable part. It expires ``retention_s`` after creation.
- ``acl:approval:op:<operation>``: hash ``{id, gen}``, the operation's slot.
- ``acl:approval:open``: sorted set of pending and approved ids by deadline (the sweeper's
  work list); ``acl:approval:pending``: set of pending ids (``acl_approvals_pending``);
  ``acl:approval:all``: sorted set of every id by creation time (listings).

Create-or-get and every transition are Lua scripts: the check (slot open? state a legal
source? deadline passed?) and the write happen in one step no other client can interleave
with. Both scripts check everything before their first write.
"""

import json
from datetime import UTC, datetime
from typing import Any, ClassVar, Final, cast

from redis.asyncio import Redis
from redis.commands.core import AsyncScript
from redis.exceptions import RedisError

from gateway.approvals.model import (
    Approval,
    ApprovalConflictError,
    ApprovalDraft,
    ApprovalState,
    Transition,
    approval_id_for,
)
from gateway.approvals.store import LIST_LIMIT, ApprovalStore, ApprovalStoreUnavailableError

RECORD_PREFIX: Final = "acl:approval:rec:"
SLOT_PREFIX: Final = "acl:approval:op:"
OPEN_KEY: Final = "acl:approval:open"
PENDING_KEY: Final = "acl:approval:pending"
ALL_KEY: Final = "acl:approval:all"
SCAN_LIMIT: Final = 5000  # listings look at most this many of the newest records

_OK: Final = 1
_CONFLICT: Final = 0
_MISSING: Final = -1

# KEYS[...]: per script. `expire_if_due` moves a pending or approved record whose deadline is
# at or before `now` to expired and drops it from the open and pending indexes.
_PRELUDE: Final = """
local function expire_if_due(rkey, id, now, open_key, pending_key)
  local f = redis.call('HMGET', rkey, 'state', 'expires_at')
  local state = f[1]
  if not state then return nil, false end
  if (state == 'pending' or state == 'approved') and tonumber(f[2]) <= tonumber(now) then
    redis.call('HSET', rkey, 'state', 'expired', 'updated_at', now, 'outcome', 'approval_timeout')
    redis.call('ZREM', open_key, id)
    redis.call('SREM', pending_key, id)
    return 'expired', true
  end
  return state, false
end
"""

# KEYS: slot, open, pending, all. ARGV: record prefix, base id, now, expires_at, doc,
# retention_s. Returns {id, created (0/1)}.
_CREATE_OR_GET: Final = (
    _PRELUDE
    + """
local prefix, base, now, expires, doc = ARGV[1], ARGV[2], ARGV[3], ARGV[4], ARGV[5]
local retention = tonumber(ARGV[6])
local current = redis.call('HGET', KEYS[1], 'id')
local gen = tonumber(redis.call('HGET', KEYS[1], 'gen') or '0')
if current then
  local state = expire_if_due(prefix .. current, current, now, KEYS[2], KEYS[3])
  if state == 'pending' or state == 'approved' or state == 'executing' then
    return {current, 0}
  end
end
gen = gen + 1
local id = base
if gen > 1 then id = base .. '-' .. gen end
local rkey = prefix .. id
redis.call('HSET', rkey, 'doc', doc, 'state', 'pending', 'expires_at', expires, 'updated_at', now)
redis.call('EXPIRE', rkey, retention)
redis.call('HSET', KEYS[1], 'id', id, 'gen', gen)
redis.call('EXPIRE', KEYS[1], retention)
redis.call('ZADD', KEYS[2], expires, id)
redis.call('SADD', KEYS[3], id)
redis.call('ZADD', KEYS[4], now, id)
redis.call('ZREMRANGEBYSCORE', KEYS[4], '-inf', tonumber(now) - retention)
return {id, 1}
"""
)

# KEYS: record, open, pending. ARGV: id, now, target, comma-separated sources, new deadline
# ('' keeps it), then field/value pairs to set. Returns {status, state}.
_TRANSITION: Final = (
    _PRELUDE
    + """
local id, now, target, sources, deadline = ARGV[1], ARGV[2], ARGV[3], ARGV[4], ARGV[5]
local state, just_expired = expire_if_due(KEYS[1], id, now, KEYS[2], KEYS[3])
if not state then return {-1, ''} end
if just_expired and target == 'expired' then return {1, state} end
if target == 'expired' then return {0, state} end  -- only the deadline expires a record
if not string.find(',' .. sources .. ',', ',' .. state .. ',', 1, true) then
  return {0, state}
end
local fields = {'state', target, 'updated_at', now}
for i = 6, #ARGV do fields[#fields + 1] = ARGV[i] end
if deadline ~= '' then
  fields[#fields + 1] = 'expires_at'
  fields[#fields + 1] = deadline
end
redis.call('HSET', KEYS[1], unpack(fields))
if target == 'pending' then redis.call('SADD', KEYS[3], id) else redis.call('SREM', KEYS[3], id) end
if target == 'pending' or target == 'approved' then
  local score = deadline
  if score == '' then score = redis.call('HGET', KEYS[1], 'expires_at') end
  redis.call('ZADD', KEYS[2], score, id)
else
  redis.call('ZREM', KEYS[2], id)
end
return {1, target}
"""
)


def _epoch(moment: datetime) -> str:
    return f"{moment.timestamp():.6f}"


def _moment(value: bytes | str) -> datetime:
    return datetime.fromtimestamp(float(value), UTC)


def _text(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


class RedisApprovalStore(ApprovalStore):
    """Approval records in Redis. Any Redis or connection error fails closed."""

    kind: ClassVar[str] = "redis"

    def __init__(self, client: Redis) -> None:
        self._client = client
        self._create = client.register_script(_CREATE_OR_GET)
        self._transition = client.register_script(_TRANSITION)

    async def create_or_get(
        self, draft: ApprovalDraft, *, expires_at: datetime, retention_s: int
    ) -> tuple[Approval, bool]:
        keys = [SLOT_PREFIX + draft.operation, OPEN_KEY, PENDING_KEY, ALL_KEY]
        args = [
            RECORD_PREFIX,
            approval_id_for(draft.operation, 1),
            _epoch(draft.created_at),
            _epoch(expires_at),
            draft.model_dump_json(),
            str(retention_s),
        ]
        reply = cast("list[Any]", await self._run(self._create, keys, args))
        approval_id, created = _text(reply[0]), int(reply[1]) == 1
        record = await self.get(approval_id)
        if record is None:  # evicted between the two round trips: nothing usable exists
            raise ApprovalStoreUnavailableError
        return record, created

    async def get(self, approval_id: str) -> Approval | None:
        try:
            fields = await self._client.hgetall(RECORD_PREFIX + approval_id)  # pyright: ignore[reportUnknownMemberType] -- redis-py's hgetall is untyped
        except (RedisError, OSError) as exc:
            raise ApprovalStoreUnavailableError from exc
        return _parse(approval_id, cast("dict[bytes, bytes]", fields))

    async def transition(self, approval_id: str, change: Transition) -> Approval:
        extra: list[str] = []
        if change.decided_by is not None:
            extra += ["decided_by", change.decided_by, "decided_at", _epoch(change.at)]
        if change.note is not None:
            extra += ["note", change.note]
        if change.outcome is not None:
            extra += ["outcome", change.outcome]
        keys = [RECORD_PREFIX + approval_id, OPEN_KEY, PENDING_KEY]
        args = [
            approval_id,
            _epoch(change.at),
            change.target.value,
            ",".join(sorted(state.value for state in change.sources)),
            _epoch(change.expires_at) if change.expires_at is not None else "",
            *extra,
        ]
        reply = cast("list[Any]", await self._run(self._transition, keys, args))
        status = int(reply[0])
        record = await self.get(approval_id)
        if status != _OK or record is None:
            raise ApprovalConflictError(approval_id, record)
        return record

    async def records(
        self, states: frozenset[ApprovalState] | None = None, limit: int = LIST_LIMIT
    ) -> list[Approval]:
        try:
            raw_ids = await self._client.zrevrange(ALL_KEY, 0, SCAN_LIMIT - 1)  # pyright: ignore[reportUnknownMemberType] -- untyped in redis-py
            ids = [_text(i) for i in cast("list[bytes]", raw_ids)]
            async with self._client.pipeline(transaction=False) as pipe:
                for approval_id in ids:
                    pipe.hgetall(RECORD_PREFIX + approval_id)  # pyright: ignore[reportUnknownMemberType] -- untyped in redis-py
                rows = cast("list[dict[bytes, bytes]]", await pipe.execute())  # pyright: ignore[reportUnknownMemberType] -- untyped in redis-py
        except (RedisError, OSError) as exc:
            raise ApprovalStoreUnavailableError from exc
        chosen: list[Approval] = []
        for approval_id, fields in zip(ids, rows, strict=True):
            record = _parse(approval_id, fields)
            if record is None or (states is not None and record.state not in states):
                continue
            chosen.append(record)
            if len(chosen) >= limit:
                break
        return chosen

    async def due(self, now: datetime) -> list[str]:
        try:
            ids = await self._client.zrangebyscore(OPEN_KEY, "-inf", _epoch(now))  # pyright: ignore[reportUnknownMemberType] -- untyped in redis-py
        except (RedisError, OSError) as exc:
            raise ApprovalStoreUnavailableError from exc
        return [_text(i) for i in cast("list[bytes]", ids)]

    async def pending_count(self) -> int:
        try:
            return int(await self._client.scard(PENDING_KEY))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportGeneralTypeIssues] -- untyped in redis-py
        except (RedisError, OSError) as exc:
            raise ApprovalStoreUnavailableError from exc

    async def healthy(self) -> bool:
        try:
            return bool(await self._client.ping())  # pyright: ignore[reportUnknownMemberType] -- untyped **kwargs in redis-py
        except (RedisError, OSError):
            return False

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _run(self, script: AsyncScript, keys: list[str], args: list[str]) -> object:
        try:
            return await script(keys=keys, args=args)
        except (RedisError, OSError) as exc:
            raise ApprovalStoreUnavailableError from exc


def _parse(approval_id: str, fields: dict[bytes, bytes]) -> Approval | None:
    if b"doc" not in fields:
        return None
    draft: dict[str, Any] = json.loads(fields[b"doc"])
    optional = {
        key: _text(fields[key.encode()])
        for key in ("decided_by", "note", "outcome")
        if key.encode() in fields
    }
    decided_at = fields.get(b"decided_at")
    return Approval.model_validate(
        {
            **draft,
            **optional,
            "id": approval_id,
            "state": _text(fields[b"state"]),
            "expires_at": _moment(fields[b"expires_at"]),
            "updated_at": _moment(fields[b"updated_at"]),
            "decided_at": _moment(decided_at) if decided_at is not None else None,
        }
    )
