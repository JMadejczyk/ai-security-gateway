"""Budget counters in Redis (``redis:7.2``, BSD-3), on the ``state`` network only the gateway joins.

Each scope is one hash (``acl:budget:<kind>:<window>:<subject>``) with one integer field per
meter; each reservation is one hash (``acl:budget:op:<id>``) holding its amounts while it is
``held``, then a ``done`` tombstone. Reserve and settle are Lua scripts, so every check and
write of one operation happens in one round trip no other client can interleave with.

A Lua error midway would leave the writes before it in place (Redis does not roll back), so
both scripts work in two passes: compute and check every resulting value first (limits,
``MAX_COUNTER``, clamping at zero), and only then write, each scope with one ``HSET`` of
already-formatted integers followed by its TTL. The checks leave nothing that can fail
between the first write and the last.

TTLs are extended on every write and never shortened (``TTL`` compared in the script: Redis's
``EXPIRE ... GT`` ignores keys without a TTL).
"""

from collections.abc import Sequence
from typing import ClassVar, Final, cast

from redis.asyncio import Redis
from redis.commands.core import AsyncScript
from redis.exceptions import RedisError

from gateway.budget.model import (
    MAX_COUNTER,
    BudgetScope,
    BudgetStoreUnavailableError,
    Meter,
    ReserveOutcome,
    Spend,
    op_key,
)
from gateway.budget.store import OP_TTL_S, BudgetStore

_METERS: Final = tuple(Meter)
_FIELDS: Final = tuple(meter.value for meter in _METERS)
_NO_LIMIT: Final = "-1"
_DUPLICATE: Final = -1
_REFUSED: Final = 0

# ARGV: the meter count, the meter fields, the op TTL, MAX_COUNTER, then the script's own
# arguments. KEYS[1] is the operation, KEYS[2..] the scopes.
_PRELUDE: Final = """
local nf = tonumber(ARGV[1])
local fields = {}
for i = 1, nf do fields[i] = ARGV[1 + i] end
local op_ttl = tonumber(ARGV[nf + 2])
local max_counter = tonumber(ARGV[nf + 3])
local head = nf + 3
local function write(key, values, ttl)
  local items = {}
  for m = 1, nf do
    items[#items + 1] = fields[m]
    items[#items + 1] = string.format('%d', values[m])
  end
  redis.call('HSET', key, unpack(items))
  if redis.call('TTL', key) < ttl then redis.call('EXPIRE', key, ttl) end
end
"""

# Then: one amount per meter, then per scope its TTL and one limit per meter (-1 = none).
# Returns {-1} for a used op id, {0, scope, meter} when refused, else {1, usage...}.
_RESERVE: Final = (
    _PRELUDE
    + """
if redis.call('EXISTS', KEYS[1]) == 1 then return {-1} end
local amounts = {}
for m = 1, nf do amounts[m] = tonumber(ARGV[head + m]) end
local base, stride = head + nf, 1 + nf
local rows = {}
for k = 2, #KEYS do
  local b = base + (k - 2) * stride
  local current = redis.call('HMGET', KEYS[k], unpack(fields))
  local row = {}
  for m = 1, nf do
    local used = tonumber(current[m]) or 0
    local after = used + amounts[m]
    local limit = tonumber(ARGV[b + 1 + m])
    if after > max_counter or (limit >= 0 and (used >= limit or after > limit)) then
      return {0, k - 1, m}
    end
    row[m] = after
  end
  rows[k] = row
end
local out = {1}
for k = 2, #KEYS do
  write(KEYS[k], rows[k], tonumber(ARGV[base + (k - 2) * stride + 1]))
  for m = 1, nf do out[#out + 1] = rows[k][m] end
end
local hold = {'state', 'held'}
for m = 1, nf do
  hold[#hold + 1] = fields[m]
  hold[#hold + 1] = ARGV[head + m]
end
redis.call('HSET', KEYS[1], unpack(hold))
redis.call('EXPIRE', KEYS[1], op_ttl)
return out
"""
)

# Then: one spent amount per meter, then one TTL per scope. Applies (spent - held) once, while
# the op is held; always leaves a `done` tombstone. Returns the usage per scope.
_SETTLE: Final = (
    _PRELUDE
    + """
local base = head + nf
local out = {}
local rows = {}
local held = nil
if redis.call('HGET', KEYS[1], 'state') == 'held' then
  held = redis.call('HMGET', KEYS[1], unpack(fields))
end
for k = 2, #KEYS do
  local current = redis.call('HMGET', KEYS[k], unpack(fields))
  local row = {}
  for m = 1, nf do
    local value = tonumber(current[m]) or 0
    if held then
      value = value + tonumber(ARGV[head + m]) - (tonumber(held[m]) or 0)
      if value < 0 then value = 0 elseif value > max_counter then value = max_counter end
    end
    row[m] = value
  end
  rows[k] = row
end
for k = 2, #KEYS do
  if held then write(KEYS[k], rows[k], tonumber(ARGV[base + k - 1])) end
  for m = 1, nf do out[#out + 1] = rows[k][m] end
end
redis.call('DEL', KEYS[1])
redis.call('HSET', KEYS[1], 'state', 'done')
redis.call('EXPIRE', KEYS[1], op_ttl)
return out
"""
)


class RedisBudgetStore(BudgetStore):
    """Atomic budget counters in Redis. Any Redis or connection error fails closed."""

    kind: ClassVar[str] = "redis"

    def __init__(self, client: Redis) -> None:
        self._client = client
        self._reserve = client.register_script(_RESERVE)
        self._settle = client.register_script(_SETTLE)

    @classmethod
    def from_url(
        cls, url: str, *, password: str | None, timeout_s: float = 1.0
    ) -> "RedisBudgetStore":
        """A client that fails fast: a dead Redis must refuse calls, not stall them."""
        client = Redis.from_url(  # pyright: ignore[reportUnknownMemberType] -- untyped **kwargs in redis-py
            url,
            password=password,
            socket_connect_timeout=timeout_s,
            socket_timeout=timeout_s,
            health_check_interval=30,
        )
        return cls(client)

    async def reserve(
        self, op_id: str, scopes: Sequence[BudgetScope], amount: Spend, meters: frozenset[Meter]
    ) -> ReserveOutcome:
        args = [*_header(), *(str(amount[meter]) for meter in _METERS)]
        for scope in scopes:
            args.append(str(scope.ttl_s))
            for meter in _METERS:
                limit = scope.limits[meter]
                args.append(str(limit) if meter in meters and limit is not None else _NO_LIMIT)
        reply = await self._run(self._reserve, op_id, scopes, args)
        if reply[0] == _DUPLICATE:
            return ReserveOutcome(granted=False, duplicate=True)
        if reply[0] == _REFUSED:
            return ReserveOutcome(
                granted=False, exceeded_scope=reply[1] - 1, exceeded_meter=_METERS[reply[2] - 1]
            )
        return ReserveOutcome(granted=True, usage=_usage(reply[1:], len(scopes)))

    async def settle(
        self, op_id: str, scopes: Sequence[BudgetScope], spent: Spend
    ) -> tuple[Spend, ...]:
        args = [*_header(), *(str(spent[meter]) for meter in _METERS)]
        args += [str(scope.ttl_s) for scope in scopes]
        return _usage(await self._run(self._settle, op_id, scopes, args), len(scopes))

    async def usage(self, scope: BudgetScope) -> Spend:
        try:
            values = await self._client.hmget(scope.key, list(_FIELDS))
        except (RedisError, OSError) as exc:
            raise BudgetStoreUnavailableError from exc
        return Spend.of({m: int(v) for m, v in zip(_METERS, values, strict=True) if v is not None})

    async def healthy(self) -> bool:
        try:
            return bool(await self._client.ping())  # pyright: ignore[reportUnknownMemberType] -- untyped **kwargs in redis-py
        except (RedisError, OSError):
            return False

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    async def _run(
        script: AsyncScript, op_id: str, scopes: Sequence[BudgetScope], args: list[str]
    ) -> list[int]:
        keys = [op_key(op_id), *(scope.key for scope in scopes)]
        try:
            reply: object = await script(keys=keys, args=args)
        except (RedisError, OSError) as exc:
            raise BudgetStoreUnavailableError from exc
        return [int(value) for value in cast("list[int]", reply)]


def _header() -> list[str]:
    return [str(len(_FIELDS)), *_FIELDS, str(OP_TTL_S), str(MAX_COUNTER)]


def _usage(values: Sequence[int], scopes: int) -> tuple[Spend, ...]:
    width = len(_METERS)
    return tuple(
        Spend.of(dict(zip(_METERS, values[i * width : (i + 1) * width], strict=True)))
        for i in range(scopes)
    )
