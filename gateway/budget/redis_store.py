"""Budget counters in Redis (``redis:7.2``, BSD-3), on the ``state`` network only the gateway joins.

Each scope is one hash (``acl:budget:<kind>:<window>:<subject>``) with one integer field per
meter. Reserve and adjust are Lua scripts, so the check and the increment happen in one round
trip that no other client can interleave with. A hash gets its TTL when it is created (two
days for a daily scope, the session lifetime for a session) and it is never extended, so
counters disappear on their own.

Amounts travel as the decimal strings of exact integers and go to ``HINCRBY`` as given; the
scripts only convert them to Lua numbers to compare, which is exact below 2^53.
"""

from collections.abc import Mapping, Sequence
from typing import ClassVar, Final, cast

from redis.asyncio import Redis
from redis.commands.core import AsyncScript
from redis.exceptions import RedisError

from gateway.budget.model import (
    BudgetScope,
    BudgetStoreUnavailableError,
    Meter,
    ReserveOutcome,
    Spend,
)
from gateway.budget.store import BudgetStore

_METERS: Final = tuple(Meter)
_FIELDS: Final = tuple(meter.value for meter in _METERS)
_NO_LIMIT: Final = "-1"

# ARGV: the meter fields, then per key: ttl, then (amount, limit) per meter (limit -1 = none).
# Returns {0, key index, meter index} when refused, else {1, usage per key per meter ...}.
_RESERVE: Final = """
local fields = {}
for i = 1, tonumber(ARGV[1]) do fields[i] = ARGV[1 + i] end
local nf = #fields
local stride = 1 + 2 * nf
local function base(k) return 1 + nf + (k - 1) * stride end
for k, key in ipairs(KEYS) do
  local current = redis.call('HMGET', key, unpack(fields))
  for m = 1, nf do
    local limit = tonumber(ARGV[base(k) + 2 * m + 1])
    if limit >= 0 then
      local used = tonumber(current[m]) or 0
      if used >= limit or used + tonumber(ARGV[base(k) + 2 * m]) > limit then
        return {0, k, m}
      end
    end
  end
end
local out = {1}
for k, key in ipairs(KEYS) do
  for m = 1, nf do
    local amount = ARGV[base(k) + 2 * m]
    if tonumber(amount) ~= 0 then
      out[#out + 1] = redis.call('HINCRBY', key, fields[m], amount)
    else
      out[#out + 1] = tonumber(redis.call('HGET', key, fields[m])) or 0
    end
  end
  if redis.call('TTL', key) == -1 then redis.call('EXPIRE', key, ARGV[base(k) + 1]) end
end
return out
"""

# ARGV: the meter fields, then per key: ttl, then one signed delta per meter.
# Returns the usage per key per meter after the adjustment, floored at zero.
_ADJUST: Final = """
local fields = {}
for i = 1, tonumber(ARGV[1]) do fields[i] = ARGV[1 + i] end
local nf = #fields
local stride = 1 + nf
local out = {}
for k, key in ipairs(KEYS) do
  local b = 1 + nf + (k - 1) * stride  -- ARGV[b + 1] is the ttl, then the deltas
  for m = 1, nf do
    local delta = ARGV[b + 1 + m]
    local value
    if tonumber(delta) ~= 0 then
      value = redis.call('HINCRBY', key, fields[m], delta)
      if value < 0 then
        redis.call('HSET', key, fields[m], 0)
        value = 0
      end
    else
      value = tonumber(redis.call('HGET', key, fields[m])) or 0
    end
    out[#out + 1] = value
  end
  if redis.call('TTL', key) == -1 then redis.call('EXPIRE', key, ARGV[b + 1]) end
end
return out
"""


class RedisBudgetStore(BudgetStore):
    """Atomic budget counters in Redis. Any Redis or connection error fails closed."""

    kind: ClassVar[str] = "redis"

    def __init__(self, client: Redis) -> None:
        self._client = client
        self._reserve = client.register_script(_RESERVE)
        self._adjust = client.register_script(_ADJUST)

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
        self, scopes: Sequence[BudgetScope], amount: Spend, meters: frozenset[Meter]
    ) -> ReserveOutcome:
        args = [*_header()]
        for scope in scopes:
            args.append(str(scope.ttl_s))
            for meter in _METERS:
                limit = scope.limits[meter]
                in_force = meter in meters and limit is not None
                args += [str(amount[meter]), str(limit) if in_force else _NO_LIMIT]
        reply = await self._run(self._reserve, scopes, args)
        if reply[0] == 0:
            return ReserveOutcome(
                granted=False, exceeded_scope=reply[1] - 1, exceeded_meter=_METERS[reply[2] - 1]
            )
        return ReserveOutcome(granted=True, usage=_usage(reply[1:], len(scopes)))

    async def adjust(
        self, scopes: Sequence[BudgetScope], delta: Mapping[Meter, int]
    ) -> tuple[Spend, ...]:
        args = [*_header()]
        for scope in scopes:
            args.append(str(scope.ttl_s))
            args += [str(delta.get(meter, 0)) for meter in _METERS]
        return _usage(await self._run(self._adjust, scopes, args), len(scopes))

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
        script: AsyncScript, scopes: Sequence[BudgetScope], args: list[str]
    ) -> list[int]:
        try:
            reply: object = await script(keys=[scope.key for scope in scopes], args=args)
        except (RedisError, OSError) as exc:
            raise BudgetStoreUnavailableError from exc
        return [int(value) for value in cast("list[int]", reply)]


def _header() -> list[str]:
    return [str(len(_FIELDS)), *_FIELDS]


def _usage(values: Sequence[int], scopes: int) -> tuple[Spend, ...]:
    width = len(_METERS)
    return tuple(
        Spend.of(dict(zip(_METERS, values[i * width : (i + 1) * width], strict=True)))
        for i in range(scopes)
    )
