"""Tools ``tool_pinning`` caught drifting from their baseline, blocked until an operator acts.

SPEC: any change "blocks the tool until re-approved". A mismatch seen in one listing is not
enough on its own: another session may still hold an earlier, matching listing, and a server
that restores its metadata would otherwise be trusted again without anyone looking. So the
first time a pinned tool is advertised with a different digest (or twice), the gateway
records ``(server, tool, baseline digest)`` here, shared by every session and every gateway,
and every ``tools/list`` and ``tools/call`` of that server refuses the tool while the record
stands. It is cleared only by an operator:

- re-approving a new baseline (``acl pin <server> --write``): a record whose baseline digest
  is no longer the pinned one is stale and is dropped on the next check;
- or explicitly (``acl pin <server> --clear-quarantine <tool>``, ``POST
  /admin/mcp/{server}/quarantine/{tool}/clear``) after confirming the tool is the approved one.

A tool that is merely missing from a listing is refused for that listing but not quarantined:
absence exposes nothing, and an identical tool coming back is exactly the approved one.

`RedisToolQuarantine` keeps one hash per server (``acl:tool_quarantine:<server>``, no TTL);
`InMemoryToolQuarantine` serves one process (tests, local development). An unreachable store
fails closed (``tool_quarantine_unavailable``, 503).
"""

from abc import ABC, abstractmethod
from typing import Final, cast, override

from pydantic import AwareDatetime, Field, ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from gateway.core.envelope import FrozenModel
from gateway.errors import RejectionError

KEY_PREFIX: Final = "acl:tool_quarantine"


class ToolQuarantineUnavailableError(RejectionError):
    status_code = 503

    def __init__(self) -> None:
        super().__init__("tool_quarantine_unavailable", "the tool quarantine cannot be read")


class QuarantineEntry(FrozenModel):
    """Why one tool of one server is quarantined (digests only, never the tool's text)."""

    tool: str = Field(min_length=1)
    pin_digest: str = Field(min_length=1)  # the baseline in force when the drift was seen
    advertised_digest: str  # what the server advertised instead ("" for a duplicate entry)
    reason: str = Field(min_length=1)
    detected_at: AwareDatetime


class ToolQuarantine(ABC):
    """Quarantine records, one per (server, tool). Every write is a compare-and-set, so two
    gateways checking the same server never lose each other's detections:

    - `record` keeps an existing entry for the *same* baseline (the first detection) and
      atomically replaces one recorded against an obsolete baseline, so a drift seen right
      after a re-approval is never dropped together with the stale entry it replaces;
    - `discard` deletes only the exact entry the caller inspected (stale-baseline cleanup);
    - `clear` is the operator's unconditional lift.
    """

    @abstractmethod
    async def record(self, server: str, entry: QuarantineEntry) -> None:
        """Quarantine a tool against ``entry.pin_digest`` (see the class docstring)."""

    @abstractmethod
    async def entries(self, server: str) -> dict[str, QuarantineEntry]:
        """Every quarantined tool of ``server``, by name."""

    @abstractmethod
    async def discard(self, server: str, entry: QuarantineEntry) -> bool:
        """Delete ``entry`` only if it is still exactly the stored one."""

    @abstractmethod
    async def clear(self, server: str, tool: str) -> bool:
        """Lift one tool's quarantine (operator); False when it had none."""


class InMemoryToolQuarantine(ToolQuarantine):
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, QuarantineEntry]] = {}

    @override
    async def record(self, server: str, entry: QuarantineEntry) -> None:
        tools = self._entries.setdefault(server, {})
        current = tools.get(entry.tool)
        if current is None or current.pin_digest != entry.pin_digest:
            tools[entry.tool] = entry

    @override
    async def entries(self, server: str) -> dict[str, QuarantineEntry]:
        return dict(self._entries.get(server, {}))

    @override
    async def discard(self, server: str, entry: QuarantineEntry) -> bool:
        tools = self._entries.get(server, {})
        if tools.get(entry.tool) != entry:
            return False
        del tools[entry.tool]
        return True

    @override
    async def clear(self, server: str, tool: str) -> bool:
        return self._entries.get(server, {}).pop(tool, None) is not None


# Values are "<pin digest> <entry JSON>", so the scripts compare baselines without parsing JSON.
# KEYS[1]: the server's hash. ARGV: tool, pin digest, value.
_RECORD: Final = """
local current = redis.call('HGET', KEYS[1], ARGV[1])
if current and string.sub(current, 1, string.len(ARGV[2]) + 1) == ARGV[2] .. ' ' then
  return 0
end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[3])
return 1
"""

# KEYS[1]: the server's hash. ARGV: tool, the exact value inspected.
_DISCARD: Final = """
if redis.call('HGET', KEYS[1], ARGV[1]) == ARGV[2] then
  return redis.call('HDEL', KEYS[1], ARGV[1])
end
return 0
"""


def _value(entry: QuarantineEntry) -> str:
    return f"{entry.pin_digest} {entry.model_dump_json()}"


class RedisToolQuarantine(ToolQuarantine):
    def __init__(self, client: Redis) -> None:
        self._client = client
        self._record = client.register_script(_RECORD)
        self._discard = client.register_script(_DISCARD)

    @staticmethod
    def _key(server: str) -> str:
        return f"{KEY_PREFIX}:{server}"

    @override
    async def record(self, server: str, entry: QuarantineEntry) -> None:
        args = [entry.tool, entry.pin_digest, _value(entry)]
        try:
            await self._record(keys=[self._key(server)], args=args)
        except (RedisError, OSError) as error:
            raise ToolQuarantineUnavailableError from error

    @override
    async def entries(self, server: str) -> dict[str, QuarantineEntry]:
        try:
            raw = cast("dict[bytes, bytes]", await self._client.hgetall(self._key(server)))  # pyright: ignore[reportUnknownMemberType] -- untyped in redis-py
            return {
                name.decode(): QuarantineEntry.model_validate_json(value.partition(b" ")[2])
                for name, value in raw.items()
            }
        except (RedisError, OSError, ValidationError, UnicodeDecodeError) as error:
            raise ToolQuarantineUnavailableError from error

    @override
    async def discard(self, server: str, entry: QuarantineEntry) -> bool:
        try:
            removed = await self._discard(
                keys=[self._key(server)], args=[entry.tool, _value(entry)]
            )
        except (RedisError, OSError) as error:
            raise ToolQuarantineUnavailableError from error
        return bool(removed)

    @override
    async def clear(self, server: str, tool: str) -> bool:
        try:
            removed = await self._client.hdel(self._key(server), tool)  # pyright: ignore[reportUnknownMemberType] -- untyped in redis-py
        except (RedisError, OSError) as error:
            raise ToolQuarantineUnavailableError from error
        return bool(removed)
