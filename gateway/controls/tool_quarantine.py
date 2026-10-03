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
    @abstractmethod
    async def add(self, server: str, entry: QuarantineEntry) -> None:
        """Quarantine a tool; an existing record (the first detection) is kept."""

    @abstractmethod
    async def entries(self, server: str) -> dict[str, QuarantineEntry]:
        """Every quarantined tool of ``server``, by name."""

    @abstractmethod
    async def clear(self, server: str, tool: str) -> bool:
        """Lift one tool's quarantine; False when it had none."""


class InMemoryToolQuarantine(ToolQuarantine):
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, QuarantineEntry]] = {}

    @override
    async def add(self, server: str, entry: QuarantineEntry) -> None:
        self._entries.setdefault(server, {}).setdefault(entry.tool, entry)

    @override
    async def entries(self, server: str) -> dict[str, QuarantineEntry]:
        return dict(self._entries.get(server, {}))

    @override
    async def clear(self, server: str, tool: str) -> bool:
        return self._entries.get(server, {}).pop(tool, None) is not None


class RedisToolQuarantine(ToolQuarantine):
    def __init__(self, client: Redis) -> None:
        self._client = client

    @staticmethod
    def _key(server: str) -> str:
        return f"{KEY_PREFIX}:{server}"

    @override
    async def add(self, server: str, entry: QuarantineEntry) -> None:
        try:
            await self._client.hsetnx(self._key(server), entry.tool, entry.model_dump_json())  # pyright: ignore[reportUnknownMemberType] -- untyped in redis-py
        except (RedisError, OSError) as error:
            raise ToolQuarantineUnavailableError from error

    @override
    async def entries(self, server: str) -> dict[str, QuarantineEntry]:
        try:
            raw = cast("dict[bytes, bytes]", await self._client.hgetall(self._key(server)))  # pyright: ignore[reportUnknownMemberType] -- untyped in redis-py
            return {
                name.decode(): QuarantineEntry.model_validate_json(value)
                for name, value in raw.items()
            }
        except (RedisError, OSError, ValidationError, UnicodeDecodeError) as error:
            raise ToolQuarantineUnavailableError from error

    @override
    async def clear(self, server: str, tool: str) -> bool:
        try:
            removed = await self._client.hdel(self._key(server), tool)  # pyright: ignore[reportUnknownMemberType] -- untyped in redis-py
        except (RedisError, OSError) as error:
            raise ToolQuarantineUnavailableError from error
        return bool(removed)
