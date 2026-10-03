"""Session state: binding, lifetime, per-session serialization, risk and taint.

SPEC "Risk score and taint". A session is bound on first use to ``(sub, act.sub, mode)``;
refreshing the token keeps its state. Risk lives in ``[0, 1]`` and decays exponentially with
``risk.half_life_s``; taint is sticky until the session ends. A session ends on
``DELETE /v1/session``, after ``sessions.idle_ttl_s`` without calls or after
``sessions.max_lifetime_s``, and an ended session is never revived.

The stored state *is* the read-only `SessionContext` controls see: every update builds a new
snapshot. `InMemorySessionStore` serves a single gateway process; a Redis store implements the
same interface once state must survive restarts or be shared.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final

from pydantic import AwareDatetime, Field

from gateway.clock import Clock, utc_now
from gateway.core.envelope import (
    CallRecord,
    Cooldown,
    FlaggedToolCall,
    FrozenModel,
    SessionContext,
)
from gateway.core.types import SessionMode
from gateway.errors import RejectionError
from gateway.policy.schema import Sessions

MAX_CALL_HISTORY: Final = 256
# Newest kept. Each flag comes from a judged-misaligned tool call that also adds risk, so a
# session flooding past this cap is frozen by its risk rules long before it gets there.
MAX_FLAGGED_TOOL_CALLS: Final = 1024
MAX_GOAL_CHARS: Final = 4000


class SessionReason(StrEnum):
    BINDING_MISMATCH = "session_binding_mismatch"
    ENDED = "session_ended"


class SessionError(RejectionError):
    def __init__(self, reason: SessionReason) -> None:
        messages = {
            SessionReason.BINDING_MISMATCH: "session belongs to another principal, agent or mode",
            SessionReason.ENDED: "session has ended; request a token with a new session_id",
        }
        super().__init__(reason.value, messages[reason])
        self.reason = reason
        self.status_code = 401 if reason is SessionReason.ENDED else 403


class SessionBinding(FrozenModel):
    """What a session is bound to on first use."""

    principal: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    mode: SessionMode

    def matches(self, ctx: SessionContext) -> bool:
        return (ctx.principal, ctx.actor, ctx.mode) == (self.principal, self.actor, self.mode)


class SessionUpdate(FrozenModel):
    """What one call changes in its session, persisted before the result is released."""

    risk_delta: float = Field(default=0.0, ge=0.0)  # raw sum; the store clamps the result
    taint: bool = False
    freeze_until: AwareDatetime | None = None
    cooldowns: tuple[Cooldown, ...] = ()
    calls: tuple[CallRecord, ...] = ()
    flagged_tool_calls: tuple[FlaggedToolCall, ...] = ()
    goal: str | None = Field(default=None, max_length=MAX_GOAL_CHARS, repr=False)  # set once


def apply_update(
    ctx: SessionContext, update: SessionUpdate, *, now: datetime, half_life_s: float
) -> SessionContext:
    """The session after ``update``: decay, add, clamp; taint and running timers are kept, and
    so is the goal once set (a later call can never replace it)."""
    risk = min(max(ctx.risk_at(now, half_life_s) + update.risk_delta, 0.0), 1.0)
    cooldowns: dict[str, Cooldown] = {}
    for cooldown in (*ctx.cooldowns, *update.cooldowns):
        if cooldown.until > now and (
            cooldown.key not in cooldowns or cooldown.until > cooldowns[cooldown.key].until
        ):
            cooldowns[cooldown.key] = cooldown
    freezes = [f for f in (ctx.freeze_until, update.freeze_until) if f is not None and f > now]
    return ctx.model_copy(
        update={
            "risk": risk,
            "risk_updated_at": now,
            "taint": ctx.taint or update.taint,
            "last_seen": now,
            "freeze_until": max(freezes, default=None),
            "cooldowns": tuple(cooldowns.values()),
            "call_history": (*ctx.call_history, *update.calls)[-MAX_CALL_HISTORY:],
            "flagged_tool_calls": tuple(
                dict.fromkeys((*ctx.flagged_tool_calls, *update.flagged_tool_calls))
            )[-MAX_FLAGGED_TOOL_CALLS:],
            "goal": ctx.goal if ctx.goal is not None else update.goal,
        }
    )


class SessionStore(ABC):
    """Session state shared by the LLM and MCP entry points."""

    @abstractmethod
    def lock(self, session_id: str) -> AbstractAsyncContextManager[None]:
        """Serialize calls in one session: hold this from admission until state is persisted."""

    @abstractmethod
    async def open(
        self, session_id: str, binding: SessionBinding, limits: Sessions
    ) -> SessionContext:
        """Get or create the session, enforcing its binding and lifetime; counts as activity."""

    @abstractmethod
    async def get(self, session_id: str) -> SessionContext | None:
        """Current state of a live session, without touching it."""

    @abstractmethod
    async def apply(
        self, session_id: str, update: SessionUpdate, *, half_life_s: float
    ) -> SessionContext:
        """Persist one call's effects and return the new state."""

    @abstractmethod
    async def end(self, session_id: str) -> None:
        """End the session for good; later tokens naming it are refused."""

    @abstractmethod
    async def is_retired(self, session_id: str) -> bool:
        """True once the session ended or expired. A retired id is never live again: its
        taint and risk must not come back clean under a fresh session."""

    @abstractmethod
    async def tainted_count(self) -> int:
        """Number of live tainted sessions (the ``acl_tainted_sessions`` gauge)."""


@dataclass(slots=True)
class _LockEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class InMemorySessionStore(SessionStore):
    """Single-process store. Expired sessions are retired lazily, when next touched, and
    never while a call holds or waits for their lock: that call persists its state first.

    Tombstones of ended and expired sessions are kept for the life of the process (a short
    string each), so an id can never be revived with clean state, however late a token naming
    it arrives. A shared store must keep them at least ``max(sessions.max_lifetime_s, maximum
    token lifetime)`` and the demo issuer must keep refusing retired ids.
    """

    def __init__(self, *, clock: Clock = utc_now) -> None:
        self._clock = clock
        self._sessions: dict[str, SessionContext] = {}
        self._ended: dict[str, datetime] = {}  # tombstones: session_id -> retired at
        self._locks: dict[str, _LockEntry] = {}
        self._limits = Sessions()

    @asynccontextmanager
    async def _held(self, session_id: str) -> AsyncGenerator[None]:
        entry = self._locks.setdefault(session_id, _LockEntry())
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0:
                del self._locks[session_id]

    def lock(self, session_id: str) -> AbstractAsyncContextManager[None]:
        return self._held(session_id)

    async def open(
        self, session_id: str, binding: SessionBinding, limits: Sessions
    ) -> SessionContext:
        now = self._clock()
        self._limits = limits
        self._retire_expired(now)  # other sessions; those in use are skipped
        current = self._sessions.get(session_id)
        if current is not None and self._expired(current, now):
            # The caller holds this session's lock to admit a new call: nothing is in flight.
            self._retire(session_id, now)
        if session_id in self._ended:
            raise SessionError(SessionReason.ENDED)
        ctx = self._sessions.get(session_id)
        if ctx is None:
            ctx = SessionContext(
                session_id=session_id,
                principal=binding.principal,
                actor=binding.actor,
                mode=binding.mode,
                risk_updated_at=now,
                created_at=now,
                last_seen=now,
            )
        elif not binding.matches(ctx):
            raise SessionError(SessionReason.BINDING_MISMATCH)
        else:
            ctx = ctx.model_copy(update={"last_seen": now})
        self._sessions[session_id] = ctx
        return ctx

    async def get(self, session_id: str) -> SessionContext | None:
        self._retire_expired(self._clock())
        return self._sessions.get(session_id)

    async def apply(
        self, session_id: str, update: SessionUpdate, *, half_life_s: float
    ) -> SessionContext:
        ctx = self._sessions.get(session_id)
        if ctx is None:
            raise SessionError(SessionReason.ENDED)
        updated = apply_update(ctx, update, now=self._clock(), half_life_s=half_life_s)
        self._sessions[session_id] = updated
        return updated

    async def end(self, session_id: str) -> None:
        self._retire(session_id, self._clock())

    async def is_retired(self, session_id: str) -> bool:
        self._retire_expired(self._clock())
        return session_id in self._ended

    async def tainted_count(self) -> int:
        self._retire_expired(self._clock())
        return sum(1 for ctx in self._sessions.values() if ctx.taint)

    def _expired(self, ctx: SessionContext, now: datetime) -> bool:
        idle = now - ctx.last_seen >= timedelta(seconds=self._limits.idle_ttl_s)
        too_old = now - ctx.created_at >= timedelta(seconds=self._limits.max_lifetime_s)
        return idle or too_old

    def _retire(self, session_id: str, now: datetime) -> None:
        self._sessions.pop(session_id, None)
        self._ended.setdefault(session_id, now)

    def _retire_expired(self, now: datetime) -> None:
        for session_id, ctx in list(self._sessions.items()):
            if session_id not in self._locks and self._expired(ctx, now):
                self._retire(session_id, now)
