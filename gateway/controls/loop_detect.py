"""``loop_detect``: the same call repeated too often in one session is an agent stuck in a loop.

A call's identity is its `CallFingerprint`: channel, MCP server, action, resource and a digest
of the canonical (sorted-key JSON) payload with volatile fields removed, so re-sending the
identical operation matches however the client formats it. Each pre evaluation records one
occurrence; the call that makes it more than ``max_repeats`` occurrences within ``window_s``
is blocked (``loop_detected``), and so is every further repeat until older ones leave the
window. Any other call (different arguments, tool or resource) has its own count.

Calls retrying a held operation with its ``approval_id`` are neither counted nor blocked: the
retry is how an approval is consumed (SPEC "Human in the loop").

Counts live behind `CallCounter`: `InMemoryCallCounter` serves one gateway process,
`gateway.controls.loop_counter_redis.RedisCallCounter` every gateway sharing the ``state``
Redis. A counter that cannot count fails closed (``loop_store_unavailable``).
"""

import hashlib
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, ClassVar, Final, Self, cast, override

from gateway.clock import Clock, utc_now
from gateway.controls.scope import current_scope
from gateway.core.envelope import FrozenModel, Interaction, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import Action, Channel, ControlKind, ControlMode, Decision, Stage
from gateway.policy.schema import LoopDetectConfig
from gateway.telemetry import canonical_json

LOOP_DETECTED: Final = "loop_detected"
STORE_UNAVAILABLE: Final = "loop_store_unavailable"
# Top-level payload fields that change how an answer is delivered or attributed, not what the
# call does: an LLM client toggling streaming is still repeating the same request.
VOLATILE_FIELDS: Final = frozenset({"stream", "stream_options", "user", "metadata", "_meta"})
SWEEP_EVERY: Final = 1024


class CallFingerprint(FrozenModel):
    """What makes two calls "the same call"."""

    channel: Channel
    server: str | None  # MCP upstream; None on other channels
    action: Action
    resource: str
    digest: str  # SHA-256 of the canonical payload without volatile fields

    @classmethod
    def of(cls, interaction: Interaction) -> Self:
        payload: object = interaction.payload
        if isinstance(payload, Mapping):
            fields = cast("Mapping[str, Any]", payload)
            payload = {k: v for k, v in fields.items() if k not in VOLATILE_FIELDS}
        return cls(
            channel=interaction.channel,
            server=interaction.server,
            action=interaction.action,
            resource=interaction.resource,
            digest=hashlib.sha256(canonical_json(payload)).hexdigest(),
        )

    @property
    def key(self) -> str:
        return "|".join((self.channel, self.server or "", self.action, self.resource, self.digest))


class CallCounterUnavailableError(Exception):
    """The counter's store cannot be reached: the call is not counted, so it is refused."""


class CallCounter(ABC):
    """Sliding-window occurrence counts per session and fingerprint."""

    @abstractmethod
    async def hit(self, session_id: str, key: str, now: datetime, window_s: float) -> int:
        """Record one occurrence of ``key`` at ``now``; return the occurrences within the
        ``window_s`` ending at ``now``, this one included. Raises
        `CallCounterUnavailableError` when it cannot count."""


@dataclass(slots=True)
class _SessionCounts:
    """One session's occurrences, plus an expiry queue so dead fingerprints leave promptly."""

    occurrences: dict[str, deque[datetime]] = field(default_factory=dict[str, deque[datetime]])
    expires: dict[str, datetime] = field(default_factory=dict[str, datetime])  # key -> last expiry
    queue: deque[tuple[datetime, str]] = field(default_factory=deque[tuple[datetime, str]])
    horizon: datetime | None = None  # when the session's last occurrence leaves its window

    def expire(self, now: datetime) -> None:
        """Drop every fingerprint whose latest occurrence has left the window (amortized O(1))."""
        while self.queue and self.queue[0][0] <= now:
            _, key = self.queue.popleft()
            if key in self.expires and self.expires[key] <= now:
                del self.expires[key]
                del self.occurrences[key]


class InMemoryCallCounter(CallCounter):
    """Single-process counts, held only while they can still matter.

    Each hit first expires the session's fingerprints whose latest occurrence has left the
    window, so a session holds at most the fingerprints seen within one window; every
    `SWEEP_EVERY` hits a sweep does the same for every session and drops the empty ones.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionCounts] = {}
        self._hits = 0

    @override
    async def hit(self, session_id: str, key: str, now: datetime, window_s: float) -> int:
        window = timedelta(seconds=window_s)
        session = self._sessions.setdefault(session_id, _SessionCounts())
        session.expire(now)
        occurrences = session.occurrences.setdefault(key, deque())
        while occurrences and occurrences[0] <= now - window:
            occurrences.popleft()
        occurrences.append(now)
        expiry = now + window
        session.expires[key] = max(session.expires.get(key, expiry), expiry)
        session.queue.append((expiry, key))
        session.horizon = max(session.horizon or expiry, expiry)
        self._hits += 1
        if self._hits % SWEEP_EVERY == 0:
            self._sweep(now)
        return len(occurrences)

    def _sweep(self, now: datetime) -> None:
        for session_id, session in list(self._sessions.items()):
            session.expire(now)
            if not session.occurrences:
                del self._sessions[session_id]

    def __len__(self) -> int:
        """Sessions with counts held (for tests)."""
        return len(self._sessions)

    def held(self, session_id: str) -> tuple[int, int]:
        """(fingerprints, queued expiries) held for a session (for tests)."""
        session = self._sessions.get(session_id)
        return (len(session.occurrences), len(session.queue)) if session else (0, 0)


class LoopDetectControl(Control):
    id: ClassVar[str] = "loop_detect"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE})
    kind: ClassVar[ControlKind] = ControlKind.DETERMINISTIC

    def __init__(self, counter: CallCounter, *, clock: Clock = utc_now) -> None:
        self._counter = counter
        self._clock = clock

    @override
    async def evaluate(self, interaction: Interaction, stage: Stage, cfg: ControlConfig) -> Verdict:
        scope = current_scope()
        if scope is not None and scope.approval_id is not None:
            return Verdict(
                decision=Decision.ALLOW, control_id=self.id, reason_code="approval_retry"
            )
        limits = cfg if isinstance(cfg, LoopDetectConfig) else LoopDetectConfig()
        try:
            count = await self._counter.hit(
                interaction.session_id,
                CallFingerprint.of(interaction).key,
                self._clock(),
                limits.window_s,
            )
        except CallCounterUnavailableError:
            return Verdict(
                decision=Decision.BLOCK, control_id=self.id, reason_code=STORE_UNAVAILABLE
            )
        if count <= limits.max_repeats:
            return Verdict(decision=Decision.ALLOW, control_id=self.id, reason_code="no_loop")
        return Verdict(
            decision=Decision.BLOCK,
            control_id=self.id,
            reason_code=LOOP_DETECTED,
            reason=f"{count} identical calls within {limits.window_s:g}s",
            enforced=cfg.mode is not ControlMode.LOG_ONLY,
            risk_delta=cfg.risk_delta or 0.0,
        )
