"""``loop_detect``: the same call repeated too often in one session is an agent stuck in a loop.

A call's identity is its `CallFingerprint`: channel, MCP server, action, resource and a digest
of the canonical (sorted-key JSON) payload with volatile fields removed, so re-sending the
identical operation matches however the client formats it. Each pre evaluation records one
occurrence; the call that makes it more than ``max_repeats`` occurrences within ``window_s``
is blocked (``loop_detected``), and so is every further repeat until older ones leave the
window. Any other call (different arguments, tool or resource) has its own count.

Calls retrying a held operation with its ``approval_id`` are neither counted nor blocked: the
retry is how an approval is consumed (SPEC "Human in the loop").

Counts live behind `CallCounter`: `InMemoryCallCounter` serves one gateway process; a shared
store (Redis on the ``state`` network) implements the same interface.
"""

import hashlib
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
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


class CallCounter(ABC):
    """Sliding-window occurrence counts per session and fingerprint."""

    @abstractmethod
    async def hit(self, session_id: str, key: str, now: datetime, window_s: float) -> int:
        """Record one occurrence of ``key`` at ``now``; return the occurrences within the
        ``window_s`` ending at ``now``, this one included."""


class InMemoryCallCounter(CallCounter):
    """Single-process counts. Expired occurrences are dropped when their key is next hit, and
    every `SWEEP_EVERY` hits a sweep drops sessions with nothing left in any window."""

    def __init__(self) -> None:
        self._seen: dict[str, dict[str, deque[datetime]]] = {}
        self._horizon: dict[str, datetime] = {}  # session -> when its last occurrence expires
        self._hits = 0

    @override
    async def hit(self, session_id: str, key: str, now: datetime, window_s: float) -> int:
        window = timedelta(seconds=window_s)
        occurrences = self._seen.setdefault(session_id, {}).setdefault(key, deque())
        while occurrences and occurrences[0] <= now - window:
            occurrences.popleft()
        occurrences.append(now)
        self._horizon[session_id] = max(self._horizon.get(session_id, now), now + window)
        self._hits += 1
        if self._hits % SWEEP_EVERY == 0:
            self._sweep(now)
        return len(occurrences)

    def _sweep(self, now: datetime) -> None:
        for session_id, horizon in list(self._horizon.items()):
            if horizon <= now:
                del self._horizon[session_id]
                self._seen.pop(session_id, None)

    def __len__(self) -> int:
        """Sessions with counts held (for tests)."""
        return len(self._seen)


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
        count = await self._counter.hit(
            interaction.session_id,
            CallFingerprint.of(interaction).key,
            self._clock(),
            limits.window_s,
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
