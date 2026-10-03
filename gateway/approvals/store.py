"""Where approval records live, behind one interface whose writes are atomic.

`ApprovalStore.create_or_get` is the idempotency story: an operation (the keyed digest of
its binding) owns one slot. While the slot's approval is open (pending, approved,
executing) every hold of the same operation gets that approval back; once it has ended a new
approval takes the slot under the next generation's id. Concurrent holds of one operation
therefore never create two pending entries.

`ApprovalStore.transition` is the state machine's guard: it applies a change only from one of
the target's source states, and first expires a pending or approved record whose deadline has
passed, so ``approved → executing`` can never consume an approval that ran out, and two
consumers can never both win.

`RedisApprovalStore` (`gateway.approvals.redis_store`) is the production store; the in-memory
one below serves tests and an explicitly chosen ``memory`` development setup. It is atomic
because no method awaits between its check and its write.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import datetime
from typing import ClassVar, Final

from gateway.approvals.model import (
    EXPIRING_STATES,
    OPEN_STATES,
    Approval,
    ApprovalConflictError,
    ApprovalDraft,
    ApprovalState,
    Transition,
    approval_id_for,
)
from gateway.errors import RejectionError

LIST_LIMIT: Final = 500


class ApprovalStoreUnavailableError(RejectionError):
    """The store cannot be reached: nothing held can be created, decided or consumed."""

    status_code = 503

    def __init__(self) -> None:
        super().__init__(
            "approval_store_unavailable", "the approval queue is unavailable; retry later"
        )


class ApprovalStore(ABC):
    """Approval records. Every method but `healthy` raises `ApprovalStoreUnavailableError`
    when the backing service cannot be reached; callers fail closed."""

    kind: ClassVar[str]

    @abstractmethod
    async def create_or_get(
        self, draft: ApprovalDraft, *, expires_at: datetime, retention_s: int
    ) -> tuple[Approval, bool]:
        """The open approval of ``draft.operation``, or a new pending one; True if created.

        An open record past its deadline is expired first (it no longer holds the slot).
        """

    @abstractmethod
    async def get(self, approval_id: str) -> Approval | None:
        """The record as stored (callers expire it lazily through `transition`)."""

    @abstractmethod
    async def transition(self, approval_id: str, change: Transition) -> Approval:
        """Apply ``change`` atomically or raise `ApprovalConflictError` with the current
        record. A pending or approved record past its deadline is expired first."""

    @abstractmethod
    async def records(
        self, states: frozenset[ApprovalState] | None = None, limit: int = LIST_LIMIT
    ) -> list[Approval]:
        """Records newest first, optionally only in ``states``."""

    @abstractmethod
    async def due(self, now: datetime) -> list[str]:
        """Ids of pending or approved records whose deadline has passed."""

    @abstractmethod
    async def pending_count(self) -> int: ...

    @abstractmethod
    async def healthy(self) -> bool:
        """True when the store answers; never raises."""

    async def aclose(self) -> None:  # noqa: B027 -- optional hook, the memory store holds nothing
        """Release connections."""


def apply_transition(record: Approval, change: Transition) -> Approval:
    """The record after ``change``; the caller has checked the move is legal."""
    update: dict[str, object] = {"state": change.target, "updated_at": change.at}
    if change.decided_by is not None:
        update |= {"decided_by": change.decided_by, "decided_at": change.at}
    if change.note is not None:
        update["note"] = change.note
    if change.outcome is not None:
        update["outcome"] = change.outcome
    if change.expires_at is not None:
        update["expires_at"] = change.expires_at
    return record.model_copy(update=update)


def expired(record: Approval, at: datetime) -> Approval:
    return record.model_copy(
        update={"state": ApprovalState.EXPIRED, "updated_at": at, "outcome": "approval_timeout"}
    )


class InMemoryApprovalStore(ApprovalStore):
    """One process, lost on restart. Never chosen implicitly."""

    kind: ClassVar[str] = "memory"

    def __init__(self) -> None:
        self._records: dict[str, Approval] = {}
        self._slots: dict[str, tuple[str, int]] = {}  # operation -> (current id, generation)

    async def create_or_get(
        self, draft: ApprovalDraft, *, expires_at: datetime, retention_s: int
    ) -> tuple[Approval, bool]:
        del retention_s  # nothing outlives the process anyway
        now = draft.created_at
        approval_id, generation = self._slots.get(draft.operation, ("", 0))
        current = self._records.get(approval_id)
        if current is not None:
            if current.expired_at(now):
                current = self._records[approval_id] = expired(current, now)
            if current.state in OPEN_STATES:
                return current, False
        generation += 1
        record = Approval(
            **draft.model_dump(),
            id=approval_id_for(draft.operation, generation),
            state=ApprovalState.PENDING,
            expires_at=expires_at,
            updated_at=now,
        )
        self._records[record.id] = record
        self._slots[draft.operation] = (record.id, generation)
        return record, True

    async def get(self, approval_id: str) -> Approval | None:
        return self._records.get(approval_id)

    async def transition(self, approval_id: str, change: Transition) -> Approval:
        record = self._records.get(approval_id)
        if record is None:
            raise ApprovalConflictError(approval_id, None)
        if record.expired_at(change.at):
            record = self._records[approval_id] = expired(record, change.at)
            if change.target is ApprovalState.EXPIRED:
                return record
        if change.target is ApprovalState.EXPIRED or record.state not in change.sources:
            raise ApprovalConflictError(approval_id, record)
        record = self._records[approval_id] = apply_transition(record, change)
        return record

    async def records(
        self, states: frozenset[ApprovalState] | None = None, limit: int = LIST_LIMIT
    ) -> list[Approval]:
        newest = sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)
        return _take(newest, states, limit)

    async def due(self, now: datetime) -> list[str]:
        return [
            r.id
            for r in self._records.values()
            if r.state in EXPIRING_STATES and now >= r.expires_at
        ]

    async def pending_count(self) -> int:
        return sum(1 for r in self._records.values() if r.state is ApprovalState.PENDING)

    async def healthy(self) -> bool:
        return True


def _take(
    records: Iterable[Approval], states: frozenset[ApprovalState] | None, limit: int
) -> list[Approval]:
    chosen: list[Approval] = []
    for record in records:
        if states is None or record.state in states:
            chosen.append(record)
            if len(chosen) >= limit:
                break
    return chosen
