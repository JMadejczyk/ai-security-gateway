"""Where budget counters live, behind one interface with an atomic check-and-hold.

`BudgetStore.reserve` is the whole correctness story: for every scope it checks each meter
the call draws on and, only if none would cross its limit (or ``MAX_COUNTER``), adds the
amounts to all scopes and records the hold under the call's operation id, in one atomic step.
Two concurrent calls can therefore never both take the last of a budget, and a check never
leaves a partial write behind.

`BudgetStore.settle` replaces an operation's hold with what the call actually spent. It is
idempotent by operation id: it applies only while the hold exists, then leaves a tombstone.
Settling with nothing spent is a release, and a release that arrives before a delayed
reservation makes that reservation refuse (the tombstone), so an ambiguous reserve outcome
can always be cleaned up.

Counter TTLs are (re)extended on every write, never shortened: daily counters outlive their
day, session counters outlive the longest session the current policy allows.

`RedisBudgetStore` (`gateway.budget.redis_store`) is the production store. The in-memory one
below serves tests and an explicitly chosen ``ACL_BUDGET_STORE=memory`` development setup: it
is atomic because a check and its update never await in between.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import ClassVar, Final

from gateway.budget.model import (
    MAX_COUNTER,
    BudgetScope,
    Meter,
    ReserveOutcome,
    Spend,
    op_key,
)
from gateway.clock import Clock, utc_now

OP_TTL_S: Final = 2 * 24 * 3600  # holds and tombstones; far longer than any upstream call


class BudgetStore(ABC):
    """Counters per scope. Every method but `healthy` raises `BudgetStoreUnavailableError`
    when the backing service cannot be reached; callers fail closed."""

    kind: ClassVar[str]

    @abstractmethod
    async def reserve(
        self, op_id: str, scopes: Sequence[BudgetScope], amount: Spend, meters: frozenset[Meter]
    ) -> ReserveOutcome:
        """Hold ``amount`` on every scope under ``op_id``, unless a limit of ``meters`` (or
        ``MAX_COUNTER``) would be crossed, or ``op_id`` was already used."""

    @abstractmethod
    async def settle(
        self, op_id: str, scopes: Sequence[BudgetScope], spent: Spend
    ) -> tuple[Spend, ...]:
        """Replace ``op_id``'s hold by ``spent`` (once); usage per scope afterwards."""

    @abstractmethod
    async def usage(self, scope: BudgetScope) -> Spend: ...

    async def usages(self, scopes: Sequence[BudgetScope]) -> tuple[Spend, ...]:
        return tuple([await self.usage(scope) for scope in scopes])

    @abstractmethod
    async def healthy(self) -> bool:
        """True when the store answers; never raises."""

    async def aclose(self) -> None:  # noqa: B027 -- optional hook, most stores hold nothing
        """Release connections."""


def first_exceeded(
    scopes: Sequence[BudgetScope],
    current: Sequence[Spend],
    amount: Spend,
    meters: frozenset[Meter],
) -> tuple[int, Meter] | None:
    """The first ``(scope index, meter)`` the reservation would push past its limit."""
    for index, (scope, usage) in enumerate(zip(scopes, current, strict=True)):
        for meter in Meter:
            used, held, limit = usage[meter], amount[meter], scope.limits[meter]
            if used + held > MAX_COUNTER:
                return index, meter
            if meter in meters and limit is not None and (used >= limit or used + held > limit):
                return index, meter
    return None


class _OpState(StrEnum):
    HELD = "held"
    DONE = "done"


@dataclass(slots=True)
class _Op:
    state: _OpState
    held: Spend
    expires_at: datetime


@dataclass(slots=True)
class _Counters:
    expires_at: datetime
    values: dict[Meter, int] = field(default_factory=dict[Meter, int])

    def spend(self) -> Spend:
        return Spend.of(self.values)


class InMemoryBudgetStore(BudgetStore):
    """One process, lost on restart. Expiry follows the injected clock, like Redis TTLs."""

    kind: ClassVar[str] = "memory"

    def __init__(self, *, clock: Clock = utc_now) -> None:
        self._clock = clock
        self._counters: dict[str, _Counters] = {}
        self._ops: dict[str, _Op] = {}

    async def reserve(
        self, op_id: str, scopes: Sequence[BudgetScope], amount: Spend, meters: frozenset[Meter]
    ) -> ReserveOutcome:
        if self._op(op_id) is not None:
            return ReserveOutcome(granted=False, duplicate=True)
        current = [self._spend(scope) for scope in scopes]
        exceeded = first_exceeded(scopes, current, amount, meters)
        if exceeded is not None:
            index, meter = exceeded
            return ReserveOutcome(
                granted=False, usage=tuple(current), exceeded_scope=index, exceeded_meter=meter
            )
        after = self._apply(scopes, dict(amount.items()))
        self._ops[op_key(op_id)] = _Op(_OpState.HELD, amount, self._expiry(OP_TTL_S))
        return ReserveOutcome(granted=True, usage=after)

    async def settle(
        self, op_id: str, scopes: Sequence[BudgetScope], spent: Spend
    ) -> tuple[Spend, ...]:
        op = self._op(op_id)
        if op is not None and op.state is _OpState.HELD:
            after = self._apply(scopes, spent.minus(op.held))
        else:  # already settled, or never reserved: leave a tombstone, change nothing
            after = tuple(self._spend(scope) for scope in scopes)
        self._ops[op_key(op_id)] = _Op(_OpState.DONE, Spend(), self._expiry(OP_TTL_S))
        return after

    async def usage(self, scope: BudgetScope) -> Spend:
        return self._spend(scope)

    async def healthy(self) -> bool:
        return True

    def _apply(self, scopes: Sequence[BudgetScope], delta: dict[Meter, int]) -> tuple[Spend, ...]:
        after: list[Spend] = []
        for scope in scopes:
            counters = self._live(scope)
            expires_at = self._expiry(scope.ttl_s)
            if counters is None:
                counters = self._counters[scope.key] = _Counters(expires_at=expires_at)
            counters.expires_at = max(counters.expires_at, expires_at)  # extend, never shorten
            for meter, change in delta.items():
                value = counters.values.get(meter, 0) + change
                counters.values[meter] = min(max(value, 0), MAX_COUNTER)
            after.append(counters.spend())
        return tuple(after)

    def _expiry(self, ttl_s: int) -> datetime:
        return self._clock() + timedelta(seconds=ttl_s)

    def _op(self, op_id: str) -> _Op | None:
        op = self._ops.get(op_key(op_id))
        if op is not None and op.expires_at <= self._clock():
            del self._ops[op_key(op_id)]
            return None
        return op

    def _spend(self, scope: BudgetScope) -> Spend:
        counters = self._live(scope)
        return counters.spend() if counters is not None else Spend()

    def _live(self, scope: BudgetScope) -> _Counters | None:
        counters = self._counters.get(scope.key)
        if counters is not None and counters.expires_at <= self._clock():
            del self._counters[scope.key]
            return None
        return counters
