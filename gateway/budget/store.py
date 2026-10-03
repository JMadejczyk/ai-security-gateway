"""Where budget counters live, behind one interface with an atomic check-and-hold.

`BudgetStore.reserve` is the whole correctness story: for every scope it checks each meter
the call draws on and, only if none would cross its limit, adds the amounts to all scopes in
the same atomic step. Two concurrent calls can therefore never both take the last of a budget.

A meter is refused when the scope is already at its limit, or when the held amount would
take it past the limit. The first rule matters for post-paid meters (GPU time), which are
held at zero: they admit a call only while headroom remains.

`RedisBudgetStore` (`gateway.budget.redis_store`) is the production store. The in-memory one
below serves tests and an explicitly chosen ``ACL_BUDGET_STORE=memory`` development setup: it
is atomic because a check and its update never await in between.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import ClassVar

from gateway.budget.model import BudgetScope, Meter, ReserveOutcome, Spend
from gateway.clock import Clock, utc_now


class BudgetStore(ABC):
    """Counters per scope. Every method raises `BudgetStoreUnavailableError` when the backing
    service cannot be reached; callers fail closed."""

    kind: ClassVar[str]

    @abstractmethod
    async def reserve(
        self, scopes: Sequence[BudgetScope], amount: Spend, meters: frozenset[Meter]
    ) -> ReserveOutcome:
        """Hold ``amount`` on every scope, unless any limit of ``meters`` would be crossed."""

    @abstractmethod
    async def adjust(
        self, scopes: Sequence[BudgetScope], delta: Mapping[Meter, int]
    ) -> tuple[Spend, ...]:
        """Add a signed ``delta`` to every scope (never below zero); usage after, per scope."""

    @abstractmethod
    async def usage(self, scope: BudgetScope) -> Spend: ...

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
            limit = scope.limits[meter]
            if meter not in meters or limit is None:
                continue
            if usage[meter] >= limit or usage[meter] + amount[meter] > limit:
                return index, meter
    return None


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

    async def reserve(
        self, scopes: Sequence[BudgetScope], amount: Spend, meters: frozenset[Meter]
    ) -> ReserveOutcome:
        current = [self._spend(scope) for scope in scopes]
        exceeded = first_exceeded(scopes, current, amount, meters)
        if exceeded is not None:
            index, meter = exceeded
            return ReserveOutcome(
                granted=False, usage=tuple(current), exceeded_scope=index, exceeded_meter=meter
            )
        return ReserveOutcome(granted=True, usage=self._add(scopes, dict(amount.items())))

    async def adjust(
        self, scopes: Sequence[BudgetScope], delta: Mapping[Meter, int]
    ) -> tuple[Spend, ...]:
        return self._add(scopes, delta)

    async def usage(self, scope: BudgetScope) -> Spend:
        return self._spend(scope)

    async def healthy(self) -> bool:
        return True

    def _add(self, scopes: Sequence[BudgetScope], delta: Mapping[Meter, int]) -> tuple[Spend, ...]:
        after: list[Spend] = []
        for scope in scopes:
            counters = self._live(scope)
            if counters is None:  # created now: the TTL starts now and is never extended
                expires_at = self._clock() + timedelta(seconds=scope.ttl_s)
                counters = self._counters[scope.key] = _Counters(expires_at=expires_at)
            for meter, change in delta.items():
                counters.values[meter] = max(counters.values.get(meter, 0) + change, 0)
            after.append(counters.spend())
        return tuple(after)

    def _spend(self, scope: BudgetScope) -> Spend:
        counters = self._live(scope)
        return counters.spend() if counters is not None else Spend()

    def _live(self, scope: BudgetScope) -> _Counters | None:
        counters = self._counters.get(scope.key)
        if counters is not None and counters.expires_at <= self._clock():
            del self._counters[scope.key]
            return None
        return counters
