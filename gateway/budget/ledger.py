"""The budget lifecycle of one call: reserve before dispatch, settle after, whatever happens.

SPEC "Budgets". The pipeline calls `BudgetLedger.reserve` inside the session lock, after
authorization, controls and throttling, right before the upstream; and `settle` in a
``finally`` around the upstream call, so completion, upstream failure and cancellation all
reconcile the hold. Settlement runs as its own task behind `asyncio.shield`: a cancelled
request (client gone) still returns its unused budget.

Scopes: ``per_user`` (the principal; an autonomous agent's ``svc:<agent>`` is its own user)
and ``per_agent`` count per UTC day, ``per_session`` for the session's lifetime. Only scopes
with a limit on a meter the call draws on are touched, so a call no budget limits never needs
the store, and only budget-limited calls fail closed when it is down.

Crossing ``soft_limit_pct`` of a limit logs one structured warning per scope, meter and
window; ``acl_budget_usage_ratio{scope,id}`` follows usage for user and agent scopes (session
ids are unbounded, so session scopes are logged, never labelled).
"""

import asyncio
import json
import logging
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from typing import Final

from gateway.budget import metering
from gateway.budget.model import (
    SESSION_WINDOW,
    BudgetedCall,
    BudgetExceededError,
    BudgetScope,
    BudgetStoreUnavailableError,
    Meter,
    Reservation,
    ReserveOutcome,
    ScopeKind,
    Spend,
    SpendLimits,
    daily_window,
)
from gateway.budget.pricing import CostModel
from gateway.budget.store import BudgetStore
from gateway.clock import Clock, utc_now
from gateway.policy.loader import PolicySnapshot
from gateway.policy.schema import Policy
from gateway.telemetry import (
    OTHER_LABEL,
    record_budget_store_error,
    record_cost,
    set_budget_store_up,
    set_budget_usage,
)
from gateway.upstream import UpstreamResult

logger = logging.getLogger(__name__)
alert_logger = logging.getLogger("gateway.alerts")

DAILY_TTL_S: Final = 2 * 24 * 3600  # a day's counters outlive the day by one more
MAX_REMEMBERED_WARNINGS: Final = 10_000
_PERCENT: Final = 100.0


def scopes_for(
    call: BudgetedCall, policy: Policy, meters: frozenset[Meter], now_window: str
) -> tuple[BudgetScope, ...]:
    """The scopes whose limits apply to a call drawing on ``meters``."""
    budgets = policy.budgets
    session_ttl = math.ceil(policy.sessions.max_lifetime_s)
    candidates = (
        (ScopeKind.USER, call.principal, now_window, DAILY_TTL_S, budgets.per_user, True),
        (ScopeKind.AGENT, call.agent, now_window, DAILY_TTL_S, budgets.per_agent, True),
        (
            ScopeKind.SESSION,
            call.session_id,
            SESSION_WINDOW,
            session_ttl,
            budgets.per_session,
            False,
        ),
    )
    scopes: list[BudgetScope] = []
    for kind, subject, window, ttl_s, budget, daily in candidates:
        limits = SpendLimits.from_policy(budget, daily=daily)
        if limits.limited() & meters:
            scopes.append(
                BudgetScope(kind=kind, subject=subject, window=window, ttl_s=ttl_s, limits=limits)
            )
    return tuple(scopes)


class BudgetLedger:
    """Reserves, settles and reports budgets against one `BudgetStore`."""

    def __init__(
        self,
        store: BudgetStore,
        *,
        clock: Clock = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._clock = clock
        self._monotonic = monotonic
        self._settling: set[asyncio.Task[object]] = set()
        self._warned: OrderedDict[str, None] = OrderedDict()

    @property
    def store(self) -> BudgetStore:
        return self._store

    async def reserve(self, call: BudgetedCall, snapshot: PolicySnapshot) -> Reservation:
        """Hold the call's estimated spend on every scope that limits it.

        Raises `BudgetExceededError` (403) when a hard limit would be crossed and
        `BudgetStoreUnavailableError` (503) when the store cannot be reached.
        """
        policy = snapshot.policy
        costs = CostModel(policy.pricing)
        plan = metering.plan(call, costs, default_max_tokens=policy.limits.default_max_tokens)
        scopes = scopes_for(call, policy, plan.meters, daily_window(self._clock()))
        reservation = Reservation(
            call=call,
            payload=plan.payload,
            scopes=(),
            held=Spend(),
            started_s=self._monotonic(),
            pricing=policy.pricing,
            policy_revision=snapshot.revision,
            soft_limit_pct=policy.budgets.soft_limit_pct,
        )
        if not scopes:
            return reservation
        outcome = await self._reserve_shielded(scopes, plan.estimate, plan.meters)
        if not outcome.granted:
            raise BudgetExceededError(outcome.exceeded_limit(scopes))
        return reservation.model_copy(
            update={"scopes": scopes, "held": plan.estimate, "usage": outcome.usage}
        )

    async def settle(self, reservation: Reservation, upstream: UpstreamResult | None) -> None:
        """Reconcile the hold with what the call spent. Never raises a store error.

        ``upstream`` is None when the upstream failed or the call was cancelled: tokens fall
        back to nothing reported and GPU time to the wall time since the reservation.
        """
        wall_s = (
            upstream.elapsed_s
            if upstream is not None
            else self._monotonic() - reservation.started_s
        )
        # If the caller is cancelled (client gone), the settlement still runs to the end.
        await asyncio.shield(self._track(self._settle(reservation, upstream, wall_s)))

    async def status(self) -> str:
        """``memory``, or ``up``/``down`` for Redis."""
        if self._store.kind == "memory":
            return "memory"
        up = await self._store.healthy()
        set_budget_store_up(up=up)
        return "up" if up else "down"

    async def drain(self) -> None:
        """Wait until every settlement (and orphaned-reservation release) has landed."""
        while self._settling:
            await asyncio.gather(*self._settling, return_exceptions=True)

    async def aclose(self) -> None:
        await self.drain()
        await self._store.aclose()

    # ------------------------------------------------------------------------- internals

    def _track[T](self, work: Coroutine[object, object, T]) -> asyncio.Task[T]:
        """Run ``work`` as a task `aclose` waits for, whatever happens to its caller."""
        task = asyncio.create_task(work)
        self._settling.add(task)
        task.add_done_callback(self._settling.discard)
        return task

    async def _reserve_shielded(
        self, scopes: tuple[BudgetScope, ...], amount: Spend, meters: frozenset[Meter]
    ) -> ReserveOutcome:
        """A reservation that lands after its caller was cancelled is handed back, not leaked."""
        task = self._track(self._guarded("reserve", self._store.reserve(scopes, amount, meters)))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self._track(self._release_orphan(task, scopes, amount))
            raise

    async def _release_orphan(
        self, task: "asyncio.Task[ReserveOutcome]", scopes: tuple[BudgetScope, ...], amount: Spend
    ) -> None:
        try:
            outcome = await task
            if outcome.granted:
                release = {meter: -held for meter, held in amount.items()}
                await self._guarded("release", self._store.adjust(scopes, release))
        except BudgetStoreUnavailableError:
            logger.warning("budget release lost: store unavailable")

    async def _settle(
        self, reservation: Reservation, upstream: UpstreamResult | None, wall_s: float
    ) -> None:
        call = reservation.call
        costs = CostModel(reservation.pricing)
        usage = upstream.usage if upstream is not None else None
        spent = metering.actual(call, reservation.held, costs, usage=usage, wall_s=wall_s)
        if usage is None and upstream is not None:  # answered without usage: keep the estimate
            spent = spent.model_copy(update={"tokens": reservation.held.tokens})
        if spent.cost_nano_usd:
            record_cost(
                call.user_label, call.agent_label, call.model or OTHER_LABEL, spent.cost_usd
            )
        if not reservation.scopes:
            return
        delta = spent.minus(reservation.held)
        after = reservation.usage
        if any(delta.values()):
            try:
                after = await self._guarded("settle", self._store.adjust(reservation.scopes, delta))
            except BudgetStoreUnavailableError:
                # The hold stays as charged: budgets over-count, they never under-count.
                logger.warning(
                    "budget settlement lost: store unavailable session=%s", call.session_id
                )
                return
        self._report(reservation, after)

    @staticmethod
    async def _guarded[T](operation: str, pending: Awaitable[T]) -> T:
        try:
            result = await pending
        except BudgetStoreUnavailableError:
            record_budget_store_error(operation)
            set_budget_store_up(up=False)
            raise
        set_budget_store_up(up=True)
        return result

    def _report(self, reservation: Reservation, usage: Sequence[Spend]) -> None:
        call = reservation.call
        labels = {ScopeKind.USER: call.user_label, ScopeKind.AGENT: call.agent_label}
        for scope, spent in zip(reservation.scopes, usage, strict=True):
            for meter in Meter:
                limit = scope.limits[meter]
                if limit is None:
                    continue
                ratio = spent[meter] / limit if limit else 1.0
                if (label := labels.get(scope.kind)) is not None:
                    set_budget_usage(scope.limit_name(meter), label, ratio)
                if ratio * _PERCENT >= reservation.soft_limit_pct:
                    self._warn_once(reservation, scope, meter, ratio)

    def _warn_once(
        self, reservation: Reservation, scope: BudgetScope, meter: Meter, ratio: float
    ) -> None:
        key = f"{scope.key}|{meter.value}"
        if key in self._warned:
            return
        self._warned[key] = None
        while len(self._warned) > MAX_REMEMBERED_WARNINGS:
            self._warned.popitem(last=False)
        alert_logger.warning(
            json.dumps(
                {
                    "event": "budget_soft_limit",
                    "limit": scope.limit_name(meter),
                    "scope": scope.kind.value,
                    "subject": scope.subject,
                    "window": scope.window,
                    "usage_ratio": round(ratio, 4),
                    "soft_limit_pct": reservation.soft_limit_pct,
                    "policy_revision": reservation.policy_revision,
                },
                sort_keys=True,
            )
        )
