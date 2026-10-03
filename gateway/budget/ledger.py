"""The budget lifecycle of one call: reserve before dispatch, settle after, whatever happens.

SPEC "Budgets". The pipeline calls `BudgetLedger.reserve` inside the session lock, after
authorization, controls and throttling, right before the upstream; and `settle` in a
``finally`` around the upstream call, so completion, upstream failure and cancellation all
reconcile the hold. Store calls run as their own tasks behind `asyncio.shield`: a cancelled
request (client gone) still settles, and a reservation whose caller vanished is released.

Scopes: ``per_user`` (the principal; an autonomous agent's ``svc:<agent>`` is its own user)
and ``per_agent`` count per UTC day, ``per_session`` for the session's lifetime (its counters
live ``sessions.max_lifetime_s`` plus an hour of slack, re-extended on every write, so a
reload that lengthens sessions never drops a live session's counters). Only scopes with a
limit on a meter the call draws on are touched, so a call no budget limits never needs the
store, and only budget-limited calls fail closed when it is down.

**GPU allowance.** An LLM call holds GPU time up front: the upstream timeout, lowered to what
every GPU-limited scope has left and to what every cost-limited scope can still pay for at
the model's GPU price (after the token cost). A budget-bound allowance below
``MIN_GPU_ALLOWANCE_S`` refuses the call; otherwise the pipeline enforces it as the upstream
call's deadline. The settlement charges the measured wall time (never more than the
allowance) and refunds the rest. The allowance is sized from a read, then reserved
atomically: a concurrent call can make the reservation refuse, never overspend, and the
ledger re-sizes and retries a couple of times.

**Operation ids.** Each reservation has one, and settlement is idempotent by id in the store.
A settlement the store did not confirm is kept in memory and retried with backoff by one
background worker; while one that *adds* charges (actual above the hold) is unconfirmed, its
scopes refuse new reservations (503 ``budget_settlement_pending``). A reservation whose
outcome is unknown (an error after the request may have reached Redis) is released by id; the
tombstone the release leaves makes a late-arriving reservation refuse.

Crossing ``soft_limit_pct`` of a limit logs one structured warning per scope, meter and
window; ``acl_budget_usage_ratio{scope,id}`` follows usage for user and agent scopes (session
ids are unbounded, so session scopes are logged, never labelled).
"""

import asyncio
import contextlib
import json
import logging
import math
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Final

from gateway.budget import metering
from gateway.budget.model import (
    MS_PER_SECOND,
    SESSION_WINDOW,
    BudgetedCall,
    BudgetExceededError,
    BudgetScope,
    BudgetSettlementPendingError,
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
SESSION_TTL_SLACK_S: Final = 3600
MIN_GPU_ALLOWANCE_S: Final = 0.25  # a budget-bound allowance below this refuses the call
RESERVE_ATTEMPTS: Final = 3
RETRY_FIRST_S: Final = 0.5
RETRY_MAX_S: Final = 30.0
MAX_REMEMBERED_WARNINGS: Final = 10_000
_PERCENT: Final = 100.0
_GPU_SIZED: Final = frozenset({Meter.GPU, Meter.COST})


def scopes_for(
    call: BudgetedCall, policy: Policy, meters: frozenset[Meter], now_window: str
) -> tuple[BudgetScope, ...]:
    """The scopes whose limits apply to a call drawing on ``meters``."""
    budgets = policy.budgets
    session_ttl = math.ceil(policy.sessions.max_lifetime_s) + SESSION_TTL_SLACK_S
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


@dataclass(frozen=True, slots=True)
class _Unsettled:
    """A settlement (or release) the store has not confirmed yet."""

    op_id: str
    scopes: tuple[BudgetScope, ...]
    spent: Spend
    blocking: bool  # it adds charges: its scopes refuse new reservations until it lands
    reservation: Reservation | None = None  # None for the release of an unconfirmed hold


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
        self._tasks: set[asyncio.Task[object]] = set()
        self._unsettled: dict[str, _Unsettled] = {}
        self._retrier: asyncio.Task[None] | None = None
        self._closing = asyncio.Event()
        self._warned: OrderedDict[str, None] = OrderedDict()

    @property
    def store(self) -> BudgetStore:
        return self._store

    async def reserve(self, call: BudgetedCall, snapshot: PolicySnapshot) -> Reservation:
        """Hold the call's estimated spend on every scope that limits it.

        Raises `BudgetExceededError` (403) when a hard limit would be crossed,
        `BudgetStoreUnavailableError` (503) when the store cannot be reached,
        `BudgetSettlementPendingError` (503) while an earlier overrun on a scope is unrecorded,
        and `InvalidRequestError` (400) for a request whose completions cannot be bounded.
        """
        policy = snapshot.policy
        costs = CostModel(policy.pricing)
        plan = metering.plan(
            call,
            costs,
            default_max_tokens=policy.limits.default_max_tokens,
            max_completion_tokens=policy.limits.max_completion_tokens,
        )
        scopes = scopes_for(call, policy, plan.meters, daily_window(self._clock()))
        reservation = Reservation(
            op_id=uuid.uuid4().hex,
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
        self._refuse_if_unsettled(scopes)
        refused = ""
        for _ in range(RESERVE_ATTEMPTS):
            amount, allowance_s = await self._sized(plan, scopes, call, costs, policy)
            outcome = await self._reserve_once(reservation.op_id, scopes, amount, plan.meters)
            if outcome.granted:
                return reservation.model_copy(
                    update={
                        "scopes": scopes,
                        "held": amount,
                        "held_token_cost": plan.estimate.cost_nano_usd,
                        "gpu_allowance_s": allowance_s,
                        "usage": outcome.usage,
                        "started_s": self._monotonic(),
                    }
                )
            if outcome.duplicate:  # released meanwhile: this id holds nothing
                raise BudgetStoreUnavailableError
            refused = outcome.exceeded_limit(scopes)
            if Meter.GPU not in plan.meters or outcome.exceeded_meter not in _GPU_SIZED:
                break  # re-sizing the GPU allowance cannot help
        raise BudgetExceededError(refused)

    async def settle(self, reservation: Reservation, upstream: UpstreamResult | None) -> None:
        """Reconcile the hold with what the call spent. Never raises a store error.

        ``upstream`` is None when the upstream failed or the call was cancelled: no tokens
        were reported, and GPU time is the wall time since the reservation.
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
        """Wait until every settlement and release has landed (retrying as long as it takes)."""
        while self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def aclose(self) -> None:
        """Stop retrying, let in-flight store calls finish, close the store."""
        self._closing.set()
        await self.drain()
        for op_id in self._unsettled:
            logger.warning("budget settlement abandoned at shutdown: op=%s", op_id)
        await self._store.aclose()

    # ------------------------------------------------------------------------- internals

    def _track[T](self, work: Coroutine[object, object, T]) -> asyncio.Task[T]:
        """Run ``work`` as a task `drain` waits for, whatever happens to its caller."""
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _refuse_if_unsettled(self, scopes: Sequence[BudgetScope]) -> None:
        blocked = {s.key for u in self._unsettled.values() if u.blocking for s in u.scopes}
        if any(scope.key in blocked for scope in scopes):
            raise BudgetSettlementPendingError

    async def _sized(
        self,
        plan: metering.Plan,
        scopes: tuple[BudgetScope, ...],
        call: BudgetedCall,
        costs: CostModel,
        policy: Policy,
    ) -> tuple[Spend, float | None]:
        """What to hold, GPU allowance included; the allowance itself when a budget bounds it."""
        if Meter.GPU not in plan.meters:
            return plan.estimate, None
        allowance_ms = math.ceil(policy.limits.upstream_timeout_s * MS_PER_SECOND)
        binding: str | None = None
        gpu_priced = costs.price(call.model).gpu_second > 0
        bounded = [
            scope
            for scope in scopes
            if scope.limits[Meter.GPU] is not None
            or (gpu_priced and scope.limits[Meter.COST] is not None)
        ]
        usages = await self._guarded("usage", self._store.usages(bounded)) if bounded else ()
        for scope, used in zip(bounded, usages, strict=True):
            for meter, room_ms in _gpu_room(scope, used, plan, call, costs):
                if room_ms < allowance_ms:
                    allowance_ms, binding = room_ms, scope.limit_name(meter)
        if binding is not None and allowance_ms < MIN_GPU_ALLOWANCE_S * MS_PER_SECOND:
            raise BudgetExceededError(binding)
        amount = Spend.bounded(
            tokens=plan.estimate.tokens,
            tool_calls=plan.estimate.tool_calls,
            cost_nano_usd=plan.estimate.cost_nano_usd
            + costs.gpu_cost(call.model, gpu_ms=allowance_ms),
            gpu_ms=allowance_ms,
        )
        return amount, (allowance_ms / MS_PER_SECOND if binding is not None else None)

    async def _reserve_once(
        self, op_id: str, scopes: tuple[BudgetScope, ...], amount: Spend, meters: frozenset[Meter]
    ) -> ReserveOutcome:
        task = self._track(
            self._guarded("reserve", self._store.reserve(op_id, scopes, amount, meters))
        )
        try:
            return await asyncio.shield(task)
        except BudgetStoreUnavailableError:
            # The script may have run with only its reply lost: release by id. Should the
            # reservation still land after that, the release's tombstone refuses it.
            self._track(self._land(_Unsettled(op_id, scopes, Spend(), blocking=False)))
            raise
        except asyncio.CancelledError:  # the caller is gone: hand back whatever landed
            self._track(self._release_after(task, op_id, scopes))
            raise

    async def _release_after(
        self, task: "asyncio.Task[ReserveOutcome]", op_id: str, scopes: tuple[BudgetScope, ...]
    ) -> None:
        with contextlib.suppress(Exception):
            await task
        await self._land(_Unsettled(op_id, scopes, Spend(), blocking=False))

    async def _settle(
        self, reservation: Reservation, upstream: UpstreamResult | None, wall_s: float
    ) -> None:
        call, held = reservation.call, reservation.held
        spent = metering.actual(
            call,
            held,
            reservation.held_token_cost,
            CostModel(reservation.pricing),
            answered=upstream is not None,
            usage=upstream.usage if upstream is not None else None,
            wall_s=wall_s,
        )
        if spent.cost_nano_usd:
            record_cost(
                call.user_label, call.agent_label, call.model or OTHER_LABEL, spent.cost_usd
            )
        if not reservation.scopes:
            return
        overrun = any(spent[meter] > held[meter] for meter in Meter)
        await self._land(
            _Unsettled(reservation.op_id, reservation.scopes, spent, overrun, reservation)
        )

    async def _land(self, unsettled: _Unsettled) -> None:
        """Settle now; if the store does not confirm, leave it to the retry worker."""
        try:
            after = await self._store_settle(unsettled)
        except BudgetStoreUnavailableError:
            logger.warning("budget settlement pending: op=%s", unsettled.op_id)
            self._unsettled[unsettled.op_id] = unsettled
            if self._retrier is None or self._retrier.done():
                self._retrier = self._track(self._retry_unsettled())
            return
        self._landed(unsettled, after)

    async def _retry_unsettled(self) -> None:
        """One worker for every unconfirmed settlement: exponential backoff, until all land."""
        delay = RETRY_FIRST_S
        while self._unsettled and not self._closing.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._closing.wait(), timeout=delay)
            if self._closing.is_set():
                return
            for unsettled in list(self._unsettled.values()):
                try:
                    after = await self._store_settle(unsettled)
                except BudgetStoreUnavailableError:
                    delay = min(delay * 2, RETRY_MAX_S)
                    break
                self._landed(unsettled, after)
            else:
                delay = RETRY_FIRST_S

    async def _store_settle(self, unsettled: _Unsettled) -> tuple[Spend, ...]:
        settling = self._store.settle(unsettled.op_id, unsettled.scopes, unsettled.spent)
        return await self._guarded("settle", settling)

    def _landed(self, unsettled: _Unsettled, after: tuple[Spend, ...]) -> None:
        self._unsettled.pop(unsettled.op_id, None)
        if unsettled.reservation is not None:
            self._report(unsettled.reservation, after)

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


def _gpu_room(
    scope: BudgetScope, used: Spend, plan: metering.Plan, call: BudgetedCall, costs: CostModel
) -> list[tuple[Meter, int]]:
    """GPU milliseconds the scope can still take, per limit that bounds them."""
    rooms: list[tuple[Meter, int]] = []
    if (gpu_limit := scope.limits[Meter.GPU]) is not None:
        rooms.append((Meter.GPU, gpu_limit - used.gpu_ms))
    if (cost_limit := scope.limits[Meter.COST]) is not None:
        left = cost_limit - used.cost_nano_usd - plan.estimate.cost_nano_usd
        affordable = costs.affordable_gpu_ms(call.model, budget_nano_usd=left)
        if affordable is not None:
            rooms.append((Meter.COST, affordable))
    return rooms
