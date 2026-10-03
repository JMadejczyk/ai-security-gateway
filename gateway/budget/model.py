"""Value objects of the budget layer: what is metered, against which scope, and what is held.

Amounts are integers in base units so a limit check is exact and atomic in Redis: tokens,
nano-USD, tool calls and GPU milliseconds. Policy values (USD, seconds) are converted once,
when a scope is built. Every amount and counter stays at or below ``MAX_COUNTER`` (2^53 - 1):
exact as a Lua number, far inside Redis's 64-bit integers, so no write can overflow.
"""

import math
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Final, Self

from pydantic import Field, NonNegativeInt

from gateway.core.envelope import FrozenModel
from gateway.core.frozen import FrozenDict
from gateway.core.types import Channel
from gateway.errors import RejectionError
from gateway.policy.schema import DailyBudget, ModelPrice, SessionBudget

NANO_USD_PER_USD: Final = 1_000_000_000
MS_PER_SECOND: Final = 1_000
KEY_PREFIX: Final = "acl:budget"
SESSION_WINDOW: Final = "session"
MAX_COUNTER: Final = 2**53 - 1  # the largest value any counter or amount may take


class Meter(StrEnum):
    """One metered quantity. The value is the `Spend` field and the Redis hash field."""

    TOKENS = "tokens"
    COST = "cost_nano_usd"
    TOOL_CALLS = "tool_calls"
    GPU = "gpu_ms"


# Policy field per meter (``daily_`` prefixed for daily scopes) and how a policy value
# converts to the meter's base unit.
_POLICY_FIELD: Final[Mapping[Meter, str]] = {
    Meter.TOKENS: "tokens",
    Meter.COST: "cost_usd",
    Meter.TOOL_CALLS: "tool_calls",
    Meter.GPU: "gpu_seconds",
}
_UNIT: Final[Mapping[Meter, int]] = {
    Meter.TOKENS: 1,
    Meter.COST: NANO_USD_PER_USD,
    Meter.TOOL_CALLS: 1,
    Meter.GPU: MS_PER_SECOND,
}


type Amount = Annotated[int, Field(ge=0, le=MAX_COUNTER)]


class Spend(FrozenModel):
    """Non-negative amounts per meter, in base units, bounded by ``MAX_COUNTER``."""

    tokens: Amount = 0
    cost_nano_usd: Amount = 0
    tool_calls: Amount = 0
    gpu_ms: Amount = 0

    def __getitem__(self, meter: Meter) -> int:
        amount: int = getattr(self, meter.value)
        return amount

    def items(self) -> Iterator[tuple[Meter, int]]:
        return ((meter, self[meter]) for meter in Meter)

    def minus(self, other: "Spend") -> dict[Meter, int]:
        """Signed difference per meter (a reconcile delta)."""
        return {meter: self[meter] - other[meter] for meter in Meter}

    @classmethod
    def of(cls, amounts: Mapping[Meter, int]) -> Self:
        return cls.model_validate({meter.value: amount for meter, amount in amounts.items()})

    @classmethod
    def bounded(cls, **amounts: int) -> Self:
        """Amounts clamped to ``[0, MAX_COUNTER]``: an absurd estimate holds the maximum."""
        return cls.model_validate({k: min(max(v, 0), MAX_COUNTER) for k, v in amounts.items()})

    @property
    def cost_usd(self) -> float:
        return self.cost_nano_usd / NANO_USD_PER_USD


class SpendLimits(FrozenModel):
    """Hard limits per meter, in base units; None means the meter is not limited."""

    tokens: NonNegativeInt | None = None
    cost_nano_usd: NonNegativeInt | None = None
    tool_calls: NonNegativeInt | None = None
    gpu_ms: NonNegativeInt | None = None

    def __getitem__(self, meter: Meter) -> int | None:
        limit: int | None = getattr(self, meter.value)
        return limit

    @classmethod
    def from_policy(cls, budget: DailyBudget | SessionBudget, *, daily: bool) -> Self:
        prefix = "daily_" if daily else ""
        limits: dict[str, int] = {}
        for meter, name in _POLICY_FIELD.items():
            value: float | None = getattr(budget, prefix + name)
            if value is not None:
                limits[meter.value] = min(math.floor(value * _UNIT[meter]), MAX_COUNTER)
        return cls.model_validate(limits)

    def limited(self) -> frozenset[Meter]:
        return frozenset(meter for meter in Meter if self[meter] is not None)


class ScopeKind(StrEnum):
    """The policy section a scope comes from; also the first part of its metric label."""

    USER = "per_user"
    AGENT = "per_agent"
    SESSION = "per_session"


class BudgetScope(FrozenModel):
    """One counter set: a user or agent for one UTC day, or one session.

    ``window`` is the UTC date (``YYYY-MM-DD``) for daily scopes, so a new day starts from zero
    without any reset job, and ``session`` for session scopes. Neither contains a colon, so
    the key is unambiguous whatever the subject (``svc:nightly_etl``) looks like.
    """

    kind: ScopeKind
    subject: str = Field(min_length=1)  # principal, agent id or session id
    window: str = Field(min_length=1, pattern=r"^[0-9a-z-]+$")
    ttl_s: int = Field(gt=0)  # set when the counters are created; never extended
    limits: SpendLimits

    @property
    def key(self) -> str:
        return f"{KEY_PREFIX}:{self.kind.value}:{self.window}:{self.subject}"

    def limit_name(self, meter: Meter) -> str:
        """The policy setting a meter's limit comes from, e.g. ``per_user.daily_tokens``."""
        prefix = "" if self.kind is ScopeKind.SESSION else "daily_"
        return f"{self.kind.value}.{prefix}{_POLICY_FIELD[meter]}"


def daily_window(now: datetime) -> str:
    """The UTC day ``now`` (an aware datetime) falls in: counters reset at 00:00 UTC."""
    return now.astimezone(UTC).date().isoformat()


def op_key(op_id: str) -> str:
    """The record of one reservation: its hold until settled, then a tombstone."""
    return f"{KEY_PREFIX}:op:{op_id}"


class ReserveOutcome(FrozenModel):
    """What a store answered to a reservation. Usage is per scope, after the reservation."""

    granted: bool
    duplicate: bool = False  # the operation id was already reserved or settled: nothing held
    usage: tuple[Spend, ...] = ()
    exceeded_scope: int | None = None  # index into the scopes, when refused
    exceeded_meter: Meter | None = None

    def exceeded_limit(self, scopes: Sequence[BudgetScope]) -> str:
        """The policy setting a refused reservation would have crossed."""
        if self.exceeded_scope is None or self.exceeded_meter is None:
            msg = "a refused reservation names the scope and meter it would exceed"
            raise ValueError(msg)
        return scopes[self.exceeded_scope].limit_name(self.exceeded_meter)


class BudgetedCall(FrozenModel):
    """One call about to be dispatched, as the budget layer sees it."""

    session_id: str = Field(min_length=1)
    principal: str = Field(min_length=1)  # a human, or svc:<agent>: its own user
    agent: str = Field(min_length=1)
    channel: Channel
    model: str | None = None  # the model identifier the call is authorized for (LLM)
    payload: Any = Field(repr=False)  # the final payload the upstream would receive
    user_label: str  # bounded metric label values
    agent_label: str


class Reservation(FrozenModel):
    """Budget held for one dispatched call until it is settled."""

    op_id: str = Field(min_length=1)  # makes settlement and release idempotent in the store
    call: BudgetedCall
    payload: Any = Field(repr=False)  # what to dispatch (the LLM request carries a token cap)
    scopes: tuple[BudgetScope, ...]  # empty when no limit applies: nothing was held
    held: Spend
    held_token_cost: NonNegativeInt = 0  # the token part of ``held.cost_nano_usd``
    gpu_allowance_s: float | None = None  # LLM: the upstream call must finish within this
    usage: tuple[Spend, ...] = ()  # per scope, right after the reservation
    started_s: float  # monotonic time of the reservation, for wall time on failure
    pricing: FrozenDict[str, ModelPrice]  # the snapshot's prices: settled at reservation terms
    policy_revision: str
    soft_limit_pct: float


class BudgetRefusalError(RejectionError):
    """The budget layer refused the call; the pipeline records it as a ``budget`` verdict."""


class BudgetExceededError(BudgetRefusalError):
    """Dispatching the call would cross a hard limit. Names the limit, never anyone's usage."""

    status_code = 403

    def __init__(self, limit_name: str) -> None:
        super().__init__("budget_exceeded", f"budget exceeded: {limit_name}")
        self.limit_name = limit_name


class BudgetStoreUnavailableError(BudgetRefusalError):
    """The budget store cannot be reached: a budget-limited call fails closed."""

    status_code = 503

    def __init__(self) -> None:
        super().__init__("budget_store_unavailable", "budget store unavailable; try again later")


class BudgetSettlementPendingError(BudgetRefusalError):
    """An earlier call's charge on one of these scopes has not reached the store yet."""

    status_code = 503

    def __init__(self) -> None:
        super().__init__(
            "budget_settlement_pending", "an earlier charge is still being recorded; try again"
        )
