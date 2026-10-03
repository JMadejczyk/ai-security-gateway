"""``sql_guard``: the mandatory pre control on every SQL tool call (SPEC "Control catalog").

For an MCP call carrying a ``sql`` argument it

1. re-validates the statement against the supported subset (`QueryPlan.parse`), independently
   of the adapter that already derived the call's resources from it (defense in depth);
2. caps the outermost ``SELECT`` at ``force_limit`` rows and renders the statement the way
   the analyzer understood it (`QueryPlan.with_row_limit`); when that text differs from the
   agent's, the verdict's ``rewrite`` carries the tool arguments with the new ``sql``, so the
   pipeline re-authorizes it and the upstream executes exactly that text;
3. prices the rewritten statement with the injected `SqlPlanner` (``EXPLAIN (FORMAT JSON)``
   without ``ANALYZE``, as the caller's principal) and blocks when the planner's top-level
   ``Total Cost`` exceeds ``max_cost``. No price, no execution: a planner failure or timeout
   blocks (``sql_plan_unavailable``).

Applicability is decided from the call itself, never from a policy lookup that a reload could
change mid-call: every MCP interaction whose arguments carry ``sql`` is guarded. A tool on a
non-sql server with a ``sql`` argument therefore cannot be priced and is refused (fail closed).
The statement timeout and result caps are not enforced here but by the SQL server, from the
limits signed into ``X-ACL-Principal`` (`gateway.proxies.mcp.upstream.sql_execution_limits`).

Reasons never contain SQL text: only reason codes and numbers.
"""

import asyncio
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, ClassVar, Final, Protocol, cast

from pydantic import Field

from gateway.adapters.sql import QueryPlan, UnsupportedSqlError
from gateway.core.envelope import FrozenModel, Interaction, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import Channel, ControlKind, Decision, Stage
from gateway.policy.schema import SqlGuardConfig

SQL_ARGUMENT: Final = "sql"
# The planner call adds an MCP handshake to the EXPLAIN itself, which the server bounds by the
# statement timeout; the control waits that long plus this much before it fails closed.
PLANNER_OVERHEAD_S: Final = 2.0


class SqlGuardReason(StrEnum):
    ALLOWED = "sql_allowed"
    NOT_APPLICABLE = "not_sql"
    UNSUPPORTED = "unsupported_sql"
    COST_EXCEEDED = "sql_cost_exceeded"
    PLAN_UNAVAILABLE = "sql_plan_unavailable"


class PlanRequest(FrozenModel):
    """What to price: the exact statement that would execute, for whom, on which server."""

    server: str = Field(min_length=1)
    principal: str = Field(min_length=1)
    sql: str = Field(min_length=1, repr=False)


class PlanUnavailableError(Exception):
    """The planner could not price the statement (upstream down or refusing, bad answer)."""


class SqlPlanner(Protocol):
    """Port: the planner's estimated top-level ``Total Cost`` of one statement."""

    async def __call__(self, request: PlanRequest) -> float:
        """Raises `PlanUnavailableError` when no trustworthy cost is available."""
        ...


def sql_argument(interaction: Interaction) -> object:
    """The ``sql`` tool argument of an MCP call, `None` when there is none."""
    payload = interaction.payload
    if interaction.channel is not Channel.MCP or not isinstance(payload, Mapping):
        return None
    arguments = cast("Mapping[str, Any]", payload).get("arguments")
    if not isinstance(arguments, Mapping):
        return None
    return cast("Mapping[str, Any]", arguments).get(SQL_ARGUMENT)


class SqlGuardControl(Control):
    id: ClassVar[str] = "sql_guard"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE})
    kind: ClassVar[ControlKind] = ControlKind.DETERMINISTIC
    mandatory: ClassVar[bool] = True

    def __init__(self, planner: SqlPlanner) -> None:
        self._planner = planner

    async def evaluate(self, interaction: Interaction, stage: Stage, cfg: ControlConfig) -> Verdict:
        sql = sql_argument(interaction)
        if stage is not Stage.PRE or sql is None:
            return self._verdict(Decision.ALLOW, SqlGuardReason.NOT_APPLICABLE)
        config = cfg if isinstance(cfg, SqlGuardConfig) else SqlGuardConfig(**cfg.model_dump())
        risk = config.risk_delta or 0.0
        if not isinstance(sql, str):
            return self._verdict(Decision.BLOCK, SqlGuardReason.UNSUPPORTED, risk_delta=risk)
        try:
            plan = QueryPlan.parse(sql).with_row_limit(config.force_limit)
        except UnsupportedSqlError as exc:
            return self._verdict(
                Decision.BLOCK, SqlGuardReason.UNSUPPORTED, exc.message, risk_delta=risk
            )
        try:
            cost = await self._price(interaction, plan.sql, config)
        except (PlanUnavailableError, TimeoutError):
            return self._verdict(Decision.BLOCK, SqlGuardReason.PLAN_UNAVAILABLE, risk_delta=risk)
        if cost > config.max_cost:
            reason = f"planner cost {cost:.1f} exceeds max_cost {config.max_cost:.1f}"
            return self._verdict(
                Decision.BLOCK, SqlGuardReason.COST_EXCEEDED, reason, risk_delta=risk
            )
        rewrite = _with_sql(interaction.payload, plan.sql) if plan.sql != sql else None
        return Verdict(
            decision=Decision.ALLOW,
            control_id=self.id,
            reason_code=SqlGuardReason.ALLOWED,
            reason=f"planner cost {cost:.1f}, row limit {plan.row_limit}",
            rewrite=rewrite,
        )

    async def _price(self, interaction: Interaction, sql: str, config: SqlGuardConfig) -> float:
        if interaction.server is None:
            raise PlanUnavailableError
        request = PlanRequest(server=interaction.server, principal=interaction.principal, sql=sql)
        async with asyncio.timeout(config.timeout_ms / 1000 + PLANNER_OVERHEAD_S):
            return await self._planner(request)

    def _verdict(
        self,
        decision: Decision,
        reason_code: SqlGuardReason,
        reason: str = "",
        *,
        risk_delta: float = 0.0,
    ) -> Verdict:
        return Verdict(
            decision=decision,
            control_id=self.id,
            reason_code=reason_code,
            reason=reason,
            risk_delta=risk_delta,
        )


def _with_sql(payload: object, sql: str) -> dict[str, Any]:
    """The tool call ``{"name", "arguments"}`` with its ``sql`` argument replaced."""
    call = cast("Mapping[str, Any]", payload)
    arguments = cast("Mapping[str, Any]", call["arguments"])
    return {**call, "arguments": {**arguments, SQL_ARGUMENT: sql}}
