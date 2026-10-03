"""``sql_guard``'s planner over MCP: the SQL server's internal ``explain`` tool.

SPEC "Row-level filtering": the plan cost comes from the same server that runs the query,
with the same principal and transaction setup (read-only, transaction-local principal,
statement timeout), as ``EXPLAIN (FORMAT JSON)`` without ``ANALYZE``.

``explain`` is deliberately absent from the operator mapping (``upstreams.mcp.<server>.tools``):
agents never see it in ``tools/list``, and their ``tools/call`` for it is refused as unmapped
before anything reaches the upstream. The gateway calls it here, through the upstream client
and past the downstream mapping on purpose, in a short-lived upstream session of its own for
the caller's principal: the request carries the same ``X-ACL-Principal`` assertion, execution
limits included, that the ``query`` call will carry.

The endpoint, trust and execution limits come from ``request.snapshot``, the admitted call's own
policy snapshot, never the store's current one: a reload mid-call cannot move the plan to
another endpoint or sign it with other limits than the statement will run under.

Wired at the composition root with `functools.partial` (connector bound), which makes it a
`gateway.controls.sql_guard.SqlPlanner`.
"""

from typing import Final

from pydantic import Field, ValidationError

from gateway.controls.sql_guard import PlanRequest, PlanUnavailableError
from gateway.core.envelope import FrozenModel
from gateway.proxies.mcp import wire
from gateway.proxies.mcp.upstream import MCPConnector
from gateway.upstream import UpstreamError

EXPLAIN_TOOL: Final = "explain"


class ExplainResult(FrozenModel):
    """The ``explain`` tool's structured result: the plan's top-level ``Total Cost``."""

    total_cost: float = Field(ge=0.0)


async def explain_cost(connector: MCPConnector, request: PlanRequest) -> float:
    """The planner cost of ``request.sql`` on ``request.server``, as ``request.principal``.

    Raises `PlanUnavailableError` unless the server is a trusted sql upstream in the call's
    policy snapshot and answers with a well-formed cost.
    """
    snapshot = request.snapshot
    config = snapshot.policy.upstreams.mcp.get(request.server)
    if config is None or config.adapter != "sql" or config.trust != "internal":
        raise PlanUnavailableError
    upstream = connector.open(request.server, config, request.principal)
    try:
        answer = await upstream.execute(
            {"name": EXPLAIN_TOOL, "arguments": {"sql": request.sql}}, snapshot
        )
    except UpstreamError as exc:
        raise PlanUnavailableError from exc
    finally:
        await upstream.aclose()
    try:
        result = wire.CallToolResult.model_validate(answer.body)
        if result.is_error or result.structured_content is None:
            raise PlanUnavailableError
        return ExplainResult.model_validate(result.structured_content).total_cost
    except ValidationError as exc:
        raise PlanUnavailableError from exc
