"""The pipeline's view of human oversight: the approval queue and the kill switch.

`Oversight` is the one collaborator `gateway.pipeline.Pipeline` talks to for both, so the
pipeline keeps only the seams (where in the step order each check happens) and this module
keeps the rules.

**Retry protocol.** A held call is answered with an ``approval_id`` (MCP: a tool error with
``_meta["ai-control-layer/approval_id"]``; LLM: a 403 whose error body carries
``approval_id``). After an operator approves it, the agent sends the *same* call again with
the id attached: MCP ``tools/call`` params ``_meta: {"ai-control-layer/approval_id": "<id>"}``;
LLM the ``X-ACL-Approval-Id`` header. The id is only a pointer, never a credential: the
gateway re-authenticates the caller, requires the record to be bound to exactly this call
(session, principal, agent, channel, server, tool and argument digest), then runs base
authorization, blocklist, kill switch and every control again under the *current* policy.
Only the approval obligation is satisfied by the record; right before dispatch the record
moves ``approved → executing`` atomically, so it executes at most once.

A record created under an older policy revision stays usable if, and only if, the current
policy still allows the call (SPEC "Hot reload": re-checked when consumed): base
authorization and every control run under the current snapshot, and an approval never grants
anything outside base authorization.
"""

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Final, cast

from gateway.approvals.kill_switch import KillRecord, KillSwitch
from gateway.approvals.model import (
    APPROVAL_ID_PATTERN,
    Approval,
    ApprovalBinding,
    ApprovalConflictError,
    ApprovalState,
)
from gateway.approvals.service import ApprovalService
from gateway.core.types import Action, Channel
from gateway.identity import TokenClaims
from gateway.policy.loader import PolicySnapshot
from gateway.upstream import UpstreamError, UpstreamResult

KILL_SWITCH: Final = "kill_switch"  # a pipeline seam's control id, like `budget`
APPROVAL: Final = "approval"  # likewise, for refusals of a presented approval
AGENT_KILLED: Final = "agent_killed"
_APPROVAL_ID: Final = re.compile(APPROVAL_ID_PATTERN)

# Upstream failures that prove the call did not run, or ran and reported its own failure.
# Every other failure after dispatch (timeouts, a lost connection, a broken or oversized
# answer) leaves the outcome unknown: `uncertain`, which is never retried automatically.
_DEFINITE_FAILURES: Final = frozenset(
    {
        "upstream_rpc_error",  # the server answered with a JSON-RPC error
        "upstream_error",  # the server answered with an HTTP error status
        "upstream_invalid_request",  # refused before sending
        "upstream_session_closed",  # refused before sending
        "upstream_session_lost",  # the server did not know the session: nothing ran
        "upstream_misconfigured",  # refused before sending
    }
)


class ApprovalRefusal(StrEnum):
    UNKNOWN = "approval_unknown"
    MISMATCH = "approval_mismatch"
    DENIED = "approval_denied"
    EXPIRED = "approval_expired"
    USED = "approval_already_used"
    UNCERTAIN = "approval_outcome_uncertain"
    UNBINDABLE = "approval_unbindable"  # the payload has no canonical form to bind


_REFUSAL_BY_STATE: Final[Mapping[ApprovalState, ApprovalRefusal]] = {
    ApprovalState.DENIED: ApprovalRefusal.DENIED,
    ApprovalState.EXPIRED: ApprovalRefusal.EXPIRED,
    ApprovalState.EXECUTING: ApprovalRefusal.USED,
    ApprovalState.SUCCEEDED: ApprovalRefusal.USED,
    ApprovalState.FAILED: ApprovalRefusal.USED,
    ApprovalState.UNCERTAIN: ApprovalRefusal.UNCERTAIN,
}


def refusal_for(state: ApprovalState | None) -> ApprovalRefusal:
    """Why a presented approval in ``state`` cannot authorize anything (any state but
    pending or approved)."""
    if state is None:
        return ApprovalRefusal.UNKNOWN
    return _REFUSAL_BY_STATE[state]


def upstream_outcome(
    result: UpstreamResult | None, error: UpstreamError | None
) -> tuple[ApprovalState, str]:
    """The state a consumed approval ends in, and its outcome reason code."""
    if error is not None:
        if error.reason_code in _DEFINITE_FAILURES:
            return ApprovalState.FAILED, error.reason_code
        return ApprovalState.UNCERTAIN, error.reason_code
    body: object = result.body if result is not None else None
    if isinstance(body, dict) and cast("dict[str, Any]", body).get("isError") is True:
        return ApprovalState.FAILED, "tool_error"  # the MCP tool ran and reported failure
    return ApprovalState.SUCCEEDED, "upstream_ok"


class Oversight:
    """Approval queue plus kill switch, as the pipeline uses them."""

    def __init__(self, approvals: ApprovalService, kill_switch: KillSwitch) -> None:
        self._approvals = approvals
        self._kill_switch = kill_switch

    @property
    def approvals(self) -> ApprovalService:
        return self._approvals

    @property
    def kill_switch(self) -> KillSwitch:
        return self._kill_switch

    async def killed(self, agent: str) -> KillRecord | None:
        """Raises `KillSwitchUnavailableError` (fail closed) when the state is unknown."""
        return await self._kill_switch.check(agent)

    def binding(
        self, claims: TokenClaims, channel: Channel, server: str | None, payload: object
    ) -> ApprovalBinding | None:
        """The exact operation ``payload`` is; None when it has no canonical form."""
        digest = self._approvals.args_digest(payload)
        if digest is None:
            return None
        tool: str | None = None
        if channel is Channel.MCP and isinstance(payload, dict):
            name = cast("dict[str, Any]", payload).get("name")
            tool = name if isinstance(name, str) else None
        return ApprovalBinding(
            session_id=claims.session_id,
            principal=claims.sub,
            agent=claims.agent,
            channel=channel,
            server=server,
            tool=tool,
            args_digest=digest,
        )

    async def presented(
        self, approval_id: str, binding: ApprovalBinding | None
    ) -> Approval | ApprovalRefusal:
        """The record a retry names, if it can still stand for this exact call (pending or
        approved, bound to ``binding``); otherwise why not. Never changes the record."""
        if binding is None:
            return ApprovalRefusal.UNBINDABLE
        if not _valid_id(approval_id):
            return ApprovalRefusal.UNKNOWN
        record = await self._approvals.get(approval_id)
        if record is None:
            return ApprovalRefusal.UNKNOWN
        if record.binding != binding:  # checked before the state: no state oracle either
            return ApprovalRefusal.MISMATCH
        if record.state in {ApprovalState.PENDING, ApprovalState.APPROVED}:
            return record
        return refusal_for(record.state)

    async def hold(
        self,
        binding: ApprovalBinding,
        snapshot: PolicySnapshot,
        *,
        actions: tuple[Action, ...],
        resources: tuple[str, ...],
        reasons: tuple[str, ...],
    ) -> Approval:
        return await self._approvals.hold(
            binding, snapshot, actions=actions, resources=resources, reasons=reasons
        )

    async def consume(self, approval: Approval) -> Approval | ApprovalRefusal:
        """``approved → executing``, atomically; or why the record no longer allows it."""
        try:
            return await self._approvals.begin(approval.id)
        except ApprovalConflictError as exc:
            return refusal_for(exc.current.state if exc.current is not None else None)

    async def finish(
        self, approval: Approval, result: UpstreamResult | None, error: UpstreamError | None
    ) -> None:
        state, outcome = upstream_outcome(result, error)
        await self._approvals.finish(approval.id, state, outcome)

    async def finish_uncertain(self, approval: Approval, outcome: str) -> None:
        await self._approvals.finish(approval.id, ApprovalState.UNCERTAIN, outcome)


def _valid_id(approval_id: str) -> bool:
    return _APPROVAL_ID.fullmatch(approval_id) is not None
