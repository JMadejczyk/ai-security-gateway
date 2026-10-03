"""Operator API bodies, shared by the ``/admin`` routes and the CLI.

An approval's view never carries call arguments: the operator sees the tool, the resources
it would touch, why it was held and the argument digest (to match it against a payload they
already hold), never the payload itself.
"""

from datetime import datetime
from typing import Annotated, Self

from pydantic import AwareDatetime, StringConstraints

from gateway.approvals.kill_switch import MAX_REASON_CHARS, KillRecord
from gateway.approvals.model import Approval, ApprovalState, Note
from gateway.core.envelope import FrozenModel
from gateway.core.types import Action, Channel

type AgentName = Annotated[str, StringConstraints(min_length=1, max_length=128)]
type Reason = Annotated[str, StringConstraints(max_length=MAX_REASON_CHARS)]


class ApprovalView(FrozenModel):
    id: str
    state: ApprovalState
    agent: str
    principal: str
    session_id: str
    channel: Channel
    server: str | None
    tool: str | None
    actions: tuple[Action, ...]
    resources: tuple[str, ...]
    reasons: tuple[str, ...]
    args_digest: str
    policy_revision: str
    approver_roles: tuple[str, ...]
    created_at: AwareDatetime
    expires_at: AwareDatetime
    updated_at: AwareDatetime
    decided_by: str | None = None
    decided_at: AwareDatetime | None = None
    note: str | None = None
    outcome: str | None = None

    @classmethod
    def of(cls, approval: Approval) -> Self:
        binding = approval.binding
        return cls(
            id=approval.id,
            state=approval.state,
            agent=binding.agent,
            principal=binding.principal,
            session_id=binding.session_id,
            channel=binding.channel,
            server=binding.server,
            tool=binding.tool,
            actions=approval.actions,
            resources=approval.resources,
            reasons=approval.reasons,
            args_digest=binding.args_digest,
            policy_revision=approval.policy_revision,
            approver_roles=approval.approver_roles,
            created_at=approval.created_at,
            expires_at=approval.expires_at,
            updated_at=approval.updated_at,
            decided_by=approval.decided_by,
            decided_at=approval.decided_at,
            note=approval.note,
            outcome=approval.outcome,
        )


class ApprovalList(FrozenModel):
    approvals: tuple[ApprovalView, ...]


class DecisionBody(FrozenModel):
    note: Note | None = None


class KillBody(FrozenModel):
    agent: AgentName
    reason: Reason = ""


class UnkillBody(FrozenModel):
    agent: AgentName


class KillView(FrozenModel):
    agent: str
    killed: bool
    reason: str = ""
    killed_by: str | None = None
    killed_at: datetime | None = None
    revoked_approvals: int = 0  # unused approvals denied by this kill

    @classmethod
    def of(cls, record: KillRecord, *, revoked: int = 0) -> Self:
        return cls(
            agent=record.agent,
            killed=True,
            reason=record.reason,
            killed_by=record.killed_by,
            killed_at=record.killed_at,
            revoked_approvals=revoked,
        )


class KillList(FrozenModel):
    kills: tuple[KillView, ...]
