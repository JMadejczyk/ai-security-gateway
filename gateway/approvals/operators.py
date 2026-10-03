"""Who may use ``/admin/*`` (SPEC "Identity": operator tokens).

An operator token (`gateway.identity.OperatorClaims`: audience ``ai-control-layer-operator``,
no agent, no session) is verified against the current policy. Role ``admin`` may do
everything. Any other caller must hold an approver role (a role some agent names in
``approvers``) and then sees and decides only the approvals of agents that name one of its
roles. Nobody decides an approval for their own session: the approver (``sub``) must differ
from the record's principal. An agent can never approve anything: agent tokens do not open
the operator API at all.

Authorization reads the current policy, not the one the approval was created under: an
approver role removed by a reload stops working at once.
"""

from enum import StrEnum

from gateway.approvals.model import Approval
from gateway.errors import RejectionError
from gateway.identity import ADMIN_ROLE, OperatorClaims
from gateway.policy.loader import PolicySnapshot


class OperatorReason(StrEnum):
    OPERATOR_ROLE_REQUIRED = "operator_role_required"
    ADMIN_REQUIRED = "admin_required"
    APPROVER_ROLE_REQUIRED = "approver_role_required"
    SELF_APPROVAL = "self_approval_forbidden"


class OperatorRefusedError(RejectionError):
    status_code = 403

    def __init__(self, reason: OperatorReason) -> None:
        super().__init__(reason.value, f"operator refused: {reason.value.replace('_', ' ')}")
        self.reason = reason


class OperatorAccess:
    """One verified operator token under one policy snapshot."""

    def __init__(self, claims: OperatorClaims, snapshot: PolicySnapshot) -> None:
        self._claims = claims
        self._snapshot = snapshot

    @property
    def principal(self) -> str:
        return self._claims.sub

    @property
    def roles(self) -> tuple[str, ...]:
        return self._claims.roles

    @property
    def is_admin(self) -> bool:
        return ADMIN_ROLE in self._claims.roles

    def approvable_agents(self) -> frozenset[str]:
        """Agents whose ``approvers`` name one of the caller's roles."""
        roles = set(self._claims.roles)
        return frozenset(
            agent_id
            for agent_id, agent in self._snapshot.policy.agents.items()
            if roles.intersection(agent.approvers)
        )

    def require_operator(self) -> None:
        """Admin, or an approver of at least one agent."""
        if not self.is_admin and not self.approvable_agents():
            raise OperatorRefusedError(OperatorReason.OPERATOR_ROLE_REQUIRED)

    def require_admin(self) -> None:
        if not self.is_admin:
            raise OperatorRefusedError(OperatorReason.ADMIN_REQUIRED)

    def can_view(self, approval: Approval) -> bool:
        return self.is_admin or approval.binding.agent in self.approvable_agents()

    def require_decider(self, approval: Approval) -> None:
        if not self.can_view(approval):
            raise OperatorRefusedError(OperatorReason.APPROVER_ROLE_REQUIRED)
        if self.principal == approval.binding.principal:
            raise OperatorRefusedError(OperatorReason.SELF_APPROVAL)
