"""`OperatorAccess`: who may see and decide which approvals, under the current policy."""

from datetime import UTC, datetime

import pytest

from gateway.approvals.model import Approval, ApprovalBinding, ApprovalState
from gateway.approvals.operators import OperatorAccess, OperatorRefusedError
from gateway.core.types import Channel
from gateway.identity import OperatorClaims

NOW = datetime(2026, 10, 4, 10, 0, tzinfo=UTC)


def claims(sub: str, *roles: str) -> OperatorClaims:
    return OperatorClaims(
        iss="ai-control-layer", aud="ai-control-layer-operator", sub=sub, roles=roles, iat=0, exp=1
    )


def approval(principal: str = "svc:nightly_etl", agent: str = "nightly_etl") -> Approval:
    return Approval(
        id="apr-" + "a" * 24,
        operation="ab" * 32,
        binding=ApprovalBinding(
            session_id="s-1",
            principal=principal,
            agent=agent,
            channel=Channel.MCP,
            server="reports",
            tool="write_report",
            args_digest="cd" * 32,
        ),
        operation_digest="ef" * 32,
        policy_revision="r",
        created_at=NOW,
        state=ApprovalState.PENDING,
        expires_at=NOW,
        updated_at=NOW,
    )


@pytest.mark.parametrize(
    ("operator", "record", "reason"),
    [
        pytest.param(claims("olga@demo", "ops-team"), approval(), None, id="approver"),
        pytest.param(claims("root@demo", "admin"), approval(), None, id="admin"),
        pytest.param(
            claims("bartek@demo", "intern"),
            approval(),
            "approver_role_required",
            id="no-approver-role",
        ),
        pytest.param(
            claims("olga@demo", "ops-team"),
            approval(principal="olga@demo", agent="nightly_etl"),
            "self_approval_forbidden",
            id="own-principal",
        ),
        pytest.param(
            claims("olga@demo", "ops-team"),
            approval(principal="anna@demo", agent="databot"),
            None,
            id="another-persons-databot-call",
        ),
        pytest.param(
            claims("root@demo", "admin"),
            approval(principal="root@demo", agent="databot"),
            "self_approval_forbidden",
            id="own-session-even-for-admin",
        ),
    ],
)
def test_who_may_decide(snapshot, operator, record, reason):
    access = OperatorAccess(operator, snapshot)
    if reason is None:
        access.require_decider(record)
        return
    with pytest.raises(OperatorRefusedError) as caught:
        access.require_decider(record)
    assert caught.value.reason_code == reason


def test_approvers_see_only_the_agents_that_name_their_role(snapshot):
    olga = OperatorAccess(claims("olga@demo", "ops-team"), snapshot)
    assert olga.approvable_agents() == frozenset({"databot", "opencode", "nightly_etl"})
    bartek = OperatorAccess(claims("bartek@demo", "intern"), snapshot)
    assert not bartek.can_view(approval())
    with pytest.raises(OperatorRefusedError, match="operator_role_required"):
        bartek.require_operator()
    with pytest.raises(OperatorRefusedError, match="admin_required"):
        olga.require_admin()
