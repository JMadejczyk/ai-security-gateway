"""`OperatorAccess`: who may see and decide which approvals, under the current policy."""

from datetime import UTC, datetime

import pytest

from gateway.approvals.model import Approval, ApprovalBinding, ApprovalState
from gateway.approvals.operators import OperatorAccess, OperatorRefusedError
from gateway.core.types import Channel, SessionMode
from gateway.identity import Actor, TokenClaims

NOW = datetime(2026, 10, 4, 10, 0, tzinfo=UTC)


def claims(sub: str, agent: str, *roles: str) -> TokenClaims:
    mode = SessionMode.AUTONOMOUS if sub.startswith("svc:") else SessionMode.INTERACTIVE
    return TokenClaims(
        iss="ai-control-layer",
        aud="ai-control-layer",
        sub=sub,
        act=Actor(sub=agent),
        roles=roles,
        mode=mode,
        session_id="s-op",
        iat=0,
        exp=600,
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
        policy_revision="r",
        created_at=NOW,
        state=ApprovalState.PENDING,
        expires_at=NOW,
        updated_at=NOW,
    )


@pytest.mark.parametrize(
    ("operator", "record", "reason"),
    [
        pytest.param(claims("olga@demo", "databot", "ops-team"), approval(), None, id="approver"),
        pytest.param(claims("root@demo", "databot", "admin"), approval(), None, id="admin"),
        pytest.param(
            claims("bartek@demo", "databot", "intern"),
            approval(),
            "approver_role_required",
            id="no-approver-role",
        ),
        pytest.param(
            claims("olga@demo", "databot", "ops-team"),
            approval(principal="olga@demo", agent="nightly_etl"),
            "self_approval_forbidden",
            id="own-principal",
        ),
        pytest.param(
            claims("root@demo", "databot", "admin"),
            approval(principal="anna@demo", agent="databot"),
            "self_approval_forbidden",
            id="own-agent-even-for-admin",
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
    olga = OperatorAccess(claims("olga@demo", "databot", "ops-team"), snapshot)
    assert olga.approvable_agents() == frozenset({"databot", "nightly_etl"})
    bartek = OperatorAccess(claims("bartek@demo", "databot", "intern"), snapshot)
    assert not bartek.can_view(approval())
    with pytest.raises(OperatorRefusedError, match="operator_role_required"):
        bartek.require_operator()
    with pytest.raises(OperatorRefusedError, match="admin_required"):
        olga.require_admin()
