"""``/admin`` routes for approvals and the kill switch (operator listener only).

- ``GET /admin/approvals[?state=pending&limit=100]``: the approvals the caller may see.
- ``GET /admin/approvals/{id}``: one approval (404 when it does not exist or is not visible).
- ``POST /admin/approvals/{id}/approve`` / ``/deny`` with an optional ``{"note": ...}``.
- ``POST /admin/kill {agent, reason}``, ``POST /admin/unkill {agent}``, ``GET /admin/kill``:
  admin only.

Every route verifies the operator token against the current policy snapshot first
(`gateway.approvals.operators`). Refusals are `RejectionError`s, rendered by the operator
app's error handler with their reason code.
"""

import re
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Query, Request

from gateway.approvals.kill_switch import KillRecord, KillSwitch
from gateway.approvals.model import APPROVAL_ID_PATTERN, ApprovalState
from gateway.approvals.operators import OperatorAccess
from gateway.approvals.service import (
    ApprovalNotFoundError,
    ApprovalService,
    log_operator_action,
)
from gateway.approvals.views import (
    ApprovalList,
    ApprovalView,
    DecisionBody,
    KillBody,
    KillList,
    KillView,
    UnkillBody,
)
from gateway.clock import Clock, utc_now
from gateway.errors import RejectionError
from gateway.identity import TokenVerifier
from gateway.policy.loader import PolicySnapshot

_APPROVAL_ID = re.compile(APPROVAL_ID_PATTERN)


class UnknownAgentError(RejectionError):
    status_code = 404

    def __init__(self) -> None:
        super().__init__("unknown_agent", "no agent with that name in the policy")


def admin_router(  # noqa: PLR0913 -- the collaborators of the admin routes, wired once
    *,
    verifier: TokenVerifier,
    policy: Callable[[], PolicySnapshot],
    approvals: ApprovalService,
    kill_switch: KillSwitch,
    token_of: Callable[[Request], str | None],
    clock: Clock = utc_now,
) -> APIRouter:
    router = APIRouter(prefix="/admin")

    def operator(request: Request) -> tuple[OperatorAccess, PolicySnapshot]:
        snapshot = policy()
        claims = verifier.verify(token_of(request), snapshot)
        return OperatorAccess(claims, snapshot), snapshot

    def approval_id_of(raw: str) -> str:
        if not _APPROVAL_ID.fullmatch(raw):
            raise ApprovalNotFoundError
        return raw

    @router.get("/approvals")
    async def list_approvals(
        request: Request,
        state: ApprovalState | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> ApprovalList:
        access, _snapshot = operator(request)
        records = await approvals.list_for(access, state, limit)
        return ApprovalList(approvals=tuple(ApprovalView.of(r) for r in records))

    @router.get("/approvals/{approval_id}")
    async def show_approval(approval_id: str, request: Request) -> ApprovalView:
        access, _snapshot = operator(request)
        return ApprovalView.of(await approvals.view(approval_id_of(approval_id), access))

    async def decide(
        approval_id: str, request: Request, body: DecisionBody | None, *, approve: bool
    ) -> ApprovalView:
        access, snapshot = operator(request)
        decided = await approvals.decide(
            approval_id_of(approval_id),
            access,
            snapshot,
            approve=approve,
            note=body.note if body is not None else None,
        )
        return ApprovalView.of(decided)

    @router.post("/approvals/{approval_id}/approve")
    async def approve(
        approval_id: str, request: Request, body: DecisionBody | None = None
    ) -> ApprovalView:
        return await decide(approval_id, request, body, approve=True)

    @router.post("/approvals/{approval_id}/deny")
    async def deny(
        approval_id: str, request: Request, body: DecisionBody | None = None
    ) -> ApprovalView:
        return await decide(approval_id, request, body, approve=False)

    @router.post("/kill")
    async def kill(body: KillBody, request: Request) -> KillView:
        access, snapshot = operator(request)
        access.require_admin()
        if body.agent not in snapshot.policy.agents:
            raise UnknownAgentError
        record = KillRecord(
            agent=body.agent, reason=body.reason, killed_by=access.principal, killed_at=clock()
        )
        await kill_switch.kill(record)
        # The kill already holds every call; denying its unused approvals makes them
        # unusable for good, even after an unkill.
        revoked = await approvals.revoke_agent(body.agent, by=access.principal)
        log_operator_action(
            "kill",
            agent=body.agent,
            operator=access.principal,
            reason=body.reason,
            revoked_approvals=revoked,
            policy_revision=snapshot.revision,
        )
        return KillView.of(record, revoked=revoked)

    @router.post("/unkill")
    async def unkill(body: UnkillBody, request: Request) -> KillView:
        access, snapshot = operator(request)
        access.require_admin()
        was_killed = await kill_switch.unkill(body.agent)
        log_operator_action(
            "unkill",
            agent=body.agent,
            operator=access.principal,
            was_killed=was_killed,
            policy_revision=snapshot.revision,
        )
        return KillView(agent=body.agent, killed=False)

    @router.get("/kill")
    async def list_kills(request: Request) -> KillList:
        access, _snapshot = operator(request)
        access.require_admin()
        active = await kill_switch.active()
        return KillList(kills=tuple(KillView.of(r) for _, r in sorted(active.items())))

    return router
