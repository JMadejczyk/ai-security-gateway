"""The approval queue: holds, decisions, consumption and expiry (SPEC "Human in the loop").

`ApprovalService` is the one path every ``require_approval`` outcome takes. The pipeline
holds an operation (`hold`: create-or-get, so retries never duplicate an entry), operators
decide it (`decide`, through ``/admin/approvals``), and a retry carrying the id consumes it
(`begin`: atomically ``approved → executing``) and records the upstream outcome (`finish`).
Pending and approved records run out at their deadline: lazily when read, and through
`expire_due`, which the container's sweeper calls every few seconds.

Deadlines: a pending approval expires ``approvals.timeout_s`` after it was requested; an
approval, once given, is usable for ``approvals.timeout_s`` after the decision. Timeout means
deny (``on_timeout: deny``, the only accepted value).

Every operator decision and kill-switch revocation leaves one structured line on the
``gateway.operator`` logger: ids, identities, states and the policy revision, never call
arguments.
"""

import json
import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import Final

from gateway.approvals.metrics import record_approval_state, set_approvals_pending
from gateway.approvals.model import (
    Approval,
    ApprovalBinding,
    ApprovalConflictError,
    ApprovalDraft,
    ApprovalState,
    Transition,
)
from gateway.approvals.operators import OperatorAccess
from gateway.approvals.store import ApprovalStore, ApprovalStoreUnavailableError
from gateway.canonical import canonical_digest
from gateway.clock import Clock, utc_now
from gateway.core.types import Action
from gateway.errors import RejectionError
from gateway.policy.loader import PolicySnapshot

logger = logging.getLogger(__name__)
operator_log = logging.getLogger("gateway.operator")

MIN_RETENTION_S: Final = 7 * 24 * 3600  # records stay readable this long after creation
RETENTION_MARGIN_S: Final = 24 * 3600


class ApprovalNotFoundError(RejectionError):
    status_code = 404

    def __init__(self) -> None:
        super().__init__("approval_not_found", "no such approval")


class ApprovalStateError(RejectionError):
    status_code = 409

    def __init__(self, state: ApprovalState | None) -> None:
        current = state.value if state is not None else "missing"
        super().__init__("approval_not_pending", f"the approval is {current}")
        self.state = state


def log_operator_action(action: str, **fields: object) -> None:
    """One structured operator-action line: identities, ids and states, never payloads."""
    operator_log.info(json.dumps({"event": "operator_action", "action": action, **fields}))


class ApprovalService:
    def __init__(self, store: ApprovalStore, *, key: bytes, clock: Clock = utc_now) -> None:
        self._store = store
        self._key = key
        self._clock = clock

    @property
    def store(self) -> ApprovalStore:
        return self._store

    # ----------------------------------------------------------------------- binding

    def args_digest(self, payload: object) -> str | None:
        """Keyed digest of the canonical payload; None when it has no canonical form."""
        return canonical_digest(self._key, payload)

    def operation_of(self, binding: ApprovalBinding) -> str:
        """The slot of one exact operation: a keyed digest of every binding field."""
        digest = canonical_digest(self._key, binding.model_dump(mode="json"))
        if digest is None:  # a binding is strings only; this cannot happen
            msg = "approval binding has no canonical form"
            raise ValueError(msg)
        return digest

    # ------------------------------------------------------------------------ queue

    async def hold(
        self,
        binding: ApprovalBinding,
        snapshot: PolicySnapshot,
        *,
        actions: Sequence[Action],
        resources: Sequence[str],
        reasons: Sequence[str],
    ) -> Approval:
        """The open approval of this exact operation, or a new pending one."""
        now = self._clock()
        policy = snapshot.policy
        agent = policy.agents.get(binding.agent)
        draft = ApprovalDraft(
            operation=self.operation_of(binding),
            binding=binding,
            policy_revision=snapshot.revision,
            actions=tuple(dict.fromkeys(actions)),
            resources=tuple(dict.fromkeys(resources)),
            reasons=tuple(dict.fromkeys(reasons)),
            approver_roles=agent.approvers if agent is not None else (),
            created_at=now,
        )
        timeout_s = policy.approvals.timeout_s
        record, created = await self._store.create_or_get(
            draft,
            expires_at=now + timedelta(seconds=timeout_s),
            retention_s=int(max(MIN_RETENTION_S, 2 * timeout_s + RETENTION_MARGIN_S)),
        )
        if created:
            record_approval_state(ApprovalState.PENDING)
            await self.refresh_pending_gauge()
        return record

    async def get(self, approval_id: str) -> Approval | None:
        """The record, expired first if its deadline has passed."""
        record = await self._store.get(approval_id)
        if record is not None and record.expired_at(self._clock()):
            return await self._expire(record.id) or record
        return record

    async def list_for(
        self, operator: OperatorAccess, state: ApprovalState | None, limit: int
    ) -> list[Approval]:
        """Records the operator may see, newest first; expired lazily on the way."""
        operator.require_operator()
        await self.expire_due()
        states = frozenset({state}) if state is not None else None
        records = await self._store.records(states)
        return [r for r in records if operator.can_view(r)][:limit]

    async def view(self, approval_id: str, operator: OperatorAccess) -> Approval:
        operator.require_operator()
        record = await self.get(approval_id)
        if record is None or not operator.can_view(record):  # no existence oracle
            raise ApprovalNotFoundError
        return record

    async def decide(
        self,
        approval_id: str,
        operator: OperatorAccess,
        snapshot: PolicySnapshot,
        *,
        approve: bool,
        note: str | None = None,
    ) -> Approval:
        """Approve a pending approval, or deny a pending or approved (unused) one."""
        record = await self.view(approval_id, operator)
        operator.require_decider(record)
        target = ApprovalState.APPROVED if approve else ApprovalState.DENIED
        allowed = record.state is ApprovalState.PENDING if approve else record.can_become(target)
        if not allowed:
            raise ApprovalStateError(record.state)
        now = self._clock()
        change = Transition(
            target=target,
            at=now,
            decided_by=operator.principal,
            note=note,
            expires_at=(
                now + timedelta(seconds=snapshot.policy.approvals.timeout_s) if approve else None
            ),
            outcome=None if approve else "operator_denied",
        )
        try:  # approving applies only from pending: two approvers cannot both succeed
            decided = await self._apply(record.id, change)
        except ApprovalConflictError as exc:
            raise ApprovalStateError(exc.current.state if exc.current else None) from None
        log_operator_action(
            "approve" if approve else "deny",
            approval_id=decided.id,
            agent=decided.binding.agent,
            principal=decided.binding.principal,
            operator=operator.principal,
            operator_roles=list(operator.roles),
            state=decided.state.value,
            policy_revision=snapshot.revision,
            created_under=decided.policy_revision,
            note_chars=len(note or ""),
        )
        return decided

    async def begin(self, approval_id: str) -> Approval:
        """Consume an approval: ``approved → executing``, atomically; raises
        `ApprovalConflictError` when it is no longer approved (used, denied, expired)."""
        change = Transition(target=ApprovalState.EXECUTING, at=self._clock())
        return await self._apply(approval_id, change)

    async def finish(self, approval_id: str, state: ApprovalState, outcome: str) -> None:
        """Record the upstream outcome of a consumed approval (best effort: the call has
        already run, and a lost write leaves it ``executing``, which never runs again)."""
        change = Transition(target=state, at=self._clock(), outcome=outcome)
        try:
            await self._apply(approval_id, change)
        except (ApprovalConflictError, ApprovalStoreUnavailableError) as exc:
            logger.warning(
                "approval %s: outcome %s not recorded (%s)", approval_id, state, type(exc).__name__
            )

    async def revoke_agent(self, agent: str, *, by: str) -> int:
        """Deny every unused approval of a killed agent; returns how many."""
        open_states = frozenset({ApprovalState.PENDING, ApprovalState.APPROVED})
        revoked = 0
        for record in await self._store.records(open_states):
            if record.binding.agent != agent:
                continue
            change = Transition(
                target=ApprovalState.DENIED, at=self._clock(), decided_by=by, outcome="agent_killed"
            )
            try:
                await self._apply(record.id, change)
            except ApprovalConflictError:
                continue  # decided, consumed or expired meanwhile
            revoked += 1
        return revoked

    async def expire_due(self) -> int:
        """Move every pending or approved record past its deadline to ``expired``."""
        expired = 0
        for approval_id in await self._store.due(self._clock()):
            if await self._expire(approval_id) is not None:
                expired += 1
        if expired:
            await self.refresh_pending_gauge()
        return expired

    async def refresh_pending_gauge(self) -> None:
        try:
            set_approvals_pending(await self._store.pending_count())
        except ApprovalStoreUnavailableError:
            logger.warning("approval store unavailable: acl_approvals_pending not refreshed")

    # --------------------------------------------------------------------- internals

    async def _expire(self, approval_id: str) -> Approval | None:
        change = Transition(target=ApprovalState.EXPIRED, at=self._clock())
        try:
            record = await self._store.transition(approval_id, change)
        except ApprovalConflictError:
            return None
        if record.state is ApprovalState.EXPIRED:
            record_approval_state(ApprovalState.EXPIRED)
        return record

    async def _apply(self, approval_id: str, change: Transition) -> Approval:
        record = await self._store.transition(approval_id, change)
        record_approval_state(change.target)
        await self.refresh_pending_gauge()
        return record
