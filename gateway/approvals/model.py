"""The approval record and its state machine (SPEC "Human in the loop").

An approval authorizes one exact pending operation. Its `ApprovalBinding` names that
operation: who (principal, agent, session), where (channel, MCP server, tool) and what (a
keyed digest of the canonical arguments). The rest of the record says why it was held, under
which policy revision, who may decide, and how far it got.

States and the only allowed moves::

    pending ──approve──▶ approved ──consume──▶ executing ──▶ succeeded | failed | uncertain
       │                    │
       ├──deny──▶ denied ◀──┤ (deny revokes an unused approval; a kill switch does the same)
       └─timeout▶ expired ◀─┘
                    denied ◀── executing (a kill landed after consumption, before dispatch:
                                          the call never ran)

``uncertain`` means the upstream outcome is unknown (timeout or lost connection after the
request was sent): it is terminal and never retried automatically.

A kill switch revokes an agent's unused approvals by bumping the agent's revocation
generation: every record carries the generation it was created under, and a pending or
approved record whose generation is stale lapses to ``denied`` (outcome ``agent_killed``)
atomically wherever it is next read or moved, however many records there are.
"""

from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Final

from pydantic import AwareDatetime, StringConstraints

from gateway.core.envelope import FrozenModel
from gateway.core.types import Action, Channel

APPROVAL_ID_PATTERN: Final = r"^apr-[0-9a-f]{24}(-[1-9][0-9]{0,8})?$"
MAX_NOTE_CHARS: Final = 500

type ApprovalId = Annotated[str, StringConstraints(pattern=APPROVAL_ID_PATTERN)]
type Note = Annotated[str, StringConstraints(max_length=MAX_NOTE_CHARS)]


class ApprovalState(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


TRANSITIONS: Final[Mapping[ApprovalState, frozenset[ApprovalState]]] = MappingProxyType(
    {
        ApprovalState.PENDING: frozenset(
            {ApprovalState.APPROVED, ApprovalState.DENIED, ApprovalState.EXPIRED}
        ),
        ApprovalState.APPROVED: frozenset(
            {ApprovalState.EXECUTING, ApprovalState.DENIED, ApprovalState.EXPIRED}
        ),
        ApprovalState.EXECUTING: frozenset(
            {
                ApprovalState.SUCCEEDED,
                ApprovalState.FAILED,
                ApprovalState.UNCERTAIN,
                ApprovalState.DENIED,  # stopped between consumption and dispatch
            }
        ),
        ApprovalState.DENIED: frozenset(),
        ApprovalState.EXPIRED: frozenset(),
        ApprovalState.SUCCEEDED: frozenset(),
        ApprovalState.FAILED: frozenset(),
        ApprovalState.UNCERTAIN: frozenset(),
    }
)

# States that still hold the operation's slot: a new call of the same operation gets this
# approval back instead of a new one.
OPEN_STATES: Final = frozenset(
    {ApprovalState.PENDING, ApprovalState.APPROVED, ApprovalState.EXECUTING}
)
# States that run out at ``expires_at``.
EXPIRING_STATES: Final = frozenset({ApprovalState.PENDING, ApprovalState.APPROVED})


def operations_of(pairs: Iterable[tuple[Action, str]]) -> tuple[str, ...]:
    """The sorted, unique ``action:resource`` set of a call's interactions."""
    return tuple(sorted({f"{action.value}:{resource}" for action, resource in pairs}))


def sources_of(target: ApprovalState) -> frozenset[ApprovalState]:
    """Every state from which ``target`` may be reached."""
    return frozenset(state for state, targets in TRANSITIONS.items() if target in targets)


class ApprovalBinding(FrozenModel):
    """The one exact operation an approval authorizes; a retry must match every field."""

    session_id: str
    principal: str
    agent: str
    channel: Channel
    server: str | None = None  # MCP upstream; None on the LLM channel
    tool: str | None = None  # MCP tool name; None on the LLM channel
    args_digest: str  # keyed HMAC of the canonical (sorted-key JSON) payload the agent sent
    # Every normalized `action:resource` of the call's interactions, sorted: the same
    # arguments normalized under another policy (a changed resource template) are another
    # operation.
    operations: tuple[str, ...] = ()


class ApprovalDraft(FrozenModel):
    """The immutable part of a record, fixed when the operation is first held."""

    operation: str  # keyed digest of the binding: one slot per exact operation
    binding: ApprovalBinding
    # Keyed HMAC of the final payload after every rewrite and redaction (before sealing,
    # which is deterministic and checked again at dispatch): what an approval lets run.
    operation_digest: str
    policy_revision: str  # the revision the hold was decided under
    actions: tuple[Action, ...] = ()
    resources: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()  # reason codes of the verdicts that required approval
    approver_roles: tuple[str, ...] = ()  # the agent's `approvers` at creation (display only)
    created_at: AwareDatetime


class Approval(ApprovalDraft):
    """A stored approval: the draft plus where it is in its lifecycle."""

    id: ApprovalId
    state: ApprovalState
    expires_at: AwareDatetime
    updated_at: AwareDatetime
    decided_by: str | None = None
    decided_at: AwareDatetime | None = None
    note: Note | None = None
    outcome: str | None = None  # reason code: why it ended (upstream_timeout, agent_killed...)
    generation: int = 0  # the agent's revocation generation when the record was created

    def expired_at(self, now: datetime) -> bool:
        """True when the record has run out but has not been moved to ``expired`` yet."""
        return self.state in EXPIRING_STATES and now >= self.expires_at

    def can_become(self, target: ApprovalState) -> bool:
        return target in TRANSITIONS[self.state]


class Transition(FrozenModel):
    """One requested state change, applied atomically by the store."""

    target: ApprovalState
    at: AwareDatetime
    decided_by: str | None = None
    note: Note | None = None
    outcome: str | None = None
    expires_at: AwareDatetime | None = None  # a new deadline (set when approving)
    # Narrows the legal sources (an operator's deny never touches a record being executed).
    only_from: frozenset[ApprovalState] | None = None

    @property
    def sources(self) -> frozenset[ApprovalState]:
        legal = sources_of(self.target)
        return legal if self.only_from is None else legal & self.only_from


class ApprovalConflictError(Exception):
    """A transition did not apply: the record is missing or no longer in a source state."""

    def __init__(self, approval_id: str, current: Approval | None) -> None:
        state = current.state.value if current is not None else "missing"
        super().__init__(f"{approval_id}: cannot transition from {state}")
        self.approval_id = approval_id
        self.current = current


def approval_id_for(operation: str, generation: int) -> str:
    """Deterministic id: the first approval of an operation is ``apr-<24 hex>``; a later one
    (after the previous ended) gets ``-<generation>``."""
    base = f"apr-{operation[:24]}"
    return base if generation <= 1 else f"{base}-{generation}"
