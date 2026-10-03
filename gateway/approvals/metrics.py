"""Approval queue and kill switch metrics, on the gateway's operator-only registry.

- ``acl_approvals_pending``: approvals waiting for a decision (the queue length, read from the
  store after every change and by the sweeper, so all gateway processes agree).
- ``acl_approvals_total{decision}``: approvals reaching a state, ``decision`` being the state
  (``pending`` = requested, ``approved``, ``denied``, ``expired``, ``executing``, ``succeeded``,
  ``failed``, ``uncertain``): bounded by the state enum.
- ``acl_kill_switch_active{agent}``: 1 per killed agent; agents outside the policy are
  bucketed as ``other``.
"""

from collections.abc import Iterable, Mapping

from prometheus_client import Counter, Gauge

from gateway.approvals.model import ApprovalState
from gateway.telemetry import REGISTRY, bounded

APPROVALS_PENDING = Gauge(
    "acl_approvals_pending",
    "Approvals waiting for a human decision.",
    registry=REGISTRY,
)
APPROVALS = Counter(
    "acl_approvals",
    "Approvals reaching a state (`decision` is the state reached).",
    ["decision"],
    registry=REGISTRY,
)
for _state in ApprovalState:  # every state at 0, so `increase()` sees the first of each
    APPROVALS.labels(decision=_state.value)
KILL_SWITCH_ACTIVE = Gauge(
    "acl_kill_switch_active",
    "1 for every agent whose kill switch is on.",
    ["agent"],
    registry=REGISTRY,
)


def set_approvals_pending(count: int) -> None:
    APPROVALS_PENDING.set(count)


def record_approval_state(state: ApprovalState) -> None:
    APPROVALS.labels(decision=state.value).inc()


def set_killed_agents(killed: Iterable[str], known: Mapping[str, object]) -> None:
    KILL_SWITCH_ACTIVE.clear()
    for agent in killed:
        KILL_SWITCH_ACTIVE.labels(agent=bounded(agent, known)).set(1)
