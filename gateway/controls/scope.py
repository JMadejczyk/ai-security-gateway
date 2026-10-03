"""Per-call facts a control may need beyond its `Interaction`, set by the pipeline.

The `Control` interface hands a control one interaction and its config. A few controls also
need the call's policy snapshot and authenticated principal (``model_allowlist`` re-checks the
model through the evaluator against the same snapshot the call was admitted under), or whether
the call retries an approved operation (approval retries are excluded from ``loop_detect``).
The pipeline sets a `CallScope` for the duration of one call; outside a call there is none.
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from gateway.policy.evaluator import PrincipalContext
from gateway.policy.loader import PolicySnapshot


@dataclass(frozen=True, slots=True)
class CallScope:
    snapshot: PolicySnapshot  # the one snapshot every decision of this call reads
    principal: PrincipalContext
    # Set by the approval queue when the call retries a held operation with its approval_id.
    approval_id: str | None = None


_SCOPE: ContextVar[CallScope | None] = ContextVar("acl_call_scope", default=None)


@contextmanager
def call_scope(scope: CallScope) -> Generator[CallScope]:
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def current_scope() -> CallScope | None:
    return _SCOPE.get()
