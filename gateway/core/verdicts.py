"""Merging the verdicts of every control into one decision plus obligations.

A plain function, not a class: merging is stateless and takes no configuration.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from enum import StrEnum

from pydantic import Field

from gateway.core.envelope import FrozenModel, Span, Verdict
from gateway.core.types import DECISION_STRICTNESS_ORDER, Decision


class Obligation(StrEnum):
    """Something that must be satisfied before (or while) an allowed call executes."""

    APPROVAL = "approval"
    REDACTION = "redaction"


class MergedVerdict(FrozenModel):
    """Outcome of one pipeline stage for one interaction."""

    decision: Decision
    obligations: frozenset[Obligation] = frozenset()
    redactions: tuple[Span, ...] = ()  # unioned per path, sorted by (path, start)
    risk_delta: float = Field(default=0.0, ge=0.0)  # raw sum; the caller clamps session risk
    verdicts: tuple[Verdict, ...] = ()  # every verdict, enforced or not, for the audit entry

    @property
    def requires_approval(self) -> bool:
        return Obligation.APPROVAL in self.obligations


def union_spans(spans: Iterable[Span]) -> tuple[Span, ...]:
    """Merge overlapping spans on the same path; the merged label lists every source label."""
    by_path: defaultdict[tuple[str, bool, str], list[Span]] = defaultdict(list)
    for span in spans:  # one string = one (path, embedded) pair; "" embedded is the root
        by_path[span.path, span.embedded is not None, span.embedded or ""].append(span)
    merged: list[Span] = []
    for location in sorted(by_path):
        current: Span | None = None
        for span in sorted(by_path[location], key=lambda s: (s.start, s.end)):
            if current is not None and span.start < current.end:
                labels = sorted({*current.label.split("+"), *span.label.split("+")})
                current = current.model_copy(
                    update={"end": max(current.end, span.end), "label": "+".join(labels)}
                )
                continue
            if current is not None:
                merged.append(current)
            current = span
        if current is not None:
            merged.append(current)
    return tuple(merged)


def merge_verdicts(verdicts: Sequence[Verdict]) -> MergedVerdict:
    """Strictest enforced decision wins; obligations and redactions accumulate.

    - ``enforced=False`` (log_only) verdicts are kept for audit but never applied.
    - ``block`` beats ``require_approval`` beats ``redact`` beats ``allow``.
    - An approval and a redaction together yield both obligations.
    - Risk deltas are summed over every verdict, enforced or not: the detection happened
      even when its decision is not applied, and risk rules are profile-independent.
    - No verdicts means allow.
    """
    enforced = [v for v in verdicts if v.enforced]
    present = {v.decision for v in enforced}
    decision = next((d for d in DECISION_STRICTNESS_ORDER if d in present), Decision.ALLOW)

    obligations: set[Obligation] = set()
    if Decision.REQUIRE_APPROVAL in present:
        obligations.add(Obligation.APPROVAL)
    redactions = union_spans(
        span for v in enforced if v.decision is Decision.REDACT for span in v.redactions
    )
    if redactions:
        obligations.add(Obligation.REDACTION)

    return MergedVerdict(
        decision=decision,
        obligations=frozenset(obligations),
        redactions=redactions,
        risk_delta=sum(v.risk_delta for v in verdicts),
        verdicts=tuple(verdicts),
    )
