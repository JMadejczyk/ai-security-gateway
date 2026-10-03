"""Verdict merging: strictest enforced decision wins, obligations accumulate."""

import pytest
from pydantic import ValidationError

from gateway.core.envelope import Span, Verdict
from gateway.core.types import Decision
from gateway.core.verdicts import MergedVerdict, Obligation, merge_verdicts

A, R, B, Q = Decision.ALLOW, Decision.REDACT, Decision.BLOCK, Decision.REQUIRE_APPROVAL


def verdict(
    decision: Decision,
    *,
    enforced: bool = True,
    risk: float = 0.0,
    spans: tuple[Span, ...] = (),
    control: str = "pii",
) -> Verdict:
    return Verdict(
        decision=decision,
        control_id=control,
        reason_code="test",
        enforced=enforced,
        risk_delta=risk,
        redactions=spans,
    )


def span(path: str, start: int, end: int, label: str = "PESEL") -> Span:
    return Span(path=path, start=start, end=end, label=label)


@pytest.mark.parametrize(
    ("decisions", "expected"),
    [
        ([], A),
        ([A], A),
        ([A, R], R),
        ([R, Q], Q),
        ([Q, B], B),
        ([B, A, R, Q], B),
        ([A, A, A], A),
    ],
)
def test_strictest_enforced_decision_wins(decisions, expected):
    assert merge_verdicts([verdict(d) for d in decisions]).decision is expected


@pytest.mark.parametrize("decision", [R, Q, B])
def test_log_only_verdicts_are_recorded_not_applied(decision):
    merged = merge_verdicts([verdict(A), verdict(decision, enforced=False)])
    assert merged.decision is A
    assert merged.obligations == frozenset()
    assert len(merged.verdicts) == 2


def test_log_only_block_does_not_mask_enforced_redact():
    merged = merge_verdicts([verdict(B, enforced=False), verdict(R, spans=(span("/a", 0, 3),))])
    assert merged.decision is R
    assert merged.obligations == {Obligation.REDACTION}


def test_approval_and_redaction_obligations_both_kept():
    merged = merge_verdicts(
        [verdict(Q, control="intent_judge"), verdict(R, spans=(span("/m/0", 2, 5),))]
    )
    assert merged.decision is Q
    assert merged.obligations == {Obligation.APPROVAL, Obligation.REDACTION}
    assert merged.requires_approval
    assert merged.redactions == (span("/m/0", 2, 5),)


def test_redaction_spans_unioned_per_path():
    merged = merge_verdicts(
        [
            verdict(R, spans=(span("/a", 0, 5, "PESEL"), span("/b", 0, 2, "NIP"))),
            verdict(R, spans=(span("/a", 3, 9, "API_KEY"), span("/a", 20, 22, "IBAN"))),
            verdict(R, enforced=False, spans=(span("/c", 0, 1),)),
        ]
    )
    assert merged.redactions == (
        span("/a", 0, 9, "API_KEY+PESEL"),
        span("/a", 20, 22, "IBAN"),
        span("/b", 0, 2, "NIP"),
    )


def test_touching_spans_stay_separate():
    merged = merge_verdicts([verdict(R, spans=(span("/a", 0, 3), span("/a", 3, 5, "NIP")))])
    assert merged.redactions == (span("/a", 0, 3), span("/a", 3, 5, "NIP"))


def test_risk_deltas_summed_over_all_verdicts_unclamped():
    merged = merge_verdicts(
        [verdict(B, risk=0.6), verdict(R, risk=0.3), verdict(B, enforced=False, risk=0.4)]
    )
    assert merged.risk_delta == pytest.approx(1.3)


def test_empty_merge_is_plain_allow():
    assert merge_verdicts([]) == MergedVerdict(decision=A)


@pytest.mark.parametrize(
    ("path", "start", "end"),
    [("no-leading-slash", 0, 1), ("/a", 3, 3), ("/a", 4, 2), ("/a", -1, 2)],
)
def test_invalid_spans_rejected(path, start, end):
    with pytest.raises(ValidationError):
        Span(path=path, start=start, end=end, label="X")
