"""The verdict of a content-detection control (``pii``, ``secrets``): spans in, decision out.

Both controls find labelled spans and react the same way per resolved mode:

- ``redact``: ``decision=redact`` carrying the spans;
- ``block``: ``decision=block`` (spans kept, so an audit consumer can count what was found);
- ``log_only``: the would-be redaction, ``enforced=False``; the pipeline records it only.

The reason names labels only (``"PL_PESEL, EMAIL_ADDRESS"``), never matched text.
"""

from collections.abc import Sequence

from gateway.core.catalog import control_spec
from gateway.core.envelope import Span, Verdict
from gateway.core.interfaces import ControlConfig
from gateway.core.types import ControlMode, Decision
from gateway.core.verdicts import union_spans

_DECISIONS: dict[ControlMode, Decision] = {
    ControlMode.BLOCK: Decision.BLOCK,
    ControlMode.REDACT: Decision.REDACT,
    ControlMode.LOG_ONLY: Decision.REDACT,
}


def detection_verdict(
    control_id: str,
    cfg: ControlConfig,
    spans: Sequence[Span],
    *,
    detected: str,
    clean: str,
) -> Verdict:
    """``allow`` with ``clean`` when nothing was found, else the mode's decision with
    ``detected`` and the configured (or catalog) risk delta.

    Raises ``ValueError`` for a mode the control's catalog entry does not support; the
    pipeline turns that into a fail-closed ``control_error``.
    """
    spec = control_spec(control_id)
    mode = cfg.mode or spec.most_enforcing
    if not spec.supports(mode) or mode not in _DECISIONS:
        msg = f"control {control_id!r} does not support mode {mode!s}"
        raise ValueError(msg)
    if not spans:
        return Verdict(decision=Decision.ALLOW, control_id=control_id, reason_code=clean)
    labels = sorted({label for span in spans for label in span.label.split("+")})
    return Verdict(
        decision=_DECISIONS[mode],
        control_id=control_id,
        reason_code=detected,
        reason=", ".join(labels),
        enforced=mode is not ControlMode.LOG_ONLY,
        risk_delta=cfg.risk_delta if cfg.risk_delta is not None else spec.default_risk_delta,
        redactions=union_spans(spans),
    )
