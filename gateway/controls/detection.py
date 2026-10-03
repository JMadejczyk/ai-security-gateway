"""The verdict of a content-detection control (``pii``, ``secrets``): hits in, decision out.

Both controls find labelled spans and react the same way per resolved mode:

- ``redact``: ``decision=redact`` carrying the spans;
- ``block``: ``decision=block`` (spans kept, so an audit consumer can count what was found);
- ``log_only``: the would-be decision, ``enforced=False``; the pipeline records it only.

The reason names labels only (``"PL_PESEL, EMAIL_ADDRESS"``), never matched text.
"""

from collections.abc import Sequence

from gateway.controls.scanning import Hit
from gateway.core.catalog import control_spec
from gateway.core.envelope import Verdict
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
    hits: Sequence[Hit],
    *,
    detected: str,
    clean: str,
) -> Verdict:
    """``allow`` with ``clean`` when nothing was found, else the mode's decision with
    ``detected`` and the configured (or catalog) risk delta.

    A hit in a segment no mask can be written into (decoded base64, a data URL) turns
    ``redact`` into ``block``: what cannot be redacted must not be released.

    Raises ``ValueError`` for a mode the control's catalog entry does not support; the
    pipeline turns that into a fail-closed ``control_error``.
    """
    spec = control_spec(control_id)
    mode = cfg.mode or spec.most_enforcing
    if not spec.supports(mode) or mode not in _DECISIONS:
        msg = f"control {control_id!r} does not support mode {mode!s}"
        raise ValueError(msg)
    if not hits:
        return Verdict(decision=Decision.ALLOW, control_id=control_id, reason_code=clean)
    spans = [h.segment.span(h.start, h.end, h.label) for h in hits if h.segment.redactable]
    decision = _DECISIONS[mode]
    if len(spans) < len(hits):
        decision = Decision.BLOCK
    return Verdict(
        decision=decision,
        control_id=control_id,
        reason_code=detected,
        reason=", ".join(sorted({hit.label for hit in hits})),
        enforced=mode is not ControlMode.LOG_ONLY,
        risk_delta=cfg.risk_delta if cfg.risk_delta is not None else spec.default_risk_delta,
        redactions=union_spans(spans),
    )
