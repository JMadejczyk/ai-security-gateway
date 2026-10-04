"""The corpus's no-leak checks are not vacuous, and not flaky.

For every case with ``absent`` (or ``absent_upstream``) values: an observation that meets the
whole ``expect`` block passes `check`; the same observation with the value planted in a
string of the agent's answer, of an audit line, or of an upstream request fails it; and the
value's digits buried in a latency float (what made a plain substring check flaky) do not.
"""

import json
from typing import Any

import pytest
from attack_kit import AttackCase, Observed, check, expand, load_corpus

LEAKY = [c for c in load_corpus() if c.expect.absent or c.expect.absent_upstream]


def _passing(case: AttackCase) -> Observed:
    """An observation that satisfies every expectation of ``case``."""
    expect = expand(case.expect.model_dump())
    verdicts = [
        {
            "control": v["control"],
            "decision": v["decision"],
            "reason_code": v["reason_code"] or "x",
        }
        for v in expect["verdicts"]
    ]
    audit: list[dict[str, Any]] = [
        {"decision": expect["decision"], "latency_ms": {"total": 1.5}, "verdicts": verdicts}
    ]
    return Observed(
        decision=expect["decision"],
        reason=expect["reason"][0] if expect["reason"] else None,
        agent_saw=json.dumps({"answer": "nothing to see"}),
        upstream_got=json.dumps({"forwarded": list(expect["present_upstream"])}),
        upstream_called=bool(expect["upstream_called"]),
        audit_raw="\n".join(json.dumps(entry) for entry in audit),
        audit=audit,
        taint=expect["taint"],
    )


def _planted(text: str, value: str) -> str:
    """``text`` (JSON or JSON lines) with one more line carrying ``value`` in a string."""
    return text + "\n" + json.dumps({"reason": f"... {value} ..."})


def _in_a_float(text: str, value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit()) or "4111"
    return text + "\n" + json.dumps({"latency_ms": {"total": float(f"0.0{digits}7")}})


@pytest.mark.parametrize("case", LEAKY, ids=[c.id for c in LEAKY])
def test_the_clean_observation_passes(case: AttackCase):
    check(case, _passing(case))


@pytest.mark.parametrize("case", LEAKY, ids=[c.id for c in LEAKY])
def test_a_planted_value_fails_the_check(case: AttackCase):
    clean = _passing(case)
    for value in expand(case.expect.absent):
        for field, message in (("agent_saw", "agent"), ("audit_raw", "audit log")):
            leaky = clean.model_copy(update={field: _planted(getattr(clean, field), value)})
            with pytest.raises(AssertionError, match=f"leaked to the {message}"):
                check(case, leaky)
    for value in expand(case.expect.absent_upstream):
        leaky = clean.model_copy(update={"upstream_got": _planted(clean.upstream_got, value)})
        with pytest.raises(AssertionError, match="reached the upstream"):
            check(case, leaky)


@pytest.mark.parametrize("case", LEAKY, ids=[c.id for c in LEAKY])
def test_digits_in_a_measurement_are_not_a_leak(case: AttackCase):
    clean = _passing(case)
    for value in expand([*case.expect.absent, *case.expect.absent_upstream]):
        noisy = clean.model_copy(
            update={
                "agent_saw": _in_a_float(clean.agent_saw, value),
                "audit_raw": _in_a_float(clean.audit_raw, value),
                "upstream_got": _in_a_float(clean.upstream_got, value),
            }
        )
        check(case, noisy)
