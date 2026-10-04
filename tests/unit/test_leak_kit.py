"""`leaked` finds a value wherever a payload could land, and not in measurements."""

import json

import pytest
from leak_kit import leaked

AUDIT_LINE = json.dumps(
    {
        "decision": "redact",
        "latency_ms": {"total": 9441.97, "controls": {"cost": 0.04904111847}},
        "risk": 0.4111,
        "status": 200,
        "verdicts": [{"control": "output_policy", "reason_code": "out_of_scope_data"}],
    }
)


@pytest.mark.parametrize(
    "text",
    [
        AUDIT_LINE,  # the flaky case: digits inside floats are measurements
        AUDIT_LINE + "\n" + AUDIT_LINE,  # JSON lines
        'data: {"choices": [{"delta": {"content": "card [REDACTED]"}}]}\n\ndata: [DONE]\n\n',
    ],
)
def test_measurements_and_redacted_text_are_not_leaks(text):
    assert not leaked("4111", text)


@pytest.mark.parametrize(
    "text",
    [
        json.dumps({"reason": "card 4111-1111 seen"}),  # in a string value
        json.dumps({"verdicts": [{"why": ["x", "4111"]}]}),  # nested in a list
        json.dumps({"4111": 1}),  # as a key
        json.dumps({"tokens": 4111}),  # a whole integer
        AUDIT_LINE + "\n" + json.dumps({"reason": "4111"}),  # in one of several lines
        'data: {"choices": [{"delta": {"content": "card 4111"}}]}\n\ndata: [DONE]\n\n',
        "not json at all: 4111",  # unparseable text is searched as written
    ],
)
def test_a_real_leak_is_found(text):
    assert leaked("4111", text)


def test_an_integer_containing_the_value_is_not_a_whole_match():
    assert not leaked("4111", json.dumps({"created": 1790041110}))
