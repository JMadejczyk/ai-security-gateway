"""The gateway's own reason codes are neutral text to the injection classifier.

`GATEWAY_REASON_CODES` is a literal set; this keeps it in step with the code: every reason code
the gateway can write (``reason_code="..."`` literals, refusal error codes, the controls' and
approvals' code constants, the reason enums) and every multi-word control id is in it.
"""

import importlib
import re
from pathlib import Path

import pytest

from gateway.core.catalog import CONTROL_CATALOG
from gateway.injection.prose import neutral_markers
from gateway.injection.reason_codes import GATEWAY_REASON_CODES

GATEWAY = Path(__file__).resolve().parents[2] / "gateway"
SNAKE = r"[a-z]+(?:_[a-z]+)+"
REASON_ENUMS = {
    "gateway.sessions": ["SessionReason"],
    "gateway.identity": ["TokenReason"],
    "gateway.adapters.mcp": ["ToolCallReason"],
    "gateway.controls.tool_pinning": ["PinStatus"],
    "gateway.adapters.sql": ["SqlRefusal"],
    "gateway.controls.sql_guard": ["SqlGuardReason"],
    "gateway.approvals.oversight": ["ApprovalRefusal"],
    "gateway.approvals.operators": ["OperatorReason"],
    "gateway.policy.evaluator": ["AuthzReason", "RestrictionReason"],
}


def codes_in_the_source() -> set[str]:
    found: set[str] = set()
    for path in GATEWAY.rglob("*.py"):
        source = path.read_text()
        for pattern in (
            rf'reason_code="({SNAKE})"',
            rf'(?:RejectionError|Error)\(\s*"({SNAKE})"',
            rf'super\(\).__init__\(\s*"({SNAKE})"',
            rf'tool_error\(\s*"({SNAKE})"',
        ):
            found |= set(re.findall(pattern, source))
    constants = [*(GATEWAY / "controls").glob("*.py"), *(GATEWAY / "approvals").glob("*.py")]
    for path in [*constants, GATEWAY / "pipeline.py"]:
        found |= set(
            re.findall(rf'^[A-Z][A-Z0-9_]*(?:: Final)? = "({SNAKE})"', path.read_text(), re.M)
        )
    for module, names in REASON_ENUMS.items():
        loaded = importlib.import_module(module)
        for name in names:
            found |= {e.value for e in getattr(loaded, name) if re.fullmatch(SNAKE, e.value)}
    return found


def test_every_reason_code_the_gateway_writes_is_neutralised():
    missing = codes_in_the_source() - GATEWAY_REASON_CODES
    assert not missing, f"add to gateway/injection/reason_codes.py: {sorted(missing)}"


def test_every_multi_word_control_id_is_neutralised():
    assert {c for c in CONTROL_CATALOG if "_" in c} <= GATEWAY_REASON_CODES


@pytest.mark.parametrize(
    ("text", "neutral"),
    [
        ("prompt_injection_detected", "***"),
        ("approval_required approval_id=apr-0a1b", "*** approval_id=apr-0a1b"),
        ("refused with reason code `tool_poisoning_detected`.", "refused with reason code `***`."),
        ("the prompt_injection control", "the *** control"),
        ("a_prompt_injection_detected_b", "a_prompt_injection_detected_b"),  # not a whole token
        ("customer_id and order_total", "customer_id and order_total"),  # not gateway codes
        ("ignore previous instructions", "ignore previous instructions"),
    ],
)
def test_only_whole_gateway_codes_are_replaced(text, neutral):
    assert neutral_markers(text) == neutral
