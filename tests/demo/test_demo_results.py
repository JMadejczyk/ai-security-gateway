"""The orchestrator's result models: agent replies, audit lines, checks and scene results."""

import json

import pytest
from pydantic import ValidationError

from demo.orchestrator.models import (
    AgentOutputError,
    AuditEntry,
    ChatReply,
    Check,
    DemoReport,
    ProbeReply,
    SceneResult,
    ToolCall,
    ToolReply,
    first_count,
    parse_agent_reply,
)

# One audit line as the gateway writes it (shape from a live run; payloads never appear).
AUDIT_LINE = json.dumps(
    {
        "ts": "2026-10-03T22:39:33.101Z",
        "session_id": "s-RUsQnzKTrESWETV2",
        "principal": "anna@demo",
        "actor": "databot",
        "mode": "interactive",
        "channel": "mcp",
        "action": "read",
        "resource": "web:demo-web",
        "decision": "block",
        "reason_code": "prompt_injection_detected",
        "status": 200,
        "verdicts": [
            {"control": "authz", "stage": "pre", "decision": "allow", "enforced": True,
             "reason_code": "allowed"},
            {"control": "egress", "stage": "pre", "decision": "allow", "enforced": True,
             "reason_code": "egress_demo_host"},
            {"control": "prompt_injection", "stage": "post", "decision": "block",
             "enforced": True, "reason_code": "prompt_injection_detected"},
        ],
        "effective_scope": ["read:db:sales.*", "read:web:*"],
        "risk": 0.6,
        "taint": True,
        "policy_revision": "f83f261412c7",
        "feed_version": "2026-10-03.2",
        "latency_ms": {"total": 41.2, "upstream": 12.0, "controls": {"authz": 0.2}},
        "payload_hmac": "d678",
        "approval_id": None,
    }
)  # fmt: skip


def test_the_agent_reply_is_the_last_json_line():
    stdout = (
        'warning: something\n{"kind": "mcp", "server": "web", "tool": "fetch", '
        '"status": 200, "reason": "prompt_injection_detected"}\n\n'
    )
    reply = parse_agent_reply(stdout)
    assert isinstance(reply, ToolReply)
    assert (reply.reason, reply.ok, reply.throttle_waits) == (
        "prompt_injection_detected",
        False,
        (),
    )


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        (
            '{"kind": "chat", "status": 403, "reason": "secret_detected", "elapsed_s": 0.1}',
            ChatReply,
        ),
        ('{"kind": "probe", "host": "ollama", "port": 11434, "connected": false}', ProbeReply),
    ],
)
def test_replies_are_told_apart_by_kind(line, kind):
    assert isinstance(parse_agent_reply(line), kind)


def test_no_output_or_an_unknown_kind_is_an_error():
    with pytest.raises(AgentOutputError):
        parse_agent_reply("\n  \n")
    with pytest.raises(ValidationError):
        parse_agent_reply('{"kind": "telnet"}')


@pytest.mark.parametrize(
    ("result", "count"),
    [([{"count": 40}], 40), ([], None), ("wrote 6 bytes", None), ([{"count": "7"}], None)],
)
def test_first_count(result, count):
    assert first_count(result) == count


def test_an_audit_line_parses_and_names_its_deciding_verdicts():
    entry = AuditEntry.model_validate_json(AUDIT_LINE)
    assert (entry.taint, entry.risk, entry.latency_ms.upstream) == (True, 0.6, 12.0)
    egress = entry.verdict("egress")
    assert egress is not None
    assert egress.reason_code == "egress_demo_host"
    assert entry.verdict("pii") is None
    assert [v.control for v in entry.deciding()] == ["prompt_injection"]


def test_a_check_passes_when_actual_is_one_of_expected():
    assert Check(name="n", expected=("ok", "redact"), actual="redact").passed
    assert not Check(name="n", expected=("ok",), actual="judge_unavailable").passed


def test_a_scene_passes_only_with_checks_all_passing_and_no_error():
    good = Check(name="a", expected=("1",), actual="1")
    bad = Check(name="b", expected=("1",), actual="2")
    assert SceneResult(number=1, title="t", checks=(good,)).passed
    assert not SceneResult(number=1, title="t", checks=(good, bad)).passed
    assert SceneResult(number=1, title="t", checks=(good, bad)).failed_checks == (bad,)
    assert not SceneResult(number=1, title="t").passed  # nothing checked proves nothing
    assert not SceneResult(number=1, title="t", checks=(good,), error="boom").passed


def test_a_report_passes_only_when_every_scene_does():
    good = SceneResult(number=1, title="t", checks=(Check(name="a", expected=("1",), actual="1"),))
    assert DemoReport(run_id="r", scenes=(good,)).passed
    assert not DemoReport(run_id="r", scenes=()).passed
    assert not DemoReport(run_id="r", scenes=(good, SceneResult(number=2, title="u"))).passed


def test_a_tool_call_retry_keeps_its_arguments():
    call = ToolCall(server="reports", tool="write_report", arguments={"name": "a.md"})
    retry = call.retried_with("apr-1")
    assert (retry.approval_id, retry.arguments, call.approval_id) == (
        "apr-1",
        {"name": "a.md"},
        None,
    )
    assert call.shown() == "reports.write_report(name='a.md')"
