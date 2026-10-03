"""The transcript's wording: pure formatting functions and the narrator's output."""

import io

from demo.orchestrator.models import (
    AuditEntry,
    ChatReply,
    Check,
    ProbeReply,
    ReloadEvent,
    SceneResult,
    ToolReply,
)
from demo.orchestrator.narration import (
    Narrator,
    describe_audit,
    describe_chat_reply,
    describe_probe,
    describe_reload,
    describe_tool_reply,
    format_check,
    format_summary,
    grafana_links,
)


def _tool(**fields: object) -> ToolReply:
    return ToolReply.model_validate(
        {"kind": "mcp", "server": "reports", "tool": "write_report", "status": 200, **fields}
    )


def test_an_allowed_call_shows_its_result():
    assert describe_tool_reply(_tool(reason="ok", result=[{"count": 40}])) == (
        "ALLOWED -> [{'count': 40}]"
    )


def test_a_held_call_shows_reason_code_and_approval_id():
    text = describe_tool_reply(_tool(reason="approval_required", approval_id="apr-1"))
    assert text == "REFUSED reason_code=approval_required, approval_id=apr-1"


def test_throttle_waits_are_shown():
    text = describe_tool_reply(_tool(reason="ok", result="x", throttle_waits=[5.5, 10.5]))
    assert text.endswith("(throttled first: waited 5.5, 10.5 s on Retry-After)")


def test_long_results_are_shortened_to_one_line():
    text = describe_tool_reply(_tool(reason="ok", result="line\n" * 100))
    assert "\n" not in text
    assert text.endswith("…")


def test_chat_replies():
    answered = ChatReply(kind="chat", status=200, reason="ok", answer="CALLER", elapsed_s=21.04)
    refused = ChatReply(kind="chat", status=403, reason="secret_detected", elapsed_s=0.1)
    assert describe_chat_reply(answered) == "ANSWERED in 21.0 s: CALLER"
    assert describe_chat_reply(refused) == "REFUSED HTTP 403 reason_code=secret_detected in 0.1 s"


def test_probe_outcomes():
    def probe(**fields: object) -> str:
        return describe_probe(ProbeReply.model_validate({"kind": "probe", **fields}))

    assert probe(host="gateway", port=8080, resolved=["172.30.0.3"], connected=True) == (
        "gateway:8080 CONNECTED (172.30.0.3)"
    )
    assert "name does not resolve" in probe(host="ollama", port=11434, connected=False, error="x")
    assert probe(
        host="172.25.0.2", port=11434, resolved=["172.25.0.2"], connected=False,
        error="Network is unreachable",
    ) == "172.25.0.2:11434 no connection: Network is unreachable"  # fmt: skip


def test_an_audit_line_shows_decision_risk_taint_revision_and_deciding_verdicts():
    entry = AuditEntry.model_validate(
        {
            "ts": "t",
            "session_id": "s-1",
            "principal": "anna@demo",
            "actor": "databot",
            "channel": "mcp",
            "action": "write",
            "resource": "fs:reports/a.md",
            "decision": "block",
            "reason_code": "action_removed_by_session_risk",
            "verdicts": [
                {"control": "authz", "stage": "pre", "decision": "block",
                 "reason_code": "action_removed_by_session_risk"},
                {"control": "pii", "stage": "pre", "decision": "allow", "reason_code": "no_pii"},
            ],
            "risk": 0.7,
            "taint": True,
            "policy_revision": "f83f261412c7",
        }
    )  # fmt: skip
    assert describe_audit(entry) == (
        "audit: write fs:reports/a.md -> block (action_removed_by_session_risk)  risk=0.70  "
        "taint=true  rev=f83f261412c7  verdicts: authz/pre=action_removed_by_session_risk"
    )


def test_reload_events():
    event = ReloadEvent(
        ts="2026-10-03T22:40:15Z",
        event="policy_reload",
        result="ok",
        revision="f4e8",
        previous_revision="f83f",
    )
    assert describe_reload(event) == "audit: policy_reload ok f83f -> f4e8 at 2026-10-03T22:40:15Z"


def test_checks_show_what_was_expected_only_on_failure():
    assert format_check(Check(name="n", expected=("40",), actual="40")) == "[PASS] n: 40"
    failed = Check(name="n", expected=("ok", "redact"), actual="judge_unavailable")
    assert format_check(failed) == "[FAIL] n: got judge_unavailable, expected ok | redact"


def test_the_summary_has_one_line_per_scene_with_timing_and_the_failures():
    passed = Check(name="a", expected=("1",), actual="1")
    failed = Check(name="b", expected=("1",), actual="2")
    lines = format_summary(
        [
            SceneResult(number=1, title="RLS", checks=(passed,), elapsed_s=10.4),
            SceneResult(number=5, title="LLM", checks=(passed, failed), elapsed_s=61.0),
            SceneResult(number=6, title="Bypass", elapsed_s=0.2, error="docker down"),
        ]
    )
    text = "\n".join(lines)
    assert "PASS  scene 1  RLS" in text
    assert "10.4 s  1/1 checks" in text
    assert "FAIL  scene 5  LLM" in text
    assert "[FAIL] b: got 2, expected 1" in text
    assert "error: docker down" in text
    assert lines[-1] == "1/3 scenes as expected, 71.6 s in total"


def test_grafana_links_encode_the_session_variable():
    links = grafana_links("http://127.0.0.1:3300/", {"scene 2": "s-RUsQ_z-1"})
    assert links[0] == (
        "Threats:        http://127.0.0.1:3300/d/acl-threats?orgId=1&from=now-30m&to=now"
    )
    assert "d/acl-session-trace?orgId=1&from=now-30m&to=now&var-session_id=s-RUsQ_z-1" in links[1]
    assert links[1].endswith("(scene 2)")


def test_the_narrator_writes_plain_text_unless_colour_is_on():
    plain, coloured = io.StringIO(), io.StringIO()
    Narrator(plain, colour=False).check(Check(name="n", expected=("1",), actual="1"))
    Narrator(coloured, colour=True).check(Check(name="n", expected=("1",), actual="1"))
    assert plain.getvalue() == "    [PASS] n: 1\n"
    assert "\033[32m" in coloured.getvalue()
    assert Narrator(io.StringIO())._colour is False  # a StringIO is not a terminal
