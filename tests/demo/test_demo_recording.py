"""`make record`: the beats as data, and the recorder against stand-ins for the stack."""

import io
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from demo import record
from demo.orchestrator.models import ApprovalInfo, AuditEntry, IssuedToken, SceneResult
from demo.orchestrator.narration import Narrator
from demo.orchestrator.recording import (
    BEATS,
    MAX_CAPTION_WORDS,
    Beat,
    GrafanaView,
    Recorder,
    RecordOptions,
    audit_ts,
    beat,
    expected_codes_seen,
    recording,
    scene_for,
)
from demo.orchestrator.scenes import Demo
from demo.orchestrator.stack import AuditLog, Operator, PolicyFile

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_DOC = REPO_ROOT / "docs" / "video.md"


# ------------------------------------------------------------------------------ beats


def test_six_beats_numbered_in_storyboard_order():
    assert [b.number for b in BEATS] == [1, 2, 3, 4, 5, 6]
    starts = [b.timecode.split("-")[0] for b in BEATS]
    assert starts == ["0:15", "0:40", "1:20", "1:50", "2:15", "2:35"]


def test_captions_have_at_most_six_words():
    assert all(len(b.caption.split()) <= MAX_CAPTION_WORDS for b in BEATS)
    with pytest.raises(ValidationError):
        Beat(number=9, timecode="x", title="t", caption="one two three four five six seven",
             scripted=False, stand_in=None)  # fmt: skip


def test_opencode_beats_have_prompts_and_reason_codes_scripted_ones_a_scene():
    for b in BEATS:
        if b.scripted:
            assert scene_for(b.stand_in) is not None, b.number
            assert not b.prompts
        elif b.prompts:
            assert b.opencode
            assert b.expect
            assert {p.who for p in b.prompts} <= set(b.principals)


def test_the_night_job_and_the_policy_edit_are_scripted():
    assert [b.number for b in BEATS if b.scripted] == [3, 5]
    assert beat(3).principals == ("svc:nightly_etl",)


def test_unknown_beat():
    with pytest.raises(ValueError, match="no beat 7"):
        beat(7)


def test_grafana_view_urls_follow_live_data_in_kiosk_mode():
    view = GrafanaView(uid="acl-session-trace", label="x", panel_hint="y")
    url = view.url("http://127.0.0.1:3300/", "s-ab_C-1")
    assert url.startswith("http://127.0.0.1:3300/d/acl-session-trace?orgId=1&from=now-5m")
    assert "refresh=5s" in url
    assert "theme=dark" in url
    assert "var-session_id=s-ab_C-1" in url
    assert url.endswith("&kiosk")
    assert "var-session_id" not in view.url("http://g")


def test_the_recording_view_hides_grafana_chrome():
    url = recording("x").url("http://g")
    assert url.startswith("http://g/d/acl-recording?")
    assert url.endswith("&kiosk&_dash.hideTimePicker&_dash.hideVariables&_dash.hideLinks")


def test_every_beat_frames_the_recording_dashboard():
    assert all(b.views[0].uid == "acl-recording" for b in BEATS)


def test_audit_ts_matches_the_audit_log_format():
    assert audit_ts(datetime(2026, 10, 4, 3, 4, 5, 678_901, tzinfo=UTC)) == (
        "2026-10-04T03:04:05.678Z"
    )


def test_expected_codes_seen():
    checks = expected_codes_seen(
        ["prompt_injection_detected", "action_removed_by_session_risk"],
        ["allowed", "prompt_injection_detected"],
    )
    assert [(c.name, c.actual, c.passed) for c in checks] == [
        ("take produced prompt_injection_detected", "seen", True),
        ("take produced action_removed_by_session_risk", "missing", False),
    ]


def test_the_video_doc_carries_every_caption_and_prompt():
    text = VIDEO_DOC.read_text()
    for b in BEATS:
        assert b.caption in text, b.caption
        assert f"make record SCENE={b.number}" in text
        for prompt in b.prompts:
            assert prompt.text in text, prompt.text


# ------------------------------------------------------------------------------ recorder


class FakeOperator(Operator):
    def __init__(self, pending: int = 0) -> None:
        self.pending = [
            ApprovalInfo(id=f"apr-{i}", state="pending", agent="nightly_etl",
                         principal="svc:nightly_etl", session_id="s-old")
            for i in range(pending)
        ]  # fmt: skip
        self.denied: list[str] = []

    def wait_for_revision(self, done, *, timeout_s=20.0, nudge_after_s=8.0, admin="root@demo"):
        return "87e5c84554e3"

    def pending_approvals(self, approver: str) -> tuple[ApprovalInfo, ...]:
        return tuple(a for a in self.pending if a.id not in self.denied)

    def deny(self, approver: str, approval_id: str) -> ApprovalInfo:
        self.denied.append(approval_id)
        return self.pending[0]

    def token(self, sub: str, *, kind: str = "agent") -> IssuedToken:
        raise AssertionError


class FakeAudit(AuditLog):
    def __init__(self, entries: dict[str, list[dict[str, object]]]) -> None:
        self.entries_by_principal = entries
        self.asked: list[tuple[str, str]] = []

    def since(self, principal: str, ts: str) -> tuple[AuditEntry, ...]:
        self.asked.append((principal, ts))
        base = {"ts": ts, "principal": principal, "actor": "databot", "channel": "mcp",
                "action": "read", "resource": "r", "decision": "allow"}  # fmt: skip
        return tuple(
            AuditEntry.model_validate({**base, **e})
            for e in self.entries_by_principal.get(principal, [])
        )


class FakePolicy(PolicyFile):
    def __init__(self, leftover: bool = False) -> None:
        self.leftover = leftover

    def recover(self) -> bool:
        return self.leftover

    def revision(self) -> str:
        return "87e5c84554e3"


def _recorder(audit=None, operator=None, wait=True, answers=None, policy=None):
    out = io.StringIO()
    demo = Demo(
        compose=None,  # type: ignore[arg-type]  -- not used by these beats
        agent=None,  # type: ignore[arg-type]
        operator=operator or FakeOperator(),
        audit=audit or FakeAudit({}),
        db=None,  # type: ignore[arg-type]
        policy=policy or FakePolicy(),
        narrator=Narrator(out, colour=False),
        run_id="t",
    )
    asked: list[str] = []
    opened: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return ""

    recorder = Recorder(
        demo,
        RecordOptions(wait=wait, pace_s=0, open_browser=True),
        ask=ask,
        sleep=lambda _s: None,
        opener=opened.append,
        now=lambda: datetime(2026, 10, 4, 3, 0, 10, tzinfo=UTC),
    )
    return recorder, out, asked, opened


def test_reset_denies_stale_approvals_and_tells_opencode_to_relaunch():
    operator = FakeOperator(pending=2)
    recorder, out, _, _ = _recorder(operator=operator)
    checks = recorder.reset(beat(2))
    assert operator.denied == ["apr-0", "apr-1"]
    assert [(c.name, c.passed) for c in checks] == [
        ("policy file in force", True),
        ("approval queue empty", True),
    ]
    text = out.getvalue()
    assert "denied 2 stale pending approval(s)" in text
    assert "make opencode AS=anna" in text


def test_reset_reports_a_restored_policy():
    recorder, out, _, _ = _recorder(policy=FakePolicy(leftover=True))
    recorder.reset(beat(5))
    assert "restored config/policy.yaml" in out.getvalue()


def test_an_opencode_take_is_checked_from_the_audit_log_after_enter():
    audit = FakeAudit(
        {
            "anna@demo": [
                {"session_id": "s-take", "reason_code": "prompt_injection_detected",
                 "decision": "block", "resource": "web:demo-web"},
                {"session_id": "s-take", "reason_code": "action_removed_by_session_risk",
                 "decision": "block", "action": "write"},
            ]
        }
    )  # fmt: skip
    recorder, out, asked, opened = _recorder(audit=audit)
    result = recorder.run(beat(2))
    assert result.passed, result.failed_checks
    assert asked == ["\n[Enter] when the take is done "]
    assert audit.asked == [("anna@demo", "2026-10-04T03:00:08.000Z")]  # 2 s clock-skew margin
    assert opened
    assert "/d/acl-recording?" in opened[0]
    text = out.getvalue()
    assert "caption: One bad page. Session loses writes." in text
    assert "type in opencode (anna@demo)  Read http://demo-web/q3-market-notes.html" in text
    assert re.search(r"Recording \(anna@demo take\).*acl-recording.*var-session_id=s-take", text)
    assert re.search(r"Session trace \(anna@demo take\).*var-session_id=s-take", text)


def test_a_take_that_missed_its_outcome_fails():
    audit = FakeAudit({"anna@demo": [{"session_id": "s-1", "reason_code": "allowed"}]})
    recorder, _, _, _ = _recorder(audit=audit)
    result = recorder.run(beat(2))
    assert not result.passed
    assert [c.name for c in result.failed_checks] == [
        "take produced prompt_injection_detected",
        "take produced action_removed_by_session_risk",
    ]


def test_the_end_shot_has_nothing_to_play():
    recorder, out, asked, _ = _recorder()
    result = recorder.run(beat(6))
    assert result.passed
    assert len(asked) == 1
    assert "acl-posture" in out.getvalue()


class FakeScene:
    """Records the hooks the recorder installs, and calls them like a scripted scene."""

    number = 3

    def __init__(self) -> None:
        self.ran_with: tuple[object, object] | None = None

    def run(self, demo: Demo) -> SceneResult:
        self.ran_with = (demo.pace, demo.hold)
        demo.pace("a step")
        demo.hold("Approve? [Enter]")
        return SceneResult(number=3, title="night job", sessions={"scene 3 nightly_etl": "s-etl"},
                           checks=())  # fmt: skip


@pytest.mark.parametrize(("wait", "enters"), [(True, 2), (False, 0)])
def test_a_scripted_beat_is_paced_and_holds_only_when_waiting(monkeypatch, wait, enters):
    scene = FakeScene()
    monkeypatch.setattr("demo.orchestrator.recording.scene_for", lambda _n: scene)
    recorder, out, asked, _ = _recorder(wait=wait)
    recorder.run(beat(3))
    assert scene.ran_with is not None
    assert len(asked) == enters  # roll + the approval hold
    text = out.getvalue()
    assert "... a step" in text
    assert "var-session_id=s-etl" in text


def test_no_wait_plays_an_opencode_beat_with_the_demo_agent(monkeypatch):
    played: list[int] = []

    class StandIn(FakeScene):
        def run(self, demo: Demo) -> SceneResult:
            played.append(1)
            return SceneResult(number=1, title="rls", checks=())

    monkeypatch.setattr("demo.orchestrator.recording.scene_for", lambda _n: StandIn())
    recorder, out, asked, _ = _recorder(wait=False)
    recorder.run(beat(1))
    assert played == [1]
    assert asked == []
    assert "rehearsal: the demo agent plays this beat" in out.getvalue()


def test_record_main_exits_2_when_the_stack_is_not_ready(monkeypatch, capsys):
    monkeypatch.setattr(record, "Compose", object)
    monkeypatch.setattr(record, "preflight", lambda _c: "not running: demo-web")
    assert record.main(["--scene", "2", "--no-color"]) == record.EXIT_NOT_READY
    assert "cannot record: not running: demo-web" in capsys.readouterr().out
