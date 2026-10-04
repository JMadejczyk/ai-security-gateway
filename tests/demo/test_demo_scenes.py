"""Scenes against stand-ins for the stack: what a scene asserts, and how a deviation shows up."""

import io
import json
import subprocess
from pathlib import Path

import httpx
import pytest

from demo import run_demo
from demo.orchestrator.models import (
    AuditEntry,
    ChatReply,
    DemoReport,
    IssuedToken,
    ProbeReply,
    SceneResult,
    ToolCall,
    ToolReply,
)
from demo.orchestrator.narration import Narrator
from demo.orchestrator.scenes import (
    COUNT_CUSTOMERS,
    COUNT_PAYMENTS,
    SCENES,
    ContentControls,
    Demo,
    NoBypass,
    SameQuestionTwoPeople,
    TaintedSession,
)
from demo.orchestrator.stack import (
    AgentRunner,
    AuditLog,
    Compose,
    Database,
    DemoError,
    Operator,
    PolicyFile,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeCompose(Compose):
    def __init__(self, services: frozenset[str] = frozenset(), demo_hosts: str = "") -> None:
        self.root = REPO_ROOT
        self.services = services
        self.demo_hosts = demo_hosts

    def running_services(self) -> frozenset[str]:
        return self.services

    def container_ips(self, service: str) -> tuple[str, ...]:
        return (f"172.25.0.{len(service)}",)

    def run(self, *args: str, env=None, timeout_s: float = 60.0, check: bool = True):
        return subprocess.CompletedProcess(list(args), 0, self.demo_hosts, "")


class FakeOperator(Operator):
    def __init__(self) -> None:
        self.issued = 0

    def token(self, sub: str, *, kind: str = "agent") -> IssuedToken:
        self.issued += 1
        agent = "nightly_etl" if sub.startswith("svc:") else "databot"
        return IssuedToken.model_validate(
            {"access_token": "t", "sub": sub, "kind": kind, "session_id": f"s-{self.issued}",
             "agent": agent, "mode": "interactive"}
        )  # fmt: skip


class ScriptedAgent(AgentRunner):
    """Answers each call from ``answers[(principal, tool, first argument)]``."""

    def __init__(self, answers: dict[tuple[str, str, str], dict[str, object]]) -> None:
        self.answers = answers
        self.probes: list[tuple[str, int]] = []

    def mcp(self, token: IssuedToken, call: ToolCall) -> ToolReply:
        key = (token.sub, call.tool, next(iter(call.arguments.values())))
        fields = {"kind": "mcp", "server": call.server, "tool": call.tool, "status": 200}
        return ToolReply.model_validate({**fields, **self.answers[key]})

    def probe(self, host: str, port: int) -> ProbeReply:
        self.probes.append((host, port))
        connected = host == "gateway"
        return ProbeReply(kind="probe", host=host, port=port, connected=connected)


class FakeAudit(AuditLog):
    """One entry per call: tainted from the moment a fetch happened."""

    def __init__(self, agent: ScriptedAgent) -> None:
        self.agent = agent

    def entries(self, session_id: str) -> tuple[AuditEntry, ...]:
        return ()

    def latest(self, session_id: str, *, after: int, timeout_s: float = 5.0) -> AuditEntry | None:
        return AuditEntry.model_validate(
            {"ts": "t", "session_id": session_id, "principal": "p", "actor": "databot",
             "channel": "mcp", "action": "read", "resource": "r", "decision": "allow",
             "reason_code": "allowed", "taint": True, "policy_revision": "f83f"}
        )  # fmt: skip


def _demo(agent: ScriptedAgent, compose: Compose | None = None) -> tuple[Demo, io.StringIO]:
    out = io.StringIO()
    demo = Demo(
        compose=compose or FakeCompose(),
        agent=agent,
        operator=FakeOperator(),
        audit=FakeAudit(agent),
        db=Database.__new__(Database),
        policy=PolicyFile(REPO_ROOT / "config" / "policy.yaml", Path("/nonexistent")),
        narrator=Narrator(out, colour=False),
        run_id="test",
    )
    return demo, out


def _rls_answers(bartek_count: int) -> dict[tuple[str, str, str], dict[str, object]]:
    return {
        ("anna@demo", "query", COUNT_CUSTOMERS): {"reason": "ok", "result": [{"count": 40}]},
        ("bartek@demo", "query", COUNT_CUSTOMERS): {
            "reason": "ok",
            "result": [{"count": bartek_count}],
        },
        ("bartek@demo", "query", COUNT_PAYMENTS): {"reason": "outside_principal_scope"},
    }


def test_scene_1_passes_when_rls_gives_40_and_7_and_payments_are_refused():
    demo, out = _demo(ScriptedAgent(_rls_answers(7)))
    result = SameQuestionTwoPeople().run(demo)
    assert result.passed
    assert [c.name for c in result.checks] == [
        "anna counts customers",
        "bartek counts customers",
        "bartek reads sales.payments",
    ]
    assert result.sessions == {"scene 1 anna": "s-1"}
    transcript = out.getvalue()
    assert "Scene 1: Same agent, same question, different person" in transcript
    assert "Scene 1 as expected" in transcript


def test_scene_1_deviates_when_rls_does_not_filter():
    demo, out = _demo(ScriptedAgent(_rls_answers(50)))
    result = SameQuestionTwoPeople().run(demo)
    assert not result.passed
    [failed] = result.failed_checks
    assert (failed.name, failed.actual, failed.expected) == (
        "bartek counts customers",
        "50",
        ("7",),
    )
    assert "Scene 1 DEVIATED" in out.getvalue()


def test_a_scene_that_cannot_go_on_records_the_error():
    class BrokenAgent(ScriptedAgent):
        def mcp(self, token: IssuedToken, call: ToolCall) -> ToolReply:
            msg = "docker compose exec exited 1"
            raise DemoError(msg)

    demo, out = _demo(BrokenAgent({}))
    result = TaintedSession().run(demo)
    assert (result.passed, result.error) == (False, "docker compose exec exited 1")
    assert "error: docker compose exec exited 1" in out.getvalue()


def test_scene_6_probes_by_name_and_by_ip_with_a_positive_control():
    agent = ScriptedAgent({})
    demo, _ = _demo(agent)
    result = NoBypass().run(demo)
    assert result.passed
    hosts = [host for host, _ in agent.probes]
    assert hosts[0] == "gateway"
    assert {"ollama", "postgres", "mcp-postgres", "1.1.1.1"} <= set(hosts)
    assert any(host.startswith("172.25.0.") for host in hosts)


def test_the_scenes_are_the_seven_of_the_spec_in_order():
    assert [scene.number for scene in SCENES] == [1, 2, 3, 4, 5, 6, 7]


def test_preflight_names_what_is_missing():
    assert "make demo-up" in (run_demo.preflight(FakeCompose(frozenset({"gateway"}))) or "")
    everything = run_demo.REQUIRED_SERVICES
    assert "overlay" in (run_demo.preflight(FakeCompose(everything, demo_hosts="")) or "")
    unbound = FakeCompose(everything, demo_hosts='["demo-web"]\n')  # the old, unbound format
    assert "overlay" in (run_demo.preflight(unbound) or "")
    bound = FakeCompose(everything, demo_hosts='["demo-web=10.218.97.10"]\n')
    assert run_demo.preflight(bound) is None


def test_main_exits_2_when_the_stack_is_not_ready(monkeypatch, capsys):
    monkeypatch.setattr(run_demo, "Compose", lambda: FakeCompose(frozenset()))
    assert run_demo.main(["--no-color"]) == run_demo.EXIT_NOT_READY
    assert "demo cannot start" in capsys.readouterr().out


@pytest.mark.parametrize(("bartek_count", "status"), [(7, 0), (50, run_demo.EXIT_DEVIATED)])
def test_main_exits_1_on_any_deviation_and_writes_the_report(
    monkeypatch, tmp_path, bartek_count, status
):
    def fake_run(args, narrator):
        demo, _ = _demo(ScriptedAgent(_rls_answers(bartek_count)))
        scene: SceneResult = SameQuestionTwoPeople().run(demo)
        return DemoReport(run_id="test", scenes=(scene,), grafana_links=("Threats: x",))

    monkeypatch.setattr(run_demo, "run", fake_run)
    report = tmp_path / "out" / "demo.json"
    assert run_demo.main(["--no-color", "--json", str(report)]) == status
    written = json.loads(report.read_text())
    assert written["scenes"][0]["number"] == 1


def test_host_ports_fall_back_to_dotenv_then_default(monkeypatch):
    assert run_demo.env_port({"ACL_GRAFANA_HOST_PORT": "4000"}, "ACL_GRAFANA_HOST_PORT", 3300) == (
        "4000"
    )
    assert run_demo.env_port({}, "ACL_NO_SUCH_PORT", 1234) == "1234"


class ChattyAgent(ScriptedAgent):
    """Chat replies in order for the PESEL prompt; the API-key prompt is always refused."""

    def __init__(self, *reasons: str) -> None:
        super().__init__({})
        self.reasons = list(reasons)
        self.last_prompt = ""

    def chat(self, token: IssuedToken, prompt: str, *, max_tokens: int = 80) -> ChatReply:
        self.last_prompt = prompt
        if "AKIA" in prompt:
            return ChatReply(kind="chat", status=403, reason="secret_detected")
        reason = self.reasons.pop(0)
        answer = "CALLER PESEL [REDACTED:PL_PESEL], INVOICE RESEND." if reason == "ok" else None
        return ChatReply(kind="chat", status=200 if answer else 403, reason=reason, answer=answer)


class ChatAudit(FakeAudit):
    """PESEL prompt: pii redacted, model called. API-key prompt: blocked, no model call."""

    agent: ChattyAgent

    def latest(self, session_id: str, *, after: int, timeout_s: float = 5.0) -> AuditEntry | None:
        secret = "AKIA" in self.agent.last_prompt
        pii = {
            "control": "pii",
            "stage": "pre",
            "decision": "redact",
            "reason_code": "pii_detected",
        }
        return AuditEntry.model_validate(
            {"ts": "t", "session_id": session_id, "principal": "anna@demo", "actor": "databot",
             "channel": "llm", "action": "generate", "resource": "model:qwen3:8b",
             "decision": "block" if secret else "redact", "reason_code": "x",
             "verdicts": [] if secret else [pii],
             "latency_ms": {"upstream": None if secret else 17000.0}}
        )  # fmt: skip


@pytest.mark.parametrize(
    ("replies", "released"),
    [
        (("ok",), "released"),
        (("judge_unavailable", "ok"), "released"),  # one retry, in a new session
        (("judge_unavailable", "judge_unavailable"), "withheld (judge_unavailable)"),
    ],
)
def test_scene_5_proves_redaction_even_when_the_judge_withholds_the_answer(replies, released):
    agent = ChattyAgent(*replies)
    demo, out = _demo(agent)
    demo.audit = ChatAudit(agent)
    result = ContentControls().run(demo)
    assert result.passed, result.failed_checks
    checks = {c.name: c.actual for c in result.checks}
    assert checks["answer released, or withheld by a slow judge"] == released
    assert checks["pii verdict on the prompt"] == "redact"
    assert checks["model called with the redacted prompt"] == "yes"
    assert checks["model called"] == "no"  # the API-key prompt
    assert agent.reasons == []  # no retry after a released answer, one after a withheld one
    if released != "released":
        assert "fails closed" in out.getvalue()


def test_scene_5_deviates_when_the_prompt_was_not_redacted():
    class Unredacted(ChatAudit):
        def latest(self, session_id: str, *, after: int, timeout_s: float = 5.0):
            entry = super().latest(session_id, after=after)
            assert entry is not None
            return entry.model_copy(update={"verdicts": ()})

    agent = ChattyAgent("ok")
    demo, _ = _demo(agent)
    demo.audit = Unredacted(agent)
    result = ContentControls().run(demo)
    assert [c.name for c in result.failed_checks] == ["pii verdict on the prompt"]


def test_a_transport_failure_is_recorded_and_the_run_goes_on():
    class Unreachable(FakeOperator):
        def token(self, sub: str, *, kind: str = "agent") -> IssuedToken:
            raise httpx.ConnectError("connection refused")

    demo, _ = _demo(ScriptedAgent({}))
    demo.operator = Unreachable()
    result = SameQuestionTwoPeople().run(demo)
    assert result.error == "ConnectError: connection refused"
