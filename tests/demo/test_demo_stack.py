"""Stack access without a stack: the policy edit, the operator client and the agent runner."""

import subprocess
from pathlib import Path

import httpx
import pytest

from demo.orchestrator.models import IssuedToken, ToolCall
from demo.orchestrator.stack import (
    AgentRunner,
    Compose,
    DemoError,
    Operator,
    PolicyFile,
    max_cost,
    set_max_cost,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_POLICY = REPO_ROOT / "config" / "policy.yaml"


@pytest.fixture
def policy(tmp_path: Path) -> PolicyFile:
    path = tmp_path / "config" / "policy.yaml"
    path.parent.mkdir()
    path.write_bytes(ROOT_POLICY.read_bytes())
    path.chmod(0o640)
    return PolicyFile(path, tmp_path / "reports" / ".backup.yaml")


def test_max_cost_is_read_and_set_on_the_root_policy():
    text = ROOT_POLICY.read_text()
    edited = set_max_cost(text, 100_000)
    assert (max_cost(text), max_cost(edited)) == (10_000, 100_000)
    assert edited.replace("100000", "10000", 1) == text  # nothing else changes


def test_a_policy_without_sql_guard_cost_is_refused():
    with pytest.raises(DemoError):
        set_max_cost("controls: {}\n", 1)
    with pytest.raises(DemoError):
        max_cost("controls: {}\n")


def test_the_edit_replaces_the_file_and_always_restores_it(policy: PolicyFile):
    original = policy.path.read_bytes()
    inode = policy.path.stat().st_ino
    with policy.edited(lambda text: set_max_cost(text, 100_000)):
        assert max_cost(policy.path.read_text()) == 100_000
        assert policy.path.stat().st_ino != inode  # a new file: what the directory watch sees
        assert policy.backup.read_bytes() == original
    assert policy.path.read_bytes() == original
    assert policy.path.stat().st_mode & 0o777 == 0o640
    assert not policy.backup.exists()
    assert list(policy.path.parent.iterdir()) == [policy.path]  # no temp files left


def test_the_edit_is_restored_when_the_scene_fails(policy: PolicyFile):
    original = policy.path.read_bytes()
    with pytest.raises(RuntimeError), policy.edited(lambda text: set_max_cost(text, 1)):
        raise RuntimeError
    assert policy.path.read_bytes() == original


def test_a_backup_left_by_an_interrupted_run_is_restored(policy: PolicyFile):
    original = policy.path.read_bytes()
    policy.backup.parent.mkdir()
    policy.backup.write_bytes(original)
    policy.path.write_text(set_max_cost(original.decode(), 5))
    assert policy.recover() is True
    assert policy.path.read_bytes() == original
    assert not policy.backup.exists()
    assert policy.recover() is False


METRICS = """# HELP acl_policy_info Active policy revision
acl_policy_info{revision="f83f261412c7"} 0.0
acl_policy_info{revision="f4e8ebf64543"} 1.0
"""


def _operator(handler) -> Operator:
    return Operator("http://ops", transport=httpx.MockTransport(handler))


def test_the_active_revision_comes_from_metrics():
    operator = _operator(lambda _request: httpx.Response(200, text=METRICS))
    assert operator.policy_revision() == "f4e8ebf64543"


def test_waiting_for_a_revision_nudges_a_reload_when_the_watcher_is_slow():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/auth/demo-token":
            return httpx.Response(
                200, json={"access_token": "t", "sub": "root@demo", "kind": "operator"}
            )
        if request.url.path == "/admin/reload":
            return httpx.Response(200, json={"result": "ok"})
        revision = "bbb1" if "/admin/reload" in seen else "aaa0"
        return httpx.Response(200, text=f'acl_policy_info{{revision="{revision}"}} 1.0\n')

    operator = _operator(handler)
    assert operator.wait_for_revision(lambda rev: rev == "bbb1", nudge_after_s=0.0) == "bbb1"
    assert "/admin/reload" in seen


def test_a_refused_operator_call_is_a_demo_error():
    operator = _operator(lambda _request: httpx.Response(422, json={"detail": "not_an_operator"}))
    with pytest.raises(DemoError, match="422"):
        operator.token("anna@demo", kind="operator")


class FakeCompose(Compose):
    """Records the argv and answers with a canned agent reply."""

    def __init__(self, stdout: str) -> None:
        self.root = REPO_ROOT
        self.stdout = stdout
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def run(self, *args: str, env=None, timeout_s: float = 60.0, check: bool = True):
        self.calls.append((args, env or {}))
        return subprocess.CompletedProcess(list(args), 0, self.stdout, "")


TOKEN = IssuedToken.model_validate(
    {"access_token": "secret-jwt", "sub": "anna@demo", "kind": "agent", "session_id": "s-1"}
)


def test_the_agent_runs_in_its_container_with_the_token_in_the_environment():
    compose = FakeCompose('{"kind": "mcp", "server": "reports", "tool": "write_report", '
                          '"status": 200, "reason": "ok"}')  # fmt: skip
    call = ToolCall(server="reports", tool="write_report", arguments={"name": "a.md"})
    reply = AgentRunner(compose).mcp(TOKEN, call.retried_with("apr-1"))
    assert reply.ok
    [(args, env)] = compose.calls
    assert args[:7] == ("exec", "-T", "-e", "ACL_TOKEN", "agent", "python", "-m")
    assert args[7:] == ("acl_agent", "mcp", "reports", "write_report", '{"name": "a.md"}',
                        "--approval-id", "apr-1")  # fmt: skip
    assert "secret-jwt" not in args  # never on the command line
    assert env == {"ACL_TOKEN": "secret-jwt"}


def test_a_probe_needs_no_token_and_a_wrong_reply_kind_is_an_error():
    compose = FakeCompose('{"kind": "probe", "host": "ollama", "port": 11434, "connected": false}')
    runner = AgentRunner(compose)
    assert runner.probe("ollama", 11434).connected is False
    assert compose.calls[0][1] == {}
    with pytest.raises(DemoError, match="expected a chat reply"):
        runner.chat(TOKEN, "hi")
