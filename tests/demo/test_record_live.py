"""`make record SCENE=n --no-wait`, live: every beat resets and lands without a person.

Needs the stack with the demo overlay (``make demo-up``, with or without ``REMOTE=1``) and
``ACL_DOCKER_TESTS=1``. Beat 4 calls the model; the others take a few seconds each (beat 3
waits out the throttle, about 20 s).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from demo.orchestrator.scenes import ETL, INJECTION_PAGE, OLGA, fetch, report
from demo.orchestrator.stack import (
    AgentRunner,
    Compose,
    Operator,
    max_cost,
    set_max_cost,
)
from demo.run_demo import POLICY, POLICY_BACKUP, env_port

pytestmark = pytest.mark.docker

REPO_ROOT = Path(__file__).resolve().parents[2]
OPERATOR_URL = f"http://127.0.0.1:{env_port(os.environ, 'ACL_OPERATOR_HOST_PORT', 9090)}"


def _record(scene: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- fixed argv: this interpreter, our module
        [sys.executable, "-m", "demo.record", "--scene", str(scene), "--no-wait", "--pace", "0",
         "--no-color"],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )  # fmt: skip


@pytest.fixture(scope="module")
def operator():
    client = Operator(OPERATOR_URL)
    yield client
    client.close()


@pytest.mark.parametrize("scene", [1, 2, 4, 6])
def test_an_opencode_beat_or_the_end_shot_lands(scene: int):
    completed = _record(scene)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-1000:]
    assert "[PASS] approval queue empty: 0" in completed.stdout


def test_beat_3_denies_a_stale_approval_then_runs_the_night_job_once(operator: Operator):
    agent = AgentRunner(Compose())
    etl = operator.token(ETL)
    agent.mcp(etl, fetch(INJECTION_PAGE, wait_throttle=True))
    stale = agent.mcp(etl, report("stale-take.md", "left over", wait_throttle=True))
    assert stale.approval_id is not None
    assert stale.approval_id in {a.id for a in operator.pending_approvals(OLGA)}

    completed = _record(3)

    assert completed.returncode == 0, completed.stdout[-3000:]
    assert operator.approval(OLGA, stale.approval_id).state == "denied"
    assert operator.pending_approvals(OLGA) == ()
    assert "[PASS] approval executed exactly once: succeeded" in completed.stdout


def test_beat_5_restores_a_policy_an_interrupted_run_left_edited(operator: Operator):
    original = POLICY.read_bytes()
    revision = operator.policy_revision()
    try:
        # What a run killed mid-edit leaves behind: the backup, and the edited policy.
        POLICY_BACKUP.parent.mkdir(parents=True, exist_ok=True)
        POLICY_BACKUP.write_bytes(original)
        edited = POLICY.with_name(".policy-test-edit.tmp")
        edited.write_text(set_max_cost(original.decode(), 123_456))
        edited.replace(POLICY)  # a new inode, like an editor's save
        assert max_cost(POLICY.read_text()) == 123_456

        completed = _record(5)

        assert completed.returncode == 0, completed.stdout[-3000:]
        assert "restored config/policy.yaml left edited by an interrupted run" in completed.stdout
        assert POLICY.read_bytes() == original
        assert not POLICY_BACKUP.exists()
        assert operator.policy_revision() == revision
        assert "[PASS] heavy query before the edit: sql_cost_exceeded" in completed.stdout
    finally:
        if max_cost(POLICY.read_text()) == 123_456:  # only ever undo this test's own edit
            POLICY.write_bytes(original)
        POLICY_BACKUP.unlink(missing_ok=True)


def test_record_without_a_scene_is_a_usage_error():
    completed = subprocess.run(
        ["make", "--no-print-directory", "record"],  # noqa: S607 -- make from PATH
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "usage: make record SCENE=1..6" in completed.stdout + completed.stderr
