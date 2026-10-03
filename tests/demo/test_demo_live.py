"""The whole demo, live: ``run_demo`` against the running stack with the demo overlay.

Needs ``make demo-up`` and ``ACL_DOCKER_TESTS=1``. Takes 2-5 minutes (scene 5 waits for the
CPU model). ``run_demo`` asserts every scene itself; this test checks its exit status and the
report it writes, and that ``config/policy.yaml`` is byte-for-byte what it was.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from demo.orchestrator.models import DemoReport

pytestmark = pytest.mark.docker

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY = REPO_ROOT / "config" / "policy.yaml"


def test_every_scene_reaches_its_expected_outcome(tmp_path: Path):
    report_path = tmp_path / "demo.json"
    policy_before = POLICY.read_bytes()
    completed = subprocess.run(  # noqa: S603 -- fixed argv: this interpreter, our module
        [sys.executable, "-m", "demo.run_demo", "--no-color", "--json", str(report_path)],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-2000:]
    report = DemoReport.model_validate_json(report_path.read_text())
    assert [scene.number for scene in report.scenes] == [1, 2, 3, 4, 5, 6, 7]
    assert all(scene.passed for scene in report.scenes), [
        (scene.number, scene.failed_checks, scene.error) for scene in report.scenes
    ]
    assert POLICY.read_bytes() == policy_before
