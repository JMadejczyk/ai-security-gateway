"""Observability suites: dashboards and configs (static), metric presence and the audit stream
(in-process), and the live stack (``docker``-marked, only with ``ACL_DOCKER_TESTS=1``).

The dashboard builder and the observability tools live in top-level ``grafana/`` and
``observability/`` packages, so the repository root goes on ``sys.path`` here.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DOCKER_TESTS_ENV = "ACL_DOCKER_TESTS"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    if os.environ.get(DOCKER_TESTS_ENV) == "1":
        return
    skip = pytest.mark.skip(reason=f"live stack test; set {DOCKER_TESTS_ENV}=1 to run")
    here = Path(__file__).parent
    for item in items:
        if "docker" in item.keywords and Path(item.path).is_relative_to(here):
            item.add_marker(skip)
