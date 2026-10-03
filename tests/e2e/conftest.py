"""Live end-to-end tests: real requests through the gateway container to the demo upstreams.

They need the compose stack running (``docker compose up -d --build``) and run only with
``ACL_DOCKER_TESTS=1``. ``docker compose`` runs from the repo root and inherits the
environment, so ``COMPOSE_PROJECT_NAME`` / ``COMPOSE_ENV_FILES`` select the stack, and
``ACL_OPERATOR_HOST_PORT`` the host port of the operator listener (default 9090).
"""

import os
import shutil
from pathlib import Path

import pytest
from live_stack import LiveStack

DOCKER_TESTS_ENV = "ACL_DOCKER_TESTS"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    if os.environ.get(DOCKER_TESTS_ENV) == "1":
        return
    skip = pytest.mark.skip(reason=f"live stack test; set {DOCKER_TESTS_ENV}=1 to run")
    for item in items:
        if "docker" in item.keywords and Path(item.path).is_relative_to(Path(__file__).parent):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def live_stack() -> LiveStack:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker CLI not installed")
    port = os.environ.get("ACL_OPERATOR_HOST_PORT", "9090")
    return LiveStack(docker=docker, operator_url=f"http://127.0.0.1:{port}")
