"""Fixtures for the gateway-bypass suite.

* Static checks parse `docker compose config` and only need the docker CLI.
* `@pytest.mark.docker` checks probe a running stack from inside its containers and run only
  with ACL_DOCKER_TESTS=1. They use `docker compose` from the repo root, so COMPOSE_FILE /
  COMPOSE_PROJECT_NAME in the environment select the stack (e.g. with a local override file).
"""

from __future__ import annotations

import os

import pytest
from compose_support import ComposeConfig, StackProbe, find_docker

DOCKER_TESTS_ENV = "ACL_DOCKER_TESTS"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "docker: needs the live docker compose stack (set ACL_DOCKER_TESTS=1)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    if os.environ.get(DOCKER_TESTS_ENV) == "1":
        return
    skip = pytest.mark.skip(reason=f"live stack test; set {DOCKER_TESTS_ENV}=1 to run")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)


def _docker_or_skip() -> str:
    docker = find_docker()
    if docker is None:
        pytest.skip("docker CLI not installed")
    return docker


@pytest.fixture(scope="session")
def compose_config() -> ComposeConfig:
    return ComposeConfig.render(_docker_or_skip())


@pytest.fixture(scope="session")
def stack_probe() -> StackProbe:
    return StackProbe(_docker_or_skip())
