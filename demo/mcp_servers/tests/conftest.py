"""A throwaway Postgres seeded from demo/db, for the tests that need the real database.

Docker-backed and opt-in, like the gateway's live tests: they run only with
``ACL_DOCKER_TESTS=1`` and a docker CLI, and are skipped otherwise.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from acl_demo_mcp.postgres_server import PostgresSettings

DOCKER_TESTS_ENV = "ACL_DOCKER_TESTS"
POSTGRES_IMAGE = "postgres:16.15"  # as in docker-compose.yml
DB_INIT = Path(__file__).resolve().parents[2] / "db"
READY_TIMEOUT_S = 90.0


def _docker(docker: str, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603 -- fixed argv, docker resolved from PATH
        [docker, *args], capture_output=True, text=True, check=True, timeout=120
    )
    return completed.stdout.strip()


def _wait_until_seeded(settings: PostgresSettings) -> None:
    """The entrypoint seeds on a socket-only server, so a TCP query means the seed is done."""
    deadline = time.monotonic() + READY_TIMEOUT_S
    while True:
        try:
            with psycopg.connect(
                host=settings.host,
                port=settings.port,
                dbname=settings.dbname,
                user=settings.user,
                password=settings.password.get_secret_value(),
                connect_timeout=2,
            ) as conn:
                conn.execute("SELECT 1 FROM sales.orders LIMIT 1")
                return
        except psycopg.Error:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


@pytest.fixture(scope="session")
def demo_postgres() -> Iterator[PostgresSettings]:
    docker = shutil.which("docker")
    if os.environ.get(DOCKER_TESTS_ENV) != "1" or docker is None:
        pytest.skip(f"needs docker; set {DOCKER_TESTS_ENV}=1 to run")
    app_password = secrets.token_urlsafe(24)
    name = f"acl-demo-pg-{secrets.token_hex(4)}"
    _docker(
        docker,
        "run",
        "--detach",
        "--rm",
        "--name",
        name,
        "--env",
        "POSTGRES_DB=acl_demo",
        "--env",
        "POSTGRES_USER=postgres",
        "--env",
        f"POSTGRES_PASSWORD={secrets.token_urlsafe(24)}",
        "--env",
        f"ACL_APP_DB_PASSWORD={app_password}",
        "--publish",
        "127.0.0.1::5432",
        "--volume",
        f"{DB_INIT}:/docker-entrypoint-initdb.d:ro",
        POSTGRES_IMAGE,
    )
    try:
        port = int(_docker(docker, "port", name, "5432/tcp").rsplit(":", 1)[1])
        settings = PostgresSettings.model_validate(
            {"host": "127.0.0.1", "port": port, "password": app_password}
        )
        _wait_until_seeded(settings)
        yield settings
    finally:
        _docker(docker, "rm", "--force", name)
