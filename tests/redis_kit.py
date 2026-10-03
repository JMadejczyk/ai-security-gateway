"""A real Redis for the store contracts (marker ``redis``), shared by every suite.

The session fixture ``real_redis`` (root ``tests/conftest.py``) uses ``ACL_TEST_REDIS_URL``
(and ``ACL_TEST_REDIS_PASSWORD``) when set, else starts a throwaway ``redis:7.2.16`` container
on a random loopback port when docker is usable, and skips the test otherwise.
"""

import contextlib
import secrets
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass

REDIS_IMAGE = "redis:7.2.16"
REDIS_URL_ENV = "ACL_TEST_REDIS_URL"
REDIS_AUTH_ENV = "ACL_TEST_REDIS_PASSWORD"


@dataclass(frozen=True)
class RealRedis:
    url: str
    password: str | None


@contextlib.contextmanager
def docker_redis() -> Iterator[RealRedis | None]:
    """A throwaway Redis on a random loopback port, removed afterwards; None without docker."""
    docker = shutil.which("docker")
    if docker is None:
        yield None
        return
    password = secrets.token_urlsafe(24)
    run = subprocess.run(  # noqa: S603 -- fixed argv, docker resolved from PATH
        [
            docker, "run", "--rm", "-d", "-p", "127.0.0.1::6379", REDIS_IMAGE,
            "redis-server", "--save", "", "--appendonly", "no", "--requirepass", password,
        ],
        capture_output=True, text=True, check=False, timeout=120,
    )  # fmt: skip
    if run.returncode != 0:
        yield None
        return
    container = run.stdout.strip()
    try:
        port = subprocess.run(  # noqa: S603 -- fixed argv
            [docker, "port", container, "6379/tcp"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.split()[0].rsplit(":", 1)[1]  # fmt: skip
        yield RealRedis(url=f"redis://127.0.0.1:{port}/0", password=password)
    finally:
        subprocess.run(  # noqa: S603 -- fixed argv
            [docker, "rm", "-f", container], capture_output=True, check=False, timeout=60
        )
