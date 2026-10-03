"""Helpers for the gateway-bypass suite: compose config view and in-container TCP probes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"

# `docker compose config` refuses to render while required secrets are unset; any value works.
_PLACEHOLDER_SECRETS = {
    "ACL_JWT_SECRET": "placeholder-jwt-secret",
    "ACL_INTERNAL_KEY": "placeholder-internal-key",
    "POSTGRES_PASSWORD": "placeholder-postgres",
    "ACL_APP_DB_PASSWORD": "placeholder-app",
}


def find_docker() -> str | None:
    return shutil.which("docker")


@dataclass(frozen=True)
class ComposeConfig:
    """Typed view over the JSON that `docker compose config --format json` renders."""

    raw: Mapping[str, Any]

    @property
    def services(self) -> dict[str, dict[str, Any]]:
        return cast(dict[str, dict[str, Any]], self.raw["services"])

    @property
    def networks(self) -> dict[str, dict[str, Any]]:
        return cast(dict[str, dict[str, Any]], self.raw.get("networks", {}))

    def service(self, name: str) -> dict[str, Any]:
        return self.services[name]

    def networks_of(self, service: str) -> set[str]:
        return set(cast(dict[str, Any], self.service(service).get("networks", {})))

    def environment_of(self, service: str) -> dict[str, str | None]:
        return cast(dict[str, str | None], self.service(service).get("environment") or {})

    @classmethod
    def render(cls, docker: str) -> ComposeConfig:
        """Render the committed compose file with every profile enabled."""
        env = {**os.environ, **_PLACEHOLDER_SECRETS}
        env.pop("COMPOSE_FILE", None)  # the committed file only, never a local override
        argv = [docker, "compose", "-f", str(COMPOSE_FILE), "--profile", "*"]
        completed = subprocess.run(  # noqa: S603 - fixed argv, docker resolved from PATH
            [*argv, "config", "--format", "json"],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
            env=env,
            timeout=60,
        )
        if completed.returncode != 0:
            raise ComposeConfigError(completed.stderr)
        return cls(raw=cast(dict[str, Any], json.loads(completed.stdout)))


class ComposeConfigError(RuntimeError):
    """`docker compose config` rejected the compose file."""


@dataclass(frozen=True)
class ProbeResult:
    target: str
    outcome: str  # connected | dns_failure | refused | timeout | unreachable | error
    detail: str

    @property
    def connected(self) -> bool:
        return self.outcome == "connected"


# Runs inside the containers (python:3.12 images); stdlib only. Prints one JSON object per target.
_PROBE_SCRIPT = """
import errno, json, socket, sys
for target in json.loads(sys.argv[1]):
    host, port = target.rsplit(":", 1)
    try:
        infos = socket.getaddrinfo(host, int(port), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        print(json.dumps([target, "dns_failure", str(exc)])); continue
    outcome, detail = "error", "no address"
    for family, kind, proto, _, addr in infos:
        sock = socket.socket(family, kind, proto)
        sock.settimeout(3)
        try:
            sock.connect(addr)
            outcome, detail = "connected", str(addr)
            break
        except socket.timeout as exc:
            outcome, detail = "timeout", str(exc)
        except ConnectionRefusedError as exc:
            outcome, detail = "refused", str(exc)
        except OSError as exc:
            unreachable = exc.errno in (errno.ENETUNREACH, errno.EHOSTUNREACH)
            outcome, detail = ("unreachable" if unreachable else "error"), str(exc)
        finally:
            sock.close()
    print(json.dumps([target, outcome, detail]))
"""


class ProbeError(RuntimeError):
    def __init__(self, service: str, output: str) -> None:
        super().__init__(f"probe in {service} failed: {output}")


class StackProbe:
    """Attempts TCP connections from inside a running compose service."""

    def __init__(self, docker: str) -> None:
        self._docker = docker

    def from_service(self, service: str, targets: list[str]) -> dict[str, ProbeResult]:
        completed = subprocess.run(  # noqa: S603 - fixed argv, docker resolved from PATH
            [
                self._docker,
                "compose",
                "exec",
                "-T",
                service,
                "python",
                "-c",
                _PROBE_SCRIPT,
                json.dumps(targets),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
            timeout=30 + 10 * len(targets),
        )
        if completed.returncode != 0:
            raise ProbeError(service, completed.stderr)
        results: dict[str, ProbeResult] = {}
        for line in completed.stdout.splitlines():
            target, outcome, detail = cast(list[str], json.loads(line))
            results[target] = ProbeResult(target=target, outcome=outcome, detail=detail)
        if set(results) != set(targets):
            raise ProbeError(service, completed.stdout)
        return results
