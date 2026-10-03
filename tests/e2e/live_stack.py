"""Talking to a running compose stack: demo tokens from the operator API, commands in services."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LiveStack:
    docker: str
    operator_url: str

    def token(self, sub: str) -> str:
        response = httpx.post(f"{self.operator_url}/auth/demo-token", json={"sub": sub}, timeout=10)
        response.raise_for_status()
        return response.json()["access_token"]

    def run_in(self, service: str, script: str, *args: str, env: dict[str, str]) -> str:
        """``python -c script args`` inside a running service; ``env`` goes in via ``-e``."""
        flags = [arg for name in env for arg in ("-e", name)]
        completed = subprocess.run(  # noqa: S603 -- fixed argv, docker resolved from PATH
            [self.docker, "compose", "exec", "-T", *flags, service, "python", "-c", script, *args],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
            env={**os.environ, **env},
            timeout=90,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout
