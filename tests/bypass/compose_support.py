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


# Minimal streamable-HTTP MCP client (stdlib only), so it runs in any python:3.12 container.
# argv: url, tool, JSON arguments. Prints {"is_error": bool, "text": str}.
_MCP_CALL_SCRIPT = """
import json, sys, urllib.request
url, tool, arguments = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
def rpc(payload):
    request = urllib.request.Request(url, json.dumps(payload).encode(), headers, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        session = response.headers.get("mcp-session-id")
        if session:
            headers["Mcp-Session-Id"] = session
        body = response.read().decode()
        kind = response.headers.get("content-type", "")
    if "id" not in payload or not body:
        return None
    if kind.startswith("text/event-stream"):
        for line in body.splitlines():
            if line.startswith("data:") and line[5:].strip():
                message = json.loads(line[5:])
                if message.get("id") == payload["id"]:
                    return message
    return json.loads(body)
init = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {},
    "clientInfo": {"name": "bypass-tests", "version": "0"}}})
headers["MCP-Protocol-Version"] = init["result"]["protocolVersion"]
rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
reply = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": tool, "arguments": arguments}})
if "error" in reply:
    print(json.dumps({"is_error": True, "text": json.dumps(reply["error"])}))
else:
    result = reply["result"]
    text = " ".join(item.get("text", "") for item in result.get("content", []))
    print(json.dumps({"is_error": bool(result.get("isError")), "text": text[:2000]}))
"""


@dataclass(frozen=True)
class ToolCallResult:
    is_error: bool
    text: str


class ProbeError(RuntimeError):
    def __init__(self, service: str, output: str) -> None:
        super().__init__(f"probe in {service} failed: {output}")


class StackProbe:
    """Runs probes inside running compose services and inspects their network addresses."""

    def __init__(self, docker: str) -> None:
        self._docker = docker

    def _run(self, service: str, argv: list[str], timeout: float) -> str:
        completed = subprocess.run(  # noqa: S603 - fixed argv, docker resolved from PATH
            [self._docker, *argv],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise ProbeError(service, completed.stderr)
        return completed.stdout

    def _exec_python(self, service: str, script: str, *args: str, timeout: float) -> str:
        argv = ["compose", "exec", "-T", service, "python", "-c", script, *args]
        return self._run(service, argv, timeout)

    def from_service(self, service: str, targets: list[str]) -> dict[str, ProbeResult]:
        """TCP-connect from `service` to each `host:port` target."""
        output = self._exec_python(
            service, _PROBE_SCRIPT, json.dumps(targets), timeout=30 + 10 * len(targets)
        )
        results: dict[str, ProbeResult] = {}
        for line in output.splitlines():
            target, outcome, detail = cast(list[str], json.loads(line))
            results[target] = ProbeResult(target=target, outcome=outcome, detail=detail)
        if set(results) != set(targets):
            raise ProbeError(service, output)
        return results

    def addresses(self, service: str) -> dict[str, str]:
        """IPv4 address of `service`'s container on each compose network, by short network name."""
        container = self._run(service, ["compose", "ps", "-q", service], 30).strip()
        if not container or "\n" in container:
            raise ProbeError(service, f"expected one running container, got {container!r}")
        fmt = (
            "{{json .NetworkSettings.Networks}}"
            '|{{index .Config.Labels "com.docker.compose.project"}}'
        )
        raw, project = self._run(service, ["inspect", "--format", fmt, container], 30).rsplit(
            "|", 1
        )
        networks = cast(dict[str, dict[str, Any]], json.loads(raw))
        prefix = f"{project.strip()}_"
        return {
            name.removeprefix(prefix): str(spec["IPAddress"]) for name, spec in networks.items()
        }

    def call_tool(
        self, service: str, url: str, tool: str, arguments: Mapping[str, object]
    ) -> ToolCallResult:
        """Call an MCP tool at `url` from inside `service`."""
        output = self._exec_python(
            service, _MCP_CALL_SCRIPT, url, tool, json.dumps(arguments), timeout=60
        )
        data = cast(dict[str, Any], json.loads(output))
        return ToolCallResult(is_error=bool(data["is_error"]), text=str(data["text"]))
