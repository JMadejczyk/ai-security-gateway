"""The running stack, as the orchestrator touches it.

* `Compose`: ``docker compose`` with the demo overlay, from the repo root.
* `AgentRunner`: agent actions, executed **inside** the ``agent`` container (``edge`` only),
  one ``python -m acl_agent ...`` per action, the bearer token in the environment.
* `Operator`: the operator API on the host (demo tokens, approvals, reload, ``/metrics``).
* `AuditLog`: the gateway's audit JSONL, read where it is written (operator-only volume).
* `Database`: ``EXPLAIN`` as a principal, the same statement setup sql_guard's explain uses.
* `PolicyFile`: an atomic, always-restored edit of ``config/policy.yaml`` (demo step 7).
"""

import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import Final

import httpx
from pydantic import ValidationError

from demo.orchestrator.models import (
    AgentReply,
    ApprovalInfo,
    AuditEntry,
    ChatReply,
    IssuedToken,
    ProbeReply,
    ReloadEvent,
    ToolCall,
    ToolReply,
    parse_agent_reply,
)
from gateway.policy.loader import PolicyLoader
from observability.smoke_traffic import MAX_COST

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
COMPOSE_FILES: Final = ("docker-compose.yml", "demo/compose.demo.yml")
AGENT_TIMEOUT_S: Final = 240.0
AUDIT_GLOB: Final = "/var/log/acl/audit-*.jsonl"
REVISION_METRIC: Final = re.compile(r'^acl_policy_info\{revision="([0-9a-f]+)"\} 1\.0$', re.M)
TOTAL_COST: Final = re.compile(r'"Total Cost":\s*([0-9.]+)')
SAFE_PRINCIPAL: Final = re.compile(r"^[A-Za-z0-9@._:-]+$")
POLL_S: Final = 0.5


class DemoError(RuntimeError):
    """The stack did not answer the way a scene needs to go on."""


class CommandError(DemoError):
    def __init__(self, argv: Sequence[str], completed: subprocess.CompletedProcess[str]) -> None:
        tail = (completed.stderr or completed.stdout).strip()[-500:]
        super().__init__(f"{' '.join(argv[:6])} ... exited {completed.returncode}: {tail}")


class Compose:
    """``docker compose -f docker-compose.yml -f demo/compose.demo.yml`` in the repo root."""

    def __init__(self, root: Path = REPO_ROOT, docker: str | None = None) -> None:
        found = docker or shutil.which("docker")
        if found is None:
            msg = "the docker CLI is not installed"
            raise DemoError(msg)
        self.root = root
        self._base = [found, "compose", *(arg for f in COMPOSE_FILES for arg in ("-f", f))]
        self._docker = found

    def run(
        self,
        *args: str,
        env: dict[str, str] | None = None,
        timeout_s: float = 60.0,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        argv = [*self._base, *args]
        completed = subprocess.run(  # noqa: S603 -- fixed argv; docker resolved from PATH
            argv,
            capture_output=True,
            text=True,
            check=False,
            cwd=self.root,
            env={**os.environ, **(env or {})},
            stdin=subprocess.DEVNULL,
            timeout=timeout_s,
        )
        if check and completed.returncode != 0:
            raise CommandError(argv, completed)
        return completed

    def running_services(self) -> frozenset[str]:
        out = self.run("ps", "--status", "running", "--services").stdout
        return frozenset(line.strip() for line in out.splitlines() if line.strip())

    def container_ips(self, service: str) -> tuple[str, ...]:
        """Every address ``service``'s container has, one per network it joins."""
        container = self.run("ps", "-q", service).stdout.strip()
        if not container:
            return ()
        completed = subprocess.run(  # noqa: S603 -- fixed argv; docker resolved from PATH
            [
                self._docker,
                "inspect",
                "-f",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}",
                container,
            ],
            capture_output=True,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL,
            timeout=30,
        )
        return tuple(ip for ip in completed.stdout.split() if ip)


class AgentRunner:
    """Runs DataBot actions inside the ``agent`` container, which can reach only the gateway."""

    def __init__(self, compose: Compose) -> None:
        self._compose = compose

    def _exec(self, token: str | None, *args: str) -> AgentReply:
        env = {"ACL_TOKEN": token} if token is not None else {}
        flags = ("-e", "ACL_TOKEN") if token is not None else ()
        completed = self._compose.run(
            "exec",
            "-T",
            *flags,
            "agent",
            "python",
            "-m",
            "acl_agent",
            *args,
            env=env,
            timeout_s=AGENT_TIMEOUT_S,
        )
        try:
            return parse_agent_reply(completed.stdout)
        except (ValidationError, ValueError) as exc:
            raise DemoError(str(exc)) from exc

    def mcp(self, token: IssuedToken, call: ToolCall) -> ToolReply:
        args = ["mcp", call.server, call.tool, json.dumps(call.arguments)]
        if call.approval_id is not None:
            args += ["--approval-id", call.approval_id]
        if call.wait_throttle:
            args.append("--wait-throttle")
        reply = self._exec(token.access_token, *args)
        if not isinstance(reply, ToolReply):
            msg = f"expected an MCP reply, got {reply.kind}"
            raise DemoError(msg)
        return reply

    def chat(self, token: IssuedToken, prompt: str, *, max_tokens: int = 80) -> ChatReply:
        reply = self._exec(token.access_token, "chat", prompt, "--max-tokens", str(max_tokens))
        if not isinstance(reply, ChatReply):
            msg = f"expected a chat reply, got {reply.kind}"
            raise DemoError(msg)
        return reply

    def probe(self, host: str, port: int) -> ProbeReply:
        reply = self._exec(None, "probe", host, str(port))
        if not isinstance(reply, ProbeReply):
            msg = f"expected a probe reply, got {reply.kind}"
            raise DemoError(msg)
        return reply


class Operator:
    """The operator API (``/auth/demo-token``, ``/admin/*``, ``/metrics``) from the host."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def close(self) -> None:
        self._http.close()

    def token(self, sub: str, *, kind: str = "agent") -> IssuedToken:
        body: dict[str, str] = {"sub": sub}
        if kind != "agent":
            body["kind"] = kind
        response = self._http.post("/auth/demo-token", json=body)
        if response.is_error:
            msg = f"demo token for {sub} ({kind}): HTTP {response.status_code} {response.text}"
            raise DemoError(msg)
        return IssuedToken.model_validate_json(response.content)

    def _admin(self, sub: str, method: str, path: str) -> httpx.Response:
        token = self.token(sub, kind="operator").access_token
        response = self._http.request(method, path, headers={"authorization": f"Bearer {token}"})
        if response.is_error:
            msg = f"{method} {path} as {sub}: HTTP {response.status_code} {response.text}"
            raise DemoError(msg)
        return response

    def pending_approvals(self, approver: str) -> tuple[ApprovalInfo, ...]:
        body = self._admin(approver, "GET", "/admin/approvals?state=pending").json()
        return tuple(ApprovalInfo.model_validate(item) for item in body["approvals"])

    def approval(self, approver: str, approval_id: str) -> ApprovalInfo:
        response = self._admin(approver, "GET", f"/admin/approvals/{approval_id}")
        return ApprovalInfo.model_validate_json(response.content)

    def approve(self, approver: str, approval_id: str) -> ApprovalInfo:
        response = self._admin(approver, "POST", f"/admin/approvals/{approval_id}/approve")
        return ApprovalInfo.model_validate_json(response.content)

    def deny(self, approver: str, approval_id: str) -> ApprovalInfo:
        response = self._admin(approver, "POST", f"/admin/approvals/{approval_id}/deny")
        return ApprovalInfo.model_validate_json(response.content)

    def reload(self, admin: str) -> str:
        return str(self._admin(admin, "POST", "/admin/reload").json().get("result", "error"))

    def policy_revision(self) -> str:
        """The active revision, from ``acl_policy_info{revision}`` on ``/metrics``."""
        match = REVISION_METRIC.search(self._http.get("/metrics").text)
        if match is None:
            msg = "no acl_policy_info series on /metrics"
            raise DemoError(msg)
        return match.group(1)

    def wait_for_revision(
        self,
        done: Callable[[str], bool],
        *,
        timeout_s: float = 20.0,
        nudge_after_s: float = 8.0,
        admin: str = "root@demo",
    ) -> str:
        """Poll until ``done(revision)``. The file watcher normally reacts within a few
        seconds; after ``nudge_after_s`` the operator asks for a reload once."""
        started = time.monotonic()
        nudged = False
        while True:
            revision = self.policy_revision()
            if done(revision):
                return revision
            elapsed = time.monotonic() - started
            if not nudged and elapsed >= nudge_after_s:
                self.reload(admin)
                nudged = True
                continue
            if elapsed > timeout_s:
                msg = f"policy revision still {revision} after {timeout_s:.0f} s"
                raise DemoError(msg)
            time.sleep(POLL_S)


class AuditLog:
    """The gateway's audit JSONL segments, read inside the gateway container."""

    def __init__(self, compose: Compose) -> None:
        self._compose = compose

    def _grep(self, needle: str) -> list[str]:
        completed = self._compose.run(
            "exec",
            "-T",
            "gateway",
            "sh",
            "-c",
            f'grep -h -F -- "$1" {AUDIT_GLOB} || true',
            "_",
            needle,
        )
        return [line for line in completed.stdout.splitlines() if line.strip()]

    def entries(self, session_id: str) -> tuple[AuditEntry, ...]:
        """Every decision of one session, oldest first."""
        entries: list[AuditEntry] = []
        for line in self._grep(f'"session_id":"{session_id}"'):
            with contextlib.suppress(ValidationError):
                entries.append(AuditEntry.model_validate_json(line))
        return tuple(sorted(entries, key=lambda e: e.ts))

    def latest(self, session_id: str, *, after: int, timeout_s: float = 5.0) -> AuditEntry | None:
        """The newest entry of the session once it has more than ``after`` entries."""
        deadline = time.monotonic() + timeout_s
        while True:
            found = self.entries(session_id)
            if len(found) > after:
                return found[-1]
            if time.monotonic() > deadline:
                return found[-1] if found else None
            time.sleep(POLL_S)

    def since(self, principal: str, ts: str) -> tuple[AuditEntry, ...]:
        """Decisions made for ``principal`` at or after ``ts`` (ISO 8601, UTC), oldest first."""
        entries: list[AuditEntry] = []
        for line in self._grep(f'"principal":"{principal}"'):
            with contextlib.suppress(ValidationError):
                entry = AuditEntry.model_validate_json(line)
                if entry.ts >= ts:
                    entries.append(entry)
        return tuple(sorted(entries, key=lambda e: e.ts))

    def reloads(self, last: int = 3) -> tuple[ReloadEvent, ...]:
        events: list[ReloadEvent] = []
        for line in self._grep('"event":"policy_reload"'):
            with contextlib.suppress(ValidationError):
                events.append(ReloadEvent.model_validate_json(line))
        return tuple(sorted(events, key=lambda e: e.ts)[-last:])


class Database:
    """``EXPLAIN (FORMAT JSON)`` as a principal: role ``acl_app``, ``acl.set_principal`` first,
    in a transaction that is rolled back. The operator's view of what sql_guard priced."""

    def __init__(self, compose: Compose) -> None:
        self._compose = compose

    def explain_cost(self, principal: str, sql: str) -> float:
        if not SAFE_PRINCIPAL.fullmatch(principal):
            msg = f"unexpected principal {principal!r}"
            raise DemoError(msg)
        completed = self._compose.run(
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "postgres",
            "-d",
            "acl_demo",
            "-q",
            "-At",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            "BEGIN",
            "-c",
            "SET LOCAL ROLE acl_app",
            "-c",
            f"SELECT acl.set_principal('{principal}')",
            "-c",
            f"EXPLAIN (FORMAT JSON) {sql}",
            "-c",
            "ROLLBACK",
        )
        match = TOTAL_COST.search(completed.stdout)
        if match is None:
            msg = "EXPLAIN printed no Total Cost"
            raise DemoError(msg)
        return float(match.group(1))


class PolicyFile:
    """Atomic edits of the policy the gateway mounts, always restored.

    The gateway mounts ``config/`` as a directory, so a replaced file (new inode, what editors
    and ``os.replace`` do) is seen by its watcher. The original bytes are also kept in
    ``backup`` until the restore lands, so an interrupted run is undone by the next one.
    """

    def __init__(self, path: Path, backup: Path) -> None:
        self.path = path
        self.backup = backup

    def revision(self) -> str:
        """The revision the gateway reports once it has loaded this file (same loader)."""
        return PolicyLoader().load(self.path).revision

    def recover(self) -> bool:
        """Restore a backup an interrupted run left behind; True if there was one."""
        if not self.backup.exists():
            return False
        self._replace(self.backup.read_bytes())
        self.backup.unlink()
        return True

    def _replace(self, data: bytes) -> None:
        mode = self.path.stat().st_mode & 0o777
        fd, tmp = tempfile.mkstemp(prefix=".policy-demo-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            Path(tmp).chmod(mode)
            Path(tmp).replace(self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @contextlib.contextmanager
    def edited(self, edit: Callable[[str], str]) -> Generator[str]:
        """Inside: the file holds ``edit(original)``. Yields the edited text."""
        original = self.path.read_bytes()
        edited = edit(original.decode())
        self.backup.parent.mkdir(parents=True, exist_ok=True)
        self.backup.write_bytes(original)
        try:
            self._replace(edited.encode())
            yield edited
        finally:
            self._replace(original)
            self.backup.unlink(missing_ok=True)


def set_max_cost(text: str, value: int) -> str:
    """``controls.sql_guard.max_cost`` set to ``value`` (the flow-style ``sql_guard: {...}``)."""
    edited, count = MAX_COST.subn(lambda m: f"{m.group(1)}{value}", text, count=1)
    if count != 1:
        msg = "no controls.sql_guard.max_cost in the policy"
        raise DemoError(msg)
    return edited


def max_cost(text: str) -> int:
    match = MAX_COST.search(text)
    if match is None:
        msg = "no controls.sql_guard.max_cost in the policy"
        raise DemoError(msg)
    return int(match.group(2))
