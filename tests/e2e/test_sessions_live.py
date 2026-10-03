"""Session state in Redis and ``acl pin``, live against the compose stack.

The payoff of Redis sessions: a session tainted through the untrusted ``web`` server stays
tainted across a gateway restart (its report write is refused by the session's risk rules,
not by anything held in the old process). And ``acl pin`` runs inside the gateway container
(``docker compose exec``), where it sees the upstreams on the internal networks: the committed
``pins/`` must match what the demo servers advertise.
"""

import json
import os
import subprocess
import time

import httpx
import pytest
from live_stack import REPO_ROOT, LiveStack

pytestmark = pytest.mark.docker

GATEWAY = "http://gateway:8080/mcp"

# Runs in the agent container: one MCP session, one tools/call. argv: url, tool, JSON
# arguments; the bearer token arrives in $ACL_TOKEN. Prints the tools/call result as JSON.
_CALL = """
import json, os, sys
import httpx
url, tool, arguments = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
headers = {"authorization": "Bearer " + os.environ["ACL_TOKEN"],
           "accept": "application/json", "content-type": "application/json"}
with httpx.Client(timeout=60) as client:
    init = client.post(url, headers=headers, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "e2e", "version": "0"}}})
    init.raise_for_status()
    headers["mcp-session-id"] = init.headers["mcp-session-id"]
    headers["mcp-protocol-version"] = "2025-06-18"
    client.post(url, headers=headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"}).raise_for_status()
    reply = client.post(url, headers=headers, json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments}})
print(json.dumps(reply.json()))
"""


def call(stack: LiveStack, token: str, server: str, tool: str, **arguments: object) -> dict:
    output = stack.run_in(
        "agent",
        _CALL,
        f"{GATEWAY}/{server}",
        tool,
        json.dumps(arguments),
        env={"ACL_TOKEN": token},
    )
    return json.loads(output)


def reason(reply: dict) -> str:
    result = reply["result"]
    assert result["isError"] is True, result
    return result["_meta"]["ai-control-layer/reason_code"]


def compose(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- fixed argv, docker resolved from PATH
        ["docker", "compose", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env=dict(os.environ),
        timeout=180,
    )


def wait_healthy(stack: LiveStack, timeout_s: float = 90) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{stack.operator_url}/healthz", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    pytest.fail("the gateway did not come back")


def test_taint_survives_a_gateway_restart(live_stack: LiveStack):
    token = live_stack.token("anna@demo")  # one gateway session for every call below
    fetched = call(live_stack, token, "web", "fetch", url="https://example.com/")
    assert "result" in fetched, fetched  # whatever the page did, untrusted content arrived

    restarted = compose("restart", "gateway")
    assert restarted.returncode == 0, restarted.stderr
    wait_healthy(live_stack)

    # Same token, new process: the write is refused by the taint rule, which only the
    # session state persisted in Redis can know about.
    written = call(live_stack, token, "reports", "write_report", name="q3.md", content="ok")
    assert reason(written) == "action_removed_by_session_risk"
    fresh = live_stack.token("anna@demo")  # another session is not tainted
    assert reason(
        call(live_stack, fresh, "reports", "write_report", name="q3.md", content="ok")
    ) != ("action_removed_by_session_risk")


def test_acl_pin_runs_in_the_gateway_container(live_stack: LiveStack):
    issued = httpx.post(
        f"{live_stack.operator_url}/auth/demo-token",
        json={"sub": "root@demo", "kind": "operator"},
        timeout=10,
    )
    issued.raise_for_status()
    token = issued.json()["access_token"]
    for server in ("sales_db", "web", "reports"):
        script = (
            'exec python -m gateway.cli --url "http://$ACL_OPERATOR_HOST:$ACL_OPERATOR_PORT" '
            f"pin {server}"
        )
        completed = subprocess.run(  # noqa: S603 -- fixed argv
            [live_stack.docker, "compose", "exec", "-T", "-e", "ACL_OPERATOR_TOKEN",
             "gateway", "sh", "-c", script],
            capture_output=True, text=True, check=False, cwd=REPO_ROOT,
            env={**os.environ, "ACL_OPERATOR_TOKEN": token}, timeout=90,
        )  # fmt: skip
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert completed.stdout.strip() == f"{server}: pins/{server}.json is up to date"
