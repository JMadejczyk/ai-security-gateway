"""The observability stack, live: Prometheus scrapes the gateway, the audit log reaches Loki
through Alloy, Grafana is up with its four dashboards, and every panel query runs.

Run against a started stack (``make up``) with ``ACL_DOCKER_TESTS=1``. Host ports come from
``ACL_AGENT_HOST_PORT`` / ``ACL_OPERATOR_HOST_PORT`` / ``ACL_GRAFANA_HOST_PORT`` and the
Grafana password from ``ACL_GRAFANA_ADMIN_PASSWORD`` (as in ``.env``).
"""

import os
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from observability.smoke_traffic import MAX_COST
from observability.verify_panels import DASHBOARD_UIDS, Grafana, check

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.docker

AGENT = f"http://127.0.0.1:{os.environ.get('ACL_AGENT_HOST_PORT', '8080')}"
OPERATOR = f"http://127.0.0.1:{os.environ.get('ACL_OPERATOR_HOST_PORT', '9090')}"
GRAFANA = f"http://127.0.0.1:{os.environ.get('ACL_GRAFANA_HOST_PORT', '3300')}"
PROXY = "/api/datasources/proxy/uid"
FAKE_PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"


@pytest.fixture(scope="module")
def grafana_password() -> str:
    password = os.environ.get("ACL_GRAFANA_ADMIN_PASSWORD")
    if not password:
        pytest.skip("ACL_GRAFANA_ADMIN_PASSWORD is not set")
    return password


@pytest.fixture(scope="module")
def grafana(grafana_password: str) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=GRAFANA, auth=("admin", grafana_password), timeout=30) as client:
        yield client


@pytest.fixture(scope="module")
def audited_block() -> str:
    """One blocked LLM call (a private key in the prompt): an audit line with a known reason."""
    token = httpx.post(f"{OPERATOR}/auth/demo-token", json={"sub": "bartek@demo"}, timeout=10)
    token.raise_for_status()
    response = httpx.post(
        f"{AGENT}/v1/chat/completions",
        headers={"authorization": f"Bearer {token.json()['access_token']}"},
        json={
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "user",
                    "content": FAKE_PRIVATE_KEY,
                }
            ],
        },
        timeout=60,
    )
    assert response.status_code == 403, response.text
    return str(token.json()["session_id"])


def test_grafana_is_healthy_and_refuses_anonymous_access(grafana: httpx.Client) -> None:
    assert grafana.get("/api/health").json()["database"] == "ok"
    anonymous = httpx.get(f"{GRAFANA}/api/search", timeout=10)
    assert anonymous.status_code == 401


def test_the_four_dashboards_are_provisioned(grafana: httpx.Client) -> None:
    for uid in DASHBOARD_UIDS:
        response = grafana.get(f"/api/dashboards/uid/{uid}")
        assert response.status_code == 200, uid
        assert response.json()["meta"]["provisioned"] is True


def test_prometheus_scrapes_the_gateway(grafana: httpx.Client) -> None:
    response = grafana.get(
        f"{PROXY}/acl-prometheus/api/v1/query", params={"query": 'up{job="acl-gateway"}'}
    )
    (series,) = response.json()["data"]["result"]
    assert series["value"][1] == "1"


def test_audit_lines_reach_loki(grafana: httpx.Client, audited_block: str) -> None:
    query = f'{{job="acl", decision="block"}} |= "{audited_block}"'
    deadline = time.monotonic() + 60
    lines: list[object] = []
    while time.monotonic() < deadline and not lines:
        now = time.time_ns()
        response = grafana.get(
            f"{PROXY}/acl-loki/loki/api/v1/query_range",
            params={"query": query, "start": str(now - 600 * 10**9), "end": str(now)},
        )
        lines = [v for stream in response.json()["data"]["result"] for v in stream["values"]]
        if not lines:
            time.sleep(2)
    assert lines, "the blocked call never reached Loki"
    assert '"reason_code":"secret_detected"' in str(lines[0])


def test_every_panel_query_runs(grafana_password: str, audited_block: str) -> None:
    client = Grafana(GRAFANA, "admin", grafana_password)
    try:
        results = check(client, audited_block)
    finally:
        client.close()
    assert len(results) >= 40
    assert [r for r in results if r.failed] == []
    timeline = next(r for r in results if r.panel == "Timeline")
    assert timeline.values > 0  # the session that was just blocked has a trace


# Counts audit lines in every segment with start < ts <= end (runs in the gateway container).
_COUNT_LINES = """
import datetime, glob, json, sys
start, end = float(sys.argv[1]), float(sys.argv[2])
count = 0
for name in glob.glob("/var/log/acl/audit-*.jsonl"):
    for line in open(name, encoding="utf-8"):
        ts = datetime.datetime.fromisoformat(json.loads(line)["ts"].replace("Z", "+00:00"))
        count += start < ts.timestamp() <= end
print(count)
"""
OUTAGE_CALLS = int(os.environ.get("ACL_OUTAGE_CALLS", "80"))


def _compose(*args: str) -> str:
    completed = subprocess.run(  # noqa: S603 -- fixed argv, docker resolved from PATH
        ["docker", "compose", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env=dict(os.environ),
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _loki_count(grafana: httpx.Client, start: int, end: int) -> int:
    response = grafana.get(
        f"{PROXY}/acl-loki/loki/api/v1/query",
        params={"query": f'sum(count_over_time({{job="acl"}}[{end - start}s]))', "time": end},
    )
    result = response.json()["data"]["result"]
    return int(result[0]["value"][1]) if result else 0


def test_audit_lines_written_while_alloy_is_down_all_reach_loki(grafana: httpx.Client) -> None:
    """Stop Alloy, write audit lines (with a small ACL_AUDIT_MAX_BYTES this spans segment
    rollovers), start Alloy: every line in the files is in Loki, none twice."""
    token = httpx.post(f"{OPERATOR}/auth/demo-token", json={"sub": "bartek@demo"}, timeout=10)
    headers = {"authorization": f"Bearer {token.json()['access_token']}"}
    body = {"model": "qwen3:8b", "messages": [{"role": "user", "content": FAKE_PRIVATE_KEY}]}
    start = int(time.time())
    time.sleep(1.1)
    _compose("stop", "alloy")
    try:
        for _ in range(OUTAGE_CALLS):
            httpx.post(f"{AGENT}/v1/chat/completions", headers=headers, json=body, timeout=60)
        time.sleep(1.1)
        end = int(time.time())
    finally:
        _compose("start", "alloy")
    written = int(
        _compose("exec", "-T", "gateway", "python", "-c", _COUNT_LINES, str(start), str(end))
    )
    assert written >= OUTAGE_CALLS
    deadline = time.monotonic() + 90
    shipped = _loki_count(grafana, start, end)
    while shipped < written and time.monotonic() < deadline:
        time.sleep(3)
        shipped = _loki_count(grafana, start, end)
    assert shipped == written


POLICY_FILE = Path(os.environ.get("ACL_LIVE_POLICY_PATH", REPO_ROOT / "config" / "policy.yaml"))


def _replace_atomically(path: Path, text: str) -> None:
    """What editors and `git checkout` do: write a temp file, rename it over (a new inode)."""
    temp = path.with_name(f".{path.name}.tmp-live-test")
    temp.write_text(text)
    temp.replace(path)


def _revision() -> str:
    return str(httpx.get(f"{OPERATOR}/healthz", timeout=10).json()["policy_revision"])


def _wait_for_revision(predicate: Callable[[str], bool], timeout_s: float = 20) -> str:
    deadline = time.monotonic() + timeout_s
    revision = _revision()
    while not predicate(revision) and time.monotonic() < deadline:
        time.sleep(0.5)
        revision = _revision()
    return revision


def test_a_host_edit_of_the_policy_reloads_without_restart(grafana: httpx.Client) -> None:
    """Demo step 7: edit config/policy.yaml on the host the way an editor saves it; the
    running gateway picks it up by itself (no restart, no /admin/reload) and the reload
    reaches Loki for the dashboards' annotation."""
    original = POLICY_FILE.read_text()
    before = _revision()
    edited, count = MAX_COST.subn(lambda m: f"{m.group(1)}{int(m.group(2)) + 1}", original, 1)
    assert count == 1
    try:
        _replace_atomically(POLICY_FILE, edited)
        after = _wait_for_revision(lambda revision: revision != before)
        assert after != before, "the gateway never saw the host edit"
    finally:
        _replace_atomically(POLICY_FILE, original)
    assert _wait_for_revision(lambda revision: revision == before) == before

    query = (
        '{job="acl", event="policy_reload"} | json revision="revision", result="result" '
        f'| result="ok" | revision="{after}"'
    )
    deadline = time.monotonic() + 60
    lines: list[object] = []
    while time.monotonic() < deadline and not lines:
        now = time.time_ns()
        response = grafana.get(
            f"{PROXY}/acl-loki/loki/api/v1/query_range",
            params={"query": query, "start": str(now - 600 * 10**9), "end": str(now)},
        )
        lines = [v for stream in response.json()["data"]["result"] for v in stream["values"]]
        if not lines:
            time.sleep(2)
    assert lines, "the reload event never reached Loki"
