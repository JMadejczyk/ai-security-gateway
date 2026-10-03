"""The observability stack, live: Prometheus scrapes the gateway, the audit log reaches Loki
through Alloy, Grafana is up with its four dashboards, and every panel query runs.

Run against a started stack (``make up``) with ``ACL_DOCKER_TESTS=1``. Host ports come from
``ACL_AGENT_HOST_PORT`` / ``ACL_OPERATOR_HOST_PORT`` / ``ACL_GRAFANA_HOST_PORT`` and the
Grafana password from ``ACL_GRAFANA_ADMIN_PASSWORD`` (as in ``.env``).
"""

import os
import time
from collections.abc import Iterator

import httpx
import pytest

from observability.verify_panels import DASHBOARD_UIDS, Grafana, check

pytestmark = pytest.mark.docker

AGENT = f"http://127.0.0.1:{os.environ.get('ACL_AGENT_HOST_PORT', '8080')}"
OPERATOR = f"http://127.0.0.1:{os.environ.get('ACL_OPERATOR_HOST_PORT', '9090')}"
GRAFANA = f"http://127.0.0.1:{os.environ.get('ACL_GRAFANA_HOST_PORT', '3300')}"
PROXY = "/api/datasources/proxy/uid"
FAKE_PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"


@pytest.fixture(scope="module")
def grafana() -> Iterator[httpx.Client]:
    password = os.environ.get("ACL_GRAFANA_ADMIN_PASSWORD")
    if not password:
        pytest.skip("ACL_GRAFANA_ADMIN_PASSWORD is not set")
    with httpx.Client(base_url=GRAFANA, auth=("admin", password), timeout=30) as client:
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


def test_every_panel_query_runs(audited_block: str) -> None:
    password = os.environ.get("ACL_GRAFANA_ADMIN_PASSWORD", "")
    client = Grafana(GRAFANA, "admin", password)
    try:
        results = check(client, audited_block)
    finally:
        client.close()
    assert len(results) >= 40
    assert [r for r in results if r.failed] == []
    timeline = next(r for r in results if r.panel == "Timeline")
    assert timeline.values > 0  # the session that was just blocked has a trace
