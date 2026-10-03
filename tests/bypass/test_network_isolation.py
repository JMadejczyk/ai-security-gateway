"""Live bypass checks: from inside the agent (and mcp-fetch) containers, nothing but the gateway's
agent listener is reachable. DNS failure, refusal, timeout and unreachable all count as "did not
get through"; the assertion is only that no TCP connection was established.

Run against a started stack: ACL_DOCKER_TESTS=1 pytest tests/bypass -m docker
"""

from __future__ import annotations

import pytest
from compose_support import StackProbe

pytestmark = pytest.mark.docker

AGENT_BLOCKED = [
    "ollama:11434",
    "mcp-postgres:8000",
    "mcp-files:8000",
    "mcp-fetch:8000",
    "postgres:5432",
    "gateway:9090",  # operator listener: /auth/demo-token, /admin/*, /metrics
    "1.1.1.1:443",  # internet, by address (no DNS needed)
    "example.com:443",  # internet, by name
]

FETCH_BLOCKED = [
    "postgres:5432",
    "ollama:11434",
    "mcp-postgres:8000",
    "mcp-files:8000",
    "gateway:9090",
]


UPSTREAMS = ["ollama:11434", "mcp-postgres:8000", "mcp-files:8000", "mcp-fetch:8000"]


@pytest.mark.parametrize("target", UPSTREAMS)
def test_gateway_reaches_every_upstream(stack_probe: StackProbe, target: str) -> None:
    """Positive control: the blocked probes below fail because of the topology, not downtime."""
    result = stack_probe.from_service("gateway", [target])[target]
    assert result.connected, result


def test_agent_reaches_the_gateway_agent_listener(stack_probe: StackProbe) -> None:
    result = stack_probe.from_service("agent", ["gateway:8080"])["gateway:8080"]
    assert result.connected, result


@pytest.mark.parametrize("target", AGENT_BLOCKED)
def test_agent_cannot_bypass_the_gateway(stack_probe: StackProbe, target: str) -> None:
    result = stack_probe.from_service("agent", [target])[target]
    assert not result.connected, f"agent connected to {target}: {result}"


@pytest.mark.parametrize("target", FETCH_BLOCKED)
def test_fetch_server_cannot_reach_internal_services(stack_probe: StackProbe, target: str) -> None:
    result = stack_probe.from_service("mcp-fetch", [target])[target]
    assert not result.connected, f"mcp-fetch connected to {target}: {result}"
