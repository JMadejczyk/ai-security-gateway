"""Live bypass checks against a running stack.

Probes go to the containers' real IP addresses on every network they are attached to (from
`docker inspect`), not to Docker DNS names: a failed name lookup proves nothing about whether
the address is routable. Each blocked probe has a positive control showing the same listener
is up when reached from the gateway, so a pass cannot come from a stopped service.

Run against a started stack: ACL_DOCKER_TESTS=1 pytest tests/bypass -m docker
"""

from __future__ import annotations

import pytest
from compose_support import ProbeResult, StackProbe

pytestmark = pytest.mark.docker

OPERATOR_ADDRESS = "172.29.90.10"  # docker-compose.yml: gateway ipv4_address on `ops`
OPERATOR_PORT = 9090
AGENT_PORT = 8080

# Every listener the agent and mcp-fetch must never reach directly.
PROTECTED_PORTS = {
    "ollama": 11434,
    "mcp-postgres": 8000,
    "mcp-files": 8000,
    "mcp-fetch": 8000,
    "postgres": 5432,
}

StackAddresses = dict[str, dict[str, str]]


@pytest.fixture(scope="module")
def addresses(stack_probe: StackProbe) -> StackAddresses:
    services = [*PROTECTED_PORTS, "gateway", "agent"]
    return {service: stack_probe.addresses(service) for service in services}


def _protected_targets(addresses: StackAddresses, *, exclude: str) -> list[str]:
    """Every protected listener on every one of its addresses, plus the operator port on every
    gateway address."""
    targets = [
        f"{ip}:{port}"
        for service, port in PROTECTED_PORTS.items()
        if service != exclude
        for ip in addresses[service].values()
    ]
    targets += [f"{ip}:{OPERATOR_PORT}" for ip in addresses["gateway"].values()]
    return sorted(set(targets))


def _connected(results: dict[str, ProbeResult]) -> list[ProbeResult]:
    return [result for result in results.values() if result.connected]


def test_gateway_operator_address_is_the_pinned_one(addresses: StackAddresses) -> None:
    assert addresses["gateway"]["ops"] == OPERATOR_ADDRESS


def test_every_protected_listener_is_up(stack_probe: StackProbe, addresses: StackAddresses) -> None:
    """Positive controls, from the gateway over the network it shares with each service."""
    gateway_networks = set(addresses["gateway"])
    targets = [f"{OPERATOR_ADDRESS}:{OPERATOR_PORT}"]
    for service, port in PROTECTED_PORTS.items():
        shared = sorted(gateway_networks & set(addresses[service]))
        assert shared, f"gateway shares no network with {service}"
        targets.append(f"{addresses[service][shared[0]]}:{port}")
    results = stack_probe.from_service("gateway", targets)
    down = [result for result in results.values() if not result.connected]
    assert down == []


def test_agent_reaches_the_gateway_agent_listener(
    stack_probe: StackProbe, addresses: StackAddresses
) -> None:
    target = f"{addresses['gateway']['edge']}:{AGENT_PORT}"
    result = stack_probe.from_service("agent", [target])[target]
    assert result.connected, result


def test_agent_cannot_reach_any_protected_address(
    stack_probe: StackProbe, addresses: StackAddresses
) -> None:
    targets = _protected_targets(addresses, exclude="agent")
    results = stack_probe.from_service("agent", targets)
    assert _connected(results) == []
    assert all(result.outcome != "dns_failure" for result in results.values())


def test_agent_has_no_internet(stack_probe: StackProbe) -> None:
    results = stack_probe.from_service("agent", ["1.1.1.1:443", "example.com:443"])
    assert _connected(results) == []


def test_fetch_server_cannot_reach_any_protected_address(
    stack_probe: StackProbe, addresses: StackAddresses
) -> None:
    targets = _protected_targets(addresses, exclude="mcp-fetch")
    results = stack_probe.from_service("mcp-fetch", targets)
    assert _connected(results) == []
    assert all(result.outcome != "dns_failure" for result in results.values())


def _fetch_url(addresses: StackAddresses) -> str:
    return f"http://{addresses['mcp-fetch']['mcp_untrusted']}:8000/mcp"


def test_fetch_tool_works_for_a_public_url(
    stack_probe: StackProbe, addresses: StackAddresses
) -> None:
    """Positive control for the SSRF cases below (needs internet on the host)."""
    result = stack_probe.call_tool(
        "gateway", _fetch_url(addresses), "fetch", {"url": "https://example.com/"}
    )
    assert not result.is_error, result.text
    assert "Example Domain" in result.text


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        f"http://{OPERATOR_ADDRESS}/healthz",
        "http://localhost/",
        "http://[::ffff:127.0.0.1]/",
    ],
)
def test_fetch_tool_refuses_internal_destinations(
    stack_probe: StackProbe, addresses: StackAddresses, url: str
) -> None:
    result = stack_probe.call_tool("gateway", _fetch_url(addresses), "fetch", {"url": url})
    assert result.is_error
    assert "destination not allowed" in result.text


def test_fetch_tool_refuses_container_addresses(
    stack_probe: StackProbe, addresses: StackAddresses
) -> None:
    for ip in (addresses["postgres"]["mcp_backend"], addresses["ollama"]["llm_backend"]):
        result = stack_probe.call_tool(
            "gateway", _fetch_url(addresses), "fetch", {"url": f"http://{ip}/"}
        )
        assert result.is_error
        assert "destination not allowed" in result.text, ip


@pytest.mark.parametrize(
    "url",
    [
        "http://postgres:5432/",
        f"http://{OPERATOR_ADDRESS}:{OPERATOR_PORT}/healthz",
        "http://127.0.0.1:8000/mcp",
    ],
)
def test_fetch_tool_refuses_non_web_ports(
    stack_probe: StackProbe, addresses: StackAddresses, url: str
) -> None:
    result = stack_probe.call_tool("gateway", _fetch_url(addresses), "fetch", {"url": url})
    assert result.is_error
    assert "port 80 or 443" in result.text
