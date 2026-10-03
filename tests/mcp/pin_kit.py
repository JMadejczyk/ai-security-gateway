"""Pin files for the in-process upstreams, captured the way ``acl pin`` captures them.

The MCP suites run with ``require_pin: true`` (as policy.yaml ships), so every server needs a
baseline. Capturing it from the test servers themselves (through `capture_pin`, the code behind
``GET /admin/mcp/{server}/tools``) keeps the baselines in step with `upstreams.py`.
"""

from pathlib import Path
from typing import Any

from gateway_testkit import INTERNAL_KEY, ROOT_POLICY, T0
from upstreams import RoutingTransport

from gateway.policy.loader import PolicyLoader
from gateway.proxies.mcp.admin import capture_pin
from gateway.proxies.mcp.pins import PinFile, ToolBaseline, tool_digest
from gateway.proxies.mcp.upstream import MCPConnector

OPERATOR = "root@demo"


async def capture_pins(transport: RoutingTransport) -> dict[str, PinFile]:
    """A baseline of every policy server, from what it advertises right now."""
    snapshot = PolicyLoader().load(ROOT_POLICY)
    connector = MCPConnector(INTERNAL_KEY.encode(), transport=transport)
    await connector.start()
    try:
        return {
            server: await capture_pin(connector, server, snapshot, principal=OPERATOR, now=T0)
            for server in snapshot.policy.upstreams.mcp
        }
    finally:
        await connector.aclose()
        transport.requests.clear()  # suites count upstream requests from a clean slate


def write_pins(pins_dir: Path, pins: dict[str, PinFile]) -> None:
    pins_dir.mkdir(parents=True, exist_ok=True)
    for server, pin in pins.items():
        (pins_dir / f"{server}.json").write_text(pin.to_json())


def pin_with_schema(pins_dir: Path, server: str, tool: str, schema: dict[str, Any]) -> None:
    """Re-pin one tool of ``server``'s baseline with another input schema (digest recomputed):
    the upstream now differs from its baseline, so ``tool_pinning`` alone would block it."""
    path = pins_dir / f"{server}.json"
    pin = PinFile.model_validate_json(path.read_text())

    def repinned(baseline: ToolBaseline) -> ToolBaseline:
        if baseline.name != tool:
            return baseline
        fields = baseline.model_dump(by_alias=True) | {"inputSchema": schema}
        fields["digest"] = tool_digest(
            fields["name"], fields["description"], schema, fields["annotations"]
        )
        return ToolBaseline.model_validate(fields)

    tools = [repinned(baseline) for baseline in pin.tools]
    path.write_text(pin.model_copy(update={"tools": tuple(tools)}).to_json())
