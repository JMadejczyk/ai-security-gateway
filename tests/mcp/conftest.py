"""Fixtures for the MCP suites: real upstreams behind a routing transport."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from gateway_testkit import running_gateway
from mcp_harness import MCPStack
from upstreams import running_upstreams


@pytest.fixture
async def stack(tmp_path: Path) -> AsyncIterator[MCPStack]:
    async with (
        running_upstreams() as (transport, log),
        running_gateway(tmp_path, transport=transport) as gateway,
    ):
        yield MCPStack(gateway, transport, log)
