"""``tools/list`` screens: controls that hide advertised tools before the agent sees them.

``tools/list`` does not run through the pipeline (it names no resource), so the controls that
judge tool definitions (``signatures``, ``tool_poisoning``) screen the listing instead. The
MCP proxy runs every screen concurrently on the listing ``tool_pinning`` verified and hides
the union of what they flag. Each screen keeps its own deadline and fails closed.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from gateway.controls.signatures import SignaturesControl
from gateway.policy.loader import PolicySnapshot
from gateway.proxies.mcp import wire

type ToolScreen = Callable[
    [str, Sequence[wire.ToolDefinition], PolicySnapshot], Awaitable[set[str]]
]
"""Given the server name and its tools as listed (duplicates kept, advertised order), the names
to hide from ``tools/list``."""


def described(tools: Sequence[wire.ToolDefinition]) -> list[tuple[str, str | None]]:
    """``(name, description)`` per entry, in order: a duplicate name must not hide the first."""
    advertised: list[tuple[str, str | None]] = []
    for tool in tools:
        description = (tool.model_extra or {}).get("description")
        advertised.append((tool.name, description if isinstance(description, str) else None))
    return advertised


def signatures_screen(control: SignaturesControl) -> ToolScreen:
    """The ``signatures`` listing screen, which matches names and descriptions only."""

    async def screen(
        server: str, tools: Sequence[wire.ToolDefinition], snapshot: PolicySnapshot
    ) -> set[str]:
        del server  # the feed's mcp_tool patterns are not per server
        return await control.screen_listing(described(tools), snapshot)

    return screen


async def run_screens(
    screens: Sequence[ToolScreen],
    server: str,
    tools: Sequence[wire.ToolDefinition],
    snapshot: PolicySnapshot,
) -> set[str]:
    """Every screen at once; the names any of them flags."""
    flagged = await asyncio.gather(*(screen(server, tools, snapshot) for screen in screens))
    return set[str]().union(*flagged)
