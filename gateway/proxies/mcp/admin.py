"""What the gateway sees upstream, for operators approving a baseline (``acl pin <server>``).

``GET /admin/mcp/{server}/tools`` (operator listener, admin role) lists the server's tools in a
short-lived upstream session of its own, opened as the operator (internal servers get the
operator's ``X-ACL-Principal``), and answers a candidate `PinFile`: the listing with every
tool's digest. Nothing is written: the operator reviews the diff against the committed pin file
and writes it with ``acl pin <server> --write`` on the host (the gateway mounts ``pins/``
read-only).
"""

from datetime import datetime

from pydantic import ValidationError

from gateway.errors import RejectionError
from gateway.policy.loader import PolicySnapshot
from gateway.proxies.mcp.pins import PinFile
from gateway.proxies.mcp.sessions import UnknownServerError
from gateway.proxies.mcp.upstream import MCPConnector


class UnpinnableListingError(RejectionError):
    """The upstream's listing is ambiguous or malformed (a duplicate tool name, a description
    that is not a string): it cannot be approved as a baseline."""

    status_code = 502

    def __init__(self) -> None:
        super().__init__("tool_listing_unpinnable", "the upstream tool listing cannot be pinned")


async def capture_pin(
    connector: MCPConnector,
    server: str,
    snapshot: PolicySnapshot,
    *,
    principal: str,
    now: datetime,
) -> PinFile:
    """A candidate baseline from a fresh ``tools/list`` of ``server``. Raises
    `UnknownServerError`, `UpstreamError` or `UnpinnableListingError`."""
    config = snapshot.policy.upstreams.mcp.get(server)
    if config is None:
        raise UnknownServerError
    upstream = connector.open(server, config, principal)
    try:
        tools = await upstream.list_tools(snapshot)
    finally:
        await upstream.aclose()
    try:
        return PinFile.capture(server, tools, now)
    except ValidationError:
        raise UnpinnableListingError from None
