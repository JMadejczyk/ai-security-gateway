"""`mcp-fetch`: an untrusted web fetcher. Lives alone on `mcp_untrusted`.

Redirects are not followed: the gateway's egress control validates every destination, so a
redirect is reported back and the agent must fetch the new URL explicitly (through the gateway).
"""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class UnsupportedUrlError(ToolError):
    def __init__(self) -> None:
        super().__init__("only absolute http(s) URLs are supported")


class RedirectNotFollowedError(ToolError):
    def __init__(self, status: int, location: str) -> None:
        super().__init__(f"HTTP {status} redirect to {location!r}; not followed")


class UpstreamHTTPError(ToolError):
    def __init__(self, status: int, snippet: str) -> None:
        super().__init__(f"HTTP {status}: {snippet}")


class FetchSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_bytes: int = Field(default=256 * 1024, gt=0)
    timeout_s: float = Field(default=10.0, gt=0)


class Fetcher:
    def __init__(self, settings: FetchSettings) -> None:
        self._settings = settings

    async def fetch(self, url: str) -> str:
        """Fetch a web page with HTTP GET and return its body as text (size-capped)."""
        parts = urlsplit(url)
        if parts.scheme not in _ALLOWED_SCHEMES or not parts.hostname:
            raise UnsupportedUrlError
        timeout = httpx.Timeout(self._settings.timeout_s)
        async with (
            httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client,
            client.stream("GET", url) as response,
        ):
            if response.status_code in _REDIRECT_STATUSES:
                location = response.headers.get("location", "")
                raise RedirectNotFollowedError(response.status_code, location)
            body = await self._read_capped(response)
        text = body.decode(response.encoding or "utf-8", errors="replace")
        if response.is_error:
            raise UpstreamHTTPError(response.status_code, text[:500])
        return text

    async def _read_capped(self, response: httpx.Response) -> bytes:
        limit = self._settings.max_bytes
        chunks: list[bytes] = []
        received = 0
        async for chunk in response.aiter_bytes():
            chunks.append(chunk[: limit - received])
            received += len(chunk)
            if received >= limit:
                break
        return b"".join(chunks)


def build_fetch_server(settings: FetchSettings | None = None) -> MCPServer:
    fetcher = Fetcher(settings or FetchSettings())
    server = MCPServer(name="mcp-fetch")
    server.tool(
        name="fetch",
        annotations=ToolAnnotations(
            title="Fetch URL",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
    )(fetcher.fetch)
    return server
