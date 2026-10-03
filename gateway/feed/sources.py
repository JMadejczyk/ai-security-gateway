"""Where the feed bytes come from: a file, or an http(s) URL.

Both read at most ``max_bytes + 1`` bytes, so an oversized feed is refused without being
buffered whole. The HTTP source never follows redirects (a redirect could point the gateway at
an internal address the operator never named) and asks for an uncompressed body.

Timeouts: httpx's timeout applies to each connect and each read, so a server trickling one byte
just inside it would hold a fetch open indefinitely (headers included). The HTTP source
therefore also runs the whole exchange under one overall deadline (`HTTP_DEADLINE_S`) that
cancels the I/O when it expires.
"""

import asyncio
from abc import ABC, abstractmethod
from http import HTTPStatus
from pathlib import Path
from typing import Final, override
from urllib.parse import urlsplit

import httpx

from gateway.feed.schema import FeedInvalidError, FeedUnavailableError

HTTP_TIMEOUT_S: Final = 5.0  # each connect / read
HTTP_DEADLINE_S: Final = 15.0  # the whole fetch, headers to last byte
_HTTP_SCHEMES: Final = frozenset({"http", "https"})


class FeedSource(ABC):
    """Reads the raw feed document."""

    @abstractmethod
    async def fetch(self, max_bytes: int) -> bytes:
        """At most ``max_bytes + 1`` bytes; raises `FeedUnavailableError`."""

    @abstractmethod
    def describe(self) -> str:
        """Where the feed comes from, for logs."""


class FileFeedSource(FeedSource):
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    @override
    async def fetch(self, max_bytes: int) -> bytes:
        try:
            return await asyncio.to_thread(self._read, max_bytes)
        except OSError as exc:
            msg = f"cannot read feed {self._path}: {exc.strerror or type(exc).__name__}"
            raise FeedUnavailableError(msg) from None

    @override
    def describe(self) -> str:
        return str(self._path)

    def _read(self, max_bytes: int) -> bytes:
        with self._path.open("rb") as handle:
            return handle.read(max_bytes + 1)


class HttpFeedSource(FeedSource):
    def __init__(
        self,
        url: str,
        *,
        timeout_s: float = HTTP_TIMEOUT_S,
        deadline_s: float = HTTP_DEADLINE_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._deadline_s = deadline_s
        self._transport = transport

    @property
    def url(self) -> str:
        return self._url

    @override
    async def fetch(self, max_bytes: int) -> bytes:
        headers = {"accept": "application/json", "accept-encoding": "identity"}
        try:
            async with (
                asyncio.timeout(self._deadline_s),
                httpx.AsyncClient(
                    transport=self._transport,
                    timeout=self._timeout_s,
                    follow_redirects=False,
                ) as client,
                client.stream("GET", self._url, headers=headers) as response,
            ):
                return await _bounded_body(response, max_bytes)
        except TimeoutError:
            msg = f"feed {self._url} took longer than {self._deadline_s:g}s"
            raise FeedUnavailableError(msg) from None
        except httpx.HTTPError as exc:
            msg = f"feed {self._url} is unreachable ({type(exc).__name__})"
            raise FeedUnavailableError(msg) from None

    @override
    def describe(self) -> str:
        return self._url


async def _bounded_body(response: httpx.Response, max_bytes: int) -> bytes:
    if response.is_redirect:
        msg = f"feed answered with a redirect ({response.status_code}); redirects are refused"
        raise FeedUnavailableError(msg)
    if response.status_code != HTTPStatus.OK:
        msg = f"feed answered with HTTP {response.status_code}"
        raise FeedUnavailableError(msg)
    if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
        msg = "feed answered with a compressed body; only identity encoding is accepted"
        raise FeedUnavailableError(msg)
    body = bytearray()
    async for chunk in response.aiter_raw():
        body.extend(chunk)
        if len(body) > max_bytes:
            break
    return bytes(body[: max_bytes + 1])


def resolve_source(
    spec: str, *, base_dir: Path, transport: httpx.AsyncBaseTransport | None = None
) -> FeedSource:
    """An http(s) URL, or a file path; a relative path is resolved against ``base_dir``
    (the policy file's directory, so the same policy works in a checkout and in compose)."""
    scheme = urlsplit(spec).scheme.lower()
    if scheme in _HTTP_SCHEMES:
        return HttpFeedSource(spec, transport=transport)
    if "://" in spec:
        msg = f"feed {spec!r}: only http(s) URLs and file paths are supported"
        raise FeedInvalidError(msg)
    path = Path(spec)
    return FileFeedSource(path if path.is_absolute() else base_dir / path)
