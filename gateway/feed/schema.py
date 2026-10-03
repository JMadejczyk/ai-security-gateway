"""Signature feed format and the compiled, immutable feed (SPEC "Historical attacks").

The feed is JSON: ``{version, signatures: [{id, source, pattern_type, pattern, severity,
channels}]}``. It is external input, so it is size-bounded, validated strictly (unknown fields
rejected) and every pattern is compiled up front: one bad entry rejects the whole feed and the
last valid one stays in effect (`gateway.feed.store`).

Pattern types:

- ``regex``: searched in every string of the payload (pre) or result (post) on the channels
  the signature names. Python ``regex`` syntax; write ``(?i)`` for case-insensitive.
- ``mcp_tool``: a regex searched in tool names and descriptions: the tool a ``tools/call``
  names, the tools an agent declares to the LLM, and the tools an MCP server advertises in
  ``tools/list`` (flagged tools are hidden from the listing).
- ``path_glob``: matched against the whole of every string argument of an MCP ``tools/call``
  (a leading ``file://`` is dropped). ``*`` matches within one path segment, ``**`` across
  segments; every other character is literal (no ``?``, ``[...]`` or ``{a,b}``) and matching
  is case-sensitive, so ``**/.env`` matches ``/app/.env`` and ``config/.env``.

ReDoS: every match runs with a per-pattern timeout (``regex``'s ``timeout=``) and every scan
has an overall time budget. A scan that hits either is *incomplete*, and the control fails
closed on it rather than letting a pathological input skip the remaining signatures.
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Final, Self

import regex
from pydantic import (
    Field,
    PrivateAttr,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from gateway.core.envelope import FrozenModel
from gateway.core.types import Channel

logger = logging.getLogger(__name__)

MAX_FEED_BYTES: Final = 256 * 1024
MAX_SIGNATURES: Final = 1000
MAX_PATTERN_CHARS: Final = 1024
MAX_GLOB_WILDCARDS: Final = 8
PATTERN_TIMEOUT_S: Final = 0.05  # one pattern against one string
SCAN_BUDGET_S: Final = 0.25  # every pattern against every string of one interaction
FILE_SCHEME: Final = "file://"

_GLOB_TOKEN: Final = regex.compile(r"\*\*|\*|[^*]+")

type SignatureId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")]
type FeedVersion = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")]


class FeedError(Exception):
    """A feed could not be read, parsed or validated. The message is operator-facing."""


class FeedUnavailableError(FeedError):
    """The feed could not be read: missing file, unreachable URL, HTTP error, redirect."""


class FeedInvalidError(FeedError):
    """The feed was read but is not acceptable: oversized, not JSON, or fails the schema."""


class PatternType(StrEnum):
    REGEX = "regex"
    MCP_TOOL = "mcp_tool"
    PATH_GLOB = "path_glob"


class Severity(StrEnum):
    """Informational: recorded with the feed and in logs, it does not scale risk."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def glob_to_regex(glob: str) -> str:
    """Translate a path glob (``*`` within a segment, ``**`` across) into a regex body."""
    parts: list[str] = []
    wildcards = 0
    for piece in _GLOB_TOKEN.findall(glob):
        if piece == "**":
            parts.append(".*")
        elif piece == "*":
            parts.append("[^/]*")
        else:
            parts.append(regex.escape(piece))
            continue
        wildcards += 1
    if wildcards > MAX_GLOB_WILDCARDS:
        msg = f"a path glob may use at most {MAX_GLOB_WILDCARDS} wildcards"
        raise ValueError(msg)
    return "".join(parts)


class Signature(FrozenModel):
    """One feed entry, compiled when validated."""

    id: SignatureId
    source: str = Field(min_length=1, max_length=200)
    pattern_type: PatternType
    pattern: str = Field(min_length=1, max_length=MAX_PATTERN_CHARS)
    severity: Severity
    channels: frozenset[Channel] = Field(min_length=1)

    _compiled: regex.Pattern[str] = PrivateAttr()

    @model_validator(mode="after")
    def _compile(self) -> Self:
        source = (
            glob_to_regex(self.pattern)
            if self.pattern_type is PatternType.PATH_GLOB
            else self.pattern
        )
        try:
            self._compiled = regex.compile(source)
        except regex.error as exc:
            msg = f"signature {self.id!r}: invalid pattern ({exc})"
            raise ValueError(msg) from None
        return self

    def applies(self, pattern_type: PatternType, channel: Channel) -> bool:
        return self.pattern_type is pattern_type and channel in self.channels

    def matches(self, text: str, *, timeout_s: float) -> bool:
        """Globs match the whole string, everything else anywhere in it.

        Raises `TimeoutError` when the match takes longer than ``timeout_s``.
        """
        if self.pattern_type is PatternType.PATH_GLOB:
            return self._compiled.fullmatch(text, timeout=timeout_s) is not None
        return self._compiled.search(text, timeout=timeout_s) is not None


class FeedDocument(FrozenModel):
    """The feed as published: a version of its own plus at most `MAX_SIGNATURES` entries."""

    version: FeedVersion
    signatures: tuple[Signature, ...] = Field(max_length=MAX_SIGNATURES)

    @field_validator("signatures")
    @classmethod
    def _unique_ids(cls, value: tuple[Signature, ...]) -> tuple[Signature, ...]:
        seen: set[str] = set()
        for signature in value:
            if signature.id in seen:
                msg = f"signature id {signature.id!r} appears twice"
                raise ValueError(msg)
            seen.add(signature.id)
        return value


@dataclass(frozen=True, slots=True)
class ScanResult:
    matched: tuple[Signature, ...] = ()
    incomplete: bool = False  # a pattern timed out or the scan budget ran out

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(signature.id for signature in self.matched)


class Scan:
    """One time-bounded scan of one interaction's strings against one feed and channel."""

    def __init__(
        self,
        feed: "SignatureFeed",
        channel: Channel,
        *,
        budget_s: float = SCAN_BUDGET_S,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        self._feed = feed
        self._channel = channel
        self._timer = timer
        self._deadline = timer() + budget_s
        self._hits: dict[str, Signature] = {}
        self._incomplete = False

    def check(self, pattern_type: PatternType, texts: Iterable[str]) -> None:
        """Test every signature of ``pattern_type`` on this channel against ``texts``."""
        candidates = [t for t in texts if t]
        if not candidates:
            return
        for signature in self._feed.signatures(pattern_type, self._channel):
            if signature.id in self._hits:
                continue
            for text in candidates:
                if not self._within_budget():
                    return
                if self._matches(signature, text):
                    self._hits[signature.id] = signature
                    break

    def result(self) -> ScanResult:
        return ScanResult(matched=tuple(self._hits.values()), incomplete=self._incomplete)

    def _within_budget(self) -> bool:
        if self._timer() < self._deadline:
            return True
        self._incomplete = True
        return False

    def _matches(self, signature: Signature, text: str) -> bool:
        timeout_s = min(PATTERN_TIMEOUT_S, self._deadline - self._timer())
        if timeout_s > 0:
            try:
                return signature.matches(text, timeout_s=timeout_s)
            except TimeoutError:
                logger.warning(
                    "signature %s timed out on a %d-char string", signature.id, len(text)
                )
        self._incomplete = True
        return False


class SignatureFeed:
    """One validated feed, immutable. `EMPTY_FEED` stands in when none is configured."""

    __slots__ = ("_digest", "_document")

    def __init__(self, document: FeedDocument | None = None) -> None:
        self._document = document
        canonical = json.dumps(
            document.model_dump(mode="json") if document is not None else None,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._digest = hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def version(self) -> str | None:
        return self._document.version if self._document is not None else None

    @property
    def digest(self) -> str:
        """SHA-256 of the canonical document: tells a changed feed from a re-read one."""
        return self._digest

    def __len__(self) -> int:
        return len(self._document.signatures) if self._document is not None else 0

    def signatures(self, pattern_type: PatternType, channel: Channel) -> tuple[Signature, ...]:
        if self._document is None:
            return ()
        return tuple(s for s in self._document.signatures if s.applies(pattern_type, channel))

    def scan(self, channel: Channel, *, budget_s: float = SCAN_BUDGET_S) -> Scan:
        return Scan(self, channel, budget_s=budget_s)


EMPTY_FEED: Final = SignatureFeed()


def parse_feed(data: bytes, *, max_bytes: int = MAX_FEED_BYTES) -> SignatureFeed:
    """Validate feed bytes into a compiled feed; raises `FeedInvalidError` (operator-facing)."""
    if len(data) > max_bytes:
        msg = f"feed is larger than {max_bytes} bytes"
        raise FeedInvalidError(msg)
    try:
        document: object = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        msg = f"feed is not valid UTF-8 JSON ({type(exc).__name__})"
        raise FeedInvalidError(msg) from None
    try:
        return SignatureFeed(FeedDocument.model_validate(document))
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors(include_url=False, include_input=False)
        ]
        msg = f"invalid feed: {'; '.join(problems[:10])}"
        raise FeedInvalidError(msg) from None
