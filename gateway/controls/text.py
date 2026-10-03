"""The content of an interaction, addressed by JSON pointer, for content-scanning controls.

Content controls (``pii``, ``secrets``, ``signatures``, later ``prompt_injection``) scan what
an interaction carries and answer with spans: a JSON pointer into the document the pipeline
redacts at that stage, plus code-point offsets in the string it names (`gateway.redaction`).

What is scanned is everything that crosses the gateway, not a list of known fields, so a new
or legacy field (``tools`` definitions, ``function_call``, ``reasoning_details``, an MCP
``resource``) cannot carry data past the controls:

- llm, pre: every string in the chat request (``payload``), ``tools`` included;
- llm, post: every string in the chat completion (``result``); the SSE re-emission is built
  from that same (redacted) document;
- mcp, pre: every string and number under ``/arguments`` of ``{"name", "arguments"}``;
- mcp, post: every string and number in the ``CallToolResult``, ``isError`` results too.

Three encodings are opened before scanning, because a regex over the encoded form misses
what the encoding hides:

- ``arguments`` strings in LLM documents are JSON text (``\\u0041KIA...`` is ``AKIA...``):
  they are parsed and their leaves yielded with ``embedded`` pointers, so a span is applied
  to the decoded value and the JSON serialized again. Arguments that do not parse are
  scanned as written, unless they contain a backslash: then an escape may hide content the
  scan cannot see, and the segment is `SegmentKind.UNSCANNABLE` (``secrets`` blocks it).
  Malformed JSON with an escape in it is rare, and a block is the only safe answer to it.
- base64 ``blob`` of an MCP resource, ``data`` of an MCP content item (next to its
  ``mimeType``) and
  ``data:`` URLs are decoded when they are text, judged by the bytes and not by the claimed
  type (a ``text/plain`` labelled ``image/png`` is still text): valid UTF-8 without NUL.
  Decoded text is `SegmentKind.OPAQUE`: a mask cannot be written back into base64, so a
  detection there blocks. Binary media (longer than a credential could be) is not scanned:
  regexes over image bytes find noise, not text. Text larger than `MAX_DECODED_BYTES` is
  UNSCANNABLE rather than skipped.
- numbers (MCP only, and inside decoded arguments) are rendered as digits: a PESEL sent as
  ``44051401359`` is still a PESEL. A span on a number replaces the whole value.

Every segment carries ``key``, the nearest object key above it, for key-sensitive detection
(``{"password": "..."}``). Shapes the gateway does not recognize yield nothing.
"""

import base64
import binascii
import codecs
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final, cast
from urllib.parse import unquote_to_bytes

from gateway.core.envelope import Interaction, Span
from gateway.core.types import Channel, Stage

MAX_DECODED_BYTES: Final = 1 << 20  # decoded text above this is not scanned: UNSCANNABLE
_PROBE_BYTES: Final = 4096  # bytes decoded to tell text from binary in an oversized payload
_RAW_SCAN_CHARS: Final = 4096  # binary payloads up to this length are still scanned as written
_ARGUMENTS: Final = "arguments"
_BLOB: Final = "blob"  # an MCP resource's base64 payload
_DATA: Final = "data"  # an MCP image/audio item's base64 payload, next to its mimeType
_DATA_URL: Final = re.compile(r"data:([^,]{0,256}),", re.IGNORECASE)
# Protocol fields (identifiers and enums at known places of the chat and MCP documents), not
# text a person, model or tool wrote. They are scanned like any value, but left out when
# consecutive segments are joined into one text: ``role: "user"`` between two messages would
# otherwise break a value split across them. Matched by pointer, never by key alone: a ``name``
# or ``type`` inside tool arguments, ``structuredContent`` or a resource link is data.
_PROTOCOL_POINTER: Final = re.compile(
    r"/(?:model|object|id|system_fingerprint|service_tier|reasoning_effort|tool_choice"
    r"|response_format/type)"
    r"|/(?:messages/\d+|choices/\d+/(?:message|delta))"
    r"/(?:role|tool_call_id|content/\d+/type|tool_calls/\d+/(?:id|type))"
    r"|/choices/\d+/finish_reason"
    r"|/tools/\d+/type"
    r"|/content/\d+/(?:type|mimeType|resource/mimeType|annotations/audience/\d+)"
)


class SegmentKind(StrEnum):
    """How a detection in a segment can be acted on."""

    TEXT = "text"  # a string: spans redact by offset
    NUMBER = "number"  # a number rendered as digits: a span replaces the whole value
    OPAQUE = "opaque"  # text decoded from base64 or a data URL: no mask fits, a hit blocks
    UNSCANNABLE = "unscannable"  # could not be decoded reliably: content controls fail closed


@dataclass(frozen=True, slots=True)
class TextSegment:
    """One scannable value of the stage document and where it lives."""

    pointer: str  # JSON pointer (RFC 6901) into the document; "" is the document itself
    text: str  # what is scanned: the string, the decoded text or the rendered number
    key: str | None = None  # nearest enclosing object key, if any
    embedded: str | None = None  # pointer into the JSON text at `pointer` (tool arguments)
    kind: SegmentKind = SegmentKind.TEXT

    @property
    def redactable(self) -> bool:
        return self.kind in {SegmentKind.TEXT, SegmentKind.NUMBER}

    @property
    def joinable(self) -> bool:
        """Part of the running text a split value is rebuilt from (not a protocol field)."""
        return self.embedded is not None or _PROTOCOL_POINTER.fullmatch(self.pointer) is None

    def span(self, start: int, end: int, label: str) -> Span:
        """A span over ``text[start:end]`` (a number is always replaced whole)."""
        if self.kind is SegmentKind.NUMBER:
            start, end = 0, len(self.text)
        return Span(path=self.pointer, embedded=self.embedded, start=start, end=end, label=label)


def pointer(*tokens: str | int) -> str:
    """RFC 6901 pointer from reference tokens: ``pointer("a/b", 0)`` is ``"/a~1b/0"``."""
    return "".join("/" + str(t).replace("~", "~0").replace("/", "~1") for t in tokens)


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    return cast("Mapping[str, Any]", value) if isinstance(value, Mapping) else None


def _as_list(value: object) -> Sequence[Any]:
    return cast("list[Any]", value) if isinstance(value, list) else ()


def string_leaves(value: object, base: str = "") -> Iterator[TextSegment]:
    """Every non-empty string inside a JSON value, depth first, keys in document order."""
    if isinstance(value, str):
        if value:
            yield TextSegment(base, value)
    elif (mapping := _as_mapping(value)) is not None:
        for key, child in mapping.items():
            yield from string_leaves(child, base + pointer(key))
    else:
        for index, child in enumerate(_as_list(value)):
            yield from string_leaves(child, base + pointer(index))


_EXACT_FLOAT_INTEGERS: Final = 2**53


def render_number(value: float) -> str:
    """Digits as a person would write the number: ``44051401359``, not ``4.4051401359e10``."""
    if isinstance(value, float) and value.is_integer() and abs(value) < _EXACT_FLOAT_INTEGERS:
        return str(int(value))
    return str(value)


def _text_of(raw: bytes, *, partial: bool = False) -> str | None:
    """``raw`` as text when it is UTF-8 without NUL; None for binary. ``partial``: ``raw`` is
    a prefix, so a character cut at its end is not an error."""
    if b"\x00" in raw:
        return None
    try:
        return codecs.getincrementaldecoder("utf-8")().decode(raw, final=not partial)
    except UnicodeDecodeError:
        return None


def decode_blob(encoded: str, *, is_base64: bool = True) -> tuple[SegmentKind, str] | None:
    """What to scan for an encoded payload.

    ``(OPAQUE, text)`` when it decodes to text; ``(UNSCANNABLE, encoded)`` for text larger
    than `MAX_DECODED_BYTES`; ``(TEXT, encoded)`` when it is not valid encoding at all, or is
    binary but short (an ``AKIA...`` key is valid base64 too: what is short enough to be a
    credential is scanned as written); None for binary media, which is left alone.
    """
    oversized = is_base64 and len(encoded) // 4 * 3 > MAX_DECODED_BYTES
    try:
        if not is_base64:
            raw = unquote_to_bytes(encoded)
        else:  # an oversized payload: decode a prefix, enough to tell text from binary
            raw = base64.b64decode(
                encoded[: _PROBE_BYTES // 3 * 4] if oversized else encoded, validate=True
            )
    except (binascii.Error, ValueError):
        return SegmentKind.TEXT, encoded
    if oversized or len(raw) > MAX_DECODED_BYTES:
        is_text = _text_of(raw[:_PROBE_BYTES], partial=True) is not None
        return (SegmentKind.UNSCANNABLE, encoded) if is_text else None
    text = _text_of(raw)
    if text is not None:
        return SegmentKind.OPAQUE, text
    return (SegmentKind.TEXT, encoded) if len(encoded) <= _RAW_SCAN_CHARS else None


@dataclass(frozen=True, slots=True)
class _LeafWalker:
    """Depth-first walk over a JSON value yielding every scannable leaf."""

    numbers: bool  # yield numbers too (tool data), not just strings
    decode_arguments: bool  # parse ``arguments`` strings as JSON (LLM tool calls)

    def walk(
        self, value: object, base: str, key: str | None, parent: Mapping[str, Any] | None
    ) -> Iterator[TextSegment]:
        if isinstance(value, str):
            yield from self._string(value, base, key, parent)
        elif isinstance(value, bool) or value is None:
            return
        elif isinstance(value, int | float):
            if self.numbers:
                yield TextSegment(base, render_number(value), key, kind=SegmentKind.NUMBER)
        elif (mapping := _as_mapping(value)) is not None:
            for child_key, child in mapping.items():
                yield from self.walk(child, base + pointer(child_key), child_key, mapping)
        else:
            for index, child in enumerate(_as_list(value)):  # items inherit the list's key
                yield from self.walk(child, base + pointer(index), key, parent)

    def _string(
        self, value: str, base: str, key: str | None, parent: Mapping[str, Any] | None
    ) -> Iterator[TextSegment]:
        if not value:
            return
        if self.decode_arguments and key == _ARGUMENTS:
            yield from self._arguments(value, base)
            return
        decoded: tuple[SegmentKind, str] | None = SegmentKind.TEXT, value
        if key == _BLOB or (key == _DATA and parent is not None and "mimeType" in parent):
            decoded = decode_blob(value)
        elif (url := _DATA_URL.match(value)) is not None:
            is_base64 = url[1].lower().endswith(";base64")
            decoded = decode_blob(value[url.end() :], is_base64=is_base64)
            if decoded is not None and decoded[0] is SegmentKind.TEXT:
                decoded = SegmentKind.TEXT, value  # scanned as written: offsets into the URL
        if decoded is not None and decoded[1]:
            yield TextSegment(base, decoded[1], key, kind=decoded[0])

    def _arguments(self, value: str, base: str) -> Iterator[TextSegment]:
        try:
            document: object = json.loads(value)
        except ValueError:
            kind = SegmentKind.UNSCANNABLE if "\\" in value else SegmentKind.TEXT
            yield TextSegment(base, value, _ARGUMENTS, kind=kind)
            return
        inner = _LeafWalker(numbers=True, decode_arguments=False)
        for leaf in inner.walk(document, "", _ARGUMENTS, None):
            yield replace(leaf, pointer=base, embedded=leaf.pointer)


_LLM: Final = _LeafWalker(numbers=False, decode_arguments=True)  # numbers there are parameters
_MCP: Final = _LeafWalker(numbers=True, decode_arguments=False)


class TextExtractor:
    """Yields the scannable values of an interaction at one stage (see the module docstring)."""

    def segments(self, interaction: Interaction, stage: Stage) -> list[TextSegment]:
        """The values controls scan, each with its pointer into `document`, in document order."""
        document = self.document(interaction, stage)
        match interaction.channel, stage:
            case Channel.LLM, _:
                if _as_mapping(document) is None:
                    return []
                return list(_LLM.walk(document, "", None, None))
            case Channel.MCP, Stage.PRE:
                arguments = (_as_mapping(document) or {}).get(_ARGUMENTS)
                return list(_MCP.walk(arguments, pointer(_ARGUMENTS), None, None))
            case Channel.MCP, Stage.POST:
                if _as_mapping(document) is None:
                    return []
                return list(_MCP.walk(document, "", None, None))
            case _:
                return []

    @staticmethod
    def document(interaction: Interaction, stage: Stage) -> object:
        """The value spans address: the payload before the upstream, its result after."""
        return interaction.payload if stage is Stage.PRE else interaction.result
