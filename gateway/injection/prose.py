"""Turning what an interaction carries into the text the injection classifier should see.

The classifier was trained on natural language. These pure functions decide what of a
segment is language and present it the way a reader (or a model) reads it:

- `readable_blocks`: an HTML document as its readable text, one block per block element,
  comment or text attribute. Tags, ``<script>`` and ``<style>`` bodies go; text nodes,
  comments (a favourite hiding place) and the text attributes ``alt``/``title``/
  ``aria-label``/``placeholder`` stay, hidden elements included. The raw markup is still
  what ``signatures`` scans (it matches markup such as ``<IMPORTANT>``). Script bodies are
  not classified: an instruction placed in inline JavaScript is left to ``signatures``.
- `json_fragments`: a text that is a JSON object or array (a SQL tool's rows as JSON text),
  as its keys and string values: structured data is classified field by field, not as one
  long run of punctuation.
- `neutral_markers`: the gateway's own redaction markers (``[REDACTED:PL_PESEL]``) as
  ``***``. The model reads the marker as an instruction-like token ("Customer
  [REDACTED:EMAIL_ADDRESS] has 3 orders." scored 0.97, with ``***`` 0.00; ``(redacted)``
  and the bare label scored high too), and it is gateway output, not attacker text. The
  same goes for the gateway's own reason codes and control ids
  (`gateway.injection.reason_codes`): a refused tool call's ``prompt_injection_detected``
  (1.0) or the model quoting it (0.87, with ``***`` 0.06), re-sent as history, would
  otherwise block every later turn. Only that fixed set of whole tokens is replaced.
- `looks_like_prose`: at least `MIN_WORDS` words of two or more letters and `MIN_LETTERS`
  letters in them. A word is a whitespace-separated token made of letters (with inner
  hyphens or apostrophes), trailing punctuation allowed: numbers, ISO dates, UUIDs, emails,
  URLs, ``snake_case`` identifiers and single tokens are not prose. Calibrated on the corpus
  in ``tests/injection/corpus.py``: its shortest injection has 9 words.
"""

import json
import re
from collections.abc import Iterator, Mapping
from html.parser import HTMLParser
from typing import Any, Final, NamedTuple, cast, override

from gateway.injection.reason_codes import GATEWAY_REASON_CODES

MIN_WORDS: Final = 3
MIN_LETTERS: Final = 10
REDACTED_PLACEHOLDER: Final = "***"  # measured: "(redacted)" and the label still score high
MAX_JSON_CHARS: Final = 4 << 20  # texts above this are not parsed as JSON

# Quotes and brackets around a token, and punctuation after it (escapes: typographic quotes).
_LEADING_MARKS: Final = "\"'([{<\u00ab\u201e\u201c\u2018*_`"
_TRAILING_MARKS: Final = "\"')]}>\u00bb\u201d\u2019.,;:!?*_`\u2026"
_WORD: Final = re.compile(r"[^\W\d_]{2,}(?:[-'\u2019][^\W\d_]+)*")
_MARKER: Final = re.compile(r"\[REDACTED:[A-Z0-9_+]{1,200}\]")
_REASON_CODE: Final = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    + "|".join(re.escape(code) for code in sorted(GATEWAY_REASON_CODES, key=len, reverse=True))
    + r")(?![A-Za-z0-9_])"
)
_HTML_HINT: Final = re.compile(r"<(?:!doctype\s|html[\s>]|body[\s>]|head[\s>])", re.IGNORECASE)
_TAG: Final = re.compile(r"</?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^<>]*)?/?>")
_MIN_TAGS: Final = 3  # without a doctype/html/body/head tag, this many tags make it HTML
_DROPPED: Final = frozenset({"script", "style", "template", "noscript", "svg"})
_INLINE: Final = frozenset(
    {"a", "abbr", "b", "bdi", "bdo", "cite", "code", "data", "dfn", "em", "i", "kbd", "mark",
     "q", "s", "samp", "small", "span", "strong", "sub", "sup", "time", "u", "var", "wbr"}
)  # fmt: skip
_TEXT_ATTRIBUTES: Final = frozenset({"alt", "title", "aria-label", "placeholder"})
_SPACE: Final = re.compile(r"\s+")


def words(text: str) -> list[str]:
    """The words of ``text`` (see the module docstring)."""
    found: list[str] = []
    for token in text.split():
        core = token.lstrip(_LEADING_MARKS).rstrip(_TRAILING_MARKS)
        if _WORD.fullmatch(core):
            found.append(core)
    return found


def looks_like_prose(text: str) -> bool:
    """At least `MIN_WORDS` words and `MIN_LETTERS` letters in them."""
    found = words(text)
    return len(found) >= MIN_WORDS and sum(len(w) for w in found) >= MIN_LETTERS


def neutral_markers(text: str) -> str:
    """``text`` with every ``[REDACTED:LABEL]`` marker and every gateway reason code or control
    id (whole tokens of `GATEWAY_REASON_CODES`) replaced by `REDACTED_PLACEHOLDER`."""
    if "[REDACTED:" in text:
        text = _MARKER.sub(REDACTED_PLACEHOLDER, text)
    return _REASON_CODE.sub(REDACTED_PLACEHOLDER, text) if "_" in text else text


def looks_like_html(text: str) -> bool:
    if "<" not in text:
        return False
    if _HTML_HINT.search(text):
        return True
    return sum(1 for _ in zip(range(_MIN_TAGS), _TAG.finditer(text), strict=False)) >= _MIN_TAGS


class _Readable(HTMLParser):
    """Collects the readable blocks of a page: block elements, comments and text attributes
    each become their own block; inline elements (``<a>``, ``<b>``, ``<span>``...) do not."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._current: list[str] = []
        self._dropped = 0

    def flush(self) -> None:
        text = _SPACE.sub(" ", "".join(self._current)).strip()
        if text:
            self.blocks.append(text)
        self._current = []

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DROPPED:
            self._dropped += 1
        if tag not in _INLINE:
            self.flush()
        for name, value in attrs:
            if name in _TEXT_ATTRIBUTES and value and value.strip():
                self.blocks.append(_SPACE.sub(" ", value).strip())

    @override
    def handle_endtag(self, tag: str) -> None:
        if tag in _DROPPED and self._dropped:
            self._dropped -= 1
        if tag not in _INLINE:
            self.flush()

    @override
    def handle_data(self, data: str) -> None:
        if not self._dropped:
            self._current.append(data)

    @override
    def handle_comment(self, data: str) -> None:
        self.flush()
        self._current.append(data)
        self.flush()


def readable_blocks(html: str) -> list[str]:
    """The readable text of an HTML document, block by block (see the module docstring).

    Blocks, not one string: a hidden instruction is classified on its own, not diluted by
    the visible page around it (one hidden Polish paragraph scored 0.998 alone and 0.0004
    inside its page's whole text)."""
    parser = _Readable()
    parser.feed(html)
    parser.close()
    parser.flush()
    return parser.blocks


def readable_text(html: str) -> str:
    """The readable text of an HTML document as one string."""
    return " ".join(readable_blocks(html))


def json_fragments(text: str) -> list[str] | None:
    """Keys and string values of ``text`` when it is a JSON object or array, else None."""
    stripped = text.strip()
    if stripped[:1] not in {"{", "["} or len(stripped) > MAX_JSON_CHARS:
        return None
    try:
        document: object = json.loads(stripped)
    except ValueError:
        return None
    if not isinstance(document, dict | list):
        return None
    return [fragment for fragment in _json_strings(cast("object", document)) if fragment]


def _json_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in cast("Mapping[str, Any]", value).items():
            yield key
            yield from _json_strings(child)
    elif isinstance(value, list):
        for child in cast("list[Any]", value):
            yield from _json_strings(child)


class Fragments(NamedTuple):
    """What one segment contributes, in reading order, and how its pieces are joined when
    the pieces are read as one text."""

    pieces: list[str]
    separator: str


def fragments(text: str) -> Fragments:
    """What one segment contributes to classification (see the module docstring). An HTML
    page's blocks are joined with a line break, as a reader sees them; a JSON text's fields
    as written, so a run of keys never reads as a sentence."""
    parsed = json_fragments(text)
    if parsed is not None:
        return Fragments([neutral_markers(piece) for piece in parsed], "")
    if looks_like_html(text):
        return Fragments([neutral_markers(block) for block in readable_blocks(text)], "\n")
    return Fragments([neutral_markers(text)], "")
