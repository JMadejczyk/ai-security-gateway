"""Applying merged redaction spans to a payload or result (SPEC "Pipeline and interfaces").

Spans address strings by JSON pointer and code-point offsets. A span with ``embedded`` lands
in a JSON document serialized into the string at its path: that string is parsed, redacted
and serialized again. A span on a number replaces the whole number with the mask (digits have
no partial redaction). A span that lands anywhere else is a control bug; the call fails closed
rather than releasing unredacted data.
"""

import copy
import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, cast

from gateway.core.envelope import Span
from gateway.errors import RejectionError


class RedactionError(RejectionError):
    def __init__(self) -> None:
        super().__init__("redaction_failed", "a redaction could not be applied")


def mask(label: str) -> str:
    return f"[REDACTED:{label}]"


def _tokens(pointer: str) -> list[str]:
    if not pointer:
        return []
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _child(container: object, token: str) -> object:
    if isinstance(container, dict):
        return cast("dict[str, Any]", container)[token]
    if isinstance(container, list) and token.isdigit():
        return cast("list[Any]", container)[int(token)]
    raise KeyError(token)


def _replace(container: object, token: str, value: object) -> None:
    if isinstance(container, dict):
        cast("dict[str, Any]", container)[token] = value
    else:
        cast("list[Any]", container)[int(token)] = value


def _masked(text: str, spans: list[Span]) -> str:
    for span in sorted(spans, key=lambda s: s.start, reverse=True):  # unioned: no overlaps
        if span.end > len(text):
            raise RedactionError
        text = text[: span.start] + mask(span.label) + text[span.end :]
    return text


def _labels(spans: Sequence[Span]) -> str:
    return "+".join(sorted({label for span in spans for label in span.label.split("+")}))


def _redacted(target: object, spans: list[Span]) -> object:
    """``target`` (the value at one path) with every span applied."""
    embedded = [span for span in spans if span.embedded is not None]
    if embedded:
        if len(embedded) != len(spans) or not isinstance(target, str):
            raise RedactionError  # one string is either text or JSON text, never both
        try:
            document: object = json.loads(target)
        except ValueError:
            raise RedactionError from None
        inner = [
            span.model_copy(update={"path": span.embedded, "embedded": None}) for span in spans
        ]
        return json.dumps(apply_redactions(document, inner), ensure_ascii=False)
    if isinstance(target, str):
        return _masked(target, spans)
    if isinstance(target, int | float) and not isinstance(target, bool):
        return mask(_labels(spans))
    raise RedactionError


def apply_redactions(document: object, spans: Sequence[Span]) -> object:
    """A copy of ``document`` with every span replaced by ``[REDACTED:<label>]``."""
    if not spans:
        return document
    by_path: defaultdict[str, list[Span]] = defaultdict(list)
    for span in spans:
        by_path[span.path].append(span)
    result = copy.deepcopy(document)
    for path, group in by_path.items():
        *parents, last = ["", *_tokens(path)]
        try:
            container: object = result
            for token in parents[1:]:
                container = _child(container, token)
            target = _child(container, last) if path else result
        except (KeyError, IndexError):
            raise RedactionError from None
        masked = _redacted(target, group)
        if path:
            _replace(container, last, masked)
        else:
            result = masked
    return result
