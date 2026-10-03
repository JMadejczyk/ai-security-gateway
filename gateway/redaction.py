"""Applying merged redaction spans to a payload or result (SPEC "Pipeline and interfaces").

Spans address strings by JSON pointer and code-point offsets. A span that does not land on
a string is a control bug; the call fails closed rather than releasing unredacted data.
"""

import copy
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


def _replace(container: object, token: str, value: str) -> None:
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
        if not isinstance(target, str):
            raise RedactionError
        masked = _masked(target, group)
        if path:
            _replace(container, last, masked)
        else:
            result = masked
    return result
