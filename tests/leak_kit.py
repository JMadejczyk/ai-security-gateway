"""No-leak assertions that look only where a payload could land.

A plain ``value not in raw_text`` over an audit line or a response is flaky: measurements
(``"cost": 0.04904111847``) contain arbitrary digit runs, so a short numeric value such as
``4111`` matches by chance. `leaked` parses the text (one JSON document, JSON lines, or SSE
``data:`` events) and searches every string (keys too) for ``value``, and every integer for
an exact match (a number can carry a PESEL or a card number whole). Floats are measurements
(latency, risk, cost) and are skipped. Text that does not parse is searched as written.
"""

import json
from collections.abc import Iterator
from typing import Any, cast


def _documents(text: str) -> list[Any] | None:
    """The JSON documents in ``text``, or None when it is not JSON, JSON lines or SSE."""
    try:
        return [json.loads(text)]
    except ValueError:
        pass
    documents: list[Any] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line in {"data: [DONE]", "[DONE]"}:
            continue
        try:
            documents.append(json.loads(line.removeprefix("data:").strip()))
        except ValueError:
            return None
    return documents


def _leaves(value: object) -> Iterator[str | int]:
    if isinstance(value, bool) or value is None or isinstance(value, float):
        return
    if isinstance(value, str | int):
        yield value
    elif isinstance(value, dict):
        for key, child in cast("dict[str, Any]", value).items():
            yield key
            yield from _leaves(child)
    elif isinstance(value, list):
        for child in cast("list[Any]", value):
            yield from _leaves(child)


def leaked(value: str, text: str) -> bool:
    """True when ``value`` appears in a string of ``text`` or is the whole of an integer."""
    documents = _documents(text)
    if documents is None:
        return value in text
    for leaf in _leaves(documents):
        if isinstance(leaf, str) and value in leaf:
            return True
        if isinstance(leaf, int) and str(leaf) == value:
            return True
    return False
