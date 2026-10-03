"""Applying merged redaction spans by JSON pointer; anything unresolvable fails closed."""

import pytest

from gateway.core.envelope import Span
from gateway.redaction import RedactionError, apply_redactions


def span(path: str, start: int, end: int, label: str = "PESEL") -> Span:
    return Span(path=path, start=start, end=end, label=label)


def test_spans_on_several_paths_and_the_original_is_untouched():
    doc = {"messages": [{"content": "id 123 and 456"}], "a/b": {"~k": "key sk-1"}}
    out = apply_redactions(
        doc,
        [
            span("/messages/0/content", 3, 6),
            span("/messages/0/content", 11, 14, "NIP"),
            span("/a~1b/~0k", 4, 8, "API_KEY"),
        ],
    )
    assert out == {
        "messages": [{"content": "id [REDACTED:PESEL] and [REDACTED:NIP]"}],
        "a/b": {"~k": "key [REDACTED:API_KEY]"},
    }
    assert doc["messages"][0]["content"] == "id 123 and 456"


def test_whole_document_string():
    assert apply_redactions("call 555", [span("", 5, 8, "PHONE")]) == "call [REDACTED:PHONE]"


def test_no_spans_returns_the_same_object():
    doc = {"x": "y"}
    assert apply_redactions(doc, []) is doc


@pytest.mark.parametrize(
    ("doc", "bad"),
    [
        ({"a": "text"}, span("/missing", 0, 1)),
        ({"a": ["text"]}, span("/a/5", 0, 1)),
        ({"a": 42}, span("/a", 0, 1)),
        ({"a": "abc"}, span("/a", 1, 9)),
    ],
    ids=["missing-key", "index-out-of-range", "not-a-string", "past-the-end"],
)
def test_unresolvable_spans_fail_closed(doc, bad):
    with pytest.raises(RedactionError):
        apply_redactions(doc, [bad])
