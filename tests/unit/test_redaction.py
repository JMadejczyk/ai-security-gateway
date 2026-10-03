"""Applying merged redaction spans by JSON pointer; anything unresolvable fails closed."""

import json

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
        ({"a": {"b": "x"}}, span("/a", 0, 1)),
        ({"a": True}, span("/a", 0, 1)),
        ({"a": "abc"}, span("/a", 1, 9)),
        ({"a": "not json"}, Span(path="/a", embedded="/x", start=0, end=1, label="K")),
        ({"a": '{"x": "v"}'}, Span(path="/a", embedded="/y", start=0, end=1, label="K")),
        ({"a": 7}, Span(path="/a", embedded="", start=0, end=1, label="K")),
    ],
    ids=[
        "missing-key",
        "index-out-of-range",
        "object",
        "boolean",
        "past-the-end",
        "embedded-in-non-json",
        "embedded-missing",
        "embedded-in-a-number",
    ],
)
def test_unresolvable_spans_fail_closed(doc, bad):
    with pytest.raises(RedactionError):
        apply_redactions(doc, [bad])


def test_a_number_is_replaced_whole():
    """Digits have no partial mask: a span on a number replaces it with the mask string."""
    doc = {"pesel": 44051401359, "n": 2.5, "keep": 7}
    out = apply_redactions(
        doc, [span("/pesel", 0, 11, "PL_PESEL"), span("/n", 0, 1, "A"), span("/n", 0, 3, "B")]
    )
    assert out == {"pesel": "[REDACTED:PL_PESEL]", "n": "[REDACTED:A+B]", "keep": 7}


def test_embedded_json_is_decoded_redacted_and_serialized_again():
    """A span inside JSON text (a tool call's arguments) lands on the decoded value."""
    arguments = '{"key": "\\u0041KIA-secret", "n": 44051401359, "keep": "x"}'
    doc = {"function": {"arguments": arguments}}
    out = apply_redactions(
        doc,
        [
            Span(path="/function/arguments", embedded="/key", start=0, end=4, label="API_KEY"),
            Span(path="/function/arguments", embedded="/n", start=0, end=11, label="PL_PESEL"),
        ],
    )
    assert json.loads(out["function"]["arguments"]) == {  # type: ignore[index]
        "key": "[REDACTED:API_KEY]-secret",
        "n": "[REDACTED:PL_PESEL]",
        "keep": "x",
    }


def test_text_and_embedded_spans_on_one_string_fail_closed():
    doc = {"a": '{"x": "secret"}'}
    with pytest.raises(RedactionError):
        apply_redactions(
            doc,
            [
                Span(path="/a", embedded="/x", start=0, end=1, label="K"),
                Span(path="/a", start=0, end=1, label="K"),
            ],
        )
