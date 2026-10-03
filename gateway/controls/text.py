"""The free text of an interaction, addressed by JSON pointer, for content-scanning controls.

Content controls (``pii``, ``secrets``, later ``signatures`` and ``prompt_injection``) scan
the strings an interaction carries and answer with spans. A span is a JSON pointer into the
document the pipeline redacts at that stage plus code-point offsets inside the string it
names (`gateway.redaction`), so every pointer yielded here addresses exactly that document:

- llm, pre: the chat request (``payload``): ``/messages/i/content`` (a string, or
  ``/messages/i/content/j/text`` of a content-part list), the message's other prose fields
  (e.g. ``reasoning``) and ``/messages/i/tool_calls/k/function/arguments``;
- llm, post: the chat completion (``result``): the same fields under ``/choices/i/message``;
- mcp, pre: ``{"name", "arguments"}`` (``payload``): every string leaf under ``/arguments``;
- mcp, post: the ``CallToolResult`` (``result``): ``/content/i/text`` and every string leaf
  under ``/structuredContent``.

Tool-call arguments are JSON text: offsets point into that text, and a mask written over a
match inside one of its string values keeps it valid JSON. Shapes the gateway does not
recognize yield nothing rather than fail; the adapters and upstreams have validated them.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

from gateway.core.envelope import Interaction
from gateway.core.types import Channel, Stage

# Message fields that name or route, never carry prose a person or a model wrote.
_MESSAGE_METADATA: Final = frozenset({"role", "name", "tool_call_id"})


@dataclass(frozen=True, slots=True)
class TextSegment:
    """One string of the scanned document and where it lives."""

    pointer: str  # JSON pointer (RFC 6901) into the document; "" is the document itself
    text: str


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


class TextExtractor:
    """Yields the scannable strings of an interaction at one stage (see the module table)."""

    def segments(self, interaction: Interaction, stage: Stage) -> list[TextSegment]:
        """The strings controls scan, each with its pointer into `document`."""
        document = self.document(interaction, stage)
        match interaction.channel, stage:
            case Channel.LLM, Stage.PRE:
                return list(self._chat_request(document))
            case Channel.LLM, Stage.POST:
                return list(self._chat_completion(document))
            case Channel.MCP, Stage.PRE:
                return list(self._tool_arguments(document))
            case Channel.MCP, Stage.POST:
                return list(self._tool_result(document))
            case _:
                return []

    @staticmethod
    def document(interaction: Interaction, stage: Stage) -> object:
        """The value spans address: the payload before the upstream, its result after."""
        return interaction.payload if stage is Stage.PRE else interaction.result

    # -------------------------------------------------------------------- llm

    def _chat_request(self, request: object) -> Iterator[TextSegment]:
        body = _as_mapping(request) or {}
        for index, message in enumerate(_as_list(body.get("messages"))):
            yield from self._message(message, pointer("messages", index))

    def _chat_completion(self, completion: object) -> Iterator[TextSegment]:
        body = _as_mapping(completion) or {}
        for index, choice in enumerate(_as_list(body.get("choices"))):
            fields = _as_mapping(choice) or {}
            yield from self._message(fields.get("message"), pointer("choices", index, "message"))

    def _message(self, message: object, base: str) -> Iterator[TextSegment]:
        fields = _as_mapping(message) or {}
        for key, value in fields.items():
            if key in _MESSAGE_METADATA:
                continue
            here = base + pointer(key)
            if key == "content":
                yield from self._content(value, here)
            elif key == "tool_calls":
                yield from self._tool_calls(value, here)
            elif isinstance(value, str) and value:  # reasoning, refusal, ...
                yield TextSegment(here, value)

    @staticmethod
    def _content(content: object, base: str) -> Iterator[TextSegment]:
        if isinstance(content, str):
            if content:
                yield TextSegment(base, content)
            return
        for index, part in enumerate(_as_list(content)):
            text = (_as_mapping(part) or {}).get("text")
            if isinstance(text, str) and text:
                yield TextSegment(base + pointer(index, "text"), text)

    @staticmethod
    def _tool_calls(calls: object, base: str) -> Iterator[TextSegment]:
        for index, call in enumerate(_as_list(calls)):
            function = _as_mapping((_as_mapping(call) or {}).get("function")) or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments:
                yield TextSegment(base + pointer(index, "function", "arguments"), arguments)

    # -------------------------------------------------------------------- mcp

    @staticmethod
    def _tool_arguments(payload: object) -> Iterator[TextSegment]:
        body = _as_mapping(payload) or {}
        yield from string_leaves(body.get("arguments"), pointer("arguments"))

    @staticmethod
    def _tool_result(result: object) -> Iterator[TextSegment]:
        body = _as_mapping(result) or {}
        for index, item in enumerate(_as_list(body.get("content"))):
            text = (_as_mapping(item) or {}).get("text")
            if isinstance(text, str) and text:
                yield TextSegment(pointer("content", index, "text"), text)
        yield from string_leaves(body.get("structuredContent"), pointer("structuredContent"))
