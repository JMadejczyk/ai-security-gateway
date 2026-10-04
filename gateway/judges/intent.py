"""What the intent judge compares, and how its flags match later (SPEC "Intent vs enforcement").

- The session **goal** is the first user message the gateway forwarded to the LLM in the
  session (the redacted, dispatched request), recorded once in session state. A later
  transcript the agent sends can never replace it.
- A flagged ``tool_call`` is ``(tool name, digest of its canonical arguments)``. The same
  digest is computed for an MCP ``tools/call``: SHA-256 of the canonical JSON
  (`gateway.canonical`) of the arguments object. LLM tool-call ``arguments`` are JSON text
  and are parsed first; text that does not parse is digested as the string it is (it can
  never equal an MCP arguments object).
  Missing or empty arguments are ``{}`` on both sides.

Matching is by tool name, including the names MCP clients show a server's tools under to
their model: ``<server>_<tool>`` (opencode) and ``mcp__<server>__<tool>`` (Claude Code), so a
flag raised on ``sales_db_query`` holds ``query`` on ``sales_db``. Extra names can only add an
approval obligation, never remove one. A client with another naming scheme does not get its
flags matched; that is acceptable only because the judge is advisory: every MCP call is
authorized on its own, whatever the LLM said (SPEC).
"""

import hashlib
import json
from collections.abc import Iterator, Mapping
from typing import Any, Final, cast

from gateway.canonical import canonical_bytes
from gateway.core.envelope import FlaggedToolCall, SessionContext
from gateway.sessions import MAX_GOAL_CHARS

_EMPTY_ARGUMENTS: Final[dict[str, Any]] = {}


def _mapping(value: object) -> Mapping[str, Any] | None:
    return cast("Mapping[str, Any]", value) if isinstance(value, Mapping) else None


def _items(value: object) -> list[Any]:
    return cast("list[Any]", value) if isinstance(value, list) else []


def arguments_digest(arguments: object) -> str:
    """SHA-256 of the canonical JSON of ``arguments`` (``None`` and ``""`` count as ``{}``)."""
    if arguments is None or arguments == "":
        arguments = _EMPTY_ARGUMENTS
    canonical = canonical_bytes(arguments)
    if canonical is None:  # not plain JSON (NaN, ...): digest its repr, it matches nothing else
        canonical = repr(arguments).encode()
    return hashlib.sha256(canonical).hexdigest()


def parse_llm_arguments(arguments: object) -> object:
    """An LLM tool call's ``arguments`` as the object an MCP call would carry."""
    if arguments is None:
        return _EMPTY_ARGUMENTS
    if isinstance(arguments, str):
        if not arguments.strip():
            return _EMPTY_ARGUMENTS
        try:
            parsed: object = json.loads(arguments)
        except ValueError:
            return arguments
        return parsed
    return arguments


def llm_tool_calls(completion: object) -> Iterator[tuple[str, object]]:
    """``(name, parsed arguments)`` of every tool call in a chat completion, every choice,
    including the legacy ``function_call``."""
    for choice in _items((_mapping(completion) or {}).get("choices")):
        message = _mapping((_mapping(choice) or {}).get("message")) or {}
        functions = [
            (_mapping(call) or {}).get("function") for call in _items(message.get("tool_calls"))
        ]
        functions.append(message.get("function_call"))
        for function in functions:
            if (spec := _mapping(function)) is None:
                continue
            name = spec.get("name")
            if isinstance(name, str) and name:
                yield name, parse_llm_arguments(spec.get("arguments"))


def flag_for(name: str, arguments: object) -> FlaggedToolCall:
    return FlaggedToolCall(tool=name, args_digest=arguments_digest(arguments))


def exposed_names(tool: str, server: str | None) -> tuple[str, ...]:
    """The names a model may know an MCP server's tool by (see the module docstring)."""
    if not server:
        return (tool,)
    return (tool, f"{server}_{tool}", f"mcp__{server}__{tool}")


def mcp_flag_candidates(*payloads: object, server: str | None = None) -> set[FlaggedToolCall]:
    """Flags an MCP ``tools/call`` on ``server`` would match: per distinct ``{"name",
    "arguments"}`` payload given (the agent's own arguments and the adapter's canonical
    form), one per name the tool may have been exposed under."""
    candidates: set[FlaggedToolCall] = set()
    for payload in payloads:
        params = _mapping(payload)
        if params is None:
            continue
        name = params.get("name")
        if isinstance(name, str) and name:
            arguments = params.get("arguments")
            candidates.update(flag_for(alias, arguments) for alias in exposed_names(name, server))
    return candidates


def _message_text(content: object) -> str | None:
    if isinstance(content, str):
        return content
    texts = [
        part["text"]
        for part in (_mapping(item) for item in _items(content))
        if part is not None and isinstance(part.get("text"), str)
    ]
    return "\n".join(texts) if texts else None


def first_user_message(request: object) -> str | None:
    """Text of the first ``user`` message of a chat request, capped at `MAX_GOAL_CHARS`."""
    for message in _items((_mapping(request) or {}).get("messages")):
        fields = _mapping(message) or {}
        if fields.get("role") == "user":
            text = _message_text(fields.get("content"))
            if text is not None and text.strip():
                return text[:MAX_GOAL_CHARS]
    return None


def session_goal(ctx: SessionContext, request: object) -> str | None:
    """The goal recorded for the session, or (on its first LLM call, before it is persisted)
    the one this request is about to record."""
    return ctx.goal if ctx.goal is not None else first_user_message(request)
