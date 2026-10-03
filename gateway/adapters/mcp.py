"""MCP adapters: a ``tools/call`` becomes one or more ``(action, resource)`` interactions.

SPEC "GenericMCPAdapter". The mapping from tool to action and resource is operator-owned
(``upstreams.mcp.<server>.tools`` in the policy). Tool annotations come from the server, are
untrusted and are never read here. A tool missing from the mapping is refused.

Before any resource is derived, the arguments are validated against the tool's input schema
(pinned, or as the upstream first advertised it). Validation is strict: an object schema that
does not explicitly allow ``additionalProperties`` gets ``false``, so an argument the schema
does not declare cannot reach the upstream unchecked.

How an argument enters a resource template depends on the adapter:

- ``generic``: the value is one percent-encoded segment (it can never add ``/``, ``*`` or
  whitespace to the resource);
- ``http``: ``{url}`` is reduced to its host, canonicalized exactly as httpx (and so the
  demo fetcher) puts it on the wire: IDNA 2008, lower case, no user info, no trailing dot.
  The canonical URL replaces the agent's in the forwarded arguments, so the host that was
  authorized is the host that gets fetched (``faß.de`` is ``xn--fa-hia.de``, not ``fass.de``);
- ``fs``: a relative path, normalized first (no ``..``, ``.``, empty segments, absolute paths,
  backslashes or control characters), then percent-encoded per segment. Normalizing first
  matters: the permission ``write:fs:reports/*`` would otherwise accept ``reports/../x``;
- ``sql``: resources come from the tables the statement reads (`gateway.adapters.sql`).
"""

import logging
import re
import unicodedata
from collections.abc import Mapping
from enum import StrEnum
from ipaddress import ip_address
from typing import Any, ClassVar, Final
from urllib.parse import quote

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import ValidationError

from gateway.adapters.sql import UnsupportedSqlError, referenced_tables
from gateway.core.envelope import Interaction, RawCall, SessionContext
from gateway.core.interfaces import Adapter
from gateway.core.types import Channel
from gateway.errors import RejectionError
from gateway.policy.permissions import Resource
from gateway.policy.schema import McpServer, McpTool
from gateway.proxies.mcp.wire import CallToolParams, ToolSchemas

logger = logging.getLogger(__name__)

_PLACEHOLDER: Final = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_LABEL: Final = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOSTNAME: Final = re.compile(rf"{_LABEL}(?:\.{_LABEL})*")
MAX_HOSTNAME: Final = 253
MAX_PORT: Final = 65535
SQL_ARGUMENT: Final = "sql"


class ToolCallReason(StrEnum):
    TOOL_NOT_MAPPED = "tool_not_mapped"
    TOOL_NOT_ADVERTISED = "tool_not_advertised"  # mapped, but neither pinned nor advertised
    INVALID_ARGUMENTS = "invalid_arguments"


class ToolCallRejectedError(RejectionError):
    """The adapter refuses the call before authorization: it names no checkable resource."""

    def __init__(self, reason: ToolCallReason) -> None:
        super().__init__(reason.value, f"tool call refused: {reason.value.replace('_', ' ')}")


def _rejected(reason: ToolCallReason = ToolCallReason.INVALID_ARGUMENTS) -> ToolCallRejectedError:
    return ToolCallRejectedError(reason)


def _has_control_chars(value: str) -> bool:
    return any(unicodedata.category(ch) == "Cc" for ch in value)


def validate_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> None:
    """Raise unless ``arguments`` satisfy ``schema`` (strictly, see the module docstring)."""
    strict = dict(schema)
    if strict.get("type") == "object" and "additionalProperties" not in strict:
        strict["additionalProperties"] = False
    validator_class = validator_for(strict, default=Draft202012Validator)
    try:
        validator_class.check_schema(strict)
    except SchemaError:
        logger.warning("an MCP tool advertises an invalid input schema")
        raise _rejected() from None
    validator = validator_class(strict, format_checker=validator_class.FORMAT_CHECKER)
    if next(iter(validator.iter_errors(dict(arguments))), None) is not None:
        raise _rejected()


class GenericMCPAdapter(Adapter):
    """Maps a tool through its operator template; one interaction per derived resource."""

    kind: ClassVar[str] = "generic"

    def __init__(self, server: str, config: McpServer, schemas: ToolSchemas) -> None:
        self._server = server
        self._config = config
        self._schemas = schemas

    def matches(self, raw: RawCall) -> bool:
        return raw.channel is Channel.MCP and raw.server == self._server

    def normalize(self, raw: RawCall, ctx: SessionContext) -> list[Interaction]:
        try:
            call = CallToolParams.model_validate(raw.data)
        except ValidationError:
            raise _rejected() from None
        tool = self._config.tools.get(call.name)
        if tool is None:
            raise _rejected(ToolCallReason.TOOL_NOT_MAPPED)
        schema = self._schemas.get(call.name)
        if schema is None:
            raise _rejected(ToolCallReason.TOOL_NOT_ADVERTISED)
        validate_arguments(schema, call.arguments)
        arguments = self.canonical_arguments(call.arguments)
        payload = {"name": call.name, "arguments": arguments}
        return [
            Interaction(
                session_id=ctx.session_id,
                principal=ctx.principal,
                actor=ctx.actor,
                mode=ctx.mode,
                channel=Channel.MCP,
                action=tool.action,
                resource=resource,
                payload=payload,
                context=ctx,
            )
            for resource in self.resources(tool, arguments)
        ]

    def canonical_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """The arguments as they will be forwarded; resources are derived from these."""
        return arguments

    def resources(self, tool: McpTool, arguments: Mapping[str, Any]) -> list[str]:
        """Concrete resources of one call: the tool's template with every argument rendered."""
        if tool.resource is None:  # the schema requires a template for non-sql adapters
            raise _rejected(ToolCallReason.TOOL_NOT_MAPPED)

        def fill(match: re.Match[str]) -> str:
            value = arguments.get(match[1])
            if not isinstance(value, str) or not value:
                raise _rejected()
            return self.render(match[1], value)

        return [_concrete(_PLACEHOLDER.sub(fill, tool.resource))]

    def render(self, name: str, value: str) -> str:
        """How the argument ``name`` enters a resource: one opaque, encoded segment."""
        del name
        if _has_control_chars(value):
            raise _rejected()
        return quote(value, safe="")


class HttpMCPAdapter(GenericMCPAdapter):
    """``{url}`` becomes the URL's host: ``web:{url}`` → ``web:example.com``."""

    kind: ClassVar[str] = "http"
    URL_ARGUMENT: ClassVar[str] = "url"

    def canonical_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        url = arguments.get(self.URL_ARGUMENT)
        if not isinstance(url, str):
            return arguments
        return {**arguments, self.URL_ARGUMENT: str(canonical_url(url))}

    def render(self, name: str, value: str) -> str:
        if name != self.URL_ARGUMENT:
            return super().render(name, value)
        return url_host(value)


class FsMCPAdapter(GenericMCPAdapter):
    """Every placeholder is a relative path, normalized before it enters the resource."""

    kind: ClassVar[str] = "fs"

    def render(self, name: str, value: str) -> str:
        del name
        return normalized_path(value)


class SqlMCPAdapter(GenericMCPAdapter):
    """One ``db:<schema>.<table>`` interaction per table the ``sql`` argument reads."""

    kind: ClassVar[str] = "sql"

    def resources(self, tool: McpTool, arguments: Mapping[str, Any]) -> list[str]:
        sql = arguments.get(SQL_ARGUMENT)
        if not isinstance(sql, str):
            raise _rejected()
        return [_concrete(f"db:{table}", sql=True) for table in referenced_tables(sql)]


def canonical_url(url: str) -> httpx.URL:
    """An absolute http(s) URL parsed the way httpx sends it; anything ambiguous is refused.

    httpx (and the demo fetcher, which uses it) encodes hosts with IDNA 2008. Parsing with
    the same library, rather than ``urlsplit`` plus Python's IDNA 2003 codec, removes the
    parser differential: the host checked here is byte for byte the one on the wire.
    """
    if _has_control_chars(url) or any(ch.isspace() or ch == "\\" for ch in url):
        raise _rejected()
    try:
        parsed = httpx.URL(url)
        host = parsed.raw_host.decode("ascii")
    except (httpx.InvalidURL, UnicodeError):
        raise _rejected() from None
    if parsed.scheme not in {"http", "https"} or parsed.userinfo:
        raise _rejected()
    if parsed.port is not None and not 0 < parsed.port <= MAX_PORT:
        raise _rejected()
    try:
        ip_address(host)
    except ValueError:
        if len(host) > MAX_HOSTNAME or not _HOSTNAME.fullmatch(host):
            raise _rejected() from None  # empty, trailing dot, upper case, odd characters
    return parsed


def url_host(url: str) -> str:
    """The canonical (ASCII, lower-case) host of an absolute http(s) URL."""
    return canonical_url(url).raw_host.decode("ascii")


def normalized_path(path: str) -> str:
    """A relative path with every segment percent-encoded; traversal and ambiguity refused."""
    if not path or path.startswith("/") or "\\" in path or _has_control_chars(path):
        raise _rejected()
    segments = path.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise _rejected()
    return "/".join(quote(segment, safe="") for segment in segments)


def _concrete(resource: str, *, sql: bool = False) -> str:
    try:
        return str(Resource.parse(resource))
    except ValueError:
        if sql:
            raise UnsupportedSqlError from None
        raise _rejected() from None


_ADAPTERS: Final[Mapping[str, type[GenericMCPAdapter]]] = {
    cls.kind: cls for cls in (GenericMCPAdapter, HttpMCPAdapter, FsMCPAdapter, SqlMCPAdapter)
}


def mcp_adapter(server: str, config: McpServer, schemas: ToolSchemas) -> GenericMCPAdapter:
    """The adapter the policy names for ``server``, bound to this call's schemas."""
    return _ADAPTERS[config.adapter](server, config, schemas)
