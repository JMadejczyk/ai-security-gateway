"""JSON-RPC 2.0 envelopes and the MCP ``2025-06-18`` subset the gateway speaks.

Hand-written instead of taken from the SDK's ``mcp-types``: that package models every protocol
revision at once (it always serializes 2026 fields such as ``resultType``), while the gateway
pins one revision and must never emit or accept more than its tools-only subset.

Agent-facing envelopes forbid unknown members (JSON-RPC defines none). Upstream results are
validated only in the fields the gateway reads; everything else passes through untouched, so
tool definitions and results reach the agent as the server sent them.
"""

from collections.abc import Mapping
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

type ToolSchemas = Mapping[str, Mapping[str, Any]]  # tool name -> JSON Schema of its arguments

PROTOCOL_VERSION: Final = "2025-06-18"
JSONRPC_VERSION: Final = "2.0"

SESSION_HEADER: Final = "mcp-session-id"
PROTOCOL_HEADER: Final = "mcp-protocol-version"
PRINCIPAL_HEADER: Final = "x-acl-principal"
META_PREFIX: Final = "ai-control-layer/"  # `_meta` keys the gateway adds to tool errors
# The `_meta` key of a held call's id: the gateway sets it on the tool error, and the agent
# sets it in `tools/call` params to retry the approved call (`gateway.approvals.oversight`).
APPROVAL_ID_META: Final = f"{META_PREFIX}approval_id"

# JSON-RPC 2.0 error codes.
PARSE_ERROR: Final = -32700
INVALID_REQUEST: Final = -32600
METHOD_NOT_FOUND: Final = -32601
INVALID_PARAMS: Final = -32602
INTERNAL_ERROR: Final = -32603
# Implementation-defined band: the gateway refused the HTTP request (auth, session, origin).
GATEWAY_REFUSAL: Final = -32000

type RequestId = Annotated[int, Field(strict=True)] | Annotated[str, Field(strict=True)]


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    jsonrpc: Literal["2.0"]


class JsonRpcRequest(_Envelope):
    id: RequestId
    method: str = Field(min_length=1)
    params: dict[str, Any] | None = None


class JsonRpcNotification(_Envelope):
    method: str = Field(min_length=1)
    params: dict[str, Any] | None = None


class ErrorObject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: int
    message: str
    data: Any = None


class JsonRpcResult(_Envelope):
    id: RequestId
    result: dict[str, Any]


class JsonRpcError(_Envelope):
    id: RequestId | None
    error: ErrorObject


type JsonRpcMessage = JsonRpcRequest | JsonRpcNotification | JsonRpcResult | JsonRpcError

_MESSAGE: Final[TypeAdapter[JsonRpcMessage]] = TypeAdapter(JsonRpcMessage)


def parse_message(document: object) -> JsonRpcMessage:
    """One JSON-RPC message; raises `ValidationError` for anything else (batches included)."""
    return _MESSAGE.validate_python(document)


def result(request_id: RequestId, payload: dict[str, Any]) -> dict[str, Any]:
    return JsonRpcResult(jsonrpc="2.0", id=request_id, result=payload).model_dump(mode="json")


def error(
    request_id: RequestId | None, code: int, message: str, data: object = None
) -> dict[str, Any]:
    envelope = JsonRpcError(
        jsonrpc="2.0", id=request_id, error=ErrorObject(code=code, message=message, data=data)
    )
    return envelope.model_dump(mode="json", exclude_none=True) | {"id": request_id}


# ----------------------------------------------------------------------- MCP messages


class _Lenient(BaseModel):
    """Upstream data: the fields the gateway reads are validated, the rest passes through."""

    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)


class Implementation(_Lenient):
    name: str = Field(min_length=1)
    version: str


class InitializeParams(_Lenient):
    protocol_version: str = Field(alias="protocolVersion", min_length=1)
    capabilities: dict[str, Any]
    client_info: Implementation = Field(alias="clientInfo")


class InitializeResult(_Lenient):
    protocol_version: str = Field(alias="protocolVersion", min_length=1)
    capabilities: dict[str, Any]
    server_info: Implementation = Field(alias="serverInfo")


class ToolDefinition(_Lenient):
    """A tool as the upstream advertised it. Annotations are hints and are never read."""

    name: str = Field(min_length=1)
    input_schema: dict[str, Any] = Field(alias="inputSchema")

    def as_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_unset=True)


class ListToolsResult(_Lenient):
    tools: list[ToolDefinition]
    next_cursor: str | None = Field(default=None, alias="nextCursor")


class CallToolParams(BaseModel):
    """``tools/call`` params from the agent: a tool name and an arguments object."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict[str, Any])
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")  # progress token etc.

    def payload(self) -> dict[str, Any]:
        """What the pipeline authorizes and the upstream executes: name and arguments only."""
        return {"name": self.name, "arguments": self.arguments}

    def approval_id(self) -> str | None:
        """The approval this call retries under, if ``_meta`` names one. A non-string value
        is passed on as ``""`` so the pipeline refuses it instead of ignoring it."""
        if self.meta is None or APPROVAL_ID_META not in self.meta:
            return None
        value = self.meta[APPROVAL_ID_META]
        return value if isinstance(value, str) else ""


class CallToolResult(_Lenient):
    content: list[dict[str, Any]]
    structured_content: dict[str, Any] | None = Field(default=None, alias="structuredContent")
    is_error: bool = Field(default=False, alias="isError")


def server_capabilities() -> dict[str, Any]:
    """Tools only: no resources, prompts, logging or completions are offered."""
    return {"tools": {"listChanged": False}}


def tool_error(reason_code: str, *, approval_id: str | None = None) -> dict[str, Any]:
    """A tool-level error (``isError: true``) carrying only a reason code, per MCP.

    Policy refusals are tool errors, not protocol errors, so the agent's model sees them.
    The text never contains upstream output or payload data.
    """
    text = reason_code if approval_id is None else f"{reason_code} approval_id={approval_id}"
    meta = {f"{META_PREFIX}reason_code": reason_code}
    if approval_id is not None:
        meta[f"{META_PREFIX}approval_id"] = approval_id
    return {"content": [{"type": "text", "text": text}], "isError": True, "_meta": meta}
