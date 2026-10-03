"""LLM adapter: an OpenAI chat-completions request is one ``generate`` on ``model:<model>``."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gateway.core.envelope import Interaction, RawCall, SessionContext
from gateway.core.interfaces import Adapter
from gateway.core.types import Action, Channel
from gateway.errors import InvalidRequestError
from gateway.policy.permissions import Resource


class ChatMessage(BaseModel):
    """One message. Content shapes beyond text (parts, tool results) pass through untouched."""

    model_config = ConfigDict(extra="allow", frozen=True)

    role: str = Field(min_length=1)
    content: str | list[Any] | None = None


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """The subset of ``POST /v1/chat/completions`` the gateway reads; other fields pass through."""

    model_config = ConfigDict(extra="allow", frozen=True)

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False  # honoured by re-emitting the buffered answer as SSE
    stream_options: StreamOptions | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None

    @property
    def resource(self) -> str:
        return f"model:{self.model}"


def parse_chat_request(data: dict[str, Any]) -> ChatCompletionRequest:
    """Validate a request body; refusals name the fields, never echo their values."""
    try:
        request = ChatCompletionRequest.model_validate(data)
        Resource.parse(request.resource)
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) or "<body>" for e in exc.errors()})
        raise InvalidRequestError(
            "invalid_request", f"invalid chat completion request: {', '.join(fields)}"
        ) from None
    except ValueError:
        raise InvalidRequestError("invalid_model", "model is not a valid identifier") from None
    return request


class LLMAdapter(Adapter):
    """Normalizes a chat completion into one interaction carrying the request as payload."""

    def matches(self, raw: RawCall) -> bool:
        return raw.channel is Channel.LLM

    def normalize(self, raw: RawCall, ctx: SessionContext) -> list[Interaction]:
        request = parse_chat_request(raw.data)
        return [
            Interaction(
                session_id=ctx.session_id,
                principal=ctx.principal,
                actor=ctx.actor,
                mode=ctx.mode,
                channel=Channel.LLM,
                action=Action.GENERATE,
                resource=request.resource,
                payload=request.model_dump(mode="json", exclude_unset=True),
                context=ctx,
            )
        ]
