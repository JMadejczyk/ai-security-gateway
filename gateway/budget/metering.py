"""What a call is expected to spend before dispatch, and what it spent after.

LLM calls (``generate``):

- Tokens are reserved as an upper bound: a prompt estimate plus the completion cap. The cap
  is the smaller of the request's ``max_tokens`` / ``max_completion_tokens`` (positive integers
  only), ``limits.default_max_tokens`` when it names neither, and never more than
  ``limits.max_completion_tokens``. The held cap is written into ``max_tokens`` (the cap every
  OpenAI-compatible upstream, Ollama included, honours) and into ``max_completion_tokens`` when
  the request carries one, so no upstream can pick a larger cap than the one held.
- ``n`` and ``best_of`` other than 1 are refused (400 ``unsupported_parameter``): each extra
  choice is another full completion the hold would not cover. Refusing beats multiplying the
  hold: Ollama ignores ``n`` (one choice, always), so honouring it would hold n times what the
  demo upstream can spend, and every gateway path (post controls, SSE re-emission) is built
  around one answer per call.
- The prompt estimate is ``ceil(characters / 4)`` over the message text, tool-call arguments
  and tool definitions: the usual characters-per-token rule of thumb for English, a little
  generous for code and short for Polish. It only sizes the hold; the settlement charges
  the upstream's reported ``usage``.
- GPU time (upstream wall time, an estimate of GPU seconds for a local model) is reserved as
  an allowance the ledger sizes and the pipeline enforces as the call's deadline
  (`gateway.budget.ledger`); the settlement charges the measured wall time, never more than
  the allowance, and refunds the rest.

MCP calls reserve one tool call, which a dispatched call keeps even if the upstream fails:
the attempt reached the server.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

from gateway.budget.model import MS_PER_SECOND, BudgetedCall, Meter, Spend
from gateway.budget.pricing import CostModel
from gateway.core.types import Channel
from gateway.errors import InvalidRequestError
from gateway.upstream import TokenUsage

CHARS_PER_TOKEN: Final = 4
_COMPLETION_CAPS: Final = ("max_tokens", "max_completion_tokens")
_ONE_CHOICE: Final = ("n", "best_of")
_LLM_METERS: Final = frozenset({Meter.TOKENS, Meter.COST, Meter.GPU})
_MCP_METERS: Final = frozenset({Meter.TOOL_CALLS})


@dataclass(frozen=True, slots=True)
class Plan:
    """The payload to dispatch, what to hold for it, and which meters it draws on."""

    payload: object
    estimate: Spend  # tokens, tool calls and the token cost; the ledger adds GPU time
    meters: frozenset[Meter]


def charged_tokens(usage: TokenUsage) -> int:
    """Tokens one call is charged (and counted in ``acl_tokens_total``) for."""
    return max(usage.total_tokens, usage.prompt_tokens + usage.completion_tokens)


def plan(
    call: BudgetedCall, costs: CostModel, *, default_max_tokens: int, max_completion_tokens: int
) -> Plan:
    """Raises `InvalidRequestError` for a request whose completions cannot be bounded."""
    match call.channel:
        case Channel.LLM:
            return _plan_llm(call, costs, default_max_tokens, max_completion_tokens)
        case Channel.MCP:
            return Plan(payload=call.payload, estimate=Spend(tool_calls=1), meters=_MCP_METERS)
        case Channel.A2A:  # not served; nothing metered
            return Plan(payload=call.payload, estimate=Spend(), meters=frozenset())


def actual(  # noqa: PLR0913 -- what was held, what came back, and how long it took
    call: BudgetedCall,
    held: Spend,
    held_token_cost: int,
    costs: CostModel,
    *,
    answered: bool,
    usage: TokenUsage | None,
    wall_s: float,
) -> Spend:
    """What the call spent.

    - answered with ``usage``: the reported tokens at their price;
    - answered without ``usage``: the held tokens *and* their held cost (nothing says less);
    - not answered (upstream failure, cancellation): no tokens.

    GPU time is the wall time, capped at the allowance the call was held (and timed out) at.
    """
    if call.channel is not Channel.LLM:
        return held
    gpu_ms = max(round(wall_s * MS_PER_SECOND), 0)
    if held.gpu_ms:
        gpu_ms = min(gpu_ms, held.gpu_ms)
    gpu_cost = costs.gpu_cost(call.model, gpu_ms=gpu_ms)
    if not answered:
        tokens, token_cost = 0, 0
    elif usage is None:
        tokens, token_cost = held.tokens, held_token_cost
    else:
        tokens = charged_tokens(usage)
        token_cost = costs.token_cost(
            call.model, prompt=usage.prompt_tokens, completion=usage.completion_tokens
        )
    return Spend.bounded(tokens=tokens, cost_nano_usd=token_cost + gpu_cost, gpu_ms=gpu_ms)


def estimate_prompt_tokens(request: Mapping[str, Any]) -> int:
    characters = sum(len(text) for text in _prompt_texts(request))
    return math.ceil(characters / CHARS_PER_TOKEN)


def _plan_llm(
    call: BudgetedCall, costs: CostModel, default_max_tokens: int, max_completion_tokens: int
) -> Plan:
    original: object = call.payload
    if not isinstance(original, dict):  # the LLM adapter always produces an object
        return Plan(payload=original, estimate=Spend(), meters=_LLM_METERS)
    request = cast("dict[str, Any]", original)
    for name in _ONE_CHOICE:
        value: object = request.get(name)
        if value is not None and (type(value) is not int or value != 1):
            raise InvalidRequestError(
                "unsupported_parameter", f"{name} must be 1: one completion per call"
            )
    requested = min(_completion_caps(request), default=default_max_tokens)
    completion = min(requested, max_completion_tokens)
    payload: dict[str, Any] = {
        **request,
        **{
            name: completion for name in _COMPLETION_CAPS if name == "max_tokens" or name in request
        },
    }
    prompt = estimate_prompt_tokens(request)
    estimate = Spend.bounded(
        tokens=prompt + completion,
        cost_nano_usd=costs.token_cost(call.model, prompt=prompt, completion=completion),
    )
    return Plan(payload=payload, estimate=estimate, meters=_LLM_METERS)


def _completion_caps(request: Mapping[str, Any]) -> list[int]:
    """Positive integer caps the request names; anything else is the upstream's to refuse."""
    values = [request.get(name) for name in _COMPLETION_CAPS]
    return [value for value in values if type(value) is int and value > 0]


def _prompt_texts(request: Mapping[str, Any]) -> list[str]:
    texts = [text for message in _objects(request.get("messages")) for text in _texts(message)]
    tools: object = request.get("tools")
    if tools:
        texts.append(json.dumps(tools, separators=(",", ":"), ensure_ascii=False))
    return texts


def _texts(message: Mapping[str, Any]) -> list[str]:
    """Text a message puts in the prompt: content (string or text parts), tool-call calls."""
    content: object = message.get("content")
    texts = [content] if isinstance(content, str) else []
    texts += [part["text"] for part in _objects(content) if isinstance(part.get("text"), str)]
    for call in _objects(message.get("tool_calls")):
        for function in _objects([call.get("function")]):
            texts += [
                v for v in (function.get("name"), function.get("arguments")) if isinstance(v, str)
            ]
    return texts


def _objects(value: object) -> list[dict[str, Any]]:
    """The JSON objects in a JSON array; anything else is skipped."""
    items = cast("list[object]", value) if isinstance(value, list) else []
    return [cast("dict[str, Any]", item) for item in items if isinstance(item, dict)]
