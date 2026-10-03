"""Pricing, estimates and the policy's ``pricing`` / ``default_max_tokens`` schema."""

import pytest
from budget_kit import scope
from pydantic import ValidationError

from gateway.budget.metering import actual, charged_tokens, estimate_prompt_tokens, plan
from gateway.budget.model import BudgetedCall, Meter, ScopeKind, Spend, SpendLimits
from gateway.budget.pricing import CostModel
from gateway.core.types import Channel
from gateway.policy.loader import PolicyLoadError
from gateway.policy.schema import DailyBudget, ModelPrice, SessionBudget
from gateway.upstream import TokenUsage

PRICED = CostModel(
    {"qwen3:8b": ModelPrice(prompt_per_1k=0.5, completion_per_1k=1.5, gpu_second=0.0005)}
)


def call(payload: object, channel: Channel = Channel.LLM, model: str | None = "qwen3:8b"):
    return BudgetedCall(
        session_id="s-1",
        principal="anna@demo",
        agent="databot",
        channel=channel,
        model=model,
        payload=payload,
        user_label="anna@demo",
        agent_label="databot",
    )


def request(content: object = "x" * 40, **extra: object) -> dict[str, object]:
    return {"model": "qwen3:8b", "messages": [{"role": "user", "content": content}], **extra}


# ------------------------------------------------------------------------------- pricing


def test_costs_are_exact_decimals_rounded_up_to_the_nano_dollar():
    assert PRICED.token_cost("qwen3:8b", prompt=1000, completion=1000) == 2_000_000_000
    assert PRICED.gpu_cost("qwen3:8b", gpu_ms=2000) == 1_000_000  # 2 s * $0.0005, exactly
    assert PRICED.token_cost("qwen3:8b", prompt=1, completion=0) == 500_000
    assert (
        CostModel({"m": ModelPrice(prompt_per_1k=1e-9)}).token_cost("m", prompt=1, completion=0)
        == 1
    )


def test_an_unpriced_model_costs_nothing():
    assert PRICED.token_cost("llama3:70b", prompt=10_000, completion=10_000) == 0
    assert PRICED.gpu_cost(None, gpu_ms=60_000) == 0


# ----------------------------------------------------------------------------- estimates


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (request("abcd"), 1),
        (request("abcde"), 2),  # rounded up
        (request([{"type": "text", "text": "a" * 8}, {"type": "image_url"}]), 2),
        (
            {
                "messages": [
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"function": {"name": "query", "arguments": '{"sql":"SELECT 1"}'}},
                    ]},
                ]
            },
            6,  # "query" + the arguments: 23 characters
        ),
        ({"messages": "not a list"}, 0),
    ],
)  # fmt: skip
def test_prompt_estimate_is_characters_over_four_rounded_up(body, expected):
    assert estimate_prompt_tokens(body) == expected


def test_tool_definitions_count_towards_the_prompt():
    tools = [{"type": "function", "function": {"name": "query"}}]
    assert estimate_prompt_tokens(request(tools=tools)) > estimate_prompt_tokens(request())


def test_a_request_without_a_cap_gets_the_default_injected_and_held():
    planned = plan(call(request()), PRICED, default_max_tokens=4096)
    assert planned.payload == request(max_tokens=4096)
    assert planned.estimate.tokens == 10 + 4096
    assert planned.estimate.cost_nano_usd == PRICED.token_cost(
        "qwen3:8b", prompt=10, completion=4096
    )
    assert planned.meters == {Meter.TOKENS, Meter.COST, Meter.GPU}


@pytest.mark.parametrize(
    ("caps", "held"),
    [
        ({"max_tokens": 64}, 64),
        ({"max_completion_tokens": 32}, 32),
        ({"max_tokens": 64, "max_completion_tokens": 32}, 32),
        (
            {"max_completion_tokens": -1},
            4096,
        ),  # not a cap: the upstream refuses it, we hold the default
        ({"max_completion_tokens": True}, 4096),
    ],
)
def test_the_smallest_cap_is_held_and_sent_as_max_tokens(caps, held):
    planned = plan(call(request(**caps)), PRICED, default_max_tokens=4096)
    assert planned.estimate.tokens == 10 + held
    assert planned.payload["max_tokens"] == held  # type: ignore[index]


def test_an_mcp_call_holds_one_tool_call_and_is_forwarded_unchanged():
    payload = {"name": "query", "arguments": {"sql": "SELECT 1"}}
    planned = plan(call(payload, Channel.MCP, None), PRICED, default_max_tokens=4096)
    assert planned.payload is payload
    assert (planned.estimate, planned.meters) == (Spend(tool_calls=1), {Meter.TOOL_CALLS})


def test_actual_spend_charges_reported_usage_and_wall_time():
    usage = TokenUsage(model="qwen3:8b", prompt_tokens=12, completion_tokens=7, total_tokens=19)
    spent = actual(call(request()), Spend(tokens=500), PRICED, usage=usage, wall_s=1.5)
    assert spent == Spend(
        tokens=19,
        gpu_ms=1500,
        cost_nano_usd=PRICED.token_cost("qwen3:8b", prompt=12, completion=7)
        + PRICED.gpu_cost("qwen3:8b", gpu_ms=1500),
    )


def test_a_failed_call_is_charged_its_wall_time_only():
    spent = actual(call(request()), Spend(tokens=500), PRICED, usage=None, wall_s=0.25)
    assert spent == Spend(gpu_ms=250, cost_nano_usd=PRICED.gpu_cost("qwen3:8b", gpu_ms=250))


def test_a_dispatched_tool_call_keeps_its_hold():
    held = Spend(tool_calls=1)
    assert actual(call({}, Channel.MCP, None), held, PRICED, usage=None, wall_s=3) == held


def test_charged_tokens_never_trust_a_low_total():
    usage = TokenUsage(model="m", prompt_tokens=10, completion_tokens=5, total_tokens=3)
    assert charged_tokens(usage) == 15


# -------------------------------------------------------------------------------- limits


def test_policy_limits_convert_to_base_units():
    daily = SpendLimits.from_policy(DailyBudget(daily_tokens=10, daily_cost_usd=2.0), daily=True)
    assert (daily.tokens, daily.cost_nano_usd, daily.gpu_ms) == (10, 2_000_000_000, None)
    session = SpendLimits.from_policy(SessionBudget(tool_calls=50, gpu_seconds=120), daily=False)
    assert (session.tool_calls, session.gpu_ms) == (50, 120_000)


def test_limit_names_point_at_the_policy_setting():
    assert scope().limit_name(Meter.COST) == "per_user.daily_cost_usd"
    session = scope(ScopeKind.SESSION, "s-1", window="session")
    assert session.limit_name(Meter.GPU) == "per_session.gpu_seconds"


# -------------------------------------------------------------------------------- schema


def test_the_root_policy_prices_qwen_by_gpu_time(snapshot):
    price = snapshot.policy.pricing["qwen3:8b"]
    assert (price.prompt_per_1k, price.completion_per_1k, price.gpu_second) == (0.0, 0.0, 0.0005)
    assert snapshot.policy.limits.default_max_tokens == 4096


@pytest.mark.parametrize(
    "price",
    [
        {"prompt_per_1k": -0.1},
        {"gpu_second": float("inf")},
        {"completion_per_1k": float("nan")},
        {"per_request": 1.0},
    ],
    ids=["negative", "infinite", "nan", "unknown-field"],
)
def test_invalid_prices_are_rejected(price):
    with pytest.raises(ValidationError):
        ModelPrice.model_validate(price)


@pytest.mark.parametrize("model", ["", "has space", "*"])
def test_pricing_keys_must_be_model_identifiers(snapshot_from, policy_doc, model):
    policy_doc["pricing"] = {model: {"prompt_per_1k": 0.1}}
    with pytest.raises((PolicyLoadError, ValidationError)):
        snapshot_from(policy_doc)


def test_default_max_tokens_must_be_positive(snapshot_from, policy_doc):
    policy_doc["limits"]["default_max_tokens"] = 0
    with pytest.raises((PolicyLoadError, ValidationError)):
        snapshot_from(policy_doc)
