"""Session restrictions: the same event narrows an interactive session but only conditions an
autonomous one (approval, throttle), and nothing here ever widens base authorization."""

from datetime import timedelta

import pytest

from gateway.core.envelope import Cooldown
from gateway.core.types import Action, Channel, Decision, SessionMode
from gateway.policy.evaluator import AuthzReason, PrincipalContext, RestrictionReason
from gateway.policy.schema import Throttle

INT, AU = SessionMode.INTERACTIVE, SessionMode.AUTONOMOUS
MCP, LLM = Channel.MCP, Channel.LLM
REPORT = "fs:reports/q3.md"

ANNA = PrincipalContext(principal="anna@demo", roles=("analyst",), agent="databot", mode=INT)
BARTEK = PrincipalContext(principal="bartek@demo", roles=("intern",), agent="databot", mode=INT)
ETL = PrincipalContext(principal="svc:nightly_etl", agent="nightly_etl", mode=AU)


@pytest.fixture
def restrict(evaluator, snapshot, now):
    def run(ctx, action, resource=REPORT, channel=MCP, at=None):
        return evaluator.session_restrictions(
            snapshot, ctx, channel=channel, action=action, resource=resource, now=at or now
        )

    return run


@pytest.fixture
def interactive(make_ctx):
    return lambda **kw: make_ctx(INT, **kw)


@pytest.fixture
def autonomous(make_ctx):
    return lambda **kw: make_ctx(AU, principal="svc:nightly_etl", actor="nightly_etl", **kw)


# ----------------------------------------------------------------------------- interactive


@pytest.mark.parametrize("action", [Action.WRITE, Action.DELETE, Action.EGRESS])
def test_interactive_taint_removes_write_delete_egress(restrict, interactive, action):
    result = restrict(interactive(taint=True), action)
    assert result.denied
    assert result.reason_code is RestrictionReason.ACTION_REMOVED


@pytest.mark.parametrize(
    ("action", "resource", "channel"),
    [
        (Action.READ, "db:sales.orders", MCP),
        (Action.READ, "web:example.com", MCP),
        (Action.GENERATE, "model:qwen3:8b", LLM),
    ],
)
def test_interactive_taint_keeps_reads_and_generate(
    restrict, interactive, action, resource, channel
):
    result = restrict(interactive(taint=True), action, resource, channel)
    assert not result.denied
    assert not result.requires_approval


def test_clean_session_has_no_restrictions(restrict, interactive):
    result = restrict(interactive(), Action.WRITE)
    assert not result.denied
    assert not result.requires_approval
    assert result.matched_rules == ()
    assert result.cooldown_on_deny is None
    assert result.start_freeze is None


def test_risk_exactly_at_threshold_does_not_trigger(restrict, interactive):
    result = restrict(interactive(risk=0.5), Action.WRITE)
    assert not result.requires_approval
    assert result.matched_rules == ()


def test_risk_above_half_requires_approval_for_write_only(restrict, interactive, now):
    write = restrict(interactive(risk=0.51), Action.WRITE)
    assert write.requires_approval
    assert not write.denied
    assert write.cooldown_on_deny is not None
    assert write.cooldown_on_deny.key == f"write:{REPORT}"
    assert write.cooldown_on_deny.until == now + timedelta(seconds=300)

    read = restrict(interactive(risk=0.51), Action.READ, "db:sales.orders")
    assert not read.requires_approval
    assert not read.denied


def test_risk_decays_with_half_life(restrict, interactive, now):
    # 0.9 one half-life (600 s) ago is 0.45 now: below every threshold.
    ctx = interactive(risk=0.9, risk_updated_at=now - timedelta(seconds=600))
    result = restrict(ctx, Action.WRITE)
    assert result.risk == pytest.approx(0.45)
    assert result.matched_rules == ()


def test_active_cooldown_denies_the_same_action_resource(restrict, interactive, now):
    ctx = interactive(
        risk=0.6, cooldowns=(Cooldown(key=f"write:{REPORT}", until=now + timedelta(seconds=60)),)
    )
    same = restrict(ctx, Action.WRITE)
    assert same.denied
    assert same.reason_code is RestrictionReason.COOLDOWN_ACTIVE
    assert same.cooldown_on_deny is None  # a running cooldown is never refreshed

    other = restrict(ctx, Action.WRITE, "fs:reports/other.md")
    assert not other.denied
    assert other.requires_approval


def test_active_cooldown_outlives_the_risk_that_caused_it(restrict, interactive, now):
    ctx = interactive(
        risk=0.0, cooldowns=(Cooldown(key=f"write:{REPORT}", until=now + timedelta(seconds=1)),)
    )
    assert restrict(ctx, Action.WRITE).reason_code is RestrictionReason.COOLDOWN_ACTIVE


def test_expired_cooldown_no_longer_denies(restrict, interactive, now):
    ctx = interactive(cooldowns=(Cooldown(key=f"write:{REPORT}", until=now),))
    assert not restrict(ctx, Action.WRITE).denied


def test_freeze_above_point_eight_blocks_mcp_tool_calls_not_generate(restrict, interactive, now):
    ctx = interactive(risk=0.81)
    tool = restrict(ctx, Action.READ, "db:sales.orders", MCP)
    assert tool.denied
    assert tool.reason_code is RestrictionReason.TOOLS_FROZEN
    assert tool.start_freeze is not None
    assert tool.start_freeze.until == now + timedelta(seconds=300)

    generate = restrict(ctx, Action.GENERATE, "model:qwen3:8b", LLM)
    assert not generate.denied
    assert generate.start_freeze is not None  # threshold crossed: the timer starts anyway


def test_running_freeze_denies_until_it_ends_and_is_not_extended(restrict, interactive, now):
    ctx = interactive(risk=0.1, freeze_until=now + timedelta(seconds=30))
    during = restrict(ctx, Action.READ, "db:sales.orders")
    assert during.reason_code is RestrictionReason.TOOLS_FROZEN
    assert during.start_freeze is None

    still_high = restrict(
        interactive(risk=0.9, freeze_until=now + timedelta(seconds=30)), Action.READ
    )
    assert still_high.start_freeze is None

    assert not restrict(ctx, Action.GENERATE, "model:qwen3:8b", LLM).denied


def test_freeze_is_reentered_only_if_risk_is_still_high(restrict, interactive, now):
    expired = now - timedelta(seconds=1)
    high = restrict(interactive(risk=0.9, freeze_until=expired), Action.READ, "db:sales.orders")
    assert high.denied
    assert high.start_freeze is not None

    low = restrict(interactive(risk=0.3, freeze_until=expired), Action.READ, "db:sales.orders")
    assert not low.denied
    assert low.start_freeze is None


def test_freeze_applies_to_a2a_tool_calls_too(restrict, interactive):
    result = restrict(interactive(risk=0.9), Action.READ, "db:sales.orders", Channel.A2A)
    assert result.reason_code is RestrictionReason.TOOLS_FROZEN


# ------------------------------------------------------------------------------ autonomous


@pytest.mark.parametrize("action", [Action.WRITE, Action.DELETE, Action.EGRESS])
def test_autonomous_taint_requires_approval_instead_of_denying(restrict, autonomous, action):
    result = restrict(autonomous(taint=True), action)
    assert not result.denied
    assert result.requires_approval


def test_autonomous_taint_keeps_reads_unconditioned(restrict, autonomous):
    result = restrict(autonomous(taint=True), Action.READ, "db:sales.orders")
    assert not result.denied
    assert not result.requires_approval


def test_autonomous_elevated_risk_throttles_and_alerts(restrict, autonomous):
    result = restrict(autonomous(risk=0.6), Action.READ, "db:sales.orders")
    assert not result.denied
    assert not result.requires_approval
    assert result.throttle == Throttle(max_actions=1, per_s=10)
    assert result.alert


def test_autonomous_at_threshold_is_not_throttled(restrict, autonomous):
    result = restrict(autonomous(risk=0.5), Action.READ, "db:sales.orders")
    assert result.throttle is None
    assert not result.alert


@pytest.mark.parametrize(
    ("action", "needs_approval"),
    [
        (Action.WRITE, True),
        (Action.DELETE, True),
        (Action.EXECUTE, True),
        (Action.EGRESS, True),
        (Action.READ, False),
        (Action.GENERATE, False),
    ],
)
def test_autonomous_high_risk_queues_everything_but_read_and_generate(
    restrict, autonomous, action, needs_approval
):
    result = restrict(autonomous(risk=0.85), action)
    assert result.requires_approval is needs_approval
    assert not result.denied  # autonomous agents never lose rights, they wait
    assert result.alert
    assert result.throttle is not None


# --------------------------------------------------------------- restrictions only narrow


def test_decide_tainted_interactive_write_is_blocked(evaluator, snapshot, interactive, now):
    decision = evaluator.decide(
        snapshot,
        ANNA,
        interactive(taint=True),
        channel=MCP,
        action=Action.WRITE,
        resource=REPORT,
        now=now,
    )
    assert decision.authorization.allowed
    assert decision.decision is Decision.BLOCK
    assert decision.reason_code == RestrictionReason.ACTION_REMOVED


def test_decide_same_event_on_autonomous_requires_approval(evaluator, snapshot, autonomous, now):
    decision = evaluator.decide(
        snapshot,
        ETL,
        autonomous(taint=True),
        channel=MCP,
        action=Action.WRITE,
        resource="fs:reports/nightly.md",
        now=now,
    )
    assert decision.decision is Decision.REQUIRE_APPROVAL


SESSIONS = [
    {"risk": 0.0},
    {"risk": 0.6},
    {"risk": 0.95},
    {"taint": True},
    {"taint": True, "risk": 0.95},
]


@pytest.mark.parametrize("state", SESSIONS)
@pytest.mark.parametrize(
    ("principal", "mode", "action", "resource"),
    [
        (BARTEK, INT, Action.READ, "db:sales.payments"),
        (ANNA, INT, Action.EGRESS, "http:pastebin.com"),
        (ANNA, INT, Action.DELETE, "db:sales.orders"),
        (ETL, AU, Action.WRITE, "fs:secrets/keys.txt"),
        (ETL, AU, Action.GENERATE, "model:llama3:70b"),
    ],
)
def test_restrictions_never_turn_a_base_denied_call_into_allowed(
    evaluator, snapshot, make_ctx, now, state, principal, mode, action, resource
):
    ctx = make_ctx(mode, principal=principal.principal, actor=principal.agent, **state)
    decision = evaluator.decide(
        snapshot, principal, ctx, channel=MCP, action=action, resource=resource, now=now
    )
    assert decision.decision is Decision.BLOCK
    assert not decision.authorization.allowed
    assert decision.reason_code == decision.authorization.reason_code  # never an approval


@pytest.mark.parametrize("state", SESSIONS)
def test_restrictions_only_remove_or_condition_allowed_calls(
    evaluator, snapshot, make_ctx, now, state
):
    for principal, mode, action, resource in [
        (ANNA, INT, Action.WRITE, REPORT),
        (ANNA, INT, Action.READ, "db:sales.orders"),
        (ETL, AU, Action.WRITE, "fs:reports/nightly.md"),
    ]:
        ctx = make_ctx(mode, principal=principal.principal, actor=principal.agent, **state)
        decision = evaluator.decide(
            snapshot, principal, ctx, channel=MCP, action=action, resource=resource, now=now
        )
        assert decision.authorization.reason_code is AuthzReason.ALLOWED
        assert decision.decision in {Decision.ALLOW, Decision.BLOCK, Decision.REQUIRE_APPROVAL}
        if state == {"risk": 0.0}:
            assert decision.decision is Decision.ALLOW
        if mode is AU:
            assert decision.decision is not Decision.BLOCK  # autonomous: conditioned, not denied


def test_base_deny_still_reports_the_cooldown_it_triggers(evaluator, snapshot, interactive, now):
    # "A call denied in this state starts a cooldown" applies to base denials too.
    decision = evaluator.decide(
        snapshot,
        BARTEK,
        interactive(principal="bartek@demo", risk=0.6),
        channel=MCP,
        action=Action.READ,
        resource="db:sales.payments",
        now=now,
    )
    assert decision.decision is Decision.BLOCK
    assert decision.restriction.cooldown_on_deny is not None
    assert decision.restriction.cooldown_on_deny.key == "read:db:sales.payments"
