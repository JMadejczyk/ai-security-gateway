"""Which actions a tools/list may show: risk-rule removals and freezes, without side effects."""

from datetime import timedelta

from gateway.core.types import Action, SessionMode


def test_no_freeze_at_low_risk(evaluator, snapshot, make_ctx, now):
    assert not evaluator.tools_frozen(snapshot, make_ctx(risk=0.3), now)


def test_freeze_threshold_holding_freezes_listing(evaluator, snapshot, make_ctx, now):
    assert evaluator.tools_frozen(snapshot, make_ctx(risk=0.9), now)


def test_running_freeze_freezes_listing_even_after_risk_decays(evaluator, snapshot, make_ctx, now):
    ctx = make_ctx(freeze_until=now + timedelta(seconds=60))
    assert evaluator.tools_frozen(snapshot, ctx, now)
    assert not evaluator.tools_frozen(snapshot, ctx, now + timedelta(seconds=61))


def test_autonomous_rules_have_no_freeze(evaluator, snapshot, make_ctx, now):
    ctx = make_ctx(
        SessionMode.AUTONOMOUS, principal="svc:nightly_etl", actor="nightly_etl", risk=0.9
    )
    assert not evaluator.tools_frozen(snapshot, ctx, now)


def test_taint_removes_write_for_interactive_sessions(evaluator, snapshot, make_ctx, now):
    assert evaluator.removed_actions(snapshot, make_ctx(taint=True), now) == {
        Action.WRITE,
        Action.DELETE,
        Action.EGRESS,
    }
