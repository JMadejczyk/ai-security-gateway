"""Throttler: every cap enforced, rejections not counted, exponential backoff with reset."""

from datetime import timedelta

import pytest
from gateway_testkit import T0

from gateway.policy.schema import Throttle, ThrottleBackoff
from gateway.throttle import ThrottledError, Throttler

BACKOFF = ThrottleBackoff(base_s=5, max_s=30)


def admit(throttler, caps, at_s: float, agent: str = "etl") -> int | None:
    """None when admitted, else the Retry-After seconds."""
    try:
        throttler.admit(agent, caps, BACKOFF, T0 + timedelta(seconds=at_s))
    except ThrottledError as exc:
        return exc.retry_after_s
    return None


def test_sliding_window_and_backoff_cap():
    throttler, caps = Throttler(), [Throttle(max_actions=2, per_s=10)]
    assert [admit(throttler, caps, t) for t in (0, 1)] == [None, None]
    assert [admit(throttler, caps, t) for t in (2, 3, 4, 5)] == [5, 10, 20, 30]  # capped
    assert admit(throttler, caps, 60) is None  # backoff over plus a clean window: reset
    assert admit(throttler, caps, 60.5) is None
    assert admit(throttler, caps, 61) == 5


def test_every_cap_applies():
    throttler = Throttler()
    caps = [Throttle(max_actions=1, per_s=1), Throttle(max_actions=2, per_s=60)]
    assert admit(throttler, caps, 0) is None
    assert admit(throttler, caps, 30) is None
    assert admit(throttler, caps, 45) == 5  # the per-second cap is clear, the per-minute is not


def test_agents_are_counted_separately():
    throttler, caps = Throttler(), [Throttle(max_actions=1, per_s=10)]
    assert admit(throttler, caps, 0, "a") is None
    assert admit(throttler, caps, 0, "b") is None
    assert admit(throttler, caps, 1, "a") == 5


@pytest.mark.parametrize("seconds", [0.2, 4.01])
def test_retry_after_is_a_whole_second_at_least(seconds):
    assert ThrottledError(seconds).retry_after_s == max(1, -(-seconds // 1))
