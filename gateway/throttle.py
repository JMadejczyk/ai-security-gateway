"""Throttling of autonomous agents at elevated risk (SPEC "The reaction depends on session mode").

``throttle`` in a risk rule caps an agent at ``max_actions`` per ``per_s``. Every call over a
cap is rejected with ``Retry-After``; consecutive rejections grow it exponentially from
``throttle.base_s`` up to ``throttle.max_s``, and the growth resets once a full window passes
after the backoff without a violation.

State is per agent and per cap (a sliding window of admitted calls), held in memory by the one
gateway process; checks are synchronous, so two calls on the event loop never interleave
inside one check.
"""

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Final

from gateway.errors import RejectionError
from gateway.policy.schema import Throttle, ThrottleBackoff

_MAX_DOUBLINGS: Final = 62  # 2**62 * base_s is far past any max_s; avoids float overflow


class ThrottledError(RejectionError):
    """The call exceeds a throttle cap: HTTP 429 with ``Retry-After``."""

    status_code = 429

    def __init__(self, retry_after_s: float) -> None:
        self.retry_after_s = max(1, math.ceil(retry_after_s))
        super().__init__("throttled", f"rate limited; retry after {self.retry_after_s} s")


@dataclass(slots=True)
class _Window:
    admitted: deque[datetime] = field(default_factory=deque[datetime])
    rejections: int = 0  # consecutive
    blocked_until: datetime | None = None


class Throttler:
    """Sliding-window caps per ``(agent, cap)`` with exponential backoff on rejection."""

    def __init__(self) -> None:
        self._windows: dict[tuple[str, int, float], _Window] = {}

    def admit(
        self, agent: str, caps: Sequence[Throttle], backoff: ThrottleBackoff, now: datetime
    ) -> None:
        """Record the call against every cap, or raise `ThrottledError` if any is exceeded.

        A rejected call is not recorded: it did not run.
        """
        windows = [(cap, self._window(agent, cap, now)) for cap in caps]
        violated = [(cap, window) for cap, window in windows if _over(cap, window, now)]
        if not violated:
            for _, window in windows:
                window.admitted.append(now)
            return
        delay = 0.0
        for _, window in violated:
            window.rejections += 1
            doublings = min(window.rejections - 1, _MAX_DOUBLINGS)
            step = min(backoff.base_s * 2.0**doublings, backoff.max_s)
            window.blocked_until = now + timedelta(seconds=step)
            delay = max(delay, step)
        raise ThrottledError(delay)

    def _window(self, agent: str, cap: Throttle, now: datetime) -> _Window:
        window = self._windows.setdefault((agent, cap.max_actions, cap.per_s), _Window())
        span = timedelta(seconds=cap.per_s)
        while window.admitted and window.admitted[0] <= now - span:
            window.admitted.popleft()
        # One full window after the backoff without a violation: start over from base_s.
        if window.blocked_until is not None and now >= window.blocked_until + span:
            window.rejections, window.blocked_until = 0, None
        return window


def _over(cap: Throttle, window: _Window, now: datetime) -> bool:
    backing_off = window.blocked_until is not None and now < window.blocked_until
    return backing_off or len(window.admitted) >= cap.max_actions
