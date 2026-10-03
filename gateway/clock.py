"""Wall-clock time, injectable so tests can move it.

Every component that reasons about time (token expiry, session TTLs, risk decay, timers)
takes a `Clock` instead of calling `datetime.now` itself.
"""

from collections.abc import Callable
from datetime import UTC, datetime

type Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)
