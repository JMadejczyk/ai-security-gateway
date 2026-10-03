"""The background sweep the container runs: expire approvals past their deadline, close
approvals stuck in ``executing`` (their gateway died mid-call: ``uncertain``, never run again)
and keep ``acl_approvals_pending`` and ``acl_kill_switch_active`` in step with the store.

A consumed approval is ``executing`` for at most one upstream call, which every upstream
bounds by ``limits.upstream_timeout_s``; one still ``executing`` after that plus
``STUCK_SLACK_S`` has lost its outcome.

Expiry is also applied lazily whenever a record is read, so the sweep only bounds how long a
timed-out approval can sit in the pending gauge (``SWEEP_INTERVAL_S``) and makes timeouts
visible without anyone looking at the queue.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Final

from gateway.approvals.oversight import Oversight
from gateway.errors import RejectionError
from gateway.policy.loader import PolicySnapshot

logger = logging.getLogger(__name__)

SWEEP_INTERVAL_S: Final = 5.0
STUCK_SLACK_S: Final = 30.0


async def sweep_once(oversight: Oversight, policy: Callable[[], PolicySnapshot]) -> None:
    """One pass; a store outage is logged and retried on the next pass."""
    limits = policy().policy.limits
    try:
        await oversight.approvals.expire_due()
        await oversight.approvals.close_stuck(limits.upstream_timeout_s + STUCK_SLACK_S)
        await oversight.approvals.refresh_pending_gauge()
        await oversight.kill_switch.active()  # refreshes the cache and the gauge
    except RejectionError as exc:
        logger.warning("approval sweep skipped: %s", exc.reason_code)


async def sweep_forever(
    oversight: Oversight,
    policy: Callable[[], PolicySnapshot],
    interval_s: float = SWEEP_INTERVAL_S,
) -> None:
    while True:
        await sweep_once(oversight, policy)
        await asyncio.sleep(interval_s)
