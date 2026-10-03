"""The background sweep the container runs: expire approvals past their deadline and keep
``acl_approvals_pending`` and ``acl_kill_switch_active`` in step with the shared store.

Expiry is also applied lazily whenever a record is read, so the sweep only bounds how long a
timed-out approval can sit in the pending gauge (``SWEEP_INTERVAL_S``) and makes timeouts
visible without anyone looking at the queue.
"""

import asyncio
import logging
from typing import Final

from gateway.approvals.oversight import Oversight
from gateway.errors import RejectionError

logger = logging.getLogger(__name__)

SWEEP_INTERVAL_S: Final = 5.0


async def sweep_once(oversight: Oversight) -> None:
    """One pass; a store outage is logged and retried on the next pass."""
    try:
        await oversight.approvals.expire_due()
        await oversight.approvals.refresh_pending_gauge()
        await oversight.kill_switch.active()  # refreshes the cache and the gauge
    except RejectionError as exc:
        logger.warning("approval sweep skipped: %s", exc.reason_code)


async def sweep_forever(oversight: Oversight, interval_s: float = SWEEP_INTERVAL_S) -> None:
    while True:
        await sweep_once(oversight)
        await asyncio.sleep(interval_s)
