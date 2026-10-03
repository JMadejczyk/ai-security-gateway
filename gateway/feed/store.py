"""The live signature feed: loaded at boot, refreshed every ``refresh_s``, last valid kept.

Boot policy: a gateway whose policy configures a feed refuses to start if that feed cannot be
loaded, like it refuses to start without a valid policy. Starting with no signatures would
silently run the ``signatures`` control open, and an operator watching the logs of a running
gateway is better placed to react to a later failure than a gateway that never got a feed is.
After boot every failure (unreachable, oversized, invalid) is logged and counted
(``acl_feed_reloads_total{result}``) and the last valid feed stays in effect.

Where the feed lives comes from the *current* policy on every refresh
(``controls.signatures.feed``), so changing it in ``policy.yaml`` takes effect at the next
refresh without a restart; a relative path is resolved against the policy file's directory.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Final

import httpx

from gateway.core.envelope import FrozenModel
from gateway.feed.schema import (
    EMPTY_FEED,
    MAX_FEED_BYTES,
    FeedError,
    FeedUnavailableError,
    SignatureFeed,
    parse_feed,
)
from gateway.feed.sources import FeedSource, resolve_source
from gateway.policy.loader import PolicySnapshot
from gateway.policy.schema import SignaturesConfig
from gateway.telemetry import FeedReloadResult, record_feed_reload, set_active_feed_version

logger = logging.getLogger(__name__)

SIGNATURES: Final = "signatures"


class FeedRefresh(FrozenModel):
    result: FeedReloadResult
    version: str | None  # the version in effect after this attempt
    error: str | None = None


def feed_config(snapshot: PolicySnapshot) -> SignaturesConfig:
    config = snapshot.policy.control_config(SIGNATURES)
    return config if isinstance(config, SignaturesConfig) else SignaturesConfig()


class FeedStore:
    """Holds the current `SignatureFeed`; readers take ``store.current`` and keep it."""

    def __init__(
        self,
        policy: Callable[[], PolicySnapshot],
        *,
        max_bytes: int = MAX_FEED_BYTES,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._policy = policy
        self._max_bytes = max_bytes
        self._transport = transport
        self._current = EMPTY_FEED
        self._lock = asyncio.Lock()
        set_active_feed_version(None)

    @classmethod
    def boot(
        cls,
        policy: Callable[[], PolicySnapshot],
        *,
        max_bytes: int = MAX_FEED_BYTES,
        transport: httpx.BaseTransport | None = None,
    ) -> "FeedStore":
        """Load the configured feed. Raises `FeedError`: a configured feed must be valid."""
        store = cls(policy, max_bytes=max_bytes, transport=transport)
        source = store._source(policy())
        if source is not None:
            store._swap(store._load(source))
            record_feed_reload(FeedReloadResult.OK)
        return store

    @property
    def current(self) -> SignatureFeed:
        return self._current

    @property
    def version(self) -> str | None:
        return self._current.version

    async def refresh(self) -> FeedRefresh:
        """Re-read the configured feed and swap it in if it is valid and different."""
        async with self._lock:
            previous = self._current
            try:
                source = self._source(self._policy())
                candidate = (
                    EMPTY_FEED if source is None else await asyncio.to_thread(self._load, source)
                )
            except FeedError as exc:
                result = (
                    FeedReloadResult.UNAVAILABLE
                    if isinstance(exc, FeedUnavailableError)
                    else FeedReloadResult.INVALID
                )
                logger.error("signature feed refresh failed, keeping %s: %s", previous.version, exc)  # noqa: TRY400 -- the message is the operator-facing error
                return self._finish(result, error=str(exc))
            except Exception as exc:
                # A bug in loading must never take the working feed (or the refresh loop) down.
                logger.exception("signature feed refresh failed unexpectedly")
                return self._finish(FeedReloadResult.INVALID, error=type(exc).__name__)
            if candidate.digest == previous.digest:
                return self._finish(FeedReloadResult.UNCHANGED)
            if candidate.version == previous.version and previous is not EMPTY_FEED:
                logger.warning("signature feed content changed but its version did not")
            self._swap(candidate)
            logger.info("signature feed: %s -> %s", previous.version, candidate.version)
            return self._finish(FeedReloadResult.OK)

    async def run(self) -> None:
        """Refresh every ``controls.signatures.refresh_s`` until the task is cancelled."""
        while True:
            await asyncio.sleep(feed_config(self._policy()).refresh_s)
            with contextlib.suppress(Exception):  # refresh() logs and counts its own failures
                await self.refresh()

    # ----------------------------------------------------------------------- helpers

    def _source(self, snapshot: PolicySnapshot) -> FeedSource | None:
        spec = feed_config(snapshot).feed
        if spec is None:
            return None
        base_dir = snapshot.source.parent if snapshot.source is not None else Path.cwd()
        return resolve_source(spec, base_dir=base_dir, transport=self._transport)

    def _load(self, source: FeedSource) -> SignatureFeed:
        return parse_feed(source.fetch(self._max_bytes), max_bytes=self._max_bytes)

    def _swap(self, feed: SignatureFeed) -> None:
        self._current = feed
        set_active_feed_version(feed.version)

    def _finish(self, result: FeedReloadResult, *, error: str | None = None) -> FeedRefresh:
        record_feed_reload(result)
        return FeedRefresh(result=result, version=self._current.version, error=error)
