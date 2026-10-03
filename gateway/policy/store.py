"""The live policy: an atomically swapped snapshot, reloaded from disk without restart."""

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from watchfiles import (
    Change,
    awatch,  # pyright: ignore[reportUnknownVariableType] -- stop_event's union names trio's untyped Event
)

from gateway.core.envelope import FrozenModel
from gateway.policy.loader import PolicyLoader, PolicyLoadError, PolicySnapshot
from gateway.telemetry import ReloadResult, record_policy_reload, set_active_policy_revision

logger = logging.getLogger(__name__)

type ReloadListener = Callable[[PolicySnapshot, PolicySnapshot], None]
"""Called with ``(previous, current)`` after a successful swap."""


class ReloadOutcome(FrozenModel):
    result: ReloadResult
    revision: str  # the revision in effect after this attempt
    previous_revision: str
    error: str | None = None


class PolicyStore:
    """Holds the current snapshot and replaces it only with a fully validated one.

    Readers take ``store.current`` once per call and keep using that snapshot, so a
    reload mid-call never mixes two policy versions. An invalid file is logged, counted
    and never replaces the working snapshot.
    """

    def __init__(self, path: Path, snapshot: PolicySnapshot, loader: PolicyLoader) -> None:
        self._path = path
        self._loader = loader
        self._current = snapshot
        self._reload_lock = threading.Lock()
        self._listeners: list[ReloadListener] = []
        set_active_policy_revision(snapshot.revision)

    @classmethod
    def from_path(cls, path: Path, loader: PolicyLoader | None = None) -> "PolicyStore":
        """Load the initial policy. Raises `PolicyLoadError`: no valid policy, no gateway."""
        loader = loader or PolicyLoader()
        resolved = path.resolve()
        return cls(resolved, loader.load(resolved), loader)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def current(self) -> PolicySnapshot:
        return self._current

    def subscribe(self, listener: ReloadListener) -> Callable[[], None]:
        """Register a listener for successful reloads; returns an unsubscribe function."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def reload(self) -> ReloadOutcome:
        """Re-read the file and swap it in if it is valid and different."""
        with self._reload_lock:
            previous = self._current
            try:
                candidate = self._loader.load(self._path)
            except PolicyLoadError as exc:
                logger.error("policy reload rejected, keeping %s: %s", previous.revision, exc)  # noqa: TRY400 -- the message is the operator-facing error; a traceback adds nothing
                return self._finish(ReloadResult.INVALID, previous, previous, error=str(exc))
            except Exception as exc:
                # A loader bug must never take the working policy (or the watcher) down.
                logger.exception("policy reload failed unexpectedly, keeping %s", previous.revision)
                error = f"unexpected {type(exc).__name__} while loading the policy"
                return self._finish(ReloadResult.INVALID, previous, previous, error=error)
            if candidate.digest == previous.digest:
                return self._finish(ReloadResult.UNCHANGED, previous, previous)
            self._current = candidate
            set_active_policy_revision(candidate.revision)
            logger.info("policy reloaded: %s -> %s", previous.revision, candidate.revision)
        for listener in list(self._listeners):
            try:
                listener(previous, candidate)
            except Exception:
                logger.exception("policy reload listener failed")
        return self._finish(ReloadResult.OK, previous, candidate)

    async def watch(self, *, debounce_ms: int = 1_600, force_polling: bool | None = None) -> None:
        """Reload whenever the policy file changes. Runs until the task is cancelled.

        Watches the parent directory, so editors that save by rename and bind-mounted files
        that are replaced are picked up too.
        """
        name = self._path.name

        def is_policy_file(_: Change, changed: str) -> bool:
            return Path(changed).name == name

        async for _changes in awatch(
            self._path.parent,
            watch_filter=is_policy_file,
            debounce=debounce_ms,
            force_polling=force_polling,
        ):
            try:
                self.reload()
            except Exception:
                logger.exception("policy reload raised; the watcher keeps running")

    @staticmethod
    def _finish(
        result: ReloadResult,
        previous: PolicySnapshot,
        current: PolicySnapshot,
        *,
        error: str | None = None,
    ) -> ReloadOutcome:
        record_policy_reload(result)
        return ReloadOutcome(
            result=result,
            revision=current.revision,
            previous_revision=previous.revision,
            error=error,
        )
