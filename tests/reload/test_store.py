"""Hot reload: valid changes swap atomically, invalid files never replace the working policy."""

import asyncio
import contextlib
import shutil
from pathlib import Path

import pytest

from gateway.core.types import Action, SessionMode
from gateway.policy.evaluator import PolicyEvaluator, PrincipalContext
from gateway.policy.loader import PolicyLoadError, PolicySnapshot
from gateway.policy.store import PolicyStore
from gateway.telemetry import REGISTRY, ReloadResult

ROOT_POLICY = Path(__file__).resolve().parents[2] / "policy.yaml"
BARTEK = PrincipalContext(
    principal="bartek@demo", roles=("intern",), agent="databot", mode=SessionMode.INTERACTIVE
)
EVALUATOR = PolicyEvaluator()
GRANT_PAYMENTS = (
    '"read:db:sales.orders", "read:db:sales.customers",',
    '"read:db:sales.orders", "read:db:sales.customers", "read:db:sales.payments",',
)


def reloads(result: ReloadResult) -> float:
    return REGISTRY.get_sample_value("acl_policy_reloads_total", {"result": result.value}) or 0.0


def active_revision_value(revision: str) -> float | None:
    return REGISTRY.get_sample_value("acl_policy_info", {"revision": revision})


def bartek_may_read_payments(snapshot: PolicySnapshot) -> bool:
    return EVALUATOR.authorize(snapshot, BARTEK, Action.READ, "db:sales.payments").allowed


@pytest.fixture
def policy_path(tmp_path: Path) -> Path:
    path = tmp_path / "policy.yaml"
    shutil.copy(ROOT_POLICY, path)
    return path


@pytest.fixture
def store(policy_path: Path) -> PolicyStore:
    return PolicyStore.from_path(policy_path)


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def test_valid_reload_changes_revision_and_decision(store, policy_path):
    before = store.current
    assert not bartek_may_read_payments(before)
    ok_before = reloads(ReloadResult.OK)

    edit(policy_path, *GRANT_PAYMENTS)
    outcome = store.reload()

    assert outcome.result is ReloadResult.OK
    assert outcome.previous_revision == before.revision
    assert outcome.revision == store.current.revision != before.revision
    assert bartek_may_read_payments(store.current)
    assert not bartek_may_read_payments(before)  # the old snapshot is immutable
    assert reloads(ReloadResult.OK) == ok_before + 1
    assert active_revision_value(store.current.revision) == 1
    assert active_revision_value(before.revision) is None


def test_invalid_file_keeps_last_valid_snapshot(store, policy_path):
    before = store.current
    invalid_before = reloads(ReloadResult.INVALID)

    edit(policy_path, "default: deny", "default: allow")
    outcome = store.reload()

    assert outcome.result is ReloadResult.INVALID
    assert outcome.error is not None
    assert "default" in outcome.error
    assert outcome.revision == before.revision
    assert store.current is before
    assert reloads(ReloadResult.INVALID) == invalid_before + 1
    assert active_revision_value(before.revision) == 1


@pytest.mark.parametrize(
    "breakage",
    [
        "risk_rules: [unclosed",
        "\x00",
        "roles:\n  admin: { allow: ['*:*'] }\nroles: {}\n",
    ],
    ids=["broken-yaml", "binary", "duplicate-key"],
)
def test_garbage_never_replaces_working_policy(store, policy_path, breakage):
    before = store.current
    policy_path.write_text(policy_path.read_text() + "\n" + breakage)
    assert store.reload().result is ReloadResult.INVALID
    assert store.current is before


def test_deleted_file_keeps_working_policy(store, policy_path):
    before = store.current
    policy_path.unlink()
    outcome = store.reload()
    assert outcome.result is ReloadResult.INVALID
    assert store.current is before


def test_recovery_after_invalid_reload(store, policy_path):
    original = policy_path.read_text()
    policy_path.write_text("nonsense: true\n")
    assert store.reload().result is ReloadResult.INVALID
    policy_path.write_text(original.replace(*GRANT_PAYMENTS))
    assert store.reload().result is ReloadResult.OK
    assert bartek_may_read_payments(store.current)


def test_unchanged_file_reports_unchanged(store, policy_path):
    before = store.current
    unchanged_before = reloads(ReloadResult.UNCHANGED)
    # A comment and reformatting do not change the canonical document.
    policy_path.write_text("# touched\n" + policy_path.read_text())
    outcome = store.reload()
    assert outcome.result is ReloadResult.UNCHANGED
    assert store.current is before
    assert reloads(ReloadResult.UNCHANGED) == unchanged_before + 1


def test_initial_invalid_policy_refuses_to_start(tmp_path: Path):
    path = tmp_path / "policy.yaml"
    path.write_text(ROOT_POLICY.read_text().replace("default: deny", "default: allow"))
    with pytest.raises(PolicyLoadError, match="default"):
        PolicyStore.from_path(path)


def test_missing_initial_policy_refuses_to_start(tmp_path: Path):
    with pytest.raises(PolicyLoadError, match="cannot read"):
        PolicyStore.from_path(tmp_path / "absent.yaml")


def test_listeners_notified_only_on_successful_swap(store, policy_path):
    seen: list[tuple[str, str]] = []
    unsubscribe = store.subscribe(lambda old, new: seen.append((old.revision, new.revision)))

    store.reload()  # unchanged
    policy_path.write_text("broken: [")
    store.reload()  # invalid
    assert seen == []

    shutil.copy(ROOT_POLICY, policy_path)
    edit(policy_path, *GRANT_PAYMENTS)
    store.reload()
    assert len(seen) == 1
    assert seen[0][1] == store.current.revision

    unsubscribe()
    edit(policy_path, "half_life_s: 600", "half_life_s: 300")
    assert store.reload().result is ReloadResult.OK
    assert len(seen) == 1


def test_failing_listener_does_not_undo_reload(store, policy_path):
    def explode(_old: PolicySnapshot, _new: PolicySnapshot) -> None:
        raise RuntimeError("annotation backend down")

    store.subscribe(explode)
    edit(policy_path, *GRANT_PAYMENTS)
    assert store.reload().result is ReloadResult.OK
    assert bartek_may_read_payments(store.current)


async def test_watcher_picks_up_file_change(store, policy_path):
    before = store.current.revision
    swapped = asyncio.Event()
    store.subscribe(lambda _old, _new: swapped.set())

    watcher = asyncio.create_task(store.watch(debounce_ms=50))
    try:
        await asyncio.sleep(0.3)  # let the watcher register before the edit
        edit(policy_path, *GRANT_PAYMENTS)
        await asyncio.wait_for(swapped.wait(), timeout=10)
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher

    assert store.current.revision != before
    assert bartek_may_read_payments(store.current)
    assert watcher.cancelled()


async def test_watcher_ignores_invalid_edit_and_sibling_files(store, policy_path):
    before = store.current
    invalid_before = reloads(ReloadResult.INVALID)
    watcher = asyncio.create_task(store.watch(debounce_ms=50))
    try:
        await asyncio.sleep(0.3)
        (policy_path.parent / "other.yaml").write_text("unrelated: true\n")
        policy_path.write_text("default: allow\n")
        for _ in range(100):
            if reloads(ReloadResult.INVALID) > invalid_before:
                break
            await asyncio.sleep(0.1)
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher

    assert reloads(ReloadResult.INVALID) > invalid_before
    assert store.current is before
