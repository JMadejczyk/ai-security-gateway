"""Shared fixtures: the root policy, snapshots built from edited copies, session contexts."""

import copy
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from gateway.core.envelope import SessionContext
from gateway.core.types import SessionMode
from gateway.policy.evaluator import PolicyEvaluator
from gateway.policy.loader import PolicyLoader, PolicySnapshot

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_POLICY = REPO_ROOT / "policy.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
NOW = datetime(2026, 10, 4, 10, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def root_policy_doc() -> dict[str, Any]:
    return yaml.safe_load(ROOT_POLICY.read_text())


@pytest.fixture
def policy_doc(root_policy_doc: dict[str, Any]) -> dict[str, Any]:
    """A mutable deep copy of the root policy document."""
    return copy.deepcopy(root_policy_doc)


@pytest.fixture(scope="session")
def loader() -> PolicyLoader:
    return PolicyLoader()


@pytest.fixture(scope="session")
def snapshot(loader: PolicyLoader) -> PolicySnapshot:
    return loader.load(ROOT_POLICY)


@pytest.fixture
def snapshot_from(loader: PolicyLoader) -> Callable[[dict[str, Any]], PolicySnapshot]:
    def build(document: dict[str, Any]) -> PolicySnapshot:
        return loader.parse(yaml.safe_dump(document).encode())

    return build


@pytest.fixture(scope="session")
def evaluator() -> PolicyEvaluator:
    return PolicyEvaluator()


type CtxFactory = Callable[..., SessionContext]


@pytest.fixture
def now() -> datetime:
    return NOW


@pytest.fixture
def make_ctx() -> CtxFactory:
    """Build a session snapshot at NOW; keyword overrides go straight to SessionContext."""

    def build(
        mode: SessionMode = SessionMode.INTERACTIVE,
        *,
        principal: str = "anna@demo",
        actor: str = "databot",
        **overrides: Any,
    ) -> SessionContext:
        fields: dict[str, Any] = {
            "session_id": "s-test",
            "principal": principal,
            "actor": actor,
            "mode": mode,
            "risk_updated_at": NOW,
            "created_at": NOW,
            "last_seen": NOW,
        }
        fields.update(overrides)
        return SessionContext.model_validate(fields)

    return build
