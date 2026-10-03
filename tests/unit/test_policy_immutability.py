"""A published snapshot cannot change without changing its revision: every level is frozen."""

import copy
import pickle
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from gateway.core.frozen import FrozenDict
from gateway.core.types import Action, SessionMode
from gateway.policy.evaluator import PolicyEvaluator, PrincipalContext
from gateway.policy.loader import PolicySnapshot, canonical_digest
from gateway.policy.permissions import PermissionSet

BARTEK = PrincipalContext(
    principal="bartek@demo", roles=("intern",), agent="databot", mode=SessionMode.INTERACTIVE
)


def _set_role(s: PolicySnapshot) -> None:
    s.policy.roles["intern"] = s.policy.roles["analyst"]  # type: ignore[index]


def _del_role(s: PolicySnapshot) -> None:
    del s.policy.roles["intern"]  # type: ignore[attr-defined]


def _add_agent(s: PolicySnapshot) -> None:
    s.policy.agents["rogue"] = s.policy.agents["databot"]  # type: ignore[index]


def _add_server(s: PolicySnapshot) -> None:
    s.policy.upstreams.mcp["evil"] = s.policy.upstreams.mcp["web"]  # type: ignore[index]


def _remap_tool(s: PolicySnapshot) -> None:
    tools = s.policy.upstreams.mcp["sales_db"].tools
    tools["drop"] = tools["query"]  # type: ignore[index]


def _swap_permissions(s: PolicySnapshot) -> None:
    grants = s.policy.roles["intern"].allow
    grants._permissions = s.policy.roles["admin"].allow._permissions  # type: ignore[misc]


def _del_permissions(s: PolicySnapshot) -> None:
    del s.policy.roles["intern"].allow._permissions


def _replace_frozen_dict_data(s: PolicySnapshot) -> None:
    s.policy.roles._data = {}  # type: ignore[misc]


def _write_frozen_dict_data(s: PolicySnapshot) -> None:
    s.policy.roles._data["intern"] = s.policy.roles["admin"]  # type: ignore[index]


def _assign_model_field(s: PolicySnapshot) -> None:
    s.policy.roles["intern"].allow = PermissionSet.parse(["*:*"])  # type: ignore[misc]


def _assign_snapshot_policy(s: PolicySnapshot) -> None:
    s.policy = s.policy  # type: ignore[misc]


MUTATIONS: list[Callable[[PolicySnapshot], None]] = [
    _set_role,
    _del_role,
    _add_agent,
    _add_server,
    _remap_tool,
    _swap_permissions,
    _del_permissions,
    _replace_frozen_dict_data,
    _write_frozen_dict_data,
    _assign_model_field,
    _assign_snapshot_policy,
]


@pytest.mark.parametrize("mutate", MUTATIONS, ids=lambda f: f.__name__.lstrip("_"))
def test_published_snapshot_rejects_mutation(snapshot_from, policy_doc, mutate):
    snapshot = snapshot_from(policy_doc)
    evaluator = PolicyEvaluator()
    before = evaluator.authorize(snapshot, BARTEK, Action.READ, "db:sales.payments")

    with pytest.raises((TypeError, AttributeError, ValidationError)):
        mutate(snapshot)

    assert canonical_digest(snapshot.policy) == snapshot.digest
    after = evaluator.authorize(snapshot, BARTEK, Action.READ, "db:sales.payments")
    assert after == before
    assert not after.allowed


def test_published_mappings_are_frozen(snapshot):
    policy = snapshot.policy
    for mapping in (
        policy.roles,
        policy.agents,
        policy.upstreams.mcp,
        policy.upstreams.mcp["web"].tools,
    ):
        assert isinstance(mapping, FrozenDict)


def test_frozen_policy_still_copies_pickles_and_dumps(snapshot):
    policy = snapshot.policy
    assert canonical_digest(copy.deepcopy(policy)) == snapshot.digest
    assert canonical_digest(policy.model_copy(deep=True)) == snapshot.digest
    restored: Any = pickle.loads(pickle.dumps(policy))  # noqa: S301 -- round-tripping our own object
    assert canonical_digest(restored) == snapshot.digest
    assert policy.model_dump(mode="json")["roles"]["intern"]["allow"][0] == "read:db:sales.orders"


def test_frozen_dict_behaves_as_a_read_only_mapping():
    frozen = FrozenDict({"a": 1})
    assert dict(frozen) == {"a": 1}
    assert frozen == {"a": 1}
    assert "a" in frozen
    assert len(frozen) == 1
    with pytest.raises(TypeError):
        frozen["b"] = 2  # type: ignore[index]
    with pytest.raises(AttributeError):
        frozen.extra = 1  # type: ignore[attr-defined]
