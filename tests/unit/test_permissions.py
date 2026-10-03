"""Permission grammar: first-colon split, `*` as the only anchored wildcard, strict parsing."""

import pytest
from pydantic import BaseModel, ValidationError

from gateway.core.types import Action
from gateway.policy.permissions import (
    Permission,
    PermissionSet,
    Resource,
    ResourcePattern,
    glob_match,
)


@pytest.mark.parametrize(
    ("text", "action", "namespace", "identifier"),
    [
        ("generate:model:qwen3:8b", Action.GENERATE, "model", "qwen3:8b"),
        ("read:db:sales.*", Action.READ, "db", "sales.*"),
        ("write:fs:reports/*", Action.WRITE, "fs", "reports/*"),
        ("egress:http:a:b:c", Action.EGRESS, "http", "a:b:c"),
    ],
)
def test_split_at_first_colon_keeps_later_colons(text, action, namespace, identifier):
    permission = Permission.parse(text)
    assert permission.action == action
    assert permission.resource == ResourcePattern(namespace, identifier)
    assert str(permission) == text


@pytest.mark.parametrize(
    ("pattern", "action", "resource", "expected"),
    [
        # whole-resource and whole-action wildcards
        ("*:*", Action.DELETE, "db:anything.at:all", True),
        ("egress:*", Action.EGRESS, "http:pastebin.com", True),
        ("egress:*", Action.READ, "http:pastebin.com", False),
        ("*:db:sales.orders", Action.WRITE, "db:sales.orders", True),
        # '*' crosses '.' and ':'
        ("read:db:sales.*", Action.READ, "db:sales.orders", True),
        ("read:db:sales.*", Action.READ, "db:sales.orders.archive", True),
        ("generate:model:*", Action.GENERATE, "model:qwen3:8b", True),
        ("generate:model:qwen3:*", Action.GENERATE, "model:qwen3:8b", True),
        ("generate:model:*:8b", Action.GENERATE, "model:qwen3:8b", True),
        ("read:db:*.customers", Action.READ, "db:sales.customers", True),
        # anchored on the whole string
        ("read:db:sales", Action.READ, "db:sales.orders", False),
        ("read:db:orders", Action.READ, "db:sales.orders", False),
        ("read:db:sales.orders", Action.READ, "db:sales.orders2", False),
        ("generate:model:qwen3:8b", Action.GENERATE, "model:qwen3:8b-instruct", False),
        # exact concrete match
        ("generate:model:qwen3:8b", Action.GENERATE, "model:qwen3:8b", True),
        ("generate:model:qwen3:8b", Action.GENERATE, "model:llama3:70b", False),
        # namespace is literal, never wildcarded by the identifier
        ("read:db:*", Action.READ, "web:db", False),
        ("read:web:*", Action.READ, "webx:example.com", False),
        # case-sensitive
        ("read:db:Sales.*", Action.READ, "db:sales.orders", False),
        ("read:DB:sales.*", Action.READ, "db:sales.orders", False),
        # characters fnmatch would treat specially are literal in resources
        ("read:fs:reports/*", Action.READ, "fs:reports/[draft]?.md", True),
    ],
)
def test_matching(pattern, action, resource, expected):
    assert Permission.parse(pattern).matches(action, Resource.parse(resource)) is expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "read",  # no resource
        "read:",  # empty resource
        "read:db",  # resource without namespace separator
        "read:db:",  # empty identifier
        "read::sales",  # empty namespace
        ":db:sales",  # empty action
        "fetch:db:sales",  # unknown action
        "READ:db:sales",  # actions are case-sensitive
        "read:*:sales",  # namespace is literal, not a pattern
        "read:d?:sales",  # invalid namespace characters
        "read:db:sales.?",  # no fnmatch '?'
        "read:db:sales.[ab]",  # no fnmatch '[...]'
        "read:db:sales orders",  # whitespace
        "read:db:sales\norders",  # control characters
    ],
)
def test_invalid_permissions_raise(text):
    with pytest.raises(ValueError, match=r"permission|resource"):
        Permission.parse(text)


@pytest.mark.parametrize(
    "text",
    ["db", "db:", ":x", "db:sales.*", "*", "db:a b", "1db:x"],
)
def test_concrete_resource_rejects_patterns_and_bad_shapes(text):
    with pytest.raises(ValueError, match="resource"):
        Resource.parse(text)


@pytest.mark.parametrize(
    ("pattern", "text", "expected"),
    [
        ("", "", True),
        ("*", "", True),
        ("a*", "a", True),
        ("*a*a*", "aa", True),
        ("*a*a*", "a", False),
        ("ab*ba", "aba", False),
        ("a**b", "ab", True),
        ("x*y*z", "xyyzz", True),
    ],
)
def test_glob_match_edges(pattern, text, expected):
    assert glob_match(pattern, text) is expected


def test_permission_set_matches_any_entry_and_parses_strings():
    grants = PermissionSet.parse(["read:db:sales.orders", "generate:model:qwen3:8b"])
    assert grants.allows(Action.READ, "db:sales.orders")
    assert grants.allows(Action.GENERATE, Resource.parse("model:qwen3:8b"))
    assert not grants.allows(Action.READ, "db:sales.payments")
    assert not PermissionSet().allows(Action.READ, "db:sales.orders")


def test_restricted_to_narrows_wildcard_actions():
    grants = PermissionSet.parse(["*:db:sales.*", "delete:db:sales.*"])
    narrowed = grants.restricted_to([Action.READ])
    assert narrowed.allows(Action.READ, "db:sales.orders")
    assert not narrowed.allows(Action.DELETE, "db:sales.orders")
    assert not narrowed.allows(Action.WRITE, "db:sales.orders")


class _Holder(BaseModel):
    grants: PermissionSet


def test_permission_set_as_pydantic_field_round_trips():
    holder = _Holder.model_validate({"grants": ["read:db:sales.*", "*:*"]})
    assert holder.grants.allows(Action.DELETE, "fs:x")
    assert holder.model_dump(mode="json") == {"grants": ["read:db:sales.*", "*:*"]}


def test_permission_set_field_reports_grammar_errors():
    with pytest.raises(ValidationError, match="unknown action"):
        _Holder.model_validate({"grants": ["fetch:db:x"]})
