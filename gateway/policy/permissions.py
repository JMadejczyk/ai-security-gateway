"""Permission grammar (SPEC "Permission grammar").

    permission       := action ":" resource_pattern        # split at the FIRST colon only
    action           := "read" | "write" | ... | "generate" | "*"
    resource_pattern := "*" | namespace ":" identifier_pattern   # later colons preserved

``*`` is the only wildcard. It matches any run of characters (including ``.`` and ``:``),
matching is anchored on the whole string and case-sensitive. There is no fnmatch: ``?``
and ``[...]`` are rejected in patterns instead of being given a meaning.

Grants are always matched against one concrete ``(action, resource)``; pattern sets are
never intersected symbolically.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import ClassVar, Literal, Self, cast

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from gateway.core.types import Action

WILDCARD = "*"
type ActionPattern = Action | Literal["*"]

_NAMESPACE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
# No whitespace or control characters anywhere in a resource or pattern.
_FORBIDDEN_ANYWHERE = re.compile(r"[\s\x00-\x1f\x7f]")
_FNMATCH_SYNTAX = frozenset("?[]")


def glob_match(pattern: str, text: str) -> bool:
    """Anchored match where ``*`` matches any (possibly empty) run of characters.

    Linear-time segment scan instead of a regex: leftmost placement of each literal
    segment between stars is always optimal when ``*`` is the only metacharacter.
    """
    first, *rest = pattern.split(WILDCARD)
    if not rest:
        return pattern == text
    *middle, last = rest
    if len(text) < len(first) + len(last):
        return False
    if not (text.startswith(first) and text.endswith(last)):
        return False
    position, end = len(first), len(text) - len(last)
    for segment in middle:
        if not segment:
            continue
        found = text.find(segment, position, end)
        if found < 0:
            return False
        position = found + len(segment)
    return True


def _split_namespace(value: str, what: str) -> tuple[str, str]:
    namespace, sep, identifier = value.partition(":")
    if not sep:
        msg = f"{what} {value!r} must have the form 'namespace:identifier'"
        raise ValueError(msg)
    if not _NAMESPACE.fullmatch(namespace):
        msg = f"{what} {value!r} has an invalid namespace {namespace!r} (letters, digits, _ or -)"
        raise ValueError(msg)
    if not identifier:
        msg = f"{what} {value!r} has an empty identifier"
        raise ValueError(msg)
    if _FORBIDDEN_ANYWHERE.search(value):
        msg = f"{what} {value!r} contains whitespace or control characters"
        raise ValueError(msg)
    return namespace, identifier


@dataclass(frozen=True, slots=True)
class Resource:
    """A concrete resource on an interaction: ``namespace:identifier``, no wildcard."""

    namespace: str
    identifier: str

    @classmethod
    def parse(cls, value: str) -> Self:
        namespace, identifier = _split_namespace(value, "resource")
        if WILDCARD in identifier:
            msg = f"resource {value!r} is concrete and may not contain '*'"
            raise ValueError(msg)
        return cls(namespace, identifier)

    def __str__(self) -> str:
        return f"{self.namespace}:{self.identifier}"


@dataclass(frozen=True, slots=True)
class ResourcePattern:
    """``*`` (every resource) or ``namespace:identifier_pattern`` with a literal namespace."""

    namespace: str | None  # None only for the whole-resource wildcard
    identifier: str

    ANY: ClassVar["ResourcePattern"]

    @classmethod
    def parse(cls, value: str) -> "ResourcePattern":
        if value == WILDCARD:
            return cls.ANY
        namespace, identifier = _split_namespace(value, "resource pattern")
        if bad := _FNMATCH_SYNTAX.intersection(identifier):
            msg = (
                f"resource pattern {value!r} uses {''.join(sorted(bad))!r}; "
                "'*' is the only wildcard"
            )
            raise ValueError(msg)
        return cls(namespace, identifier)

    def matches(self, resource: Resource) -> bool:
        if self.namespace is None:
            return True
        return self.namespace == resource.namespace and glob_match(
            self.identifier, resource.identifier
        )

    def __str__(self) -> str:
        return WILDCARD if self.namespace is None else f"{self.namespace}:{self.identifier}"


ResourcePattern.ANY = ResourcePattern(None, WILDCARD)


def _parse_action_pattern(value: str, permission: str) -> ActionPattern:
    if value == WILDCARD:
        return WILDCARD
    try:
        return Action(value)
    except ValueError:
        allowed = ", ".join([*Action, WILDCARD])
        msg = f"permission {permission!r} has unknown action {value!r} (expected one of {allowed})"
        raise ValueError(msg) from None


@dataclass(frozen=True, slots=True)
class Permission:
    """One grant (or deny) pattern: ``action:resource_pattern``."""

    action: ActionPattern
    resource: ResourcePattern

    @classmethod
    def parse(cls, value: str) -> Self:
        action, sep, resource = value.partition(":")
        if not sep or not resource:
            msg = f"permission {value!r} must have the form 'action:resource_pattern'"
            raise ValueError(msg)
        return cls(_parse_action_pattern(action, value), ResourcePattern.parse(resource))

    def matches(self, action: Action, resource: Resource) -> bool:
        return self.action in (WILDCARD, action) and self.resource.matches(resource)

    def __str__(self) -> str:
        return f"{self.action}:{self.resource}"


class PermissionSet:
    """An ordered, immutable collection of permissions matched against concrete calls.

    Usable directly as a Pydantic field type: validated from a list of strings and
    serialized back to one.
    """

    __slots__ = ("_permissions",)

    def __init__(self, permissions: Iterable[Permission] = ()) -> None:
        self._permissions: tuple[Permission, ...] = tuple(permissions)

    @classmethod
    def parse(cls, values: Iterable[str]) -> Self:
        return cls(Permission.parse(value) for value in values)

    def allows(self, action: Action, resource: Resource | str) -> bool:
        """True when any permission matches the concrete ``(action, resource)``."""
        concrete = resource if isinstance(resource, Resource) else Resource.parse(resource)
        return any(p.matches(action, concrete) for p in self._permissions)

    def restricted_to(self, actions: Iterable[Action]) -> "PermissionSet":
        """Only the entries usable for ``actions``; an ``*`` action narrows to each of them."""
        allowed = frozenset(actions)
        narrowed: list[Permission] = []
        for permission in self._permissions:
            if permission.action == WILDCARD:
                narrowed.extend(Permission(a, permission.resource) for a in Action if a in allowed)
            elif permission.action in allowed:
                narrowed.append(permission)
        return PermissionSet(narrowed)

    def __iter__(self) -> Iterator[Permission]:
        return iter(self._permissions)

    def __len__(self) -> int:
        return len(self._permissions)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PermissionSet) and self._permissions == other._permissions

    def __hash__(self) -> int:
        return hash(self._permissions)

    def __repr__(self) -> str:
        return f"PermissionSet({self.as_strings()!r})"

    def as_strings(self) -> list[str]:
        return [str(p) for p in self._permissions]

    @classmethod
    def coerce(cls, value: object) -> "PermissionSet":
        """Accept a PermissionSet or a list of permission strings (the YAML shape)."""
        if isinstance(value, PermissionSet):
            return value
        if not isinstance(value, list | tuple):
            msg = f"expected a list of permission strings, got {type(value).__name__}"
            raise TypeError(msg)
        items = list(cast("list[object] | tuple[object, ...]", value))
        if bad := [item for item in items if not isinstance(item, str)]:
            msg = f"permissions must be strings, got {bad[0]!r}"
            raise ValueError(msg)
        return cls.parse(str(item) for item in items)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: object, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls.coerce,
            serialization=core_schema.plain_serializer_function_ser_schema(cls.as_strings),
        )
