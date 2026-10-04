"""Reading `policy.yaml` into an immutable, revisioned snapshot."""

import hashlib
import json
from collections.abc import Callable, Hashable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import AwareDatetime, Field, ValidationError
from yaml.composer import ComposerError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, Node, ScalarNode

from gateway.core.envelope import FrozenModel
from gateway.policy.schema import Policy

MAX_POLICY_BYTES = 1024 * 1024  # 1 MiB
REVISION_LENGTH = 12
MAX_YAML_DEPTH = 32  # the policy itself needs 6


class PolicyLoadError(Exception):
    """The policy could not be read, parsed or validated. The message is operator-facing."""

    def __init__(self, message: str, *, source: Path | None = None) -> None:
        self.source = source
        prefix = f"{source}: " if source is not None else ""
        super().__init__(f"{prefix}{message}")


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys, aliases and deep nesting.

    Duplicate keys would let a later line silently override an earlier grant. Aliases are
    refused because expanding them (billion laughs) bypasses the byte-size cap. Nesting is
    capped because composing, constructing and validating are all recursive: a 2 KB file of
    ``[[[[...`` would otherwise end in RecursionError.
    """

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._depth = 0

    def compose_node(self, parent: Node | None, index: int) -> Node | None:
        is_alias = self.check_event(yaml.AliasEvent)
        is_collection = self.check_event(yaml.SequenceStartEvent, yaml.MappingStartEvent)
        if is_collection:
            if self._depth >= MAX_YAML_DEPTH:
                problem = f"YAML nesting deeper than {MAX_YAML_DEPTH} levels"
                raise ComposerError(None, None, problem, parent.start_mark if parent else None)
            self._depth += 1
        try:
            node = super().compose_node(parent, index)  # an alias resolves to its anchor
        finally:
            if is_collection:
                self._depth -= 1
        if is_alias:
            problem = "YAML aliases are not allowed in the policy"
            raise ComposerError(None, None, problem, node.start_mark if node else None)
        return node

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        seen: set[tuple[str, str]] = set()
        for key_node, _ in node.value:
            if not isinstance(key_node, ScalarNode):
                problem = "mapping keys must be plain scalars"
                raise ConstructorError(None, None, problem, key_node.start_mark)
            # Tags are resolved by now, so `a` and "a" collide while `1` and "1" do not.
            identity = (key_node.tag, str(key_node.value))
            if identity in seen:
                context, problem = (
                    "while constructing a mapping",
                    f"duplicate key {key_node.value!r}",
                )
                raise ConstructorError(context, node.start_mark, problem, key_node.start_mark)
            seen.add(identity)
        return super().construct_mapping(node, deep=deep)


def canonical_digest(policy: Policy) -> str:
    """SHA-256 of the validated policy as canonical JSON (sorted keys, compact)."""
    document = json.dumps(
        policy.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(document.encode()).hexdigest()


class PolicySnapshot(FrozenModel):
    """One validated policy version. Every decision in a call reads a single snapshot."""

    policy: Policy
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    loaded_at: AwareDatetime
    source: Path | None = None

    @property
    def revision(self) -> str:
        """Short policy revision used in audit entries and metrics."""
        return self.digest[:REVISION_LENGTH]


def _format_validation_error(error: ValidationError) -> str:
    lines = [f"{error.error_count()} validation error(s):"]
    for item in error.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"  {location}: {item['msg']}")
    return "\n".join(lines)


class PolicyLoader:
    """Safe YAML loading, size cap and schema validation."""

    def __init__(
        self,
        *,
        max_bytes: int = MAX_POLICY_BYTES,
        requirement: Callable[[Policy], None] | None = None,
    ) -> None:
        """``requirement`` is a deployment check beyond the schema (it raises `ValueError`), e.g.
        that the LLM upstream the process selected is declared. It applies at startup and to
        every reload: a policy failing it never replaces a working one."""
        self._max_bytes = max_bytes
        self._requirement = requirement

    def load(self, path: Path) -> PolicySnapshot:
        """Read and validate the policy file at ``path``."""
        try:
            with path.open("rb") as handle:
                data = handle.read(self._max_bytes + 1)
        except OSError as exc:
            msg = f"cannot read policy: {exc.strerror or exc}"
            raise PolicyLoadError(msg, source=path) from exc
        return self.parse(data, source=path)

    def parse(self, data: bytes, *, source: Path | None = None) -> PolicySnapshot:
        """Validate policy bytes; ``source`` is only used in messages and the snapshot."""
        if len(data) > self._max_bytes:
            msg = f"policy is larger than {self._max_bytes} bytes"
            raise PolicyLoadError(msg, source=source)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            msg = f"policy is not valid UTF-8: {exc.reason} at byte {exc.start}"
            raise PolicyLoadError(msg, source=source) from exc
        try:
            document: object = yaml.load(text, Loader=_StrictSafeLoader)  # noqa: S506 -- _StrictSafeLoader subclasses SafeLoader
        except yaml.YAMLError as exc:
            msg = f"invalid YAML: {exc}"
            raise PolicyLoadError(msg, source=source) from exc
        except RecursionError as exc:  # backstop; the depth cap should make this unreachable
            msg = "invalid YAML: nested too deeply"
            raise PolicyLoadError(msg, source=source) from exc
        if not isinstance(document, dict):
            msg = "policy must be a YAML mapping at the top level"
            raise PolicyLoadError(msg, source=source)
        try:
            policy = Policy.model_validate(document)
        except ValidationError as exc:
            raise PolicyLoadError(_format_validation_error(exc), source=source) from exc
        if self._requirement is not None:
            try:
                self._requirement(policy)
            except ValueError as exc:
                raise PolicyLoadError(str(exc), source=source) from exc
        return PolicySnapshot(
            policy=policy,
            digest=canonical_digest(policy),
            loaded_at=datetime.now(UTC),
            source=source,
        )
