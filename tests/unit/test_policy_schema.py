"""Policy schema: the root policy loads, every schema rule rejects what it should."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from gateway.core.catalog import CONTROL_CATALOG, MANDATORY_CONTROLS
from gateway.core.types import ControlMode, Profile
from gateway.policy.loader import PolicyLoader, PolicyLoadError, canonical_digest
from gateway.policy.schema import Controls, PiiConfig, Policy

ROOT_POLICY = Path(__file__).resolve().parents[2] / "policy.yaml"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

type Doc = dict[str, Any]


def test_root_policy_loads(snapshot):
    policy = snapshot.policy
    assert policy.profile is Profile.STRICT
    assert set(policy.agents) == {"databot", "nightly_etl"}
    assert set(policy.roles) == {"analyst", "intern", "admin", "ops-team"}
    assert len(snapshot.revision) == 12
    assert snapshot.digest == canonical_digest(policy)
    assert snapshot.source == ROOT_POLICY


def test_revision_ignores_formatting_but_tracks_content(loader, policy_doc):
    reformatted = loader.parse(yaml.safe_dump(policy_doc, default_flow_style=False).encode())
    assert reformatted.digest == loader.load(ROOT_POLICY).digest
    policy_doc["controls"]["sql_guard"]["max_cost"] = 20000
    assert loader.parse(yaml.safe_dump(policy_doc).encode()).digest != reformatted.digest


def test_mandatory_controls_active_when_omitted(snapshot):
    policy = snapshot.policy
    assert policy.controls.authn is None
    for control_id in MANDATORY_CONTROLS:
        assert policy.resolved_control_mode(control_id) is not ControlMode.LOG_ONLY


def test_control_defaults_and_risk_deltas(snapshot):
    policy = snapshot.policy
    assert isinstance(policy.control_config("pii"), PiiConfig)
    assert policy.control_config("tool_pinning").mode is None
    assert policy.control_risk_delta("prompt_injection") == 0.6
    assert policy.control_risk_delta("secrets") == 0.3
    assert policy.control_risk_delta("signatures") == 0.4
    assert policy.control_risk_delta("pii") == 0.1
    assert policy.control_risk_delta("authz") == 0.1
    assert policy.control_risk_delta("loop_detect") == 0.0


def test_controls_schema_covers_exactly_the_catalog():
    assert set(Controls.model_fields) == set(CONTROL_CATALOG)
    assert {"authn", "authz", "secrets", "sql_guard"} == MANDATORY_CONTROLS


def _set(path: str, value: Any) -> Callable[[Doc], None]:
    """Mutator that sets a dotted path (list indexes as integers) in the document."""

    def mutate(doc: Doc) -> None:
        *parents, last = path.split(".")
        node: Any = doc
        for part in parents:
            node = node[int(part)] if isinstance(node, list) else node[part]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value

    return mutate


def _delete(path: str) -> Callable[[Doc], None]:
    def mutate(doc: Doc) -> None:
        *parents, last = path.split(".")
        node: Any = doc
        for part in parents:
            node = node[part]
        del node[last]

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(_set("surprise", 1), "Extra inputs are not permitted", id="unknown-top-field"),
        pytest.param(
            _set("agents.databot.colour", "red"), "Extra inputs", id="unknown-nested-field"
        ),
        pytest.param(_set("default", "allow"), "default", id="default-allow"),
        pytest.param(_set("schema_version", 1), "schema_version", id="schema-version"),
        pytest.param(_set("profile", "lenient"), "profile", id="unknown-profile"),
        pytest.param(
            _set("roles.analyst.allow", ["read:db:sales.?"]), "only wildcard", id="fnmatch-perm"
        ),
        pytest.param(
            _set("roles.analyst.allow", ["fetch:db:sales"]), "unknown action", id="unknown-action"
        ),
        pytest.param(
            _set("blocklist.use_cases", ["egress:pastebin"]),
            "namespace:identifier",
            id="bad-use-case",
        ),
        pytest.param(
            _set("agents.databot.approvers", ["security"]), "approver roles", id="unknown-approver"
        ),
        pytest.param(_set("agents.databot.type", "batch"), "type", id="unknown-agent-type"),
        pytest.param(
            _set("agents.databot.max_actions", ["read", "fly"]),
            "max_actions",
            id="unknown-max-action",
        ),
        pytest.param(
            _set("agents.databot.allow", ["delete:db:sales.*"]),
            "outside max_actions",
            id="allow-outside-max-actions",
        ),
        pytest.param(
            _set("agents.nightly_etl.principals", ["anna@demo"]),
            "can only act as",
            id="autonomous-human-principal",
        ),
        pytest.param(
            _set("agents.databot.principals", ["svc:nightly_etl"]),
            "service principals",
            id="interactive-service-principal",
        ),
        pytest.param(_set("upstreams.mcp.web.adapter", "ftp"), "adapter", id="unknown-adapter"),
        pytest.param(_set("upstreams.mcp.web.trust", "partial"), "trust", id="unknown-trust"),
        pytest.param(
            _set("upstreams.mcp.web.tools.fetch.action", "browse"),
            "action",
            id="tool-unknown-action",
        ),
        pytest.param(
            _set("upstreams.mcp.web.tools.fetch.resource", "web:{url.host}"),
            "not an argument name",
            id="template-attribute-access",
        ),
        pytest.param(
            _set("upstreams.mcp.web.tools.fetch.resource", "web:{0}"),
            "not an argument name",
            id="template-positional",
        ),
        pytest.param(
            _set("upstreams.mcp.web.tools.fetch.resource", "web:{url"),
            "unbalanced",
            id="template-unbalanced",
        ),
        pytest.param(
            _set("upstreams.mcp.web.tools.fetch.resource", "{ns}:x"),
            "namespace must be literal",
            id="template-placeholder-namespace",
        ),
        pytest.param(
            _set("upstreams.mcp.web.tools.fetch.resource", "web:*"),
            "may not contain",
            id="template-wildcard",
        ),
        pytest.param(
            _delete("upstreams.mcp.web.tools.fetch.resource"),
            "needs a resource template",
            id="http-tool-without-resource",
        ),
        pytest.param(
            _set("upstreams.mcp.sales_db.tools.query.resource", "db:{table}"),
            "derives resources",
            id="sql-tool-with-resource",
        ),
        pytest.param(_set("upstreams.llm.base_url", "ollama:11434"), "http", id="bad-url"),
        pytest.param(
            _set("controls.secrets.mode", "log_only"), "mandatory", id="log-only-mandatory"
        ),
        pytest.param(
            _set("controls.sql_guard", {"mode": "log_only"}), "mandatory", id="log-only-sql-guard"
        ),
        pytest.param(
            _set("controls.model_allowlist.mode", "redact"),
            "does not support",
            id="unsupported-mode",
        ),
        pytest.param(
            _set("controls.tool_pinning", {"mode": "log_only"}),
            "does not support",
            id="tool-pinning-log-only",
        ),
        pytest.param(_set("controls.pii.mode", "quarantine"), "mode", id="unknown-mode"),
        pytest.param(_set("controls.rate_limit", {"mode": "block"}), "Extra", id="unknown-control"),
        pytest.param(
            _set("controls.prompt_injection.judge_band", [0.9, 0.5]), "ordered", id="band-unordered"
        ),
        pytest.param(
            _set("controls.prompt_injection.judge_band", [0.5, 1.2]),
            "less than or equal",
            id="band-out",
        ),
        pytest.param(
            _set("controls.prompt_injection.judge_band", [0.5]), "judge_band", id="band-short"
        ),
        pytest.param(
            _set("controls.pii.threshold", 1.5), "less than or equal", id="threshold-above-one"
        ),
        pytest.param(
            _set("controls.pii.risk_delta", -0.1), "greater than or equal", id="negative-risk-delta"
        ),
        pytest.param(_set("risk.half_life_s", 0), "greater than 0", id="zero-half-life"),
        pytest.param(_set("approvals.timeout_s", -5), "greater than 0", id="negative-duration"),
        pytest.param(_set("approvals.on_timeout", "allow"), "on_timeout", id="timeout-allow"),
        pytest.param(_set("sessions.idle_ttl_s", 0), "greater than 0", id="zero-idle-ttl"),
        pytest.param(_set("sessions.idle_ttl_s", 100000), "cannot exceed", id="idle-over-lifetime"),
        pytest.param(_set("throttle.base_s", 600), "cannot exceed", id="throttle-base-over-max"),
        pytest.param(_set("budgets.per_user.daily_tokens", 0), "greater than 0", id="zero-budget"),
        pytest.param(
            _set("budgets.per_user.daily_cost_usd", -1), "greater than 0", id="negative-budget"
        ),
        pytest.param(_set("budgets.soft_limit_pct", 120), "less than or equal", id="soft-limit"),
        pytest.param(_set("limits.max_request_bytes", 0), "greater than 0", id="zero-limit"),
        pytest.param(
            _set("risk_rules.interactive.1.when.risk_gt", 1.5),
            "less than or equal",
            id="risk-threshold-above-one",
        ),
        pytest.param(
            _set("risk_rules.interactive.0.when", {}), "at least one condition", id="empty-when"
        ),
        pytest.param(
            _set("risk_rules.interactive.0.then", {}), "at least one effect", id="empty-then"
        ),
        pytest.param(
            _set("risk_rules.interactive.0.then", {"actions": ["write"]}),
            "given together",
            id="actions-without-mode",
        ),
        pytest.param(
            _set("risk_rules.interactive.0.then", {"mode": "require_approval"}),
            "given together",
            id="mode-without-actions",
        ),
        pytest.param(
            _set("risk_rules.interactive.0.then", {"actions": ["write"], "mode": "allow"}),
            "mode",
            id="rule-mode-allow",
        ),
        pytest.param(
            _set("risk_rules.interactive.2.then", {"freeze_tools": True}),
            "duration_s",
            id="freeze-without-duration",
        ),
        pytest.param(
            _set("risk_rules.interactive.1.then.cooldown_s", -300),
            "greater than 0",
            id="negative-cooldown",
        ),
        pytest.param(
            _set("risk_rules.autonomous.1.then.throttle", {"max_actions": 0, "per_s": 10}),
            "greater than 0",
            id="zero-throttle",
        ),
        pytest.param(_delete("risk_rules.autonomous"), "autonomous", id="missing-rule-mode"),
        pytest.param(_set("roles.bad:name", {"allow": []}), "roles", id="role-name-with-colon"),
    ],
)
def test_schema_rule_rejects(policy_doc, snapshot_from, mutate, message):
    mutate(policy_doc)
    with pytest.raises(PolicyLoadError, match=message):
        snapshot_from(policy_doc)


def test_v1_one_line_risk_rules_fail_to_parse(loader):
    with pytest.raises(PolicyLoadError, match="invalid YAML"):
        loader.load(FIXTURES / "v1_risk_rules.yaml")


def test_duplicate_keys_rejected(loader):
    with pytest.raises(PolicyLoadError, match="duplicate key 'intern'"):
        loader.load(FIXTURES / "duplicate_keys.yaml")


def test_duplicate_keys_rejected_even_when_quoted_differently(loader):
    with pytest.raises(PolicyLoadError, match="duplicate key"):
        loader.parse(b'profile: strict\n"profile": permissive\n')


def test_yaml_aliases_rejected(loader):
    text = ROOT_POLICY.read_text().replace(
        'admin:    { allow: ["*:*"] }', 'admin:    &all { allow: ["*:*"] }\n  root: *all'
    )
    with pytest.raises(PolicyLoadError, match="aliases are not allowed"):
        loader.parse(text.encode())


def test_unsafe_yaml_tags_rejected(loader):
    with pytest.raises(PolicyLoadError, match="invalid YAML"):
        loader.parse(b"profile: !!python/object/apply:os.system ['true']\n")


def test_oversize_file_rejected(tmp_path: Path):
    padding = "#" * (1024 * 1024)
    big = tmp_path / "policy.yaml"
    big.write_text(ROOT_POLICY.read_text() + "\n" + padding + "\n")
    with pytest.raises(PolicyLoadError, match="larger than 1048576 bytes"):
        PolicyLoader().load(big)


def test_non_mapping_and_non_utf8_rejected(loader):
    with pytest.raises(PolicyLoadError, match="mapping at the top level"):
        loader.parse(b"- just\n- a list\n")
    with pytest.raises(PolicyLoadError, match="UTF-8"):
        loader.parse(b"\xff\xfe")


def test_missing_file_rejected(tmp_path: Path):
    with pytest.raises(PolicyLoadError, match="cannot read policy"):
        PolicyLoader().load(tmp_path / "absent.yaml")


def test_validation_message_names_the_field(policy_doc, snapshot_from):
    policy_doc["controls"]["secrets"]["mode"] = "log_only"
    policy_doc["agents"]["databot"]["allow"] = ["read:db:sales.?"]
    with pytest.raises(PolicyLoadError) as caught:
        snapshot_from(policy_doc)
    assert "agents.databot.allow" in str(caught.value)


# Profile resolution: (control, strict, balanced, permissive), with nothing configured.
PROFILE_TABLE = [
    ("intent_judge", "require_approval", "require_approval", "log_only"),
    ("tool_pinning", "block", "block", "block"),
    ("pii", "block", "redact", "log_only"),
    ("prompt_injection", "block", "block", "log_only"),
    ("egress", "block", "require_approval", "require_approval"),
    ("output_policy", "block", "redact", "redact"),
    ("model_allowlist", "block", "block", "log_only"),
    ("secrets", "block", "block", "block"),  # mandatory: profile never applies
    ("authz", "block", "block", "block"),  # mandatory
    ("sql_guard", "block", "block", "block"),  # mandatory
]


@pytest.mark.parametrize(("control_id", "strict", "balanced", "permissive"), PROFILE_TABLE)
def test_profile_resolution(policy_doc, snapshot_from, control_id, strict, balanced, permissive):
    policy_doc["controls"] = {}
    for profile, expected in (
        ("strict", strict),
        ("balanced", balanced),
        ("permissive", permissive),
    ):
        policy_doc["profile"] = profile
        policy: Policy = snapshot_from(policy_doc).policy
        assert policy.resolved_control_mode(control_id) == expected, profile


@pytest.mark.parametrize("profile", list(Profile))
def test_explicit_mode_wins_over_profile(policy_doc, snapshot_from, profile):
    policy_doc["profile"] = str(profile)
    policy_doc["controls"]["pii"]["mode"] = "log_only"
    policy_doc["controls"]["secrets"]["mode"] = "redact"
    policy_doc["controls"]["intent_judge"] = {"mode": "log_only"}
    policy = snapshot_from(policy_doc).policy
    assert policy.resolved_control_mode("pii") is ControlMode.LOG_ONLY
    assert policy.resolved_control_mode("secrets") is ControlMode.REDACT
    assert policy.resolved_control_mode("intent_judge") is ControlMode.LOG_ONLY


def test_resolving_an_unknown_control_fails(snapshot):
    with pytest.raises(KeyError, match="unknown control id"):
        snapshot.policy.resolved_control_mode("rate_limit")
