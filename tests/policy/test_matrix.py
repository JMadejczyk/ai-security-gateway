"""Role x agent x action x resource -> allow/deny, from the root policy.yaml.

Same request, different user, different result; anything not granted is denied in every
profile.
"""

from typing import Any

import pytest

from gateway.core.types import Action, SessionMode
from gateway.policy.evaluator import AuthzReason, PrincipalContext
from gateway.policy.permissions import PermissionSet

INT, AU = SessionMode.INTERACTIVE, SessionMode.AUTONOMOUS
R, W, D, X, E, G = (
    Action.READ,
    Action.WRITE,
    Action.DELETE,
    Action.EXECUTE,
    Action.EGRESS,
    Action.GENERATE,
)

USERS: dict[str, tuple[str, ...]] = {
    "anna@demo": ("analyst",),
    "bartek@demo": ("intern",),
    "olga@demo": ("ops-team",),
    "root@demo": ("admin",),
    "svc:nightly_etl": (),
}


def who(
    principal: str,
    agent: str = "databot",
    mode: SessionMode = INT,
    *,
    roles: tuple[str, ...] | None = None,
    scope: list[str] | None = None,
) -> PrincipalContext:
    return PrincipalContext(
        principal=principal,
        roles=USERS.get(principal, ()) if roles is None else roles,
        agent=agent,
        mode=mode,
        task_scope=None if scope is None else PermissionSet.parse(scope),
    )


ANNA, BARTEK, OLGA, ROOT = who("anna@demo"), who("bartek@demo"), who("olga@demo"), who("root@demo")
ETL = who("svc:nightly_etl", "nightly_etl", AU)

ALLOWED = AuthzReason.ALLOWED
MATRIX: list[Any] = [
    # The challenge owner's case: same agent, two people, different data.
    pytest.param(ANNA, R, "db:sales.customers", ALLOWED, id="analyst-customers"),
    pytest.param(BARTEK, R, "db:sales.customers", ALLOWED, id="intern-customers"),
    pytest.param(ANNA, R, "db:sales.payments", ALLOWED, id="analyst-payments"),
    pytest.param(
        BARTEK, R, "db:sales.payments", AuthzReason.OUTSIDE_PRINCIPAL_SCOPE, id="intern-payments"
    ),
    pytest.param(BARTEK, R, "db:sales.orders", ALLOWED, id="intern-orders"),
    # Models: concrete names with colons.
    pytest.param(BARTEK, G, "model:qwen3:8b", ALLOWED, id="intern-qwen"),
    pytest.param(
        BARTEK, G, "model:llama3:70b", AuthzReason.OUTSIDE_PRINCIPAL_SCOPE, id="intern-llama"
    ),
    pytest.param(ANNA, G, "model:llama3:70b", ALLOWED, id="analyst-llama"),
    # Writes and web reads.
    pytest.param(ANNA, W, "fs:reports/q3.md", ALLOWED, id="analyst-report"),
    pytest.param(BARTEK, W, "fs:reports/q3.md", ALLOWED, id="intern-report"),
    pytest.param(
        ANNA, W, "fs:secrets/keys.txt", AuthzReason.OUTSIDE_AGENT_SCOPE, id="write-outside-reports"
    ),
    pytest.param(ANNA, R, "web:example.com", ALLOWED, id="analyst-web"),
    pytest.param(
        ANNA, W, "db:sales.orders", AuthzReason.OUTSIDE_AGENT_SCOPE, id="analyst-db-write"
    ),
    # An agent never gets more than its own grants, even for admin ("*:*").
    pytest.param(ROOT, E, "http:api.stripe.com", AuthzReason.EXPLICIT_DENY, id="admin-egress"),
    pytest.param(ANNA, E, "http:pastebin.com", AuthzReason.EXPLICIT_DENY, id="analyst-egress"),
    pytest.param(
        ROOT, D, "db:sales.orders", AuthzReason.ACTION_NOT_PERMITTED_FOR_AGENT, id="admin-delete"
    ),
    pytest.param(
        ROOT, X, "fs:reports/run.sh", AuthzReason.ACTION_NOT_PERMITTED_FOR_AGENT, id="admin-execute"
    ),
    pytest.param(ROOT, R, "db:hr.salaries", AuthzReason.OUTSIDE_AGENT_SCOPE, id="admin-other-db"),
    pytest.param(ROOT, R, "db:sales.payments", ALLOWED, id="admin-payments"),
    # Approver role grants nothing by itself.
    pytest.param(OLGA, R, "db:sales.orders", AuthzReason.OUTSIDE_PRINCIPAL_SCOPE, id="ops-read"),
    # Autonomous agent: svc principal, U = the agent's own allow list.
    pytest.param(ETL, R, "db:sales.orders", ALLOWED, id="etl-read"),
    pytest.param(ETL, W, "fs:reports/nightly.md", ALLOWED, id="etl-report"),
    pytest.param(ETL, G, "model:qwen3:8b", ALLOWED, id="etl-qwen"),
    pytest.param(ETL, G, "model:llama3:70b", AuthzReason.OUTSIDE_AGENT_SCOPE, id="etl-llama"),
    pytest.param(
        ETL, D, "db:sales.orders", AuthzReason.ACTION_NOT_PERMITTED_FOR_AGENT, id="etl-delete"
    ),
    pytest.param(
        ETL, E, "http:example.com", AuthzReason.ACTION_NOT_PERMITTED_FOR_AGENT, id="etl-egress"
    ),
    # Registration, mode and delegation.
    pytest.param(
        who("anna@demo", "nightly_etl", AU),
        R,
        "db:sales.orders",
        AuthzReason.PRINCIPAL_NOT_ALLOWED,
        id="human-on-autonomous-agent",
    ),
    pytest.param(
        who("svc:databot", "nightly_etl", AU),
        R,
        "db:sales.orders",
        AuthzReason.PRINCIPAL_NOT_ALLOWED,
        id="wrong-service-principal",
    ),
    pytest.param(
        who("svc:nightly_etl", "databot", INT),
        R,
        "db:sales.orders",
        AuthzReason.PRINCIPAL_NOT_ALLOWED,
        id="service-principal-on-interactive-agent",
    ),
    pytest.param(
        who("anna@demo", "databot", AU),
        R,
        "db:sales.orders",
        AuthzReason.MODE_MISMATCH,
        id="interactive-agent-claims-autonomous",
    ),
    pytest.param(
        who("svc:nightly_etl", "nightly_etl", INT),
        R,
        "db:sales.orders",
        AuthzReason.MODE_MISMATCH,
        id="autonomous-agent-claims-interactive",
    ),
    # Default deny: unknown agent, role, resource.
    pytest.param(
        who("anna@demo", "shadowbot"), R, "db:sales.orders", AuthzReason.UNKNOWN_AGENT, id="agent"
    ),
    pytest.param(
        who("anna@demo", roles=("analyst", "superuser")),
        R,
        "db:sales.orders",
        AuthzReason.UNKNOWN_ROLE,
        id="unknown-role-poisons-request",
    ),
    pytest.param(
        who("eve@demo", roles=()),
        R,
        "db:sales.orders",
        AuthzReason.OUTSIDE_PRINCIPAL_SCOPE,
        id="no-roles",
    ),
    pytest.param(ANNA, R, "crm:accounts", AuthzReason.OUTSIDE_AGENT_SCOPE, id="unmapped-resource"),
    pytest.param(ANNA, R, "sales.orders", AuthzReason.INVALID_RESOURCE, id="malformed-resource"),
    pytest.param(ANNA, R, "db:sales.*", AuthzReason.INVALID_RESOURCE, id="wildcard-resource"),
    # Roles are a union.
    pytest.param(
        who("bartek@demo", roles=("intern", "analyst")),
        R,
        "db:sales.payments",
        ALLOWED,
        id="role-union",
    ),
    # Task scope: None = unrestricted, [] = nothing, otherwise narrows.
    pytest.param(
        who("anna@demo", scope=[]), R, "db:sales.orders", AuthzReason.OUTSIDE_TASK_SCOPE, id="t[]"
    ),
    pytest.param(
        who("anna@demo", scope=["read:db:sales.orders"]),
        R,
        "db:sales.orders",
        ALLOWED,
        id="task-scope-match",
    ),
    pytest.param(
        who("anna@demo", scope=["read:db:sales.orders"]),
        R,
        "db:sales.customers",
        AuthzReason.OUTSIDE_TASK_SCOPE,
        id="task-scope-narrows",
    ),
    pytest.param(
        who("bartek@demo", scope=["*:*"]),
        R,
        "db:sales.payments",
        AuthzReason.OUTSIDE_PRINCIPAL_SCOPE,
        id="task-scope-never-widens",
    ),
]


@pytest.mark.parametrize(("principal", "action", "resource", "reason"), MATRIX)
def test_authorization_matrix(evaluator, snapshot, principal, action, resource, reason):
    result = evaluator.authorize(snapshot, principal, action, resource)
    assert result.reason_code is reason
    assert result.allowed is (reason is ALLOWED)
    assert result.policy_revision == snapshot.revision


def test_result_carries_effective_grants_for_audit(evaluator, snapshot):
    result = evaluator.authorize(snapshot, BARTEK, R, "db:sales.payments")
    assert "read:db:sales.orders" in result.principal_grants
    assert "egress:*" not in result.agent_grants
    assert result.agent_grants == (
        "read:db:sales.*",
        "read:web:*",
        "write:fs:reports/*",
        "generate:model:*",
    )
    assert result.task_scope is None


@pytest.mark.parametrize(
    ("blocklist", "principal", "resource", "reason"),
    [
        ({"users": ["bartek@demo"]}, BARTEK, "db:sales.orders", AuthzReason.USER_BLOCKLISTED),
        ({"users": ["bartek@demo"]}, ANNA, "db:sales.orders", ALLOWED),
        ({"agents": ["databot"]}, ANNA, "db:sales.orders", AuthzReason.AGENT_BLOCKLISTED),
        ({"agents": ["databot"]}, ETL, "db:sales.orders", ALLOWED),
        (
            {"use_cases": ["read:db:sales.payments"]},
            ANNA,
            "db:sales.payments",
            AuthzReason.USE_CASE_BLOCKLISTED,
        ),
        ({"use_cases": ["read:db:sales.payments"]}, ANNA, "db:sales.customers", ALLOWED),
        ({"use_cases": ["*:db:*"]}, ROOT, "db:sales.orders", AuthzReason.USE_CASE_BLOCKLISTED),
        ({"agents": ["nightly_etl"]}, ETL, "db:sales.orders", AuthzReason.AGENT_BLOCKLISTED),
    ],
)
def test_blocklist(evaluator, policy_doc, snapshot_from, blocklist, principal, resource, reason):
    policy_doc["blocklist"].update(blocklist)
    result = evaluator.authorize(snapshot_from(policy_doc), principal, R, resource)
    assert result.reason_code is reason


def test_agent_principals_list_restricts_delegation(evaluator, policy_doc, snapshot_from):
    policy_doc["agents"]["databot"]["principals"] = ["anna@demo"]
    snapshot = snapshot_from(policy_doc)
    assert evaluator.authorize(snapshot, ANNA, R, "db:sales.orders").allowed
    refused = evaluator.authorize(snapshot, BARTEK, R, "db:sales.orders")
    assert refused.reason_code is AuthzReason.PRINCIPAL_NOT_ALLOWED


DENIED_EVERYWHERE = [
    (ROOT, E, "http:api.stripe.com"),
    (ROOT, D, "db:sales.orders"),
    (BARTEK, R, "db:sales.payments"),
    (ANNA, R, "crm:accounts"),
    (who("anna@demo", "shadowbot"), R, "db:sales.orders"),
    (who("anna@demo", roles=("ghost",)), R, "db:sales.orders"),
    (ETL, G, "model:llama3:70b"),
    (who("anna@demo", scope=[]), R, "db:sales.orders"),
]


@pytest.mark.parametrize("profile", ["strict", "balanced", "permissive"])
@pytest.mark.parametrize(("principal", "action", "resource"), DENIED_EVERYWHERE)
def test_default_deny_holds_in_every_profile(
    evaluator, policy_doc, snapshot_from, profile, principal, action, resource
):
    policy_doc["profile"] = profile
    policy_doc["controls"] = {}
    assert not evaluator.authorize(snapshot_from(policy_doc), principal, action, resource).allowed


def test_policy_without_agents_or_roles_denies_everything(evaluator, policy_doc, snapshot_from):
    policy_doc["agents"] = {}
    policy_doc["roles"] = {}
    snapshot = snapshot_from(policy_doc)
    for principal in (ANNA, ROOT, ETL):
        assert not evaluator.authorize(snapshot, principal, R, "db:sales.orders").allowed
