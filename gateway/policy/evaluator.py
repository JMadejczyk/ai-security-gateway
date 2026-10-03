"""Decision model (SPEC "Identity and permission model" → "Decision model").

Two layers, evaluated per concrete ``(action, resource)``:

1. Base authorization: principal grants ∩ agent grants ∩ task scope, minus explicit denies,
   with registration, mode and blocklist checks. Default deny everywhere.
2. Session restrictions from ``risk_rules[mode]``: they can only remove an action or add an
   obligation (approval, throttle) on top of an allowed call, never widen it.

Everything here is pure: it reads one policy snapshot and one session snapshot and returns
data. Starting timers and mutating the session is the pipeline's job.
"""

from collections.abc import Callable
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import AwareDatetime, Field

from gateway.core.envelope import FrozenModel, SessionContext, cooldown_key
from gateway.core.types import Action, Channel, Decision, SessionMode
from gateway.policy.loader import PolicySnapshot
from gateway.policy.permissions import PermissionSet, Resource
from gateway.policy.schema import (
    SERVICE_PRINCIPAL_PREFIX,
    Agent,
    Policy,
    RiskRule,
    Throttle,
    service_principal,
)


class AuthzReason(StrEnum):
    ALLOWED = "allowed"
    INVALID_RESOURCE = "invalid_resource"
    UNKNOWN_AGENT = "unknown_agent"
    MODE_MISMATCH = "mode_mismatch"
    PRINCIPAL_NOT_ALLOWED = "principal_not_allowed_for_agent"
    UNKNOWN_ROLE = "unknown_role"
    AGENT_BLOCKLISTED = "agent_blocklisted"
    USER_BLOCKLISTED = "user_blocklisted"
    USE_CASE_BLOCKLISTED = "use_case_blocklisted"
    EXPLICIT_DENY = "explicit_deny"
    ACTION_NOT_PERMITTED_FOR_AGENT = "action_not_permitted_for_agent"
    OUTSIDE_AGENT_SCOPE = "outside_agent_scope"
    OUTSIDE_PRINCIPAL_SCOPE = "outside_principal_scope"
    OUTSIDE_TASK_SCOPE = "outside_task_scope"


class RestrictionReason(StrEnum):
    COOLDOWN_ACTIVE = "cooldown_active"
    TOOLS_FROZEN = "tools_frozen"
    ACTION_REMOVED = "action_removed_by_session_risk"


class PrincipalContext(FrozenModel):
    """Who is asking, as established by authentication."""

    principal: str = Field(min_length=1)  # sub: a human, or svc:<agent> for autonomous
    roles: tuple[str, ...] = ()
    agent: str = Field(min_length=1)  # act.sub
    mode: SessionMode
    task_scope: PermissionSet | None = None  # token `scope`: None = unrestricted, [] = nothing


class AuthorizationResult(FrozenModel):
    allowed: bool
    reason_code: AuthzReason
    policy_revision: str
    # Effective grants for the audit entry ("effective_scope").
    principal_grants: tuple[str, ...] = ()
    agent_grants: tuple[str, ...] = ()
    task_scope: tuple[str, ...] | None = None


class TimerStart(FrozenModel):
    """A session timer the caller should start: ``key`` for a cooldown, absent for a freeze."""

    until: AwareDatetime
    key: str | None = None


class SessionRestriction(FrozenModel):
    """Session-risk restrictions and obligations for one call."""

    denied: bool = False
    reason_code: RestrictionReason | None = None
    requires_approval: bool = False
    throttles: tuple[Throttle, ...] = ()  # every matching cap; the enforcer applies all
    alert: bool = False
    risk: float  # decayed risk the rules were evaluated against
    matched_rules: tuple[int, ...] = ()  # indexes into risk_rules[mode]
    # Timers that would start. A freeze starts now; a cooldown starts only if this call
    # ends up denied (by these restrictions or by any later layer).
    start_freeze: TimerStart | None = None
    cooldown_on_deny: TimerStart | None = None


class AccessDecision(FrozenModel):
    """Base authorization and session restrictions combined."""

    decision: Decision  # allow, block or require_approval
    reason_code: str
    authorization: AuthorizationResult
    restriction: SessionRestriction


class PolicyEvaluator:
    """Evaluates the decision model against a policy snapshot. Holds no state."""

    def authorize(
        self,
        snapshot: PolicySnapshot,
        principal: PrincipalContext,
        action: Action,
        resource: str,
    ) -> AuthorizationResult:
        """Base authorization of one concrete ``(action, resource)``."""
        policy = snapshot.policy
        agent = policy.agents.get(principal.agent)

        def result(
            reason: AuthzReason, principal_grants: PermissionSet | None = None
        ) -> AuthorizationResult:
            return AuthorizationResult(
                allowed=reason is AuthzReason.ALLOWED,
                reason_code=reason,
                policy_revision=snapshot.revision,
                principal_grants=tuple(principal_grants.as_strings()) if principal_grants else (),
                agent_grants=tuple(agent.grants.as_strings()) if agent else (),
                task_scope=(
                    tuple(principal.task_scope.as_strings())
                    if principal.task_scope is not None
                    else None
                ),
            )

        if agent is None:
            return result(AuthzReason.UNKNOWN_AGENT)
        if refusal := _identity_refusal(policy, principal, agent):
            return result(refusal)
        try:
            concrete = Resource.parse(resource)
        except ValueError:
            return result(AuthzReason.INVALID_RESOURCE)

        user_grants = _principal_grants(policy, principal, agent)
        task_scope = principal.task_scope
        # Ordered: the first failing check names the refusal in the audit entry.
        checks: tuple[tuple[Callable[[], bool], AuthzReason], ...] = (
            (
                lambda: policy.blocklist.use_cases.allows(action, concrete),
                AuthzReason.USE_CASE_BLOCKLISTED,
            ),
            (lambda: agent.deny.allows(action, concrete), AuthzReason.EXPLICIT_DENY),
            (lambda: action not in agent.max_actions, AuthzReason.ACTION_NOT_PERMITTED_FOR_AGENT),
            (lambda: not agent.grants.allows(action, concrete), AuthzReason.OUTSIDE_AGENT_SCOPE),
            (lambda: not user_grants.allows(action, concrete), AuthzReason.OUTSIDE_PRINCIPAL_SCOPE),
            (
                lambda: task_scope is not None and not task_scope.allows(action, concrete),
                AuthzReason.OUTSIDE_TASK_SCOPE,
            ),
        )
        refusal = next((reason for failed, reason in checks if failed()), None)
        return result(refusal or AuthzReason.ALLOWED, user_grants)

    def session_restrictions(  # noqa: PLR0913 -- every argument is an independent input of the rule evaluation
        self,
        snapshot: PolicySnapshot,
        ctx: SessionContext,
        *,
        channel: Channel,
        action: Action,
        resource: str,
        now: datetime,
    ) -> SessionRestriction:
        """Evaluate ``risk_rules[ctx.mode]`` for one call. Never grants anything."""
        policy = snapshot.policy
        rules = policy.risk_rules.for_mode(ctx.mode)
        risk = ctx.risk_at(now, policy.risk.half_life_s)
        matched = [
            (index, rule)
            for index, rule in enumerate(rules)
            if rule.when.holds(risk=risk, taint=ctx.taint)
        ]
        key = cooldown_key(action, resource)
        is_tool_call = not (channel is Channel.LLM and action is Action.GENERATE)

        start_freeze = _freeze_to_start(matched, ctx, now)
        reason: RestrictionReason | None = None
        if ctx.cooldown_until(key, now) is not None:
            reason = RestrictionReason.COOLDOWN_ACTIVE
        elif is_tool_call and (ctx.is_frozen(now) or start_freeze is not None):
            reason = RestrictionReason.TOOLS_FROZEN
        elif any(action in rule.then.deny_actions for _, rule in matched):
            reason = RestrictionReason.ACTION_REMOVED

        return SessionRestriction(
            denied=reason is not None,
            reason_code=reason,
            requires_approval=any(action in rule.then.actions for _, rule in matched),
            throttles=_throttles(matched),
            alert=any(rule.then.alert for _, rule in matched),
            risk=risk,
            matched_rules=tuple(index for index, _ in matched),
            start_freeze=start_freeze,
            cooldown_on_deny=_cooldown_to_start(matched, ctx, key, now),
        )

    def removed_actions(
        self, snapshot: PolicySnapshot, ctx: SessionContext, now: datetime
    ) -> frozenset[Action]:
        """Actions the session's risk rules currently remove (``deny_actions``).

        For listings (MCP ``tools/list``): they hide what the session can no longer do at
        all. Approvals, throttles and timers only condition a call and start nothing here.
        """
        policy = snapshot.policy
        risk = ctx.risk_at(now, policy.risk.half_life_s)
        return frozenset(
            action
            for rule in policy.risk_rules.for_mode(ctx.mode)
            if rule.when.holds(risk=risk, taint=ctx.taint)
            for action in rule.then.deny_actions
        )

    def tools_frozen(self, snapshot: PolicySnapshot, ctx: SessionContext, now: datetime) -> bool:
        """Would a tool call be refused as ``tools_frozen`` right now?

        True while a freeze is running, and while a ``freeze_tools`` rule's threshold holds
        (the next call would start one). Read-only: unlike a call, it never starts the timer.
        """
        if ctx.is_frozen(now):
            return True
        policy = snapshot.policy
        risk = ctx.risk_at(now, policy.risk.half_life_s)
        return any(
            rule.then.freeze_tools and rule.when.holds(risk=risk, taint=ctx.taint)
            for rule in policy.risk_rules.for_mode(ctx.mode)
        )

    def decide(  # noqa: PLR0913 -- the union of authorize() and session_restrictions() inputs
        self,
        snapshot: PolicySnapshot,
        principal: PrincipalContext,
        ctx: SessionContext,
        *,
        channel: Channel,
        action: Action,
        resource: str,
        now: datetime,
    ) -> AccessDecision:
        """Base authorization first; restrictions can only narrow its outcome."""
        authorization = self.authorize(snapshot, principal, action, resource)
        restriction = self.session_restrictions(
            snapshot, ctx, channel=channel, action=action, resource=resource, now=now
        )
        if not authorization.allowed:
            decision, reason = Decision.BLOCK, str(authorization.reason_code)
        elif restriction.denied and restriction.reason_code is not None:
            decision, reason = Decision.BLOCK, str(restriction.reason_code)
        elif restriction.requires_approval:
            decision, reason = Decision.REQUIRE_APPROVAL, "session_requires_approval"
        else:
            decision, reason = Decision.ALLOW, str(AuthzReason.ALLOWED)
        return AccessDecision(
            decision=decision,
            reason_code=reason,
            authorization=authorization,
            restriction=restriction,
        )


def _principal_refused(principal: PrincipalContext, agent: Agent) -> bool:
    """Delegation check: may this principal be represented by this agent at all?"""
    if agent.type is SessionMode.AUTONOMOUS:
        if principal.principal != service_principal(principal.agent):
            return True
    elif principal.principal.startswith(SERVICE_PRINCIPAL_PREFIX):
        return True
    # An explicit list always applies, for both types; `principals: []` means nobody.
    return agent.principals is not None and principal.principal not in agent.principals


def _identity_refusal(
    policy: Policy, principal: PrincipalContext, agent: Agent
) -> AuthzReason | None:
    """Registration, delegation and blocklist checks that do not depend on the resource."""
    checks: tuple[tuple[Callable[[], bool], AuthzReason], ...] = (
        (lambda: principal.mode is not agent.type, AuthzReason.MODE_MISMATCH),
        (lambda: _principal_refused(principal, agent), AuthzReason.PRINCIPAL_NOT_ALLOWED),
        (lambda: any(r not in policy.roles for r in principal.roles), AuthzReason.UNKNOWN_ROLE),
        (lambda: principal.agent in policy.blocklist.agents, AuthzReason.AGENT_BLOCKLISTED),
        (lambda: principal.principal in policy.blocklist.users, AuthzReason.USER_BLOCKLISTED),
    )
    return next((reason for failed, reason in checks if failed()), None)


def _principal_grants(policy: Policy, principal: PrincipalContext, agent: Agent) -> PermissionSet:
    """U: the union of the principal's role grants; an autonomous agent's own allow list."""
    if agent.type is SessionMode.AUTONOMOUS:
        return agent.allow
    return PermissionSet(p for role in principal.roles for p in policy.roles[role].allow)


def _freeze_to_start(
    matched: list[tuple[int, RiskRule]], ctx: SessionContext, now: datetime
) -> TimerStart | None:
    """A freeze starts when its threshold holds and no freeze is running; never extended."""
    if ctx.is_frozen(now):
        return None
    durations = [rule.then.duration_s for _, rule in matched if rule.then.duration_s is not None]
    if not durations:
        return None
    return TimerStart(until=now + timedelta(seconds=max(durations)))


def _cooldown_to_start(
    matched: list[tuple[int, RiskRule]], ctx: SessionContext, key: str, now: datetime
) -> TimerStart | None:
    """Cooldown for ``key`` if this call is denied; a running cooldown is not refreshed."""
    if ctx.cooldown_until(key, now) is not None:
        return None
    durations = [rule.then.cooldown_s for _, rule in matched if rule.then.cooldown_s is not None]
    if not durations:
        return None
    return TimerStart(until=now + timedelta(seconds=max(durations)), key=key)


def _throttles(matched: list[tuple[int, RiskRule]]) -> tuple[Throttle, ...]:
    """All throttle caps of the matched rules, deduplicated, in rule order.

    Caps over different windows (1 per 1 s and 10 per 60 s) are not comparable, so none is
    dropped: a call must satisfy every one of them.
    """
    caps: list[Throttle] = []
    for _, rule in matched:
        if rule.then.throttle is not None and rule.then.throttle not in caps:
            caps.append(rule.then.throttle)
    return tuple(caps)
