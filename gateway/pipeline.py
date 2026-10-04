"""The step order every call goes through (SPEC "Pipeline and interfaces" → "Order for every call").

1. Authentication: verify the JWT, then lock and load session state (one call per session).
2. Adapter: normalize the request into one or more `Interaction`s.
3. Base authorization plus session restrictions on every interaction; any deny stops here.
4. ``pre`` controls, deterministic first; sealing controls (`Control.seal`, e.g. ``sql_guard``)
   last, on the final payload with every rewrite and redaction applied.
5. Merge verdicts into allow / redact / block / require_approval plus obligations.
6. Execute on the upstream once, with the final (rewritten, redacted) payload.
7. ``post`` controls on the complete result.
8. Persist session state (risk deltas from every verdict, including blocked calls; timers),
   then audit and metrics, all before the result is released.

The policy snapshot is taken once per call by the caller and threaded through every step, so
a reload mid-call never mixes two policy versions. Nothing here knows about a specific
provider: a channel is an `Adapter` plus an `Upstream`.
"""

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from http import HTTPStatus
from typing import Any, Final, Literal, cast

from pydantic import Field

from gateway.approvals.kill_switch import KillSwitchUnavailableError
from gateway.approvals.model import Approval
from gateway.approvals.oversight import (
    AGENT_KILLED,
    APPROVAL,
    KILL_SWITCH,
    ApprovalRefusal,
    Oversight,
)
from gateway.budget.ledger import BudgetLedger
from gateway.budget.metering import charged_tokens
from gateway.budget.model import BudgetedCall, BudgetRefusalError
from gateway.canonical import canonical_bytes as _canonical_bytes
from gateway.clock import Clock, utc_now
from gateway.controls.registry import ControlRegistry
from gateway.controls.scope import CallScope, call_scope
from gateway.core.envelope import (
    CallRecord,
    Cooldown,
    FrozenModel,
    Interaction,
    RawCall,
    SessionContext,
    Verdict,
)
from gateway.core.interfaces import Adapter
from gateway.core.types import Channel, ControlMode, Decision, LlmUpstreamKind, Stage
from gateway.core.verdicts import MergedVerdict, merge_verdicts
from gateway.errors import InvalidRequestError, RejectionError, RequestTooLargeError
from gateway.feed.schema import EMPTY_FEED, SignatureFeed
from gateway.identity import TokenClaims, TokenVerifier
from gateway.judges.intent import first_user_message, mcp_flag_candidates
from gateway.policy.evaluator import AccessDecision, PolicyEvaluator, PrincipalContext
from gateway.policy.loader import PolicySnapshot
from gateway.policy.permissions import PermissionSet, Resource
from gateway.policy.schema import Throttle, service_principal
from gateway.redaction import apply_redactions
from gateway.sessions import SessionBinding, SessionStore, SessionUpdate
from gateway.telemetry import (
    OTHER_LABEL,
    AuditEntry,
    AuditLatency,
    AuditLogger,
    AuditVerdict,
    bounded,
    payload_hmac,
    record_alert,
    record_overhead,
    record_request,
    record_session,
    record_throttled,
    record_tokens,
    record_verdicts,
)
from gateway.throttle import ThrottledError, Throttler
from gateway.upstream import DeadlineUpstream, Upstream, UpstreamError, UpstreamResult

logger = logging.getLogger(__name__)
alert_logger = logging.getLogger("gateway.alerts")

AUTHZ: Final = "authz"
BUDGET: Final = "budget"  # a pipeline seam, not a registered control: see _execute_metered
INTENT_JUDGE: Final = "intent_judge"
INTENT_FLAGGED: Final = "intent_flagged"  # an MCP call matching a flagged tool_call
INTENT_FLAGS_OVERFLOW: Final = "intent_flags_overflow"  # more flags than a session keeps
# Controls whose detections mean untrusted content reached the agent's context.
TAINTING_CONTROLS: Final = frozenset({"prompt_injection"})
REWRITE_UNAUTHORIZED: Final = "rewrite_unauthorized"


class CallRequest(FrozenModel):
    """A call as an entry point received it."""

    channel: Channel
    token: str | None = Field(default=None, repr=False)
    body: bytes = Field(repr=False)  # read up to max_request_bytes + 1, never more
    server: str | None = None  # MCP upstream name
    # The approval a retry names (MCP `_meta` "ai-control-layer/approval_id", LLM header
    # X-ACL-Approval-Id): a pointer the pipeline verifies, never an authorization by itself.
    approval_id: str | None = None


class PipelineOutcome(FrozenModel):
    """What the entry point returns to the agent. Never contains upstream error text."""

    decision: Decision
    status_code: int
    reason_code: str
    message: str = ""
    session_id: str | None = None
    approval_id: str | None = None  # set when the call is held for approval
    retry_after_s: int | None = None  # set when the call is throttled (HTTP 429)
    request: Any = Field(default=None, repr=False)  # final payload sent upstream
    result: Any = Field(default=None, repr=False)  # final result, set only when released

    @property
    def released(self) -> bool:
        return HTTPStatus(self.status_code).is_success


@dataclass(frozen=True, slots=True)
class ChannelRoute:
    """How one channel's calls are normalized and executed."""

    adapter: Adapter
    upstream: Upstream


class SessionGate:
    """Authentication and session admission (pipeline step 1), shared by every agent route."""

    def __init__(self, verifier: TokenVerifier, sessions: SessionStore) -> None:
        self._verifier = verifier
        self._sessions = sessions

    @property
    def sessions(self) -> SessionStore:
        return self._sessions

    def authenticate(self, token: str | None, snapshot: PolicySnapshot) -> TokenClaims:
        return self._verifier.verify(token, snapshot)

    @asynccontextmanager
    async def session(
        self, claims: TokenClaims, snapshot: PolicySnapshot
    ) -> AsyncGenerator[SessionContext]:
        """Hold the session lock and yield its state; binding and lifetime enforced on entry."""
        binding = SessionBinding(principal=claims.sub, actor=claims.agent, mode=claims.mode)
        async with self._sessions.lock(claims.session_id):
            yield await self._sessions.open(claims.session_id, binding, snapshot.policy.sessions)

    @asynccontextmanager
    async def admit(
        self, token: str | None, snapshot: PolicySnapshot
    ) -> AsyncGenerator[tuple[TokenClaims, SessionContext]]:
        claims = self.authenticate(token, snapshot)
        async with self.session(claims, snapshot) as ctx:
            yield claims, ctx


class DecisionRecorder:
    """Turns finished calls into audit entries and metrics, with bounded label values."""

    def __init__(
        self,
        audit: AuditLogger,
        hmac_key: bytes,
        known_users: frozenset[str],
        feed: Callable[[], SignatureFeed] = lambda: EMPTY_FEED,
        llm_upstream: LlmUpstreamKind = LlmUpstreamKind.LOCAL,
    ) -> None:
        self._audit = audit
        self._key = hmac_key
        self._known_users = known_users
        self._feed = feed
        self.llm_upstream = llm_upstream  # stamped on every LLM audit entry

    def current_feed(self) -> SignatureFeed:
        """The signature feed in effect now: pinned per call, its version in every audit entry."""
        return self._feed()

    def digest(self, payload: object) -> str:
        return payload_hmac(self._key, payload)

    def user_label(self, principal: str | None, snapshot: PolicySnapshot) -> str:
        service = {service_principal(agent) for agent in snapshot.policy.agents}
        return bounded(principal, self._known_users | service)

    def agent_label(self, agent: str | None, snapshot: PolicySnapshot) -> str:
        return bounded(agent, snapshot.policy.agents)

    def write(self, entry: AuditEntry) -> None:
        self._audit.write(entry)


@dataclass(slots=True)
class _Step:
    """One interaction moving through the pipeline, with every verdict it collected."""

    interaction: Interaction
    access: AccessDecision
    original_payload: object = None  # the agent's payload, before any rewrite
    verdicts: list[tuple[Stage, Verdict]] = field(default_factory=list[tuple[Stage, Verdict]])

    def add(self, stage: Stage, verdicts: Sequence[Verdict]) -> None:
        self.verdicts.extend((stage, v) for v in verdicts)

    def merged(self) -> MergedVerdict:
        return merge_verdicts([v for _, v in self.verdicts])


@dataclass(frozen=True, slots=True)
class _Admitted:
    """Who is calling and through which route, fixed once the call is admitted."""

    claims: TokenClaims
    ctx: SessionContext
    principal: PrincipalContext
    route: ChannelRoute
    now: datetime


@dataclass(slots=True)
class _Trace:
    """Everything one call accumulates for persistence and the audit entry."""

    call: CallRequest
    snapshot: PolicySnapshot
    started: float
    # The signature feed pinned when the call was admitted: every control of the call sees
    # this one instance (via CallScope) and the audit entry names its version.
    feed: SignatureFeed = EMPTY_FEED
    claims: TokenClaims | None = None
    context: SessionContext | None = None
    steps: list[_Step] = field(default_factory=list[_Step])
    executed: list[Interaction] = field(default_factory=list[Interaction])  # what ran upstream
    upstream: UpstreamResult | None = None
    # The upstream failed after untrusted content may have reached the gateway (an error text,
    # a malformed or oversized body, a timeout after the request was sent): it taints.
    untrusted_failure: bool = False
    dispatched: object = None  # the final payload sent upstream (an LLM session's goal source)
    seal: "_Seal | None" = None  # set when a sealing control allowed the final payload
    # The approval the call presented, verified as bound to it (pending or approved), or why
    # it was refused; `approval_covered` once it satisfied the approval obligation.
    approval: Approval | None = None
    approval_refusal: ApprovalRefusal | None = None
    approval_covered: bool = False
    consumed: Approval | None = None  # moved approved -> executing: its outcome is recorded

    def verdicts(self) -> list[Verdict]:
        return [v for step in self.steps for _, v in step.verdicts]

    def distinct_verdicts(self) -> list[Verdict]:
        """Every verdict once: a call-wide one (``budget``, a seal refusal) is attached to each
        interaction as the same object, but it was decided, and took its time, once."""
        return list({id(v): v for v in self.verdicts()}.values())

    def control_latency_ms(self) -> dict[str, float]:
        """Time per control over the whole call, every stage and interaction summed."""
        totals: dict[str, float] = {}
        for verdict in self.distinct_verdicts():
            totals[verdict.control_id] = totals.get(verdict.control_id, 0.0) + verdict.latency_ms
        return totals


@dataclass(frozen=True, slots=True)
class _Seal:
    """The payload sealing controls allowed, as immutable canonical JSON bytes: the only
    payload that may be dispatched. Bytes, not the object: a later step could mutate a shared
    dict in place, and ``==`` would call ``true`` and ``1`` the same."""

    control_id: str
    reason_code: str  # refusal when the dispatched payload differs
    canonical: bytes


def _parse_json_object(body: bytes) -> dict[str, Any]:
    try:
        data: object = json.loads(body)
    except ValueError:
        raise InvalidRequestError("invalid_json", "request body is not valid JSON") from None
    if not isinstance(data, dict):
        raise InvalidRequestError("invalid_json", "request body must be a JSON object")
    return cast("dict[str, Any]", data)  # json.loads keys are always str


def _first_reason(merged: MergedVerdict) -> str:
    """Reason code of the first enforced verdict carrying the merged decision."""
    return next(
        (v.reason_code for v in merged.verdicts if v.enforced and v.decision is merged.decision),
        "allowed",
    )


def _final(original: object, candidates: Sequence[object], spans: MergedVerdict) -> object:
    """The one value every interaction agrees on, with the merged redactions applied."""
    rewrites: list[object] = []
    for candidate in candidates:
        if candidate != original and candidate not in rewrites:
            rewrites.append(candidate)
    if len(rewrites) > 1:
        raise RejectionError("conflicting_rewrites", "controls rewrote the call inconsistently")
    return apply_redactions(rewrites[0] if rewrites else original, spans.redactions)


def _effective_scope(snapshot: PolicySnapshot, step: _Step) -> tuple[str, ...]:
    """Grants still usable in this session: the principal's (or task) grants, narrowed to the
    agent's ``max_actions`` minus actions the session's risk rules currently remove."""
    access = step.access
    agent = snapshot.policy.agents.get(step.interaction.actor)
    if agent is None:
        return ()
    rules = snapshot.policy.risk_rules.for_mode(step.interaction.mode)
    removed = {a for i in access.restriction.matched_rules for a in rules[i].then.deny_actions}
    authz = access.authorization
    grants = authz.task_scope if authz.task_scope is not None else authz.principal_grants
    usable = set(agent.max_actions) - removed
    return tuple(PermissionSet.parse(grants).restricted_to(usable).as_strings())


def _holding(trace: _Trace) -> list[Verdict]:
    """The verdicts that decide whether a result is released. A post ``require_approval``
    that names ``flags`` (``intent_judge``) never holds a result that already exists: its
    approval obligation moves to the flagged MCP calls instead (SPEC "Intent vs enforcement").
    One without flags still holds, so no approval obligation is ever dropped."""
    return [
        verdict
        for step in trace.steps
        for stage, verdict in step.verdicts
        if not (
            stage is Stage.POST and verdict.decision is Decision.REQUIRE_APPROVAL and verdict.flags
        )
    ]


def _model_label(executed: Sequence[Interaction]) -> str:
    """``acl_tokens_total``'s model: the model the call was authorized for and ran with.

    Never the upstream's own ``model`` string, which the gateway does not control.
    """
    for interaction in executed:
        resource = Resource.parse(interaction.resource)
        if resource.namespace == "model":
            return resource.identifier
    return OTHER_LABEL


def _throttles(steps: Sequence[_Step]) -> list[Throttle]:
    caps: list[Throttle] = []
    for step in steps:
        caps.extend(cap for cap in step.access.restriction.throttles if cap not in caps)
    return caps


class Pipeline:
    """Runs one call through every step and returns what the agent gets."""

    def __init__(  # noqa: PLR0913 -- the collaborators of the pipeline, wired once at startup
        self,
        gate: SessionGate,
        channels: Mapping[Channel, ChannelRoute],
        controls: ControlRegistry,
        recorder: DecisionRecorder,
        *,
        oversight: Oversight,
        clock: Clock = utc_now,
        budgets: BudgetLedger | None = None,
    ) -> None:
        self._gate = gate
        self._oversight = oversight
        self._channels = channels
        self._controls = controls
        self._recorder = recorder
        self._clock = clock
        self._budgets = budgets
        self._evaluator = PolicyEvaluator()
        self._throttler = Throttler()

    @property
    def evaluator(self) -> PolicyEvaluator:
        return self._evaluator

    @property
    def controls(self) -> ControlRegistry:
        return self._controls

    async def handle(
        self, call: CallRequest, snapshot: PolicySnapshot, *, route: ChannelRoute | None = None
    ) -> PipelineOutcome:
        """Run ``call``; ``route`` overrides the channel's registered route for this call only
        (the MCP proxy binds the adapter to the call's server and the upstream to its session)."""
        trace = _Trace(
            call=call,
            snapshot=snapshot,
            started=time.perf_counter(),
            feed=self._recorder.current_feed(),
        )
        try:
            claims = trace.claims = self._gate.authenticate(call.token, snapshot)
            async with self._gate.session(claims, snapshot) as ctx:
                trace.context = ctx
                try:
                    outcome = await self._run(trace, claims, ctx, route)
                except RejectionError as exc:
                    outcome = self._refusal(trace, exc)
                await self._persist(trace)
        except RejectionError as exc:  # authentication or session admission
            outcome = self._refusal(trace, exc)
        self._record(trace, outcome)
        return outcome

    def record_refusal(
        self, call: CallRequest, snapshot: PolicySnapshot, exc: RejectionError
    ) -> PipelineOutcome:
        """Audit and count a call an entry point refused before it reached `handle`
        (e.g. a body over ``max_request_bytes``): with identity if the bearer verifies."""
        trace = _Trace(
            call=call,
            snapshot=snapshot,
            started=time.perf_counter(),
            feed=self._recorder.current_feed(),
        )
        try:
            trace.claims = self._gate.authenticate(call.token, snapshot)
        except RejectionError:
            trace.claims = None
        outcome = self._refusal(trace, exc)
        self._record(trace, outcome)
        return outcome

    # ------------------------------------------------------------------------- steps

    async def _run(
        self,
        trace: _Trace,
        claims: TokenClaims,
        ctx: SessionContext,
        route: ChannelRoute | None,
    ) -> PipelineOutcome:
        route, interactions = self._normalize(trace, ctx, route)
        call = _Admitted(
            claims=claims,
            ctx=ctx,
            principal=claims.principal_context(),
            route=route,
            now=self._clock(),
        )
        await self._resolve_presented(trace, claims, interactions)
        scope = CallScope(
            snapshot=trace.snapshot,
            principal=call.principal,
            feed=trace.feed,
            # Only a verified approval (bound to this exact call) excludes it from loop_detect.
            approval_id=trace.approval.id if trace.approval is not None else None,
        )
        with call_scope(scope):
            prepared = await self._prepare(trace, call, interactions)
            if isinstance(prepared, PipelineOutcome):
                return prepared
            payload, merged = prepared
            return await self._execute_metered(trace, call, payload, merged)

    def _normalize(
        self, trace: _Trace, ctx: SessionContext, route: ChannelRoute | None
    ) -> tuple[ChannelRoute, list[Interaction]]:
        """Step 2: size cap, channel route, JSON body, adapter."""
        snapshot, call = trace.snapshot, trace.call
        limit = snapshot.policy.limits.max_request_bytes
        if len(call.body) > limit:
            raise RequestTooLargeError(limit)
        route = route if route is not None else self._channels.get(call.channel)
        if route is None:
            raise InvalidRequestError("unsupported_channel", "channel not served")
        raw = RawCall(channel=call.channel, data=_parse_json_object(call.body), server=call.server)
        if not route.adapter.matches(raw):
            raise InvalidRequestError("unsupported_request", "no adapter for this request")
        interactions = route.adapter.normalize(raw, ctx)
        if not interactions:
            raise InvalidRequestError("empty_request", "the request names no operation")
        return route, interactions

    async def _prepare(
        self, trace: _Trace, call: _Admitted, interactions: list[Interaction]
    ) -> tuple[object, MergedVerdict] | PipelineOutcome:
        """Steps 3-5 plus the rewrite check and throttling: the final payload, or a refusal."""
        snapshot = trace.snapshot
        original = interactions[0].payload

        # Step 3: base authorization and session restrictions.
        trace.steps = [self._authorize(snapshot, call, i) for i in interactions]
        self._raise_alerts(trace, call.ctx)
        if (stopped := await self._admission_refusal(trace)) is not None:
            return stopped
        self._check_flags(trace, call)

        # Steps 4-5: pre controls and merge.
        for step in trace.steps:
            step.interaction, verdicts = await self._run_controls(
                snapshot, step.interaction, Stage.PRE
            )
            step.add(Stage.PRE, verdicts)
        merged = merge_verdicts(trace.verdicts())
        if merged.decision is Decision.BLOCK:
            return self._outcome(trace, merged)
        payload = _final(original, [s.interaction.payload for s in trace.steps], merged)
        if merged.requires_approval and not self._approval_covers(trace, payload):
            return await self._hold_for_approval(trace, merged, payload)
        sealed = await self._seal(trace, payload)
        if isinstance(sealed, PipelineOutcome):
            return sealed
        payload, merged = sealed, merge_verdicts(trace.verdicts())
        trace.executed = [step.interaction for step in trace.steps]
        if payload != original:  # what runs upstream is authorized too, not just what was asked
            refused = self._authorize_rewrite(trace, call, payload)
            if refused is not None:
                return refused

        # Throttle obligations: every cap, before anything executes.
        if caps := _throttles(trace.steps):
            agent = call.claims.agent
            try:
                self._throttler.admit(agent, caps, snapshot.policy.throttle, call.now)
            except ThrottledError:
                record_throttled(self._recorder.agent_label(agent, snapshot))
                raise
        return payload, merged

    async def _execute_metered(
        self, trace: _Trace, call: _Admitted, payload: object, merged: MergedVerdict
    ) -> PipelineOutcome:
        """The budget seam around steps 6-7 (SPEC "Budgets", `gateway.budget.ledger`).

        Reserve right before dispatch: inside the session lock, after authorization, controls
        and throttling, so nothing refused earlier holds budget. Settle in ``finally``, so
        completion, upstream failure and cancellation all reconcile the hold. The outcome is a
        ``budget`` verdict like any control's, so audit, metrics and merging see it the same way.
        """
        if self._budgets is None:
            return await self._execute(trace, call, payload, merged)
        snapshot, claims = trace.snapshot, call.claims
        metered = BudgetedCall(
            session_id=claims.session_id,
            principal=claims.sub,
            agent=claims.agent,
            channel=trace.call.channel,
            model=_model_label(trace.executed) if trace.call.channel is Channel.LLM else None,
            payload=payload,
            user_label=self._recorder.user_label(claims.sub, snapshot),
            agent_label=self._recorder.agent_label(claims.agent, snapshot),
        )
        started = time.perf_counter()
        try:
            reservation = await self._budgets.reserve(metered, snapshot)
        except BudgetRefusalError as exc:
            self._add_budget_verdict(trace, started, Decision.BLOCK, exc.reason_code, exc.message)
            raise
        self._add_budget_verdict(trace, started, Decision.ALLOW, "within_budget")
        if reservation.gpu_allowance_s is not None:  # the GPU time held is all it may use
            bounded = DeadlineUpstream(call.route.upstream, reservation.gpu_allowance_s)
            call = replace(call, route=replace(call.route, upstream=bounded))
        try:
            return await self._execute(trace, call, reservation.payload, merged)
        finally:
            await self._budgets.settle(reservation, trace.upstream)

    @staticmethod
    def _add_budget_verdict(
        trace: _Trace, started: float, decision: Decision, reason_code: str, reason: str = ""
    ) -> None:
        verdict = Verdict(
            decision=decision,
            control_id=BUDGET,
            reason_code=reason_code,
            reason=reason,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        for step in trace.steps:
            step.add(Stage.PRE, [verdict])

    async def _execute(
        self, trace: _Trace, call: _Admitted, payload: object, merged: MergedVerdict
    ) -> PipelineOutcome:
        """Steps 6-7: run once upstream, then post controls on the complete result."""
        snapshot = trace.snapshot
        if trace.seal is not None:
            if _canonical_bytes(payload) == trace.seal.canonical:
                # Dispatch a fresh object decoded from the sealed bytes: nothing can alias it.
                payload = json.loads(trace.seal.canonical)
            else:
                changed = Verdict(
                    decision=Decision.BLOCK,
                    control_id=trace.seal.control_id,
                    reason_code=trace.seal.reason_code,
                )
                for step in trace.steps:
                    step.add(Stage.PRE, [changed])
                return self._outcome(trace, merge_verdicts(trace.verdicts()))
        # The session lease must still be this call's, and the session stays fenced for other
        # gateways until this call persists its outcome (raises 503 session_lease_lost).
        await self._gate.sessions.before_dispatch(
            call.claims.session_id, upstream_timeout_s=snapshot.policy.limits.upstream_timeout_s
        )
        if (stopped := await self._before_dispatch(trace)) is not None:
            return stopped
        trace.dispatched = payload
        try:
            trace.upstream = await call.route.upstream.execute(payload, snapshot)
        except UpstreamError as exc:
            trace.untrusted_failure = exc.untrusted
            await self._record_approval_outcome(trace, exc)
            return self._refusal(trace, exc, decision=self._satisfied(trace, merged).decision)
        except asyncio.CancelledError:  # the request may have been sent: outcome unknown
            await self._record_approval_cancelled(trace)
            raise
        await self._record_approval_outcome(trace, None)
        post: list[Verdict] = []
        for step in trace.steps:
            with_result = step.interaction.model_copy(
                update={"payload": payload, "result": trace.upstream.body}
            )
            step.interaction, verdicts = await self._run_controls(snapshot, with_result, Stage.POST)
            step.add(Stage.POST, verdicts)
            post.extend(verdicts)
        merged_post = merge_verdicts(post)
        overall = merge_verdicts(_holding(trace))
        if overall.decision is Decision.BLOCK:
            return self._outcome(trace, overall)
        if (stopped := await self._kill_check(trace)) is not None:  # before releasing a result
            return stopped
        result = _final(
            trace.upstream.body, [s.interaction.result for s in trace.steps], merged_post
        )
        return self._outcome(trace, overall, request=payload, result=result)

    def _authorize(
        self, snapshot: PolicySnapshot, call: _Admitted, interaction: Interaction
    ) -> _Step:
        started = time.perf_counter()
        access = self._evaluator.decide(
            snapshot,
            call.principal,
            call.ctx,
            channel=interaction.channel,
            action=interaction.action,
            resource=interaction.resource,
            now=call.now,
        )
        denied = access.decision is Decision.BLOCK
        verdict = Verdict(
            decision=access.decision,
            control_id=AUTHZ,
            reason_code=access.reason_code,
            risk_delta=snapshot.policy.control_risk_delta(AUTHZ) if denied else 0.0,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        step = _Step(interaction=interaction, access=access, original_payload=interaction.payload)
        step.add(Stage.PRE, [verdict])
        return step

    def _authorize_rewrite(
        self, trace: _Trace, call: _Admitted, payload: object
    ) -> PipelineOutcome | None:
        """Re-normalize the rewritten payload and authorize every interaction it yields.

        A control may rewrite a call into one its caller could never make (another model,
        another table); anything not plainly allowed blocks the call. None = proceed.
        """
        snapshot = trace.snapshot
        original = trace.steps[0].original_payload
        rewritten: list[Interaction] = []
        if isinstance(payload, dict):
            data = cast("dict[str, Any]", payload)
            raw = RawCall(channel=trace.call.channel, data=data, server=trace.call.server)
            try:
                rewritten = call.route.adapter.normalize(raw, call.ctx)
            except RejectionError:
                rewritten = []
        refusals = [
            step
            for step in (self._authorize(snapshot, call, i) for i in rewritten)
            if step.access.decision is not Decision.ALLOW
            and not self._approved_rewrite(trace, step)
        ]
        if rewritten and not refusals:
            trace.executed = rewritten
            return None
        refused = Verdict(
            decision=Decision.BLOCK,
            control_id=AUTHZ,
            reason_code=REWRITE_UNAUTHORIZED,
            risk_delta=snapshot.policy.control_risk_delta(AUTHZ),
        )
        if not refusals:  # the rewrite is not even a valid call: refuse it on the first step
            trace.steps[0].add(Stage.PRE, [refused])
        for step in refusals:  # audited under the resource the rewrite would have reached
            step.original_payload = original
            step.verdicts = [(Stage.PRE, refused)]
            trace.steps.append(step)
        return self._outcome(
            trace, merge_verdicts(trace.verdicts()), reason_code=REWRITE_UNAUTHORIZED
        )

    @staticmethod
    def _approved_rewrite(trace: _Trace, step: _Step) -> bool:
        """A rewritten interaction whose only obligation is the approval this call already
        satisfies (same ``action:resource`` the approval names). Base-authorization denials
        and session removals are never covered."""
        approval = trace.approval
        interaction = step.interaction
        return (
            trace.approval_covered
            and approval is not None
            and step.access.decision is Decision.REQUIRE_APPROVAL
            and f"{interaction.action.value}:{interaction.resource}" in approval.binding.operations
        )

    def _check_flags(self, trace: _Trace, call: _Admitted) -> None:
        """An MCP ``tools/call`` matching a ``tool_call`` the intent judge flagged in this
        session needs approval, through the normal MCP approval flow (SPEC "Intent vs
        enforcement"). Matched on the agent's own arguments and on the adapter's canonical
        form, so neither spelling slips past. Once the session kept fewer flags than were raised
        (``flags_overflowed``), every MCP call needs approval. Honoured while ``intent_judge``
        enforces."""
        if trace.call.channel is not Channel.MCP:
            return
        if call.ctx.flags_overflowed:  # evicted flags could be this call's: hold every call
            reason = INTENT_FLAGS_OVERFLOW
        elif call.ctx.flagged_tool_calls and not mcp_flag_candidates(
            _parse_json_object(trace.call.body), *(step.original_payload for step in trace.steps)
        ).isdisjoint(call.ctx.flagged_tool_calls):
            reason = INTENT_FLAGGED
        else:
            return
        mode = trace.snapshot.policy.resolved_control_mode(INTENT_JUDGE)
        verdict = Verdict(
            decision=Decision.REQUIRE_APPROVAL,
            control_id=INTENT_JUDGE,
            reason_code=reason,
            enforced=mode is not ControlMode.LOG_ONLY,
        )
        for step in trace.steps:
            step.add(Stage.PRE, [verdict])

    def _raise_alerts(self, trace: _Trace, ctx: SessionContext) -> None:
        """One structured alert line and counter per risk rule with ``alert: true`` that holds."""
        rules = trace.snapshot.policy.risk_rules.for_mode(ctx.mode)
        raised: dict[str, float] = {}
        for step in trace.steps:
            restriction = step.access.restriction
            for index in restriction.matched_rules:
                if rules[index].then.alert:
                    raised.setdefault(f"{ctx.mode.value}.{index}", restriction.risk)
        for rule, risk in raised.items():
            record_alert(rule)
            alert_logger.warning(
                json.dumps(
                    {
                        "event": "risk_rule_alert",
                        "rule": rule,
                        "session_id": ctx.session_id,
                        "principal": ctx.principal,
                        "agent": ctx.actor,
                        "risk": round(risk, 4),
                        "policy_revision": trace.snapshot.revision,
                    },
                    sort_keys=True,
                )
            )

    async def _seal(self, trace: _Trace, payload: object) -> object | PipelineOutcome:
        """Sealing pre controls on the final payload, after every other control's rewrites and
        redactions: what they price and allow is exactly what `_execute` may dispatch."""
        channel = trace.call.channel
        sealing = [c for c in self._controls.for_stage(Stage.PRE, channel) if c.seal is not None]
        if not sealing:
            return payload
        candidates: list[object] = []
        for step in trace.steps:
            final = step.interaction.model_copy(update={"payload": payload})
            step.interaction, verdicts = await self._run_controls(
                trace.snapshot, final, Stage.PRE, sealing=True
            )
            step.add(Stage.PRE, verdicts)
            candidates.append(step.interaction.payload)
        merged = merge_verdicts(trace.verdicts())
        if merged.decision is Decision.BLOCK:
            return self._outcome(trace, merged)
        if merged.requires_approval and not self._approval_covers(trace, payload):
            return await self._hold_for_approval(trace, merged, payload)
        sealed = _final(payload, candidates, MergedVerdict(decision=Decision.ALLOW))  # no spans
        last = sealing[-1]
        canonical = _canonical_bytes(sealed)
        if canonical is None:  # not plain JSON: nothing to seal it by, so nothing runs
            refused = Verdict(
                decision=Decision.BLOCK, control_id=last.id, reason_code=str(last.seal)
            )
            for step in trace.steps:
                step.add(Stage.PRE, [refused])
            return self._outcome(trace, merge_verdicts(trace.verdicts()))
        trace.seal = _Seal(control_id=last.id, reason_code=str(last.seal), canonical=canonical)
        return json.loads(canonical)  # later steps get their own copy, never the sealed object

    async def _run_controls(
        self,
        snapshot: PolicySnapshot,
        interaction: Interaction,
        stage: Stage,
        *,
        sealing: bool = False,
    ) -> tuple[Interaction, list[Verdict]]:
        """Run the stage's ordinary (or, with ``sealing``, sealing) controls in order; an
        enforced rewrite feeds the next control."""
        policy = snapshot.policy
        current, verdicts = interaction, list[Verdict]()
        target: Literal["payload", "result"] = "payload" if stage is Stage.PRE else "result"
        controls = self._controls.for_stage(stage, interaction.channel)
        for control in (c for c in controls if (c.seal is not None) == sealing):
            mode = policy.resolved_control_mode(control.id)
            cfg = policy.control_config(control.id).model_copy(
                update={"mode": mode, "risk_delta": policy.control_risk_delta(control.id)}
            )
            started = time.perf_counter()
            try:
                verdict = await control.evaluate(current, stage, cfg)
            except Exception as exc:  # a broken control fails closed
                # Type name only: exception messages and tracebacks may quote the payload.
                logger.error(  # noqa: TRY400 -- deliberately no traceback
                    "control_error control=%s error=%s session=%s",
                    control.id,
                    type(exc).__name__,
                    interaction.session_id,
                )
                verdict = Verdict(
                    decision=Decision.BLOCK, control_id=control.id, reason_code="control_error"
                )
            enforced = verdict.enforced and mode is not ControlMode.LOG_ONLY
            verdict = verdict.model_copy(
                update={
                    "enforced": enforced,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
            if enforced and verdict.rewrite is not None:
                current = current.model_copy(update={target: verdict.rewrite})
            verdicts.append(verdict)
        return current, verdicts

    # ---------------------------------------------------------- oversight seams

    async def _resolve_presented(
        self, trace: _Trace, claims: TokenClaims, interactions: Sequence[Interaction]
    ) -> None:
        """Look up the approval a retry names; a refusal is applied after authorization."""
        approval_id = trace.call.approval_id
        if approval_id is None:
            return
        binding = self._oversight.binding(
            claims,
            trace.call.channel,
            trace.call.server,
            interactions[0].payload,
            ((i.action, i.resource) for i in interactions),
        )
        presented = await self._oversight.presented(approval_id, binding)
        if isinstance(presented, ApprovalRefusal):
            trace.approval_refusal = presented
        else:
            trace.approval = presented

    async def _admission_refusal(self, trace: _Trace) -> PipelineOutcome | None:
        """Step 3's refusals: base authorization, then the kill switch, then a refused
        presented approval (denied, expired, used, bound to another call). A refused approval
        is never consumed."""
        if any(step.access.decision is Decision.BLOCK for step in trace.steps):
            return self._outcome(trace, merge_verdicts(trace.verdicts()))
        if (stopped := await self._kill_check(trace)) is not None:
            return stopped
        if trace.approval_refusal is not None:
            return self._stop(trace, APPROVAL, trace.approval_refusal)
        return None

    def _approval_covers(self, trace: _Trace, payload: object) -> bool:
        """An approved record bound to this call and held for exactly this final payload
        (after rewrites and redactions) satisfies the approval obligation (only that one:
        every other verdict still applies)."""
        digest = self._oversight.operation_digest(payload)
        trace.approval_covered = self._oversight.covers(trace.approval, digest)
        return trace.approval_covered

    async def _kill_check(self, trace: _Trace) -> PipelineOutcome | None:
        """A killed agent's call stops here; an unknown kill state fails closed (503)."""
        if trace.claims is None:
            return None
        try:
            killed = await self._oversight.killed(trace.claims.agent)
        except KillSwitchUnavailableError as exc:
            self._add_seam_verdict(trace, KILL_SWITCH, exc.reason_code)
            raise
        if killed is None:
            return None
        return self._stop(trace, KILL_SWITCH, AGENT_KILLED)

    async def _before_dispatch(self, trace: _Trace) -> PipelineOutcome | None:
        """Right before the upstream: the kill switch, then consume a presented approval
        (``approved → executing``, atomically: at most one dispatch per approval), then the
        kill switch once more, because consuming awaits the store and a kill may land
        meanwhile. Nothing else awaits between that last check and the dispatch."""
        if (stopped := await self._kill_check(trace)) is not None:
            return stopped
        approval = trace.approval
        if not trace.approval_covered or approval is None:
            return None
        consumed = await self._oversight.consume(approval)
        if isinstance(consumed, ApprovalRefusal):
            return self._stop(trace, APPROVAL, consumed)
        try:
            stopped = await self._kill_check(trace)
        except KillSwitchUnavailableError as exc:
            await self._oversight.abandon(consumed, exc.reason_code)
            raise
        if stopped is not None:  # consumed, never dispatched: closed for good
            await self._oversight.abandon(consumed, AGENT_KILLED)
            return stopped
        trace.consumed = consumed
        return None

    async def _record_approval_outcome(self, trace: _Trace, error: UpstreamError | None) -> None:
        if trace.consumed is not None:
            await self._oversight.finish(trace.consumed, trace.upstream, error)

    async def _record_approval_cancelled(self, trace: _Trace) -> None:
        if trace.consumed is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(
                    self._oversight.finish_uncertain(trace.consumed, "dispatch_cancelled")
                )

    def _stop(self, trace: _Trace, control_id: str, reason_code: str) -> PipelineOutcome:
        self._add_seam_verdict(trace, control_id, reason_code)
        return self._outcome(trace, merge_verdicts(trace.verdicts()))

    @staticmethod
    def _add_seam_verdict(trace: _Trace, control_id: str, reason_code: str) -> None:
        verdict = Verdict(decision=Decision.BLOCK, control_id=control_id, reason_code=reason_code)
        for step in trace.steps:
            step.add(Stage.PRE, [verdict])

    @staticmethod
    def _satisfied(trace: _Trace, merged: MergedVerdict) -> MergedVerdict:
        """``merged`` without the approval obligation once an approved record covers it."""
        if not (trace.approval_covered and merged.requires_approval):
            return merged
        return merge_verdicts(
            [v for v in merged.verdicts if v.decision is not Decision.REQUIRE_APPROVAL]
        )

    async def _hold_for_approval(
        self, trace: _Trace, merged: MergedVerdict, payload: object
    ) -> PipelineOutcome:
        """The approval queue (SPEC "Human in the loop"): create or get the pending record of
        this exact operation and answer with its id. The record binds what the agent sent,
        every normalized ``action:resource`` and the final ``payload`` (after rewrites and
        redactions). A retry of the same pending operation gets the same id; nothing executes
        until an operator approves and the agent retries with the id."""
        claims = trace.claims
        binding = (
            self._oversight.binding(
                claims,
                trace.call.channel,
                trace.call.server,
                trace.steps[0].original_payload,
                ((step.interaction.action, step.interaction.resource) for step in trace.steps),
            )
            if claims is not None and trace.steps
            else None
        )
        operation_digest = self._oversight.operation_digest(payload)
        presented = trace.approval
        if presented is not None and presented.operation_digest != operation_digest:
            # The id names another final operation (redaction or rewrite changed since):
            # it cannot stand for this one. Retrying without it gets this operation's own.
            return self._stop(trace, APPROVAL, ApprovalRefusal.MISMATCH)
        if binding is None or operation_digest is None:  # nothing canonical to bind to
            return self._stop(trace, APPROVAL, ApprovalRefusal.UNBINDABLE)
        reasons = tuple(
            v.reason_code
            for v in merged.verdicts
            if v.enforced and v.decision is Decision.REQUIRE_APPROVAL
        )
        record = await self._oversight.hold(
            binding,
            trace.snapshot,
            operation_digest=operation_digest,
            actions=tuple(step.interaction.action for step in trace.steps),
            resources=tuple(step.interaction.resource for step in trace.steps),
            reasons=reasons,
        )
        held = self._outcome(trace, merged, reason_code="approval_required")
        return held.model_copy(
            update={"message": "this call needs human approval", "approval_id": record.id}
        )

    async def _persist(self, trace: _Trace) -> None:
        """Every verdict's risk, taint and timers land before the result is released."""
        if trace.context is None:
            return
        now = self._clock()
        verdicts = trace.verdicts()
        cooldowns: list[Cooldown] = []
        freezes: list[datetime] = []
        for step in trace.steps:
            restriction = step.access.restriction
            if restriction.start_freeze is not None:
                freezes.append(restriction.start_freeze.until)
            timer = restriction.cooldown_on_deny
            if timer is not None and timer.key and step.merged().decision is Decision.BLOCK:
                cooldowns.append(Cooldown(key=timer.key, until=timer.until))
        # A result from an untrusted source taints even when post controls blocked or replaced
        # it, and so does a failure after the request reached it: the content reached the
        # gateway on the agent's behalf.
        untrusted_result = trace.untrusted_failure or (
            trace.upstream is not None and trace.upstream.untrusted
        )
        update = SessionUpdate(
            risk_delta=sum(v.risk_delta for v in verdicts),
            taint=untrusted_result
            or any(
                v.control_id in TAINTING_CONTROLS and v.decision is not Decision.ALLOW
                for v in verdicts
            )
            or any(v.taint for v in verdicts),  # an allow verdict that asks for taint
            freeze_until=min(freezes, default=None),
            cooldowns=tuple(cooldowns),
            flagged_tool_calls=tuple(flag for v in verdicts if v.enforced for flag in v.flags),
            goal=(
                first_user_message(trace.dispatched)
                if trace.call.channel is Channel.LLM and trace.context.goal is None
                else None
            ),
            calls=tuple(
                CallRecord(
                    at=now,
                    channel=step.interaction.channel,
                    action=step.interaction.action,
                    resource=step.interaction.resource,
                    digest=self._recorder.digest(step.original_payload),
                )
                for step in trace.steps
            ),
        )
        trace.context = await self._gate.sessions.apply(
            trace.context.session_id,
            update,
            half_life_s=trace.snapshot.policy.risk.half_life_s,
        )
        record_session(trace.context.risk, await self._gate.sessions.tainted_count())

    # ---------------------------------------------------------------------- outcomes

    def _outcome(
        self,
        trace: _Trace,
        merged: MergedVerdict,
        *,
        reason_code: str | None = None,
        request: object = None,
        result: object = None,
    ) -> PipelineOutcome:
        merged = self._satisfied(trace, merged)
        decision = merged.decision
        reason = reason_code or _first_reason(merged)
        released = decision in {Decision.ALLOW, Decision.REDACT}
        return PipelineOutcome(
            decision=decision,
            status_code=200 if released else 403,
            reason_code=reason,
            message="" if released else f"blocked by policy: {reason}",
            session_id=trace.claims.session_id if trace.claims else None,
            request=request,
            result=result if released else None,
        )

    def _refusal(
        self, trace: _Trace, exc: RejectionError, *, decision: Decision = Decision.BLOCK
    ) -> PipelineOutcome:
        return PipelineOutcome(
            decision=decision,
            status_code=exc.status_code,
            reason_code=exc.reason_code,
            message=exc.message,
            session_id=trace.claims.session_id if trace.claims else None,
            retry_after_s=exc.retry_after_s if isinstance(exc, ThrottledError) else None,
        )

    def _record(self, trace: _Trace, outcome: PipelineOutcome) -> None:
        snapshot, claims, ctx = trace.snapshot, trace.claims, trace.context
        total = time.perf_counter() - trace.started
        upstream_s = trace.upstream.elapsed_s if trace.upstream else None
        agent = self._recorder.agent_label(claims.agent if claims else None, snapshot)
        user = self._recorder.user_label(claims.sub if claims else None, snapshot)
        channel = trace.call.channel

        record_request(channel, outcome.decision, agent)
        record_overhead(channel, total - (upstream_s or 0.0))
        record_verdicts(trace.distinct_verdicts())
        if trace.upstream is not None and trace.upstream.usage is not None:
            usage = trace.upstream.usage
            # Models without a pricing entry share `other`: a wildcard grant must not mint labels.
            model = bounded(_model_label(trace.executed), snapshot.policy.pricing)
            record_tokens(user, agent, model, charged_tokens(usage))

        latency = AuditLatency(
            total=total * 1000,
            upstream=upstream_s * 1000 if upstream_s is not None else None,
            controls=trace.control_latency_ms(),
        )
        identity: dict[str, Any] = {
            "session_id": claims.session_id if claims else None,
            "principal": claims.sub if claims else None,
            "actor": claims.agent if claims else None,
            "mode": claims.mode if claims else None,
            "risk": ctx.risk if ctx else None,
            "taint": ctx.taint if ctx else None,
        }
        common: dict[str, Any] = {
            "ts": self._clock(),
            "channel": channel,
            "decision": outcome.decision,
            "reason_code": outcome.reason_code,
            "status": outcome.status_code,
            "policy_revision": snapshot.revision,
            "feed_version": trace.feed.version,
            "latency_ms": latency,
            "approval_id": outcome.approval_id
            or (trace.approval.id if trace.approval is not None else None),
            "upstream": self._recorder.llm_upstream if channel is Channel.LLM else None,
            **identity,
        }
        if not trace.steps:
            self._recorder.write(AuditEntry(**common))
            return
        for step in trace.steps:
            self._recorder.write(
                AuditEntry(
                    **common,
                    action=step.interaction.action,
                    resource=step.interaction.resource,
                    verdicts=tuple(AuditVerdict.of(v, stage) for stage, v in step.verdicts),
                    effective_scope=_effective_scope(snapshot, step),
                    payload_hmac=self._recorder.digest(step.original_payload),
                )
            )
