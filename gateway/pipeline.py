"""The step order every call goes through (SPEC "Pipeline and interfaces" → "Order for every call").

1. Authentication: verify the JWT, then lock and load session state (one call per session).
2. Adapter: normalize the request into one or more `Interaction`s.
3. Base authorization plus session restrictions on every interaction; any deny stops here.
4. ``pre`` controls, deterministic first.
5. Merge verdicts into allow / redact / block / require_approval plus obligations.
6. Execute on the upstream once, with the final (rewritten, redacted) payload.
7. ``post`` controls on the complete result.
8. Persist session state (risk deltas from every verdict, including blocked calls; timers),
   then audit and metrics, all before the result is released.

The policy snapshot is taken once per call by the caller and threaded through every step, so
a reload mid-call never mixes two policy versions. Nothing here knows about a specific
provider: a channel is an `Adapter` plus an `Upstream`.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from typing import Any, Final, Literal, cast

from pydantic import Field

from gateway.clock import Clock, utc_now
from gateway.controls.registry import ControlRegistry
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
from gateway.core.types import Channel, ControlMode, Decision, Stage
from gateway.core.verdicts import MergedVerdict, merge_verdicts
from gateway.errors import InvalidRequestError, RejectionError, RequestTooLargeError
from gateway.identity import TokenClaims, TokenVerifier
from gateway.policy.evaluator import AccessDecision, PolicyEvaluator, PrincipalContext
from gateway.policy.loader import PolicySnapshot
from gateway.policy.permissions import PermissionSet
from gateway.policy.schema import service_principal
from gateway.redaction import apply_redactions
from gateway.sessions import SessionBinding, SessionStore, SessionUpdate
from gateway.telemetry import (
    AuditEntry,
    AuditLatency,
    AuditLogger,
    AuditVerdict,
    bounded,
    payload_hmac,
    record_overhead,
    record_request,
    record_session,
    record_tokens,
    record_verdicts,
)
from gateway.upstream import Upstream, UpstreamError, UpstreamResult

logger = logging.getLogger(__name__)

AUTHZ: Final = "authz"
# Controls whose detections mean untrusted content reached the agent's context.
TAINTING_CONTROLS: Final = frozenset({"prompt_injection"})
APPROVAL_ID_CHARS: Final = 24


class CallRequest(FrozenModel):
    """A call as an entry point received it."""

    channel: Channel
    token: str | None = Field(default=None, repr=False)
    body: bytes = Field(repr=False)  # read up to max_request_bytes + 1, never more
    server: str | None = None  # MCP upstream name


class PipelineOutcome(FrozenModel):
    """What the entry point returns to the agent. Never contains upstream error text."""

    decision: Decision
    status_code: int
    reason_code: str
    message: str = ""
    session_id: str | None = None
    approval_id: str | None = None  # set when the call is held for approval
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

    def __init__(self, audit: AuditLogger, hmac_key: bytes, known_users: frozenset[str]) -> None:
        self._audit = audit
        self._key = hmac_key
        self._known_users = known_users

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


@dataclass(slots=True)
class _Trace:
    """Everything one call accumulates for persistence and the audit entry."""

    call: CallRequest
    snapshot: PolicySnapshot
    started: float
    claims: TokenClaims | None = None
    context: SessionContext | None = None
    steps: list[_Step] = field(default_factory=list[_Step])
    upstream: UpstreamResult | None = None

    def verdicts(self) -> list[Verdict]:
        return [v for step in self.steps for _, v in step.verdicts]


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


class Pipeline:
    """Runs one call through every step and returns what the agent gets."""

    def __init__(
        self,
        gate: SessionGate,
        channels: Mapping[Channel, ChannelRoute],
        controls: ControlRegistry,
        recorder: DecisionRecorder,
        *,
        clock: Clock = utc_now,
    ) -> None:
        self._gate = gate
        self._channels = channels
        self._controls = controls
        self._recorder = recorder
        self._clock = clock
        self._evaluator = PolicyEvaluator()

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
        trace = _Trace(call=call, snapshot=snapshot, started=time.perf_counter())
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

    # ------------------------------------------------------------------------- steps

    async def _run(
        self,
        trace: _Trace,
        claims: TokenClaims,
        ctx: SessionContext,
        route: ChannelRoute | None,
    ) -> PipelineOutcome:
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
        original = interactions[0].payload

        # Step 3: base authorization and session restrictions.
        principal, now = claims.principal_context(), self._clock()
        trace.steps = [self._authorize(snapshot, principal, ctx, i, now) for i in interactions]
        if any(step.access.decision is Decision.BLOCK for step in trace.steps):
            return self._outcome(trace, merge_verdicts(trace.verdicts()))

        # Steps 4-5: pre controls and merge.
        for step in trace.steps:
            step.interaction, verdicts = await self._run_controls(
                snapshot, step.interaction, Stage.PRE
            )
            step.add(Stage.PRE, verdicts)
        merged = merge_verdicts(trace.verdicts())
        if merged.decision is Decision.BLOCK:
            return self._outcome(trace, merged)
        if merged.requires_approval:
            return self._hold_for_approval(trace, merged)
        payload = _final(original, [s.interaction.payload for s in trace.steps], merged)

        # Step 6: execute once.
        try:
            trace.upstream = await route.upstream.execute(payload, snapshot)
        except UpstreamError as exc:
            return self._refusal(trace, exc, decision=merged.decision)

        # Step 7: post controls on the complete result.
        post: list[Verdict] = []
        for step in trace.steps:
            with_result = step.interaction.model_copy(
                update={"payload": payload, "result": trace.upstream.body}
            )
            step.interaction, verdicts = await self._run_controls(snapshot, with_result, Stage.POST)
            step.add(Stage.POST, verdicts)
            post.extend(verdicts)
        merged_post = merge_verdicts(post)
        overall = merge_verdicts(trace.verdicts())
        if overall.decision is Decision.BLOCK:
            return self._outcome(trace, overall)
        result = _final(
            trace.upstream.body, [s.interaction.result for s in trace.steps], merged_post
        )
        return self._outcome(trace, overall, request=payload, result=result)

    def _authorize(
        self,
        snapshot: PolicySnapshot,
        principal: PrincipalContext,
        ctx: SessionContext,
        interaction: Interaction,
        now: datetime,
    ) -> _Step:
        access = self._evaluator.decide(
            snapshot,
            principal,
            ctx,
            channel=interaction.channel,
            action=interaction.action,
            resource=interaction.resource,
            now=now,
        )
        denied = access.decision is Decision.BLOCK
        verdict = Verdict(
            decision=access.decision,
            control_id=AUTHZ,
            reason_code=access.reason_code,
            risk_delta=snapshot.policy.control_risk_delta(AUTHZ) if denied else 0.0,
        )
        step = _Step(interaction=interaction, access=access, original_payload=interaction.payload)
        step.add(Stage.PRE, [verdict])
        return step

    async def _run_controls(
        self, snapshot: PolicySnapshot, interaction: Interaction, stage: Stage
    ) -> tuple[Interaction, list[Verdict]]:
        """Run the stage's controls in order; an enforced rewrite feeds the next control."""
        policy = snapshot.policy
        current, verdicts = interaction, list[Verdict]()
        target: Literal["payload", "result"] = "payload" if stage is Stage.PRE else "result"
        for control in self._controls.for_stage(stage, interaction.channel):
            mode = policy.resolved_control_mode(control.id)
            cfg = policy.control_config(control.id).model_copy(
                update={"mode": mode, "risk_delta": policy.control_risk_delta(control.id)}
            )
            started = time.perf_counter()
            try:
                verdict = await control.evaluate(current, stage, cfg)
            except Exception:  # a broken control fails closed
                logger.exception("control %s failed", control.id)
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

    def _hold_for_approval(self, trace: _Trace, merged: MergedVerdict) -> PipelineOutcome:
        """Seam for the approval queue (stages 10-14): until then an approval is a refusal.

        The ``approval_id`` is already bound to the one exact operation an approval will
        authorize (principal, agent, session, server, payload digest, policy revision), so a
        retry of the same pending call names the same id. The queue will persist it.
        """
        held = self._outcome(trace, merged, reason_code="approval_required")
        claims, call = trace.claims, trace.call
        binding = {
            "session_id": claims.session_id if claims else None,
            "principal": claims.sub if claims else None,
            "actor": claims.agent if claims else None,
            "channel": call.channel,
            "server": call.server,
            "payload": trace.steps[0].original_payload if trace.steps else None,
            "policy_revision": trace.snapshot.revision,
        }
        approval_id = f"apr-{self._recorder.digest(binding)[:APPROVAL_ID_CHARS]}"
        return held.model_copy(
            update={"message": "this call needs human approval", "approval_id": approval_id}
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
        # it: the content reached the gateway on the agent's behalf.
        untrusted_result = trace.upstream is not None and trace.upstream.untrusted
        update = SessionUpdate(
            risk_delta=sum(v.risk_delta for v in verdicts),
            taint=untrusted_result
            or any(
                v.control_id in TAINTING_CONTROLS and v.decision is not Decision.ALLOW
                for v in verdicts
            ),
            freeze_until=min(freezes, default=None),
            cooldowns=tuple(cooldowns),
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
        record_verdicts(trace.verdicts())
        if trace.upstream is not None and trace.upstream.usage is not None:
            usage = trace.upstream.usage
            record_tokens(user, agent, usage.model, usage.total_tokens)

        latency = AuditLatency(
            total=total * 1000,
            upstream=upstream_s * 1000 if upstream_s is not None else None,
            controls={v.control_id: v.latency_ms for v in trace.verdicts()},
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
            "latency_ms": latency,
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
