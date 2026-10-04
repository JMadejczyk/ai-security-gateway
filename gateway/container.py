"""Composition root: builds every long-lived component once and owns their lifecycle.

Both listeners (agent and operator) share one container, so they see the same policy
store, session state and upstream connection pool.
"""

import asyncio
import contextlib
import functools
import json
import logging
import os
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import Self, TextIO
from urllib.parse import urlsplit

import httpx

from gateway.adapters.llm import LLMAdapter
from gateway.approvals.factory import OperatorStores, operator_stores_from_settings
from gateway.approvals.kill_switch import KillSwitch
from gateway.approvals.metrics import set_killed_agents
from gateway.approvals.oversight import Oversight
from gateway.approvals.service import ApprovalService
from gateway.approvals.sweeper import sweep_forever
from gateway.budget.factory import budget_store_from_settings
from gateway.budget.ledger import BudgetLedger
from gateway.clock import Clock, utc_now
from gateway.controls.egress import EgressControl, HostResolver, system_resolver
from gateway.controls.intent_judge import IntentJudgeControl
from gateway.controls.loop_detect import LoopDetectControl
from gateway.controls.model_allowlist import ModelAllowlistControl
from gateway.controls.output_policy import OutputPolicyControl
from gateway.controls.pii import PiiControl
from gateway.controls.prompt_injection import PromptInjectionControl
from gateway.controls.registry import ControlRegistry
from gateway.controls.secrets import SecretsControl
from gateway.controls.signatures import SignaturesControl
from gateway.controls.sql_guard import SqlGuardControl
from gateway.controls.tool_pinning import ToolPinningControl
from gateway.controls.tool_poisoning import ToolPoisoningControl
from gateway.core.types import Channel, LlmUpstreamKind
from gateway.errors import StartupError
from gateway.feed.store import FeedStore
from gateway.identity import DemoIdentities, DemoTokenIssuer, TokenVerifier
from gateway.injection.classifier import ClassifierRunner, InjectionClassifier, load_classifier
from gateway.judges import JudgeClient
from gateway.judges.client import JudgeFactory
from gateway.metric_series import initialize_metric_series
from gateway.pipeline import ChannelRoute, DecisionRecorder, Pipeline, SessionGate
from gateway.policy.evaluator import PolicyEvaluator
from gateway.policy.loader import PolicyLoader, PolicySnapshot
from gateway.policy.store import PolicyStore
from gateway.proxies.llm import LLMProxy, require_upstream
from gateway.proxies.mcp.downstream import MCPProxy
from gateway.proxies.mcp.explain import explain_cost
from gateway.proxies.mcp.pins import PinStore
from gateway.proxies.mcp.screens import signatures_screen
from gateway.proxies.mcp.sessions import MCPSessionRegistry
from gateway.proxies.mcp.upstream import MCPConnector
from gateway.sessions import SessionStore
from gateway.settings import Settings
from gateway.state_stores import StateStores
from gateway.telemetry import AuditLogger, PolicyReloadEvent, set_llm_upstream

logger = logging.getLogger(__name__)


def check_llm_upstream(
    settings: Settings, snapshot: PolicySnapshot, env: Mapping[str, str]
) -> None:
    """Refuse to start with a remote upstream that cannot work; warn that prompts leave.

    Never falls back to the local upstream: a remote run that silently went local (or the
    reverse) would make every latency and data-handling observation wrong."""
    set_llm_upstream(settings.llm_upstream)
    if settings.llm_upstream is not LlmUpstreamKind.REMOTE:
        return
    remote = snapshot.policy.upstreams.llm_remote
    if remote is None:  # require_upstream already refused this policy; kept for clarity
        msg = "ACL_LLM_UPSTREAM=remote, but the policy declares no upstreams.llm_remote"
        raise StartupError(msg)
    if not env.get(remote.api_key_env, "").strip():
        msg = f"ACL_LLM_UPSTREAM=remote, but {remote.api_key_env} is not set"
        raise StartupError(msg)
    logger.warning(
        "LLM upstream is REMOTE (%s): agent prompts and judge content leave this machine, after "
        "pre controls (pii redaction, secrets and prompt-injection blocks); routing terms sent "
        "with every request: %s",
        urlsplit(remote.base_url).hostname,
        json.dumps(dict(remote.extra_body), sort_keys=True),
    )


@dataclass(kw_only=True, eq=False)
class GatewayContainer:
    settings: Settings
    policy_store: PolicyStore
    feed_store: FeedStore
    verifier: TokenVerifier
    issuer: DemoTokenIssuer | None  # None when ACL_DEMO_TOKENS=0
    sessions: SessionStore
    state: StateStores  # session state and loop counters (Redis on `state`, or memory)
    gate: SessionGate
    pipeline: Pipeline
    llm: LLMProxy
    mcp: MCPProxy
    mcp_connector: MCPConnector
    audit: AuditLogger
    clock: Clock
    budgets: BudgetLedger
    judges: JudgeClient  # the one LLM judge client every semantic control shares
    oversight: Oversight  # approval queue and kill switch (gateway.approvals)
    operator_stores: OperatorStores
    _users: int = field(default=0, init=False)
    _watcher: asyncio.Task[None] | None = field(default=None, init=False)
    _feed_refresher: asyncio.Task[None] | None = field(default=None, init=False)
    _approval_sweeper: asyncio.Task[None] | None = field(default=None, init=False)

    @classmethod
    def from_settings(  # noqa: PLR0913 -- each keyword is a test seam for one dependency
        cls,
        settings: Settings,
        *,
        clock: Clock = utc_now,
        transport: httpx.AsyncBaseTransport | None = None,
        env: Mapping[str, str] | None = None,
        audit_stream: TextIO | None = None,
        classifier: InjectionClassifier | None = None,
        judge_factory: JudgeFactory = JudgeClient,
        resolver: HostResolver = system_resolver,
    ) -> Self:
        """Raises `PolicyLoadError` (no valid policy, no gateway), `FeedError` (the policy names
        a signature feed that cannot be loaded), `ModelVerificationError` (the pinned injection
        classifier is missing or altered; ``classifier`` injects one instead, for tests) or an
        identities file error. ``resolver`` resolves ``egress`` destinations (tests inject a
        fake, so they make no DNS queries)."""
        # The selected LLM upstream must be declared at startup and in every reloaded policy.
        policy_store = PolicyStore.from_path(
            settings.policy_path,
            PolicyLoader(requirement=require_upstream(settings.llm_upstream)),
        )
        check_llm_upstream(settings, policy_store.current, env if env is not None else os.environ)
        feed_store = FeedStore.boot(lambda: policy_store.current)
        # prompt_injection is always active (omitted = profile default), so no verified model
        # means no gateway; tool_poisoning shares the model, its cache and its worker bound.
        injection = ClassifierRunner(
            classifier
            if classifier is not None
            else load_classifier(
                settings.models_dir,
                enabled=settings.injection_classifier == "onnx",
                threads=settings.classifier_threads,
            ),
            workers=settings.classifier_workers,
        )
        signatures = SignaturesControl(lambda: feed_store.current)
        identities = DemoIdentities.load(settings.identities_path) if settings.demo_tokens else None
        # Zero series for every bounded label set (dashboards' increase()), again on reload.
        humans = identities.subjects() if identities is not None else frozenset[str]()
        kind = settings.llm_upstream
        initialize_metric_series(policy_store.current, humans, kind)
        policy_store.subscribe(
            lambda _old, current: initialize_metric_series(current, humans, kind)
        )
        verifier = TokenVerifier(settings.jwt_key, clock=clock)
        state = StateStores.from_settings(settings, clock=clock)
        sessions = state.sessions
        issuer = (
            DemoTokenIssuer(identities, settings.jwt_key, verifier, clock=clock)
            if identities is not None
            else None
        )
        gate = SessionGate(verifier, sessions)
        llm = LLMProxy(kind=settings.llm_upstream, env=env, transport=transport)
        # Judges call the LLM upstream directly (router key, response cap), never through the
        # pipeline: not audited as agent requests, not charged to budgets.
        # ``judge_factory`` builds the client: tests hand in a deterministic stand-in.
        judges = judge_factory(llm, lambda: policy_store.current)
        tool_poisoning = ToolPoisoningControl(injection, judges)  # judges its uncertain band
        audit = AuditLogger(
            stream=audit_stream,
            path=settings.audit_path,
            max_bytes=settings.audit_max_bytes,
            backups=settings.audit_backups,
        )
        # Every reload attempt lands in the audit stream: Grafana's policy-change annotations.
        policy_store.observe(
            lambda outcome: audit.write(
                PolicyReloadEvent(
                    ts=clock(),
                    result=outcome.result,
                    revision=outcome.revision,
                    previous_revision=outcome.previous_revision,
                )
            )
        )
        recorder = DecisionRecorder(
            audit,
            settings.internal_key_bytes,
            identities.subjects() if identities is not None else frozenset(),
            feed=lambda: feed_store.current,
            llm_upstream=settings.llm_upstream,
        )
        mcp_connector = MCPConnector(settings.internal_key_bytes, clock=clock, transport=transport)
        # sql_guard prices statements through the SQL server's gateway-only `explain` tool.
        sql_guard = SqlGuardControl(functools.partial(explain_cost, mcp_connector))
        pinning = ToolPinningControl(
            PinStore(settings.pins_dir), state.tool_quarantine, clock=clock
        )
        budgets = BudgetLedger(
            budget_store_from_settings(settings, clock=clock),
            clock=clock,
            llm_upstream=settings.llm_upstream,
        )
        operator_stores = operator_stores_from_settings(settings)
        oversight = Oversight(
            ApprovalService(
                operator_stores.approvals, key=settings.internal_key_bytes, clock=clock
            ),
            KillSwitch(
                operator_stores.kills,
                on_change=lambda killed: set_killed_agents(
                    killed, policy_store.current.policy.agents
                ),
            ),
        )
        # MCP has no static route: each tools/call binds its server's adapter and the caller's
        # own upstream session (MCPProxy passes the route to Pipeline.handle).
        pipeline = Pipeline(
            gate,
            {Channel.LLM: ChannelRoute(adapter=LLMAdapter(), upstream=llm)},
            ControlRegistry(  # stage 5-10 controls register here
                [
                    sql_guard,  # sealing: runs last, on the final (redacted) SQL that executes
                    pinning,  # MCP tools must match their approved baseline (pins/)
                    # http-adapter destinations: public, allowed (demo hosts: overlay only)
                    EgressControl(resolver, demo_hosts=settings.demo_host_bindings()),
                    SecretsControl(),
                    PiiControl(),  # builds the shared Presidio analyzer once per process
                    signatures,
                    ModelAllowlistControl(PolicyEvaluator()),
                    LoopDetectControl(state.calls, clock=clock),
                    IntentJudgeControl(judges),  # advisory: flags tool calls, never holds
                    OutputPolicyControl(judges, PolicyEvaluator(), clock=clock),
                    PromptInjectionControl(injection, judges),  # judge only on doubt
                    tool_poisoning,  # the called tool's definition, at tools/call
                ]
            ),
            recorder,
            oversight=oversight,
            clock=clock,
            budgets=budgets,
        )
        mcp = MCPProxy(
            gate=gate,
            pipeline=pipeline,
            registry=MCPSessionRegistry(mcp_connector),
            pinning=pinning,
            allowed_origins=frozenset(settings.mcp_allowed_origins),
            clock=clock,
            tool_screens=(signatures_screen(signatures), tool_poisoning.screen_listing),
        )
        return cls(
            settings=settings,
            policy_store=policy_store,
            feed_store=feed_store,
            verifier=verifier,
            issuer=issuer,
            sessions=sessions,
            state=state,
            gate=gate,
            pipeline=pipeline,
            llm=llm,
            mcp=mcp,
            mcp_connector=mcp_connector,
            audit=audit,
            clock=clock,
            budgets=budgets,
            judges=judges,
            oversight=oversight,
            operator_stores=operator_stores,
        )

    @contextlib.asynccontextmanager
    async def running(self) -> AsyncGenerator[Self]:
        """Start shared resources on first entry and stop them on last exit.

        Each listener's lifespan enters this, so two servers in one process share one start.
        """
        if self._users == 0:
            await self.llm.start()
            await self.mcp_connector.start()
            if self.settings.policy_watch:
                self._watcher = asyncio.create_task(self.policy_store.watch())
            self._feed_refresher = asyncio.create_task(self.feed_store.run())
            self._approval_sweeper = asyncio.create_task(
                sweep_forever(self.oversight, lambda: self.policy_store.current)
            )
        self._users += 1
        try:
            yield self
        finally:
            self._users -= 1
            if self._users == 0:
                await self._stop()

    async def _stop(self) -> None:
        for task in (self._watcher, self._feed_refresher, self._approval_sweeper):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._watcher = self._feed_refresher = self._approval_sweeper = None
        await self.mcp.aclose()  # ends every upstream MCP session
        await self.mcp_connector.aclose()
        await self.llm.aclose()
        await self.budgets.aclose()  # after in-flight settlements land
        await self.operator_stores.aclose()
        await self.state.aclose()
