"""Composition root: builds every long-lived component once and owns their lifecycle.

Both listeners (agent and operator) share one container, so they see the same policy
store, session state and upstream connection pool.
"""

import asyncio
import contextlib
import functools
import logging
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import Self, TextIO

import httpx

from gateway.adapters.llm import LLMAdapter
from gateway.budget.factory import budget_store_from_settings
from gateway.budget.ledger import BudgetLedger
from gateway.clock import Clock, utc_now
from gateway.controls.loop_detect import InMemoryCallCounter, LoopDetectControl
from gateway.controls.model_allowlist import ModelAllowlistControl
from gateway.controls.pii import PiiControl
from gateway.controls.registry import ControlRegistry
from gateway.controls.secrets import SecretsControl
from gateway.controls.signatures import SignaturesControl
from gateway.controls.sql_guard import SqlGuardControl
from gateway.core.types import Channel
from gateway.feed.store import FeedStore
from gateway.identity import DemoIdentities, DemoTokenIssuer, TokenVerifier
from gateway.pipeline import ChannelRoute, DecisionRecorder, Pipeline, SessionGate
from gateway.policy.evaluator import PolicyEvaluator
from gateway.policy.store import PolicyStore
from gateway.proxies.llm import LLMProxy
from gateway.proxies.mcp.downstream import MCPProxy
from gateway.proxies.mcp.explain import explain_cost
from gateway.proxies.mcp.pins import PinnedSchemas
from gateway.proxies.mcp.sessions import MCPSessionRegistry
from gateway.proxies.mcp.upstream import MCPConnector
from gateway.sessions import InMemorySessionStore, SessionStore
from gateway.settings import Settings
from gateway.telemetry import AuditLogger

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, eq=False)
class GatewayContainer:
    settings: Settings
    policy_store: PolicyStore
    feed_store: FeedStore
    verifier: TokenVerifier
    issuer: DemoTokenIssuer | None  # None when ACL_DEMO_TOKENS=0
    sessions: SessionStore
    gate: SessionGate
    pipeline: Pipeline
    llm: LLMProxy
    mcp: MCPProxy
    mcp_connector: MCPConnector
    audit: AuditLogger
    clock: Clock
    budgets: BudgetLedger
    _users: int = field(default=0, init=False)
    _watcher: asyncio.Task[None] | None = field(default=None, init=False)
    _feed_refresher: asyncio.Task[None] | None = field(default=None, init=False)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        clock: Clock = utc_now,
        transport: httpx.AsyncBaseTransport | None = None,
        env: Mapping[str, str] | None = None,
        audit_stream: TextIO | None = None,
    ) -> Self:
        """Raises `PolicyLoadError` (no valid policy, no gateway), `FeedError` (the policy names
        a signature feed that cannot be loaded) or an identities file error."""
        policy_store = PolicyStore.from_path(settings.policy_path)
        feed_store = FeedStore.boot(lambda: policy_store.current)
        signatures = SignaturesControl(lambda: feed_store.current)
        identities = DemoIdentities.load(settings.identities_path) if settings.demo_tokens else None
        verifier = TokenVerifier(settings.jwt_key, clock=clock)
        sessions = InMemorySessionStore(clock=clock)
        issuer = (
            DemoTokenIssuer(
                identities, settings.jwt_key, verifier, sessions.is_retired, clock=clock
            )
            if identities is not None
            else None
        )
        gate = SessionGate(verifier, sessions)
        llm = LLMProxy(env=env, transport=transport)
        audit = AuditLogger(stream=audit_stream, path=settings.audit_path)
        recorder = DecisionRecorder(
            audit,
            settings.internal_key_bytes,
            identities.subjects() if identities is not None else frozenset(),
            feed_version=lambda: feed_store.version,
        )
        mcp_connector = MCPConnector(settings.internal_key_bytes, clock=clock, transport=transport)
        # sql_guard prices statements through the SQL server's gateway-only `explain` tool.
        sql_guard = SqlGuardControl(
            functools.partial(explain_cost, mcp_connector, lambda: policy_store.current)
        )
        budgets = BudgetLedger(budget_store_from_settings(settings, clock=clock), clock=clock)
        # MCP has no static route: each tools/call binds its server's adapter and the caller's
        # own upstream session (MCPProxy passes the route to Pipeline.handle).
        pipeline = Pipeline(
            gate,
            {Channel.LLM: ChannelRoute(adapter=LLMAdapter(), upstream=llm)},
            ControlRegistry(  # stage 5-10 controls register here
                [
                    sql_guard,  # first: later controls see (and redact) the SQL that executes
                    SecretsControl(),
                    PiiControl(),  # builds the shared Presidio analyzer once per process
                    signatures,
                    ModelAllowlistControl(PolicyEvaluator()),
                    LoopDetectControl(InMemoryCallCounter(), clock=clock),
                ]
            ),
            recorder,
            clock=clock,
            budgets=budgets,
        )
        mcp = MCPProxy(
            gate=gate,
            pipeline=pipeline,
            registry=MCPSessionRegistry(mcp_connector),
            pins=PinnedSchemas(settings.pins_dir),
            allowed_origins=frozenset(settings.mcp_allowed_origins),
            clock=clock,
            tool_screen=signatures.screen_listing,
        )
        return cls(
            settings=settings,
            policy_store=policy_store,
            feed_store=feed_store,
            verifier=verifier,
            issuer=issuer,
            sessions=sessions,
            gate=gate,
            pipeline=pipeline,
            llm=llm,
            mcp=mcp,
            mcp_connector=mcp_connector,
            audit=audit,
            clock=clock,
            budgets=budgets,
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
        self._users += 1
        try:
            yield self
        finally:
            self._users -= 1
            if self._users == 0:
                await self._stop()

    async def _stop(self) -> None:
        for task in (self._watcher, self._feed_refresher):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._watcher = self._feed_refresher = None
        await self.mcp.aclose()  # ends every upstream MCP session
        await self.mcp_connector.aclose()
        await self.llm.aclose()
        await self.budgets.aclose()  # after in-flight settlements land
