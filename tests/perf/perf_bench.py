"""Scenarios and the runner of the perf layer: the real agent app, instant upstreams, metrics.

`PerfGateway` builds both apps over one container (`gateway_testkit.running_gateway`) for one
`Configuration`, behind `InstantUpstreams`. A `Scenario` opens what it needs (a token, an MCP
session) and returns a `Call`: one request per iteration, with new text each time, returning
the gateway's decision. `measure` runs a scenario sequentially and turns each call into a
`LatencySample` from four sources: the client clock, the upstream fakes' busy time, the
``acl_overhead_seconds`` / ``acl_control_latency_seconds`` histograms (per-call deltas of their
sums: calls are sequential, so a delta is exactly that call), and the call's audit entry.
`measure_concurrent` runs many sessions at once for throughput.
"""

import asyncio
import gc
import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any, ClassVar, Final

import httpx
import yaml
from gateway_testkit import ROOT_POLICY, Harness, bearer, running_gateway
from injection_kit import MarkerClassifier
from judge_kit import FakeJudgeClient
from perf_report import (
    ConcurrencyResult,
    Configuration,
    LatencySample,
    LatencyStats,
    ScenarioResult,
    is_deterministic,
)
from perf_upstreams import InstantUpstreams, capture_pins, save_pins

from gateway.core.catalog import CONTROL_CATALOG
from gateway.core.types import Channel, ControlKind, Decision
from gateway.injection.classifier import InjectionClassifier
from gateway.policy.loader import PolicyLoader
from gateway.proxies.mcp import wire
from gateway.telemetry import CONTROL_LATENCY, OVERHEAD, ReloadResult

ANALYST: Final = "anna@demo"
MODEL: Final = "qwen3:8b"
UNLIMITED: Final = 1_000_000_000
SEMANTIC_CONTROLS: Final = tuple(
    spec.id for spec in CONTROL_CATALOG.values() if spec.kind is ControlKind.SEMANTIC
)
MCP_HEADERS: Final = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}

# Plain business prose: nothing a pii, secrets or signatures pattern matches, nothing an
# injection classifier should flag. Cycled to the requested size.
_SENTENCES: Final = (
    "The regional team reviewed the quarterly order volume and found steady growth.",
    "Most customers renewed their contracts after the spring pricing update.",
    "Shipping delays in the northern warehouse were resolved within two weeks.",
    "The analysts compared returns across product lines to find the weakest category.",
    "Average basket size rose slightly while the number of new accounts stayed flat.",
    "Our support desk closed more tickets this month than in any month last year.",
    "Marketing asked for a short summary of which campaigns brought repeat buyers.",
    "The finance group wants the report to separate online and in-store revenue.",
    "Inventory turnover improved once slow items were moved to the outlet channel.",
    "Please keep the summary brief and focus on trends rather than single orders.",
    "The board meeting next week will discuss expansion into two new regions.",
    "Customer satisfaction scores were highest for the express delivery option.",
)


def prose(size: int, nonce: int, tag: str) -> str:
    """At least ``size`` bytes of benign English prose, unique per ``tag`` and ``nonce``: the
    classifier's score cache must never answer for a measured call."""
    parts = [f"Request {tag} number {nonce} for the sales summary."]
    total = len(parts[0])
    i = nonce
    while total < size:
        sentence = _SENTENCES[i % len(_SENTENCES)]
        parts.append(sentence)
        total += len(sentence) + 1
        i += 1
    return " ".join(parts)


def html_page(size: int, nonce: int) -> str:
    """An untrusted web page of about ``size`` bytes: markup, a style block, prose paragraphs."""
    head = (
        "<!doctype html><html><head><title>Market outlook</title>"
        "<style>body{font-family:serif;margin:2em} .note{color:#555}</style></head><body>"
        f"<h1>Market outlook, edition {nonce}</h1>"
    )
    paragraphs: list[str] = []
    total = len(head)
    i = nonce
    while total < size:
        paragraph = f"<p class='note'>{prose(400, i, f'page {nonce}')}</p>"
        paragraphs.append(paragraph)
        total += len(paragraph)
        i += 1
    return head + "".join(paragraphs) + "</body></html>"


type Call = Callable[[int], Awaitable[str]]


@dataclass
class PerfGateway:
    """The gateway under measurement for one configuration, and where its numbers come from."""

    harness: Harness
    upstreams: InstantUpstreams
    configuration: Configuration

    @property
    def agent(self) -> httpx.AsyncClient:
        return self.harness.agent

    async def token(self) -> str:
        return await self.harness.token(ANALYST)

    def drain_audit(self) -> list[dict[str, Any]]:
        """Audit entries written since the last drain (the buffer is emptied: long runs)."""
        buffer = self.harness.audit
        lines = buffer.getvalue().splitlines()
        buffer.seek(0)
        buffer.truncate()
        return [json.loads(line) for line in lines if line]


def perf_policy(path: Path) -> None:
    """Raise every budget in the policy copy: the budget control still reserves and settles
    on each call, but a run of hundreds of calls in one session is not cut off."""
    document = yaml.safe_load(path.read_text())
    document["budgets"] = {
        "per_user": {"daily_tokens": UNLIMITED, "daily_cost_usd": float(UNLIMITED)},
        "per_agent": {"daily_tokens": UNLIMITED},
        "per_session": {"tool_calls": UNLIMITED, "gpu_seconds": UNLIMITED},
        "soft_limit_pct": 80,
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False))


@asynccontextmanager
async def perf_gateway(
    tmp_path: Path,
    configuration: Configuration,
    classifier: InjectionClassifier | None = None,
) -> AsyncIterator[PerfGateway]:
    """``classifier`` is the real model for `Configuration.REAL_MODEL`; the marker classifier
    otherwise. Semantic controls are removed from the registry for `DETERMINISTIC`."""
    workdir = tmp_path / configuration.value
    workdir.mkdir()
    upstreams = InstantUpstreams()
    save_pins(workdir / "pins", await capture_pins(upstreams, PolicyLoader().load(ROOT_POLICY)))
    async with running_gateway(
        workdir,
        transport=upstreams,
        classifier=classifier if classifier is not None else MarkerClassifier(),
        judge_factory=FakeJudgeClient,
    ) as harness:
        perf_policy(harness.policy_path)
        outcome = harness.container.policy_store.reload()
        assert outcome.result is ReloadResult.OK, outcome
        if configuration is Configuration.DETERMINISTIC:
            for control_id in SEMANTIC_CONTROLS:
                harness.container.pipeline.controls.remove(control_id)
        harness.audit.seek(0)
        harness.audit.truncate()
        yield PerfGateway(harness, upstreams, configuration)


# ------------------------------------------------------------------------------ scenarios


class Scenario(ABC):
    """One kind of agent request, measured in every configuration."""

    name: str
    description: str
    channel: ClassVar[Channel]
    # Bytes of new text the classifier sees per call: the real model's iterations shrink
    # with it (it runs at a few KB per second).
    classified_bytes: int

    @abstractmethod
    async def open(self, gateway: PerfGateway) -> Call:
        """Set up a fresh session; the returned call sends request ``i``."""


@dataclass(frozen=True)
class ChatScenario(Scenario):
    """``POST /v1/chat/completions`` with a user prompt of ``classified_bytes``."""

    name: str
    description: str
    classified_bytes: int
    stream: bool = False
    channel: ClassVar[Channel] = Channel.LLM

    @classmethod
    def sized(cls, name: str, prompt_bytes: int, *, stream: bool = False) -> "ChatScenario":
        mode = "stream=true, re-emitted as SSE" if stream else "non-streaming"
        description = f"LLM chat, ~{prompt_bytes} B prompt, {mode}"
        return cls(name, description, prompt_bytes, stream)

    async def open(self, gateway: PerfGateway) -> Call:
        headers = bearer(await gateway.token())

        async def call(i: int) -> str:
            body: dict[str, Any] = {
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": prose(self.classified_bytes, i, self.name)}
                ],
            }
            if self.stream:
                body["stream"] = True
            response = await gateway.agent.post("/v1/chat/completions", json=body, headers=headers)
            await response.aread()
            if response.status_code != httpx.codes.OK:
                return f"http_{response.status_code}"
            if self.stream:
                assert response.text.rstrip().endswith("data: [DONE]"), response.text[-200:]
            return Decision.ALLOW.value

        return call


@dataclass
class _MCPSession:
    gateway: PerfGateway
    server: str
    headers: dict[str, str]
    ids: Iterator[int] = field(default_factory=lambda: count(1))

    @classmethod
    async def open(cls, gateway: PerfGateway, server: str) -> "_MCPSession":
        session = cls(gateway, server, MCP_HEADERS | bearer(await gateway.token()))
        init = await session.send(
            "initialize",
            {
                "protocolVersion": wire.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "perf", "version": "1"},
            },
        )
        assert init.status_code == httpx.codes.OK, init.text
        session.headers |= {
            wire.SESSION_HEADER: init.headers[wire.SESSION_HEADER],
            wire.PROTOCOL_HEADER: wire.PROTOCOL_VERSION,
        }
        notified = await gateway.agent.post(
            f"/mcp/{server}",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=session.headers,
        )
        assert notified.status_code == httpx.codes.ACCEPTED, notified.text
        return session

    async def send(self, method: str, params: dict[str, Any] | None = None) -> httpx.Response:
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": next(self.ids), "method": method}
        if params is not None:
            message["params"] = params
        return await self.gateway.agent.post(
            f"/mcp/{self.server}", json=message, headers=self.headers
        )


@dataclass(frozen=True)
class ToolsListScenario(Scenario):
    name: str
    description: str
    server: str
    classified_bytes: int = 0
    channel: ClassVar[Channel] = Channel.MCP

    async def open(self, gateway: PerfGateway) -> Call:
        session = await _MCPSession.open(gateway, self.server)

        async def call(_: int) -> str:
            response = await session.send("tools/list")
            if response.status_code != httpx.codes.OK or "result" not in response.json():
                return f"http_{response.status_code}"
            return Decision.ALLOW.value

        return call


@dataclass(frozen=True)
class ToolCallScenario(Scenario):
    """``tools/call`` of ``tool`` on ``server``; ``arguments(i)`` varies per call. With
    ``page_bytes``, the upstream's ``fetch`` answers an HTML page of that size, new per call."""

    name: str
    description: str
    server: str
    tool: str
    arguments: Callable[[int], dict[str, Any]]
    classified_bytes: int
    page_bytes: int = 0
    channel: ClassVar[Channel] = Channel.MCP

    async def open(self, gateway: PerfGateway) -> Call:
        session = await _MCPSession.open(gateway, self.server)
        if self.page_bytes:
            size = self.page_bytes
            gateway.upstreams.fetch_page = lambda url: html_page(size, int(url.rsplit("/", 1)[1]))

        async def call(i: int) -> str:
            response = await session.send(
                "tools/call", {"name": self.tool, "arguments": self.arguments(i)}
            )
            if response.status_code != httpx.codes.OK:
                return f"http_{response.status_code}"
            if response.json().get("result", {}).get("isError"):
                return "tool_error"
            return Decision.ALLOW.value

        return call


def scenarios() -> list[Scenario]:
    return [
        ChatScenario.sized("llm_small", 200),
        ChatScenario.sized("llm_typical", 1024),
        ChatScenario.sized("llm_large", 20 * 1024),
        ChatScenario.sized("llm_typical_stream", 1024, stream=True),
        ToolsListScenario(
            "mcp_tools_list",
            "MCP tools/list on `reports` (pin check, mapping filter, listing screens)",
            server="reports",
        ),
        ToolCallScenario(
            "mcp_write_report",
            "MCP tools/call `reports.write_report`, ~1 KB report (fs adapter)",
            server="reports",
            tool="write_report",
            arguments=lambda i: {"name": f"perf-{i}.md", "content": prose(1024, i, "report")},
            classified_bytes=1024,
        ),
        ToolCallScenario(
            "mcp_fetch",
            "MCP tools/call `web.fetch` returning an untrusted ~10 KB HTML page",
            server="web",
            tool="fetch",
            arguments=lambda i: {"url": f"https://example.com/outlook/{i}"},
            classified_bytes=10 * 1024,
            page_bytes=10 * 1024,
        ),
        ToolCallScenario(
            "mcp_sql_query",
            "MCP tools/call `sales_db.query` through sql_guard (parse, forced LIMIT, `explain` "
            "priced on the instant SQL server)",
            server="sales_db",
            tool="query",
            # An integer literal in a test statement, sent to the gateway's sql_guard on purpose.
            arguments=lambda i: {"sql": f"SELECT id, amount FROM sales.orders WHERE id > {i}"},  # noqa: S608
            classified_bytes=64,
        ),
    ]


SCENARIOS: Final = {scenario.name: scenario for scenario in scenarios()}


# ------------------------------------------------------------------------------ measuring


def _histogram_sums() -> tuple[dict[str, float], dict[str, float]]:
    """Current ``_sum`` of acl_overhead_seconds per channel and acl_control_latency_seconds per
    control, in milliseconds."""
    overhead: dict[str, float] = {}
    for metric in OVERHEAD.collect():
        for sample in metric.samples:
            if sample.name.endswith("_sum"):
                overhead[sample.labels["channel"]] = sample.value * 1000
    controls: dict[str, float] = {}
    for metric in CONTROL_LATENCY.collect():
        for sample in metric.samples:
            if sample.name.endswith("_sum"):
                controls[sample.labels["control"]] = sample.value * 1000
    return overhead, controls


def _histogram_totals(channel: Channel) -> tuple[float, float]:
    """(sum in ms, count) of acl_overhead_seconds for ``channel``."""
    total, observed = 0.0, 0.0
    for metric in OVERHEAD.collect():
        for sample in metric.samples:
            if sample.labels.get("channel") != channel.value:
                continue
            if sample.name.endswith("_sum"):
                total = sample.value * 1000
            elif sample.name.endswith("_count"):
                observed = sample.value
    return total, observed


def _delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {k: v - before.get(k, 0.0) for k, v in after.items() if v - before.get(k, 0.0) > 0}


def _audit_entry(entries: list[dict[str, Any]], channel: Channel) -> dict[str, Any] | None:
    """The call's audit entry (its last one: a call writes one per interaction)."""
    return next((e for e in reversed(entries) if e.get("channel") == channel.value), None)


def _audit_numbers(entry: dict[str, Any] | None) -> tuple[float | None, float | None]:
    """(total - upstream, sum of deterministic ``latency_ms.controls``) from an audit entry."""
    if entry is None:
        return None, None
    latency = entry["latency_ms"]
    controls: dict[str, float] = latency["controls"]
    deterministic = sum(ms for c, ms in controls.items() if is_deterministic(c))
    return latency["total"] - (latency["upstream"] or 0.0), deterministic


async def measure(
    gateway: PerfGateway, scenario: Scenario, *, iterations: int, warmup: int
) -> ScenarioResult:
    call = await scenario.open(gateway)
    for i in range(warmup):
        await call(i)
    gateway.drain_audit()
    gc.collect()
    samples: list[LatencySample] = []
    for i in range(warmup, warmup + iterations):
        overhead_before, controls_before = _histogram_sums()
        busy_before = gateway.upstreams.busy_s
        started = time.perf_counter()
        decision = await call(i)
        e2e_ms = (time.perf_counter() - started) * 1000
        upstream_ms = (gateway.upstreams.busy_s - busy_before) * 1000
        overhead_after, controls_after = _histogram_sums()
        overhead = _delta(overhead_after, overhead_before).get(scenario.channel.value)
        audit_overhead, audit_deterministic = _audit_numbers(
            _audit_entry(gateway.drain_audit(), scenario.channel)
        )
        samples.append(
            LatencySample(
                e2e_ms=e2e_ms,
                upstream_ms=upstream_ms,
                decision=decision,
                overhead_ms=overhead,
                audit_overhead_ms=audit_overhead,
                audit_deterministic_ms=audit_deterministic,
                controls=_delta(controls_after, controls_before),
            )
        )
    return ScenarioResult.of(
        scenario=scenario.name,
        configuration=gateway.configuration,
        channel=scenario.channel.value,
        description=scenario.description,
        warmup=warmup,
        samples=samples,
    )


async def measure_concurrent(
    gateway: PerfGateway, scenario: Scenario, *, sessions: int, calls: int
) -> ConcurrencyResult:
    """``sessions`` sessions sending ``calls`` requests each, all at once."""
    opened = [await scenario.open(gateway) for _ in range(sessions)]
    for index, call in enumerate(opened):  # warm each session once
        await call(1_000_000 + index)
    gateway.drain_audit()
    latencies: list[float] = []
    decisions: dict[str, int] = {}

    async def run(index: int, call: Call) -> None:
        for n in range(calls):
            started = time.perf_counter()
            decision = await call(2_000_000 + index * calls + n)
            latencies.append((time.perf_counter() - started) * 1000)
            decisions[decision] = decisions.get(decision, 0) + 1

    overhead_before, count_before = _histogram_totals(scenario.channel)
    started = time.perf_counter()
    async with asyncio.TaskGroup() as group:
        for index, call in enumerate(opened):
            group.create_task(run(index, call))
    wall_s = time.perf_counter() - started
    overhead_after, count_after = _histogram_totals(scenario.channel)
    gateway.drain_audit()
    observed = count_after - count_before
    return ConcurrencyResult(
        scenario=scenario.name,
        configuration=gateway.configuration,
        sessions=sessions,
        calls_per_session=calls,
        wall_s=wall_s,
        throughput_rps=sessions * calls / wall_s,
        decisions=decisions,
        e2e=LatencyStats.of(latencies),
        mean_overhead_ms=(overhead_after - overhead_before) / observed if observed else 0.0,
    )
