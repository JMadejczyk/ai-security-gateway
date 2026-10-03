"""Prometheus metrics and the audit log. Metrics are served on the operator listener only.

A module-level registry instead of the global default one, so tests can inspect it and
nothing registered by a library leaks into `/metrics`.

Audit entries (SPEC "Audit, metrics and Grafana" → "Audit entry") are JSON lines on stdout
(→ Loki) and, when ``ACL_AUDIT_PATH`` is set, in a JSONL export file. They carry reason codes
and metadata only: payloads, messages, SQL text, tool arguments and upstream errors are never
written. A keyed HMAC of the payload lets an operator match an entry to a known payload.
The export is size-capped, never-renamed segments; Grafana Alloy tails them into Loki.
Policy reloads go to the same stream as `PolicyReloadEvent` lines
(``"event": "policy_reload"``) for dashboard annotations.
"""

import hashlib
import hmac
import json
import logging
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, TextIO

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from pydantic import AwareDatetime, Field, field_serializer

from gateway.core.envelope import FrozenModel, Verdict
from gateway.core.types import Action, Channel, Decision, SessionMode, Stage

REGISTRY = CollectorRegistry(auto_describe=True)
OTHER_LABEL: Final = "other"


class ReloadResult(StrEnum):
    OK = "ok"
    INVALID = "invalid"
    UNCHANGED = "unchanged"


POLICY_RELOADS = Counter(
    "acl_policy_reloads",
    "Policy reload attempts by result.",
    ["result"],
    registry=REGISTRY,
)
for _result in ReloadResult:  # every result at 0 (see `initialize_series`)
    POLICY_RELOADS.labels(result=_result.value)
POLICY_INFO = Gauge(
    "acl_policy_info",
    "Set to 1 for the active policy revision.",
    ["revision"],
    registry=REGISTRY,
)
REQUESTS = Counter(
    "acl_requests",
    "Calls decided by the gateway, by channel, final decision and agent.",
    ["channel", "decision", "agent"],
    registry=REGISTRY,
)
CONTROL_VERDICTS = Counter(
    "acl_control_verdicts",
    "Verdicts returned by each control (log_only ones included).",
    ["control", "decision"],
    registry=REGISTRY,
)
CONTROL_LATENCY = Histogram(
    "acl_control_latency_seconds",
    "Time spent in each control.",
    ["control"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY,
)
OVERHEAD = Histogram(
    "acl_overhead_seconds",
    "Gateway time per call, excluding the upstream.",
    ["channel"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)
TOKENS = Counter(
    "acl_tokens",
    "Tokens reported by the LLM upstream.",
    ["user", "agent", "model"],
    registry=REGISTRY,
)
TAINTED_SESSIONS = Gauge(
    "acl_tainted_sessions",
    "Live sessions whose context received untrusted content.",
    registry=REGISTRY,
)
THROTTLED = Counter(
    "acl_throttled",
    "Calls rejected by a throttle cap, by agent.",
    ["agent"],
    registry=REGISTRY,
)
ALERTS = Counter(
    "acl_alerts",
    "Risk-rule alerts raised, by rule (`<mode>.<index>` into risk_rules: bounded by the policy).",
    ["rule"],
    registry=REGISTRY,
)
SESSION_RISK = Histogram(
    "acl_session_risk",
    "Session risk after each call (per-session values live in the audit log, never labels).",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)


class FeedReloadResult(StrEnum):
    OK = "ok"
    UNCHANGED = "unchanged"
    INVALID = "invalid"  # read, but oversized, not JSON or failing the feed schema
    UNAVAILABLE = "unavailable"  # missing file, unreachable URL, HTTP error, redirect


FEED_RELOADS = Counter(
    "acl_feed_reloads",
    "Signature feed load attempts by result (the last valid feed stays in effect).",
    ["result"],
    registry=REGISTRY,
)
FEED_INFO = Gauge(
    "acl_feed_info",
    "Set to 1 for the active signature feed version.",
    ["version"],
    registry=REGISTRY,
)
SIGNATURE_HITS = Counter(
    "acl_signature_hits",
    "Signature matches by signature id (bounded by the feed: at most 1000 ids).",
    ["signature"],
    registry=REGISTRY,
)


SERVED_CHANNELS: Final = (Channel.LLM, Channel.MCP)


def initialize_series(
    *,
    agents: Iterable[str],
    controls: Iterable[str],
    spend: Iterable[tuple[str, str, str]] = (),
    alert_rules: Iterable[str] = (),
) -> None:
    """Create every bounded label set of the decision and spend counters at 0.

    Prometheus cannot see the increment that creates a series, so ``increase()`` over a
    dashboard's time range would miss each label set's first call (the one block of a short
    demo). ``spend`` is the ``(user, agent, model)`` combinations of ``acl_tokens_total`` and
    ``acl_cost_usd_total``; ``alert_rules`` the ``acl_alerts_total`` rule labels. Run at
    startup and after every policy reload (`gateway.metric_series`); idempotent.
    """
    for user, agent, model in spend:
        TOKENS.labels(user=user, agent=agent, model=model)
        COST.labels(user=user, agent=agent, model=model)
    for rule in alert_rules:
        ALERTS.labels(rule=rule)
    agent_labels = (*agents, OTHER_LABEL)
    for channel in SERVED_CHANNELS:
        for decision in Decision:
            for agent in agent_labels:
                REQUESTS.labels(channel=channel.value, decision=decision.value, agent=agent)
    for channel in SERVED_CHANNELS:  # histograms too: rate() over their first burst
        OVERHEAD.labels(channel=channel.value)
    for control in controls:
        CONTROL_LATENCY.labels(control=control)
        for decision in Decision:
            CONTROL_VERDICTS.labels(control=control, decision=decision.value)
    for agent in agent_labels:
        THROTTLED.labels(agent=agent)


def initialize_signature_series(signature_ids: Iterable[str]) -> None:
    """`acl_signature_hits_total` at 0 for every signature of a loaded feed (see above)."""
    for signature_id in signature_ids:
        SIGNATURE_HITS.labels(signature=signature_id)


def record_policy_reload(result: ReloadResult) -> None:
    POLICY_RELOADS.labels(result=result.value).inc()


def set_active_policy_revision(revision: str) -> None:
    """Only the active revision carries the info series."""
    POLICY_INFO.clear()
    POLICY_INFO.labels(revision=revision).set(1)


def bounded(value: str | None, known: Iterable[str]) -> str:
    """A label value from a known identity set, or ``other``: label cardinality stays bounded."""
    return value if value is not None and value in known else OTHER_LABEL


def record_request(channel: Channel, decision: Decision, agent: str) -> None:
    REQUESTS.labels(channel=channel.value, decision=decision.value, agent=agent).inc()


def record_verdicts(verdicts: Iterable[Verdict]) -> None:
    for verdict in verdicts:
        CONTROL_VERDICTS.labels(control=verdict.control_id, decision=verdict.decision.value).inc()
        CONTROL_LATENCY.labels(control=verdict.control_id).observe(verdict.latency_ms / 1000)


def record_overhead(channel: Channel, seconds: float) -> None:
    OVERHEAD.labels(channel=channel.value).observe(max(seconds, 0.0))


def record_tokens(user: str, agent: str, model: str, tokens: int) -> None:
    if tokens > 0:
        TOKENS.labels(user=user, agent=agent, model=model).inc(tokens)


def record_throttled(agent: str) -> None:
    THROTTLED.labels(agent=agent).inc()


def record_alert(rule: str) -> None:
    ALERTS.labels(rule=rule).inc()


def record_session(risk: float, tainted_sessions: int) -> None:
    SESSION_RISK.observe(risk)
    set_tainted_sessions(tainted_sessions)


def set_tainted_sessions(count: int) -> None:
    TAINTED_SESSIONS.set(count)


def record_feed_reload(result: FeedReloadResult) -> None:
    FEED_RELOADS.labels(result=result.value).inc()


def set_active_feed_version(version: str | None) -> None:
    """Only the active feed version carries the info series; none while no feed is loaded."""
    FEED_INFO.clear()
    if version is not None:
        FEED_INFO.labels(version=version).set(1)


def record_signature_hits(signature_ids: Iterable[str]) -> None:
    for signature_id in signature_ids:
        SIGNATURE_HITS.labels(signature=signature_id).inc()


COST = Counter(
    "acl_cost_usd",
    "Spend in USD from the policy pricing table (GPU part: upstream wall time, an estimate).",
    ["user", "agent", "model"],
    registry=REGISTRY,
)
BUDGET_USAGE = Gauge(
    "acl_budget_usage_ratio",
    "Usage over the hard limit, per limit (`per_user.daily_tokens`, ...) and user or agent.",
    ["scope", "id"],
    registry=REGISTRY,
)
BUDGET_STORE_ERRORS = Counter(
    "acl_budget_store_errors",
    "Budget store operations that failed (the call failed closed, or a settlement was lost).",
    ["operation"],
    registry=REGISTRY,
)
BUDGET_STORE_UP = Gauge(
    "acl_budget_store_up",
    "1 while the budget store answers, 0 after a failed operation or health check.",
    registry=REGISTRY,
)


def record_cost(user: str, agent: str, model: str, usd: float) -> None:
    if usd > 0:
        COST.labels(user=user, agent=agent, model=model).inc(usd)


def set_budget_usage(scope: str, subject: str, ratio: float) -> None:
    BUDGET_USAGE.labels(scope=scope, id=subject).set(ratio)


def record_budget_store_error(operation: str) -> None:
    BUDGET_STORE_ERRORS.labels(operation=operation).inc()


def set_budget_store_up(*, up: bool) -> None:
    BUDGET_STORE_UP.set(1 if up else 0)


# --------------------------------------------------------------------------------- audit


def canonical_json(value: object) -> bytes:
    """Sorted-key, compact JSON: the form payload HMACs and argument digests are taken over."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def payload_hmac(key: bytes, payload: object) -> str:
    return hmac.new(key, canonical_json(payload), hashlib.sha256).hexdigest()


class AuditVerdict(FrozenModel):
    control: str
    stage: Stage
    decision: Decision
    enforced: bool
    reason_code: str

    @classmethod
    def of(cls, verdict: Verdict, stage: Stage) -> "AuditVerdict":
        return cls(
            control=verdict.control_id,
            stage=stage,
            decision=verdict.decision,
            enforced=verdict.enforced,
            reason_code=verdict.reason_code,
        )


class AuditLatency(FrozenModel):
    total: float = Field(ge=0.0)
    upstream: float | None = Field(default=None, ge=0.0)
    controls: dict[str, float] = Field(default_factory=dict[str, float])


class AuditEntry(FrozenModel):
    """One decision. Identity fields are None when the call failed authentication."""

    ts: AwareDatetime
    session_id: str | None = None
    principal: str | None = None
    actor: str | None = None
    mode: SessionMode | None = None
    channel: Channel
    action: Action | None = None
    resource: str | None = None
    decision: Decision
    reason_code: str
    status: int
    verdicts: tuple[AuditVerdict, ...] = ()
    effective_scope: tuple[str, ...] = ()
    risk: float | None = None
    taint: bool | None = None
    policy_revision: str
    feed_version: str | None = None  # the signature feed arrives with the signatures control
    latency_ms: AuditLatency
    payload_hmac: str | None = None
    approval_id: str | None = None  # the approval the call was held under or presented

    @field_serializer("ts")
    def _utc_z(self, ts: datetime) -> str:
        return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class PolicyReloadEvent(FrozenModel):
    """One policy reload attempt, written to the audit stream next to the decisions.

    Grafana draws these as annotations (Loki ``{job="acl", event="policy_reload"}``), so a
    dashboard shows when a new revision took effect. The loader's error text stays in the
    gateway log: the audit stream carries no free text.
    """

    ts: AwareDatetime
    event: Literal["policy_reload"] = "policy_reload"
    result: ReloadResult
    revision: str  # in effect after the attempt
    previous_revision: str

    @field_serializer("ts")
    def _utc_z(self, ts: datetime) -> str:
        return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")


AUDIT_MAX_BYTES: Final = 50 * 1024 * 1024
AUDIT_BACKUPS: Final = 4


class SegmentedFileHandler(logging.Handler):
    """Append-only JSONL segments that are never renamed: ``<stem>-<UTC time>-<n><suffix>``.

    A segment is created exclusively (never reopened, never overwritten) and written until it
    would pass ``max_bytes``; then the next one starts and all but the newest ``backups + 1``
    are deleted, so the volume holds about ``max_bytes * (backups + 1)``. Names sort
    chronologically. Because no file is ever renamed, a tailer that keys its read offset by
    path (Grafana Alloy) can be down across any number of rollovers and resumes every segment
    at the right offset; a rename-based rotation would leave it with a stale offset into a
    new file at the old path.
    """

    def __init__(self, base: Path, *, max_bytes: int, backups: int) -> None:
        super().__init__()
        self._dir = base.parent
        self._stem, self._suffix = base.stem, base.suffix
        self._max_bytes = max_bytes
        self._backups = backups
        self._stream: TextIO | None = None
        self._size = 0
        self._sequence = 0

    @property
    def pattern(self) -> str:
        """Glob that matches every segment (what the log shipper tails)."""
        return f"{self._stem}-*{self._suffix}"

    def segments(self) -> list[Path]:
        return sorted(self._dir.glob(self.pattern))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record) + "\n"
            size = len(line.encode("utf-8"))
            stream = self._stream
            if stream is None or (self._size and self._size + size > self._max_bytes):
                stream = self._roll()
            stream.write(line)
            stream.flush()
            self._size += size
        except Exception:  # logging's contract: report, never raise into the caller
            self.handleError(record)

    def close(self) -> None:
        self.acquire()
        try:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        finally:
            self.release()
        super().close()

    def _roll(self) -> TextIO:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        stream: TextIO | None = None
        while stream is None:
            self._sequence += 1
            name = f"{self._stem}-{stamp}-{self._sequence:06d}{self._suffix}"
            try:
                stream = (self._dir / name).open("x", encoding="utf-8")
            except FileExistsError:
                continue
        self._stream, self._size = stream, 0
        for stale in self.segments()[: -(self._backups + 1)]:
            stale.unlink(missing_ok=True)
        return stream


class AuditLogger:
    """Writes each entry as one JSON line to stdout and, optionally, a JSONL export.

    The export is `SegmentedFileHandler` segments next to ``path``: ``/var/log/acl/audit.jsonl``
    gives ``/var/log/acl/audit-<UTC time>-<n>.jsonl``, a new segment every ``max_bytes``,
    ``backups`` old ones kept.
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        path: Path | None = None,
        max_bytes: int = AUDIT_MAX_BYTES,
        backups: int = AUDIT_BACKUPS,
    ) -> None:
        # A private logger, not registered globally: handlers never leak between instances.
        self._logger = logging.Logger("gateway.audit", logging.INFO)
        formatter = logging.Formatter("%(message)s")
        sinks: list[logging.Handler] = [logging.StreamHandler(stream or sys.stdout)]
        if path is not None:
            sinks.append(SegmentedFileHandler(path, max_bytes=max_bytes, backups=backups))
        for sink in sinks:
            sink.setFormatter(formatter)
            self._logger.addHandler(sink)

    def write(self, entry: AuditEntry | PolicyReloadEvent) -> None:
        self._logger.info(entry.model_dump_json())

    def close(self) -> None:
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)
