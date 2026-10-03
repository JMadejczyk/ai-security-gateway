"""Prometheus metrics and the audit log. Metrics are served on the operator listener only.

A module-level registry instead of the global default one, so tests can inspect it and
nothing registered by a library leaks into `/metrics`.

Audit entries (SPEC "Audit, metrics and Grafana" → "Audit entry") are JSON lines on stdout
(→ Loki) and, when ``ACL_AUDIT_PATH`` is set, in a JSONL export file. They carry reason codes
and metadata only: payloads, messages, SQL text, tool arguments and upstream errors are never
written. A keyed HMAC of the payload lets an operator match an entry to a known payload.
"""

import hashlib
import hmac
import json
import logging
import sys
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, TextIO

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

    @field_serializer("ts")
    def _utc_z(self, ts: datetime) -> str:
        return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class AuditLogger:
    """Writes each entry as one JSON line to stdout and, optionally, a JSONL export file."""

    def __init__(self, *, stream: TextIO | None = None, path: Path | None = None) -> None:
        # A private logger, not registered globally: handlers never leak between instances.
        self._logger = logging.Logger("gateway.audit", logging.INFO)
        formatter = logging.Formatter("%(message)s")
        sinks: list[logging.Handler] = [logging.StreamHandler(stream or sys.stdout)]
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            sinks.append(logging.FileHandler(path, encoding="utf-8"))
        for sink in sinks:
            sink.setFormatter(formatter)
            self._logger.addHandler(sink)

    def write(self, entry: AuditEntry) -> None:
        self._logger.info(entry.model_dump_json())

    def close(self) -> None:
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)
