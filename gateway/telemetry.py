"""Prometheus metrics. Served on the operator listener only (`/metrics`).

A module-level registry instead of the global default one, so tests can inspect it and
nothing registered by a library leaks into `/metrics`.
"""

from enum import StrEnum

from prometheus_client import CollectorRegistry, Counter, Gauge

REGISTRY = CollectorRegistry(auto_describe=True)


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


def record_policy_reload(result: ReloadResult) -> None:
    POLICY_RELOADS.labels(result=result.value).inc()


def set_active_policy_revision(revision: str) -> None:
    """Only the active revision carries the info series."""
    POLICY_INFO.clear()
    POLICY_INFO.labels(revision=revision).set(1)
