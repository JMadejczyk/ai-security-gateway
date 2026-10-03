"""Builds the four Grafana dashboards (SPEC "Audit, metrics and Grafana") as JSON files.

The dashboards are written to ``grafana/dashboards/`` and committed; Grafana provisions them
from there (``grafana/provisioning/dashboards/dashboards.yaml``). A typed builder instead of
hand-written JSON keeps the panel queries readable in one place, keeps every panel on the
``$datasource`` / ``$loki`` template variables (no hard-coded data source uids), and lets a
test assert the committed files are exactly what this module produces.

    uv run python -m grafana.build_dashboards          # rewrite grafana/dashboards/*.json
    uv run python -m grafana.build_dashboards --check  # exit 1 if a committed file is stale

Metrics come from Prometheus (``acl_*``, bounded labels only). Anything per session or per
principal (risk, reason codes, the decision timeline) comes from the audit log in Loki, where
those values stay in the line (``| json``), never in metric labels.
"""

import argparse
import json
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

DASHBOARDS_DIR: Final = Path(__file__).resolve().parent / "dashboards"
# `acl_approvals_total` counts every state an approval reaches; the panel shows the request
# ("pending", the first state) and the outcomes, not the intermediate approved/executing steps.
APPROVAL_COUNTED: Final = ("pending", "succeeded", "denied", "expired", "failed", "uncertain")
APPROVAL_COLORS: Final = {
    "requested": "blue",
    "succeeded": "green",
    "denied": "red",
    "expired": "orange",
    "failed": "dark-red",
    "uncertain": "purple",
}
GRID_COLUMNS: Final = 24
TAG: Final = "acl"

type DatasourceType = Literal["prometheus", "loki"]
type PanelType = Literal["stat", "timeseries", "bargauge", "table", "logs"]
type Json = dict[str, JsonValue]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class DatasourceRef(_Model):
    type: DatasourceType
    uid: str


PROMETHEUS: Final = DatasourceRef(type="prometheus", uid="${datasource}")
LOKI: Final = DatasourceRef(type="loki", uid="${loki}")
_SESSION_FILTER: Final = '|= "$session_id"'  # cheap line filter before the json parser


class Query(_Model):
    """One panel target. ``instant`` queries evaluate once over the dashboard range."""

    datasource: DatasourceRef
    expr: str
    legend: str | None = None
    instant: bool = False
    table: bool = False  # Prometheus ``format: table``

    def target(self, ref_id: str, *, force_range: bool = False) -> Json:
        body: Json = {
            "refId": ref_id,
            "datasource": self.datasource.model_dump(),
            "expr": self.expr,
            "editorMode": "code",
        }
        if self.legend is not None:
            body["legendFormat"] = self.legend
        if self.datasource.type == "loki":
            body["queryType"] = "instant" if self.instant and not force_range else "range"
        else:
            body["instant"] = self.instant
            body["range"] = not self.instant
            body["format"] = "table" if self.table else "time_series"
        return body


def prom(expr: str, legend: str | None = None, *, instant: bool = False) -> Query:
    return Query(datasource=PROMETHEUS, expr=expr, legend=legend, instant=instant)


def loki(expr: str, legend: str | None = None, *, instant: bool = False) -> Query:
    return Query(datasource=LOKI, expr=expr, legend=legend, instant=instant)


class Threshold(_Model):
    color: str
    value: float | None = None  # None: the base step


GREEN_AMBER_RED: Final = (
    Threshold(color="green"),
    Threshold(color="orange", value=0.5),
    Threshold(color="red", value=0.8),
)
BUDGET_STEPS: Final = (
    Threshold(color="green"),
    Threshold(color="orange", value=0.8),  # policy soft_limit_pct
    Threshold(color="red", value=1.0),
)
NEUTRAL: Final = (Threshold(color="blue"),)


class Panel(_Model):
    """A panel and its size; `Layout` gives it a position and an id."""

    type: PanelType
    title: str
    description: str
    queries: tuple[Query, ...]
    width: int = Field(ge=1, le=GRID_COLUMNS)
    height: int = Field(default=8, ge=2)
    unit: str | None = None
    decimals: int | None = None
    minimum: float | None = None
    maximum: float | None = None
    thresholds: tuple[Threshold, ...] = NEUTRAL
    color_mode: Literal["thresholds", "palette-classic"] = "palette-classic"
    stack: bool = False
    bars: bool = False
    options: Json = Field(default_factory=dict[str, JsonValue])
    transformations: tuple[Json, ...] = ()
    overrides: tuple[Json, ...] = ()
    links: tuple[Json, ...] = ()
    mappings: tuple[Json, ...] = ()
    no_value: str | None = None
    time_from: str | None = None  # relative-time override, e.g. "7d"
    interval: str | None = None  # minimum query step

    @property
    def datasource(self) -> DatasourceRef:
        if len({query.datasource.uid for query in self.queries}) != 1:
            msg = f"panel {self.title!r} needs exactly one data source"
            raise ValueError(msg)
        return self.queries[0].datasource

    def render(self, panel_id: int, x: int, y: int) -> Json:
        defaults: Json = {
            "color": {"mode": self.color_mode},
            "thresholds": {
                "mode": "absolute",
                "steps": [t.model_dump() for t in self.thresholds],
            },
            "mappings": list(self.mappings),
            "links": list(self.links),
        }
        for key, value in (
            ("unit", self.unit),
            ("decimals", self.decimals),
            ("min", self.minimum),
            ("max", self.maximum),
            ("noValue", self.no_value),
        ):
            if value is not None:
                defaults[key] = value
        legends = [query.legend for query in self.queries if query.legend]
        if self.type == "bargauge" and len(legends) == 1:
            # A bar gauge drops the series name when only one series comes back; an explicit
            # display name (the legend, as field-label references) keeps every bar labelled.
            defaults["displayName"] = re.sub(r"\{\{(\w+)\}\}", r"${__field.labels.\1}", legends[0])
        if self.type == "timeseries":
            defaults["custom"] = {
                "drawStyle": "bars" if self.bars else "line",
                "lineWidth": 1 if self.bars else 2,
                "fillOpacity": 70 if self.bars else 12,
                "showPoints": "auto",
                "pointSize": 5,
                "spanNulls": False,
                "stacking": {"mode": "normal" if self.stack else "none", "group": "A"},
            }
        # Grafana turns a Loki *instant* metric query into one table frame, so a bar gauge or a
        # label table would lose its per-series names: those panels query a coarse range and
        # reduce to the last point (each point already spans the whole $__range window).
        loki_series = self.type in {"bargauge", "table"} and self.datasource.type == "loki"
        panel: Json = {
            "id": panel_id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "gridPos": {"h": self.height, "w": self.width, "x": x, "y": y},
            "datasource": self.datasource.model_dump(),
            "targets": [
                query.target(chr(ord("A") + index), force_range=loki_series)
                for index, query in enumerate(self.queries)
            ],
            "fieldConfig": {"defaults": defaults, "overrides": list(self.overrides)},
            "options": self.options or _default_options(self.type),
            "transformations": list(self.transformations),
        }
        if self.time_from is not None:
            panel["timeFrom"] = self.time_from
        if self.interval is not None or loki_series:
            panel["interval"] = self.interval or "1m"
        return panel


def _default_options(kind: PanelType) -> Json:
    reduce: Json = {"calcs": ["lastNotNull"], "fields": "", "values": False}
    match kind:
        case "stat":
            return {
                "reduceOptions": reduce,
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto",
                "textMode": "auto",
                "orientation": "auto",
            }
        case "bargauge":
            return {
                "reduceOptions": reduce,
                "orientation": "horizontal",
                "displayMode": "gradient",
                "valueMode": "color",
                "showUnfilled": True,
                "namePlacement": "left",
                "sizing": "manual",
                "minVizHeight": 10,
                "maxVizHeight": 24,
            }
        case "timeseries":
            return {
                "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                "tooltip": {"mode": "multi", "sort": "desc"},
            }
        case "table":
            return {"showHeader": True, "cellHeight": "sm", "footer": {"show": False}}
        case "logs":
            return {
                "showTime": True,
                "wrapLogMessage": True,
                "sortOrder": "Descending",
                "enableLogDetails": True,
                "dedupStrategy": "none",
                "prettifyLogMessage": False,
                "showLabels": False,
                "showCommonLabels": False,
            }


class Layout:
    """Places panels left to right, wrapping to a new row when the 24 columns are full."""

    def __init__(self) -> None:
        self._panels: list[Json] = []
        self._x = 0
        self._y = 0
        self._row_height = 0

    def add(self, *panels: Panel) -> "Layout":
        for panel in panels:
            if self._x + panel.width > GRID_COLUMNS:
                self._x, self._y, self._row_height = 0, self._y + self._row_height, 0
            self._panels.append(panel.render(len(self._panels) + 1, self._x, self._y))
            self._x += panel.width
            self._row_height = max(self._row_height, panel.height)
        return self

    @property
    def panels(self) -> list[Json]:
        return self._panels


def _datasource_variable(name: str, label: str, kind: DatasourceType) -> Json:
    return {
        "type": "datasource",
        "name": name,
        "label": label,
        "query": kind,
        "current": {},
        "hide": 0,
        "refresh": 1,
        "regex": "",
        "includeAll": False,
        "multi": False,
        "options": [],
        "skipUrlSync": False,
    }


SESSION_VARIABLE: Final[Json] = {
    "type": "textbox",
    "name": "session_id",
    "label": "Session id",
    "description": "A session_id from the audit log (see Recent sessions, or click a session "
    "on the Threats dashboard).",
    "query": "",
    "current": {"text": "", "value": ""},
    "options": [],
    "hide": 0,
    "skipUrlSync": False,
}

_RELOADS: Final = '{job="acl", event="policy_reload"} | json result="result"'


def _annotations() -> list[JsonValue]:
    """Grafana's own annotations plus policy reloads from the audit log in Loki."""

    def reload_annotation(name: str, result: str, color: str, title: str) -> Json:
        expr = f'{_RELOADS}, revision="revision", previous_revision="previous_revision"'
        expr += f' | result="{result}"'
        return {
            "name": name,
            "datasource": LOKI.model_dump(),
            "enable": True,
            "hide": False,
            "iconColor": color,
            "expr": expr,
            "target": {"refId": "Anno", "expr": expr, "queryType": "range"},
            "titleFormat": title,
            "textFormat": "{{previous_revision}} -> {{revision}}",
            "tagKeys": "result",
            "instant": False,
        }

    return [
        {
            "builtIn": 1,
            "datasource": {"type": "grafana", "uid": "-- Grafana --"},
            "enable": True,
            "hide": True,
            "iconColor": "rgba(0, 211, 255, 1)",
            "name": "Annotations & Alerts",
            "type": "dashboard",
        },
        reload_annotation("Policy changes", "ok", "blue", "Policy reloaded"),
        reload_annotation("Rejected policy edits", "invalid", "red", "Policy edit rejected"),
    ]


class Dashboard(_Model):
    uid: str
    title: str
    description: str
    panels: tuple[Panel, ...]
    session_variable: bool = False

    def render(self) -> Json:
        variables: list[JsonValue] = [
            _datasource_variable("datasource", "Prometheus", "prometheus"),
            _datasource_variable("loki", "Loki", "loki"),
        ]
        if self.session_variable:
            variables.append(SESSION_VARIABLE)
        return {
            "uid": self.uid,
            "title": self.title,
            "description": self.description,
            "tags": [TAG],
            "editable": False,
            "graphTooltip": 1,
            "timezone": "browser",
            "time": {"from": "now-15m", "to": "now"},
            "refresh": "5s",
            "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m", "5m"]},
            "schemaVersion": 41,
            "version": 1,
            "liveNow": False,
            "fiscalYearStartMonth": 0,
            "weekStart": "",
            "links": [
                {
                    "type": "dashboards",
                    "tags": [TAG],
                    "title": "AI Control Layer",
                    "asDropdown": False,
                    "includeVars": False,
                    "keepTime": True,
                    "targetBlank": False,
                    "icon": "external link",
                    "tooltip": "",
                    "url": "",
                }
            ],
            "templating": {"list": variables},
            "annotations": {"list": _annotations()},
            "panels": list[JsonValue](Layout().add(*self.panels).panels),
        }


# --------------------------------------------------------------------------- query parts


def rate(metric: str, by: str, where: str = "") -> str:
    return f"sum by ({by}) (rate({metric}{{{where}}}[$__rate_interval]))"


def over_range(metric: str, by: str, where: str = "") -> str:
    return f"sum by ({by}) (increase({metric}{{{where}}}[$__range]))"


def quantile(q: float, histogram: str, by: str, window: str = "$__rate_interval") -> str:
    return (
        f"histogram_quantile({q}, sum by (le, {by}) (rate({histogram}_bucket[{window}])))"
        if by
        else f"histogram_quantile({q}, sum by (le) (rate({histogram}_bucket[{window}])))"
    )


def session_logs(fields: Iterable[str], *extra: str) -> str:
    """The `$session_id` session's audit lines, with ``fields`` extracted from the JSON."""
    extract = ", ".join(f'{field}="{field}"' for field in ("session_id", *fields))
    stages = " ".join(extra)
    return (
        f'{{job="acl", decision=~".+"}} {_SESSION_FILTER} | json {extract} '
        f'| session_id="$session_id" {stages}'
    ).rstrip()


RISK_DECIMALS: Final[Json] = {
    "matcher": {"id": "byName", "options": "risk"},
    "properties": [{"id": "decimals", "value": 2}],
}
SCOPE_WIDTH: Final[Json] = {
    "matcher": {"id": "byName", "options": "effective_scope"},
    "properties": [
        {"id": "custom.width", "value": 440},
        {"id": "custom.inspect", "value": True},
    ],
}


def logs_table(columns: Sequence[str]) -> tuple[Json, ...]:
    """Loki log lines as a table: the stream and extracted labels become ``columns``."""
    return (
        {"id": "extractFields", "options": {"source": "labels", "replace": False}},
        {"id": "filterFieldsByName", "options": {"include": {"names": ["Time", *columns]}}},
        {
            "id": "organize",
            "options": {
                "indexByName": {name: i for i, name in enumerate(["Time", *columns])},
                "renameByName": {"Time": "ts"},
            },
        },
        {  # Loki labels are strings; risk reads better as a number
            "id": "convertFieldType",
            "options": {"conversions": [{"targetField": "risk", "destinationType": "number"}]},
        },
        {"id": "sortBy", "options": {"sort": [{"field": "ts", "desc": True}]}},
    )


SESSION_ID_LINK: Final[Json] = {
    "matcher": {"id": "byName", "options": "session_id"},
    "properties": [
        {
            "id": "links",
            "value": [
                {
                    "title": "Trace this session",
                    "url": "/d/acl-session-trace/session-trace?"
                    "var-session_id=${__value.raw}&${__url_time_range}",
                }
            ],
        }
    ],
}

# ------------------------------------------------------------------------------- Posture


def _column_width(column: str, width: int) -> Json:
    return {
        "matcher": {"id": "byName", "options": column},
        "properties": [{"id": "custom.width", "value": width}],
    }


def _fixed_color(series: str, color: str) -> Json:
    return {
        "matcher": {"id": "byName", "options": series},
        "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}],
    }


def posture() -> Dashboard:
    decisions = {"allow": "green", "redact": "yellow", "require_approval": "orange", "block": "red"}
    decision_colors = tuple(_fixed_color(name, color) for name, color in decisions.items())
    total = "sum(increase(acl_requests_total[$__range]))"
    blocked = 'sum(increase(acl_requests_total{decision="block"}[$__range]))'
    return Dashboard(
        uid="acl-posture",
        title="Posture",
        description="Management view: how much traffic is decided which way, what threatens "
        "it, what it costs and how close each user and agent is to its budget.",
        panels=(
            Panel(
                type="stat",
                title="Decisions",
                description="Calls the gateway decided in the time range (LLM and MCP).",
                queries=(prom(f"{total} or vector(0)"),),
                width=4,
                height=5,
                decimals=0,
            ),
            Panel(
                type="stat",
                title="Block rate",
                description="Share of decided calls that were blocked.",
                queries=(prom(f"({blocked} or vector(0)) / clamp_min({total}, 1)"),),
                width=4,
                height=5,
                unit="percentunit",
                decimals=1,
                thresholds=(
                    Threshold(color="green"),
                    Threshold(color="orange", value=0.1),
                    Threshold(color="red", value=0.3),
                ),
                color_mode="thresholds",
            ),
            Panel(
                type="stat",
                title="Held for approval",
                description="Calls answered with require_approval in the time range.",
                queries=(
                    prom(
                        'sum(increase(acl_requests_total{decision="require_approval"}'
                        "[$__range])) or vector(0)"
                    ),
                ),
                width=4,
                height=5,
                decimals=0,
            ),
            Panel(
                type="stat",
                title="Spend",
                description="USD from the policy pricing table (GPU part: upstream wall time, "
                "an estimate).",
                queries=(prom("sum(increase(acl_cost_usd_total[$__range])) or vector(0)"),),
                width=4,
                height=5,
                unit="currencyUSD",
                decimals=6,
            ),
            Panel(
                type="stat",
                title="Tokens",
                description="Tokens the LLM upstream reported (prompt + completion).",
                queries=(prom("sum(increase(acl_tokens_total[$__range])) or vector(0)"),),
                width=4,
                height=5,
                unit="short",
                decimals=0,
            ),
            Panel(
                type="stat",
                title="Tainted sessions",
                description="Live sessions that received untrusted content (taint lasts until "
                "the session ends).",
                queries=(prom("max(acl_tainted_sessions) or vector(0)"),),
                width=4,
                height=5,
                decimals=0,
                thresholds=(Threshold(color="green"), Threshold(color="orange", value=1)),
                color_mode="thresholds",
            ),
            Panel(
                type="timeseries",
                title="Decisions per second",
                description="Final decision of every call, by type.",
                queries=(prom(rate("acl_requests_total", "decision"), "{{decision}}"),),
                width=12,
                unit="reqps",
                stack=True,
                overrides=decision_colors,
            ),
            Panel(
                type="bargauge",
                title="Top threat reason codes",
                description="Reason codes of blocked and held calls (from the audit log).",
                queries=(
                    loki(
                        'topk(10, sum by (reason_code) (count_over_time({job="acl", '
                        'decision=~"block|require_approval"} | json reason_code="reason_code" '
                        "[$__range])))",
                        "{{reason_code}}",
                        instant=True,
                    ),
                ),
                width=12,
                decimals=0,
                thresholds=(Threshold(color="red"),),
                color_mode="thresholds",
            ),
            Panel(
                type="bargauge",
                title="Cost per user and agent",
                description="Spend in the time range by user and agent (unknown identities are "
                "bucketed as `other`).",
                queries=(
                    prom(
                        over_range("acl_cost_usd_total", "user, agent") + " > 0",
                        "{{user}} / {{agent}}",
                        instant=True,
                    ),
                ),
                width=8,
                unit="currencyUSD",
                decimals=6,
            ),
            Panel(
                type="bargauge",
                title="Budget usage",
                description="Usage over each hard limit (daily, UTC). Orange from the soft limit "
                "(80 %), red at the hard limit, where calls are blocked.",
                queries=(
                    prom(
                        "max by (scope, id) (acl_budget_usage_ratio)",
                        "{{scope}} {{id}}",
                        instant=True,
                    ),
                ),
                width=8,
                unit="percentunit",
                decimals=1,
                minimum=0,
                maximum=1,
                thresholds=BUDGET_STEPS,
                color_mode="thresholds",
            ),
            Panel(
                type="timeseries",
                title="Tokens per agent",
                description="Token throughput by agent and model.",
                queries=(prom(rate("acl_tokens_total", "agent, model"), "{{agent}} {{model}}"),),
                width=8,
                unit="short",
            ),
            Panel(
                type="timeseries",
                title="Daily trend",
                description="Decisions in the preceding hour, every 15 minutes over the last "
                "7 days.",
                queries=(
                    prom("sum by (decision) (increase(acl_requests_total[1h]))", "{{decision}}"),
                ),
                width=24,
                stack=True,
                bars=True,
                time_from="7d",
                interval="15m",
                decimals=0,
                overrides=decision_colors,
            ),
        ),
    )


# ------------------------------------------------------------------------------- Threats


def threats() -> Dashboard:
    blocks = (
        '{job="acl", decision="block"} | json principal="principal", resource="resource", '
        'reason_code="reason_code", session_id="session_id" | line_format '
        '"{{.principal}} | {{.actor}} | {{.channel}} {{.resource}} | {{.reason_code}} '
        '| {{.session_id}}"'
    )
    risk = (
        'topk(10, max by (session_id, principal) (max_over_time({job="acl", decision=~".+"} '
        '| json session_id="session_id", principal="principal", risk="risk" '
        '| session_id != "" | risk != "" | unwrap risk | __error__="" [$__range])))'
    )
    return Dashboard(
        uid="acl-threats",
        title="Threats",
        description="Security view: blocks as they happen, which controls fire, the riskiest "
        "sessions, signature hits, the approval queue and policy changes.",
        panels=(
            Panel(
                type="stat",
                title="Blocks",
                description="Calls blocked in the time range.",
                queries=(
                    prom(
                        'sum(increase(acl_requests_total{decision="block"}[$__range])) or vector(0)'
                    ),
                ),
                width=4,
                height=5,
                decimals=0,
                thresholds=(Threshold(color="green"), Threshold(color="red", value=1)),
                color_mode="thresholds",
            ),
            Panel(
                type="stat",
                title="Approvals pending",
                description="Operations waiting for a human decision (GET /admin/approvals).",
                queries=(prom("max(acl_approvals_pending) or vector(0)"),),
                width=4,
                height=5,
                decimals=0,
                thresholds=(Threshold(color="green"), Threshold(color="orange", value=1)),
                color_mode="thresholds",
            ),
            Panel(
                type="stat",
                title="Tainted sessions",
                description="Live sessions that received untrusted content.",
                queries=(prom("max(acl_tainted_sessions) or vector(0)"),),
                width=4,
                height=5,
                decimals=0,
                thresholds=(Threshold(color="green"), Threshold(color="orange", value=1)),
                color_mode="thresholds",
            ),
            Panel(
                type="stat",
                title="Killed agents",
                description="Agents whose kill switch is on (POST /admin/kill).",
                queries=(prom("count(acl_kill_switch_active == 1) or vector(0)"),),
                width=4,
                height=5,
                decimals=0,
                thresholds=(Threshold(color="green"), Threshold(color="red", value=1)),
                color_mode="thresholds",
            ),
            Panel(
                type="stat",
                title="Throttled calls",
                description="Calls of autonomous agents rejected by a risk-rule throttle.",
                queries=(prom("sum(increase(acl_throttled_total[$__range])) or vector(0)"),),
                width=4,
                height=5,
                decimals=0,
            ),
            Panel(
                type="stat",
                title="Risk-rule alerts",
                description="Alerts raised by risk rules (`alert: true`).",
                queries=(prom("sum(increase(acl_alerts_total[$__range])) or vector(0)"),),
                width=4,
                height=5,
                decimals=0,
            ),
            Panel(
                type="logs",
                title="Blocks (live)",
                description="Every blocked call from the audit log: principal | agent | channel "
                "resource | reason code | session.",
                queries=(loki(blocks),),
                width=14,
                height=11,
            ),
            Panel(
                type="table",
                title="Highest-risk sessions",
                description="Peak session risk per session in the time range (audit log; "
                "session ids are never metric labels). Click a session for its trace.",
                queries=(loki(risk, instant=True),),
                width=10,
                height=11,
                decimals=2,
                minimum=0,
                maximum=1,
                thresholds=GREEN_AMBER_RED,
                color_mode="thresholds",
                transformations=(
                    {
                        "id": "reduce",
                        "options": {"reducers": ["lastNotNull"], "labelsToFields": True},
                    },
                    {
                        "id": "organize",
                        "options": {
                            "excludeByName": {"Field": True},
                            "indexByName": {"session_id": 0, "principal": 1},
                            "renameByName": {"Last *": "peak risk"},
                        },
                    },
                    {"id": "sortBy", "options": {"sort": [{"field": "peak risk", "desc": True}]}},
                    {"id": "limit", "options": {"limitField": 10}},
                ),
                overrides=(
                    _column_width("principal", 120),
                    {
                        "matcher": {"id": "byName", "options": "peak risk"},
                        "properties": [
                            {
                                "id": "custom.cellOptions",
                                "value": {"type": "gauge", "mode": "gradient"},
                            },
                        ],
                    },
                    SESSION_ID_LINK,
                ),
            ),
            Panel(
                type="bargauge",
                title="Top controls by block verdicts",
                description="Block verdicts per control (log_only verdicts included).",
                queries=(
                    prom(
                        "topk(10, "
                        + over_range("acl_control_verdicts_total", "control", 'decision="block"')
                        + " > 0)",
                        "{{control}}",
                        instant=True,
                    ),
                ),
                width=8,
                decimals=0,
                thresholds=(Threshold(color="red"),),
                color_mode="thresholds",
            ),
            Panel(
                type="bargauge",
                title="Signature hits",
                description="Matches of the external attack-signature feed, by signature id.",
                queries=(
                    prom(
                        "topk(10, " + over_range("acl_signature_hits_total", "signature") + " > 0)",
                        "{{signature}}",
                        instant=True,
                    ),
                ),
                width=8,
                decimals=0,
                thresholds=(Threshold(color="orange"),),
                color_mode="thresholds",
            ),
            Panel(
                type="bargauge",
                title="Approvals: requested and outcomes",
                description="Approvals requested in the time range, and how the decided ones "
                "ended: each approval counts once as requested and once in its outcome "
                "(succeeded, denied, expired, failed, uncertain). Waiting ones are on "
                "Approvals pending.",
                queries=(
                    prom(
                        'label_replace(sum by (decision) (increase(acl_approvals_total{decision=~"'
                        + "|".join(APPROVAL_COUNTED)
                        + '"}[$__range])), "decision", "requested", "decision", "pending")',
                        "{{decision}}",
                        instant=True,
                    ),
                ),
                width=8,
                decimals=0,
                overrides=tuple(
                    _fixed_color(name, color) for name, color in APPROVAL_COLORS.items()
                ),
            ),
            Panel(
                type="timeseries",
                title="Verdicts per second by control",
                description="Non-allow verdicts by control and decision.",
                queries=(
                    prom(
                        rate("acl_control_verdicts_total", "control, decision", 'decision!="allow"')
                        + " > 0",
                        "{{control}} {{decision}}",
                    ),
                ),
                width=12,
                unit="reqps",
            ),
            Panel(
                type="timeseries",
                title="Session risk after each call",
                description="Median and p95 of session risk, observed after every call.",
                queries=(
                    prom(
                        "histogram_quantile(0.5, sum by (le) "
                        "(rate(acl_session_risk_bucket[$__rate_interval])))",
                        "p50",
                    ),
                    prom(
                        "histogram_quantile(0.95, sum by (le) "
                        "(rate(acl_session_risk_bucket[$__rate_interval])))",
                        "p95",
                    ),
                ),
                width=12,
                minimum=0,
                maximum=1,
                decimals=2,
            ),
        ),
    )


# ------------------------------------------------------------------------- Session trace


def session_trace() -> Dashboard:
    timeline = (
        "action",
        "resource",
        "reason_code",
        "effective_scope",
        "risk",
        "taint",
        "policy_revision",
    )
    risk_series = session_logs(("risk",), '| risk != "" | unwrap risk | __error__=""')
    recent = (
        'topk(20, sum by (session_id, principal, actor) (count_over_time({job="acl", '
        'decision=~".+"} | json session_id="session_id", principal="principal" '
        '| session_id != "" [$__range])))'
    )
    return Dashboard(
        uid="acl-session-trace",
        title="Session trace",
        description="One session ($session_id): every decision, its effective permissions and "
        "the risk score over time, from the audit log.",
        session_variable=True,
        panels=(
            Panel(
                type="stat",
                title="Calls",
                description="Decisions recorded for this session in the time range.",
                queries=(
                    loki(f"sum(count_over_time({session_logs(())} [$__range]))", instant=True),
                ),
                width=4,
                height=5,
                decimals=0,
                no_value="0",
            ),
            Panel(
                type="stat",
                title="Blocked",
                description="Blocked calls of this session.",
                queries=(
                    loki(
                        f'sum(count_over_time({session_logs(())} | decision="block" [$__range]))',
                        instant=True,
                    ),
                ),
                width=4,
                height=5,
                decimals=0,
                no_value="0",
                thresholds=(Threshold(color="green"), Threshold(color="red", value=1)),
                color_mode="thresholds",
            ),
            Panel(
                type="stat",
                title="Peak risk",
                description="Highest session risk recorded (decays with risk.half_life_s).",
                queries=(loki(f"max(max_over_time({risk_series} [$__range]))", instant=True),),
                width=4,
                height=5,
                decimals=2,
                minimum=0,
                maximum=1,
                thresholds=GREEN_AMBER_RED,
                color_mode="thresholds",
            ),
            Panel(
                type="stat",
                title="Taint",
                description="Tainted once untrusted content reached the session; never cleared "
                "until the session ends.",
                queries=(
                    loki(
                        f"sum(count_over_time({session_logs(('taint',), '| taint="true"')} "
                        "[$__range]))",
                        instant=True,
                    ),
                ),
                width=4,
                height=5,
                no_value="clean",
                mappings=(
                    {
                        "type": "range",
                        "options": {"from": 1, "to": None, "result": {"text": "tainted"}},
                    },
                ),
                thresholds=(Threshold(color="green"), Threshold(color="red", value=1)),
                color_mode="thresholds",
            ),
            Panel(
                type="table",
                title="Recent sessions",
                description="Sessions with decisions in the time range; click an id to trace it.",
                queries=(loki(recent, instant=True),),
                width=8,
                height=10,
                transformations=(
                    {
                        "id": "reduce",
                        "options": {"reducers": ["lastNotNull"], "labelsToFields": True},
                    },
                    {
                        "id": "organize",
                        "options": {
                            "excludeByName": {"Field": True},
                            "indexByName": {"session_id": 0, "principal": 1, "actor": 2},
                            "renameByName": {"Last *": "calls"},
                        },
                    },
                    {"id": "sortBy", "options": {"sort": [{"field": "calls", "desc": True}]}},
                ),
                overrides=(
                    _column_width("principal", 110),
                    _column_width("actor", 100),
                    _column_width("calls", 60),
                    SESSION_ID_LINK,
                ),
            ),
            Panel(
                type="timeseries",
                title="Risk over time",
                description="Session risk after each call (the audit entry's `risk`).",
                queries=(
                    loki(
                        f"max(max_over_time({risk_series} [$__auto]))",
                        "risk",
                    ),
                ),
                width=16,
                height=10,
                decimals=2,
                minimum=0,
                maximum=1,
                thresholds=GREEN_AMBER_RED,
                options={
                    "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "single", "sort": "none"},
                },
                overrides=(
                    {
                        "matcher": {"id": "byName", "options": "risk"},
                        "properties": [
                            {"id": "custom.showPoints", "value": "always"},
                            {"id": "custom.spanNulls", "value": True},
                            {"id": "custom.thresholdsStyle", "value": {"mode": "line+area"}},
                        ],
                    },
                ),
            ),
            Panel(
                type="table",
                title="Timeline",
                description="Every decision of the session, newest first: what was asked, the "
                "verdict and reason, the effective scope after session restrictions, risk, "
                "taint and the policy revision that decided it.",
                queries=(loki(session_logs(timeline)),),
                width=24,
                height=12,
                transformations=logs_table(("channel", "decision", *timeline)),
                overrides=(
                    {
                        "matcher": {"id": "byName", "options": "decision"},
                        "properties": [
                            {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                            {
                                "id": "mappings",
                                "value": [
                                    {
                                        "type": "value",
                                        "options": {
                                            "allow": {"color": "green", "index": 0},
                                            "redact": {"color": "yellow", "index": 1},
                                            "require_approval": {"color": "orange", "index": 2},
                                            "block": {"color": "red", "index": 3},
                                        },
                                    }
                                ],
                            },
                        ],
                    },
                    SCOPE_WIDTH,
                    RISK_DECIMALS,
                ),
            ),
            Panel(
                type="table",
                title="Effective scope changes",
                description="The permissions left to the session at each call; taint and risk "
                "rules remove actions or attach obligations.",
                queries=(loki(session_logs(("action", "effective_scope", "taint", "risk"))),),
                width=24,
                height=8,
                transformations=logs_table(
                    ("decision", "action", "effective_scope", "taint", "risk")
                ),
                overrides=(SCOPE_WIDTH, RISK_DECIMALS),
            ),
        ),
    )


# --------------------------------------------------------------------------- Performance


def performance() -> Dashboard:
    overhead = "acl_overhead_seconds"
    upstream = (
        'quantile_over_time(0.95, {job="acl", decision=~".+"} '
        '| json upstream_ms="latency_ms.upstream" | upstream_ms != "" '
        '| unwrap upstream_ms | __error__="" [$__auto]) by (channel)'
    )
    return Dashboard(
        uid="acl-performance",
        title="Performance",
        description="Gateway overhead (excluding upstream time) per channel and per control, "
        "throughput, upstream and judge latency.",
        panels=(
            Panel(
                type="stat",
                title="p50 overhead",
                description="Gateway time per call excluding the upstream, all channels.",
                queries=(prom(quantile(0.5, overhead, "", "$__range"), instant=True),),
                width=4,
                height=5,
                unit="s",
                decimals=4,
            ),
            Panel(
                type="stat",
                title="p95 overhead",
                description="Gateway time per call excluding the upstream, all channels.",
                queries=(prom(quantile(0.95, overhead, "", "$__range"), instant=True),),
                width=4,
                height=5,
                unit="s",
                decimals=4,
            ),
            Panel(
                type="stat",
                title="p99 overhead",
                description="Gateway time per call excluding the upstream, all channels.",
                queries=(prom(quantile(0.99, overhead, "", "$__range"), instant=True),),
                width=4,
                height=5,
                unit="s",
                decimals=4,
            ),
            Panel(
                type="stat",
                title="Throughput",
                description="Decided calls per second over the time range.",
                queries=(
                    prom("sum(rate(acl_requests_total[$__range])) or vector(0)", instant=True),
                ),
                width=4,
                height=5,
                unit="reqps",
                decimals=2,
            ),
            Panel(
                type="stat",
                title="Budget store",
                description="1 while Redis answers budget operations (calls fail closed "
                "otherwise).",
                queries=(prom("min(acl_budget_store_up)"),),
                width=4,
                height=5,
                mappings=(
                    {
                        "type": "value",
                        "options": {
                            "0": {"text": "down", "color": "red"},
                            "1": {"text": "up", "color": "green"},
                        },
                    },
                ),
                thresholds=(Threshold(color="red"), Threshold(color="green", value=1)),
                color_mode="thresholds",
            ),
            Panel(
                type="stat",
                title="Policy revision",
                description="The active policy revision (acl_policy_info).",
                queries=(prom("max by (revision) (acl_policy_info)", "{{revision}}"),),
                width=4,
                height=5,
                options={
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "colorMode": "none",
                    "graphMode": "none",
                    "justifyMode": "auto",
                    "textMode": "name",
                    "orientation": "auto",
                },
            ),
            Panel(
                type="timeseries",
                title="Overhead per channel (p50 / p95 / p99)",
                description="Gateway overhead excluding the upstream, by channel.",
                queries=tuple(
                    prom(quantile(q, overhead, "channel"), f"p{round(q * 100)} {{{{channel}}}}")
                    for q in (0.5, 0.95, 0.99)
                ),
                width=12,
                unit="s",
            ),
            Panel(
                type="timeseries",
                title="Throughput by channel",
                description="Decided calls per second.",
                queries=(prom(rate("acl_requests_total", "channel"), "{{channel}}"),),
                width=12,
                unit="reqps",
            ),
            Panel(
                type="timeseries",
                title="Control latency p95",
                description="Time spent in each control (deterministic and semantic).",
                queries=(
                    prom(quantile(0.95, "acl_control_latency_seconds", "control"), "{{control}}"),
                ),
                width=12,
                unit="s",
            ),
            Panel(
                type="bargauge",
                title="Control latency p95 (time range)",
                description="Slowest controls over the whole time range.",
                queries=(
                    prom(
                        quantile(0.95, "acl_control_latency_seconds", "control", "$__range"),
                        "{{control}}",
                        instant=True,
                    ),
                ),
                width=12,
                unit="s",
                decimals=4,
                options={
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "orientation": "horizontal",
                    "displayMode": "gradient",
                    "valueMode": "color",
                    "showUnfilled": True,
                    "namePlacement": "left",
                    "minVizHeight": 10,
                    "maxVizHeight": 20,
                },
            ),
            Panel(
                type="timeseries",
                title="Upstream latency p95",
                description="Upstream (model, MCP server) time per call, from the audit "
                "entry's latency_ms.upstream.",
                queries=(loki(upstream, "{{channel}}"),),
                width=12,
                unit="ms",
            ),
            Panel(
                type="timeseries",
                title="LLM judge latency p95",
                description="Wall time of judge calls (intent_judge, output_policy, the "
                "prompt_injection judge band); not part of agent requests.",
                queries=(
                    prom(quantile(0.95, "acl_judge_latency_seconds", "control"), "{{control}}"),
                ),
                width=12,
                unit="s",
            ),
        ),
    )


DASHBOARDS: Final = (posture, threats, session_trace, performance)


def render_all() -> dict[str, str]:
    """File name -> JSON text for every dashboard."""
    rendered: dict[str, str] = {}
    for build in DASHBOARDS:
        dashboard = build()
        name = dashboard.uid.removeprefix("acl-") + ".json"
        rendered[name] = json.dumps(dashboard.render(), indent=2, ensure_ascii=False) + "\n"
    return rendered


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="fail if a committed file differs")
    args = parser.parse_args(argv)
    rendered = render_all()
    stale = [
        name
        for name, text in rendered.items()
        if not (DASHBOARDS_DIR / name).is_file() or (DASHBOARDS_DIR / name).read_text() != text
    ]
    if args.check:
        for name in stale:
            print(f"stale: grafana/dashboards/{name}", file=sys.stderr)
        return 1 if stale else 0
    DASHBOARDS_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in rendered.items():
        (DASHBOARDS_DIR / name).write_text(text)
    print(f"wrote {len(rendered)} dashboards to {DASHBOARDS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
