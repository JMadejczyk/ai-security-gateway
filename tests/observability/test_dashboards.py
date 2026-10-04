"""The provisioned Grafana dashboards and the observability configs, checked statically.

The committed JSON must be exactly what `grafana.build_dashboards` renders; every panel must
query through a template variable (never a hard-coded data source uid); PromQL and LogQL get
sanity checks (balanced brackets, only metrics the gateway exports, counters only under
``rate``/``increase``, histograms only under ``histogram_quantile`` by ``le``, LogQL stream
selectors only on labels Alloy promotes). The live suite runs every query for real.
"""

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

import gateway.approvals.metrics
import gateway.judges.client  # registers acl_judge_* on the shared registry
from gateway.telemetry import REGISTRY
from grafana.build_dashboards import DASHBOARDS_DIR, render_all

REPO_ROOT = Path(__file__).resolve().parents[2]

assert gateway.approvals.metrics  # registers acl_approvals_* and acl_kill_switch_active

EXPECTED = {
    "acl-posture": "Posture",
    "acl-threats": "Threats",
    "acl-session-trace": "Session trace",
    "acl-performance": "Performance",
    "acl-recording": "Recording",
}
DEFAULT_RANGE = {"acl-recording": "now-5m"}  # the others: now-15m
VARIABLE_UIDS = {"${datasource}", "${loki}"}
MIXED = {"type": "datasource", "uid": "-- Mixed --"}  # Grafana's built-in: per-target sources
ALLOY_LABELS = {"job", "channel", "decision", "actor", "mode", "event"}
FORBIDDEN_LABELS = {"session_id", "principal", "resource", "reason_code", "risk", "user_id"}
PROVISIONED_UIDS = ("acl-prometheus", "acl-loki")
ALLOY = REPO_ROOT / "observability" / "alloy" / "config.alloy"
PROMETHEUS = REPO_ROOT / "observability" / "prometheus" / "prometheus.yml"
LOKI = REPO_ROOT / "observability" / "loki" / "loki.yaml"
PROVISIONING = REPO_ROOT / "grafana" / "provisioning"


def _dashboards() -> dict[str, dict[str, Any]]:
    loaded = [json.loads(path.read_text()) for path in sorted(DASHBOARDS_DIR.glob("*.json"))]
    return {dashboard["uid"]: dashboard for dashboard in loaded}


DASHBOARDS = _dashboards()


def _targets() -> Iterator[tuple[str, str, dict[str, Any]]]:
    for uid, dashboard in DASHBOARDS.items():
        for panel in dashboard["panels"]:
            for target in panel["targets"]:
                yield uid, panel["title"], target


TARGETS = list(_targets())
PROM_EXPRS = [t["expr"] for _, _, t in TARGETS if t["datasource"]["type"] == "prometheus"]
LOKI_EXPRS = [t["expr"] for _, _, t in TARGETS if t["datasource"]["type"] == "loki"]
ANNOTATION_EXPRS = [
    a["expr"] for d in DASHBOARDS.values() for a in d["annotations"]["list"] if "expr" in a
]


def _exported_series() -> dict[str, str]:
    """Every sample name the gateway's registry can expose -> its metric type."""
    names: dict[str, str] = {}
    for family in REGISTRY.collect():
        if family.type == "counter":
            names[f"{family.name}_total"] = "counter"
        elif family.type == "histogram":
            for suffix in ("_bucket", "_count", "_sum"):
                names[f"{family.name}{suffix}"] = "histogram"
        else:
            names[family.name] = family.type
    return names


EXPORTED = _exported_series()


def _balanced(expr: str) -> bool:
    pairs, stack = {")": "(", "]": "[", "}": "{"}, list[str]()
    without_strings = re.sub(r'"(?:[^"\\]|\\.)*"', '""', expr)
    if without_strings.count('"') % 2:
        return False
    for char in without_strings:
        if char in "([{":
            stack.append(char)
        elif char in pairs and (not stack or stack.pop() != pairs[char]):
            return False
    return not stack


# --------------------------------------------------------------------------- generation


def test_committed_dashboards_equal_the_generator_output() -> None:
    rendered = render_all()
    committed = {path.name: path.read_text() for path in DASHBOARDS_DIR.glob("*.json")}
    assert committed == rendered, "run: uv run python -m grafana.build_dashboards"


def test_the_spec_dashboards_and_the_recording_view_exist_with_stable_uids() -> None:
    assert {uid: d["title"] for uid, d in DASHBOARDS.items()} == EXPECTED


@pytest.mark.parametrize("uid", sorted(EXPECTED))
def test_dashboard_defaults_suit_a_short_demo(uid: str) -> None:
    dashboard = DASHBOARDS[uid]
    assert dashboard["time"] == {"from": DEFAULT_RANGE.get(uid, "now-15m"), "to": "now"}
    assert dashboard["refresh"] == "5s"
    assert dashboard["editable"] is False
    variables = {v["name"]: v for v in dashboard["templating"]["list"]}
    assert variables["datasource"]["type"] == "datasource"
    assert variables["datasource"]["query"] == "prometheus"
    assert variables["loki"]["type"] == "datasource"
    assert variables["loki"]["query"] == "loki"


def test_session_trace_has_a_session_id_textbox() -> None:
    variables = {v["name"]: v for v in DASHBOARDS["acl-session-trace"]["templating"]["list"]}
    assert variables["session_id"]["type"] == "textbox"
    session_exprs = [
        t["expr"]
        for uid, title, t in TARGETS
        if uid == "acl-session-trace" and title != "Recent sessions"
    ]
    assert session_exprs
    assert all('session_id="$session_id"' in expr for expr in session_exprs)


@pytest.mark.parametrize("uid", sorted(EXPECTED))
def test_panels_have_unique_ids_targets_and_variable_data_sources(uid: str) -> None:
    panels = DASHBOARDS[uid]["panels"]
    assert len({p["id"] for p in panels}) == len(panels)
    for panel in panels:
        assert panel["targets"], panel["title"]
        mixed = panel["datasource"] == MIXED
        assert mixed or panel["datasource"]["uid"] in VARIABLE_UIDS, panel["title"]
        for target in panel["targets"]:
            assert target["datasource"]["uid"] in VARIABLE_UIDS, panel["title"]
            assert mixed or target["datasource"] == panel["datasource"], panel["title"]
            assert target["expr"].strip(), panel["title"]
        grid = panel["gridPos"]
        assert grid["x"] + grid["w"] <= 24, panel["title"]


def test_no_dashboard_hard_codes_a_data_source_uid() -> None:
    for path in DASHBOARDS_DIR.glob("*.json"):
        text = path.read_text()
        for uid in PROVISIONED_UIDS:
            assert uid not in text, f"{path.name} names {uid}"


def test_spec_panels_are_present() -> None:
    titles = {uid: {p["title"] for p in d["panels"]} for uid, d in DASHBOARDS.items()}
    assert {"Block rate", "Top threat reason codes", "Cost per user and agent"} <= titles[
        "acl-posture"
    ]
    assert {"Budget usage", "Daily trend"} <= titles["acl-posture"]
    assert {
        "Blocks (live)",
        "Top controls by block verdicts",
        "Highest-risk sessions",
        "Signature hits",
        "Approvals pending",
    } <= titles["acl-threats"]
    assert {"Timeline", "Risk over time", "Effective scope changes"} <= titles["acl-session-trace"]
    assert {"Overhead per channel (p50 / p95 / p99)", "Control latency p95"} <= titles[
        "acl-performance"
    ]


def test_every_dashboard_annotates_policy_reloads_from_loki() -> None:
    for dashboard in DASHBOARDS.values():
        names = {a["name"]: a for a in dashboard["annotations"]["list"]}
        changes = names["Policy changes"]
        assert changes["datasource"]["uid"] == "${loki}"
        assert 'event="policy_reload"' in changes["expr"]
        assert 'result="ok"' in changes["expr"]


# ------------------------------------------------------------------------------ queries


@pytest.mark.parametrize("expr", PROM_EXPRS + LOKI_EXPRS + ANNOTATION_EXPRS)
def test_queries_are_balanced(expr: str) -> None:
    assert _balanced(expr), expr


@pytest.mark.parametrize("expr", PROM_EXPRS)
def test_promql_uses_only_exported_metrics_correctly(expr: str) -> None:
    names = set(re.findall(r"\bacl_\w+", expr))
    assert names, expr
    for name in names:
        assert name in EXPORTED, f"{name} is not exported by the gateway"
        if EXPORTED[name] == "counter":
            assert re.search(rf"\b(?:rate|increase)\(\s*{name}\b", expr), f"raw counter: {expr}"
        if name.endswith("_bucket"):
            assert expr.startswith("histogram_quantile("), expr
            assert re.search(rf"sum by \(le\b[^)]*\) \(rate\({name}\[", expr), expr


@pytest.mark.parametrize("expr", PROM_EXPRS)
def test_promql_never_groups_by_per_session_values(expr: str) -> None:
    for group in re.findall(r"by \(([^)]*)\)", expr):
        labels = {label.strip() for label in group.split(",")}
        assert not labels & FORBIDDEN_LABELS, expr


@pytest.mark.parametrize("expr", LOKI_EXPRS + ANNOTATION_EXPRS)
def test_logql_selects_only_promoted_labels(expr: str) -> None:
    selectors = re.findall(r"\{([^{}]*)\}(?=\s*(?:\||\[|$))", expr)
    assert selectors, expr
    for selector in selectors:
        matchers = dict(re.findall(r'(\w+)\s*[=!~]+\s*"([^"]*)"', selector))
        assert matchers.get("job") == "acl", expr
        assert set(matchers) <= ALLOY_LABELS, expr


@pytest.mark.parametrize("expr", [e for e in LOKI_EXPRS if "unwrap" in e])
def test_logql_unwrap_drops_conversion_errors(expr: str) -> None:
    for tail in expr.split("unwrap")[1:]:
        assert re.match(r'\s+\w+\s*\|\s*__error__=""', tail), expr


# ------------------------------------------------------------------------------ configs


def test_alloy_promotes_only_low_cardinality_labels() -> None:
    config = ALLOY.read_text()
    labels_block = re.search(r"stage\.labels\s*\{\s*values\s*=\s*\{([^}]*)\}", config)
    assert labels_block is not None
    promoted = set(re.findall(r"(\w+)\s*=", labels_block.group(1)))
    assert promoted == ALLOY_LABELS - {"job"}
    assert '"job" = "acl"' in config
    assert '"__path__" = "/var/log/acl/audit-*.jsonl"' in config
    code = "\n".join(line.split("//")[0] for line in config.splitlines())  # comments out
    assert "docker" not in code.lower()  # no discovery.docker / loki.source.docker, no socket


def test_prometheus_scrapes_the_operator_listener_every_5s() -> None:
    config = yaml.safe_load(PROMETHEUS.read_text())
    assert config["global"]["scrape_interval"] == "5s"
    (job,) = config["scrape_configs"]
    assert job["metrics_path"] == "/metrics"
    assert job["static_configs"][0]["targets"] == ["gateway:9090"]


def test_loki_keeps_seven_days_on_the_filesystem() -> None:
    config = yaml.safe_load(LOKI.read_text())
    assert config["limits_config"]["retention_period"] == "168h"
    assert config["compactor"]["retention_enabled"] is True
    assert config["schema_config"]["configs"][0]["object_store"] == "filesystem"
    assert config["analytics"]["reporting_enabled"] is False


def test_grafana_provisioning_matches_the_dashboard_mount() -> None:
    sources = yaml.safe_load((PROVISIONING / "datasources" / "datasources.yaml").read_text())
    assert {d["uid"]: d["url"] for d in sources["datasources"]} == {
        "acl-prometheus": "http://prometheus:9090",
        "acl-loki": "http://loki:3100",
    }
    providers = yaml.safe_load((PROVISIONING / "dashboards" / "dashboards.yaml").read_text())
    (provider,) = providers["providers"]
    assert provider["options"]["path"] == "/etc/grafana/dashboards"
    assert provider["allowUiUpdates"] is False
    compose = (REPO_ROOT / "docker-compose.yml").read_text()
    assert "target: /etc/grafana/dashboards" in compose


def test_dashboard_files_are_the_only_ones_in_the_folder() -> None:
    assert {p.name for p in Path(DASHBOARDS_DIR).iterdir()} == set(render_all())


@pytest.mark.parametrize(
    ("uid", "title"),
    [
        ("acl-posture", "Top threat reason codes"),
        ("acl-threats", "Highest-risk sessions"),
        ("acl-session-trace", "Recent sessions"),
    ],
)
def test_loki_summaries_are_instant_queries_over_the_whole_range(uid: str, title: str) -> None:
    """Evaluated once at the end of the range: the query's topk is the final top K, and
    nothing outside the range leaks in (a range query reduced to its last point would)."""
    (panel,) = [p for p in DASHBOARDS[uid]["panels"] if p["title"] == title]
    (target,) = panel["targets"]
    assert target["queryType"] == "instant"
    assert "[$__range]" in target["expr"]
    assert target["expr"].startswith("topk(")
    ids = [t["id"] for t in panel["transformations"]]
    assert "reduce" not in ids  # no re-aggregation of the returned values over time
    assert ids[:2] == ["labelsToFields", "merge"]


# ----------------------------------------------------------------------------- Recording

# Measured at 760 px (single-column layout): ~36 px per grid unit, 18 px between panels, and
# ~50 px of page margin plus the "Powered by Grafana" footer in kiosk mode.
MOBILE_UNIT_PX, MOBILE_GAP_PX, MOBILE_CHROME_PX = 36, 18, 50


def test_recording_fits_a_760_by_1000_slot() -> None:
    """Below 769 px Grafana stacks panels one per row, so the recording view is built full
    width and its stacked height must fit the 1000 px browser slot of the video."""
    panels = DASHBOARDS["acl-recording"]["panels"]
    assert all(p["gridPos"]["w"] == 24 and p["gridPos"]["x"] == 0 for p in panels)
    height = sum(p["gridPos"]["h"] * MOBILE_UNIT_PX + MOBILE_GAP_PX for p in panels)
    assert height + MOBILE_CHROME_PX <= 1000


def test_recording_has_the_requested_panels_in_order() -> None:
    panels = DASHBOARDS["acl-recording"]["panels"]
    assert [(p["title"], p["type"]) for p in panels] == [
        ("Session risk", "gauge"),
        ("Session marked compromised", "stat"),
        ("Risk over time", "timeseries"),  # carries the policy-change annotations
        ("Last decisions", "table"),
        ("", "stat"),  # Blocked (5 min) | Approvals waiting
        ("", "stat"),  # Requests | Blocked | Redacted | p95 overhead
    ]
    gauge = panels[0]["fieldConfig"]["defaults"]
    assert (gauge["min"], gauge["max"]) == (0, 1)
    assert [t.get("value") for t in gauge["thresholds"]["steps"]] == [None, 0.5, 0.8]
    (decisions,) = panels[3]["targets"]
    assert decisions["maxLines"] == 6
    boxes = [o["properties"][0]["value"] for o in panels[4]["fieldConfig"]["overrides"]]
    assert boxes == ["Blocked (5 min)", "Approvals waiting"]


def test_recording_is_kiosk_friendly() -> None:
    dashboard = DASHBOARDS["acl-recording"]
    assert dashboard["links"] == []
    variables = {v["name"]: v for v in dashboard["templating"]["list"]}
    assert variables["session_id"]["type"] == "textbox"
    for panel in dashboard["panels"]:
        legend = panel["options"].get("legend")
        assert legend is None or legend["showLegend"] is False, panel["title"]


def test_recording_reports_check_time_split_into_rule_and_ai_checks() -> None:
    """One undifferentiated "overhead" number would mix the rule checks with the injection
    classifier and LLM judges; the closing row shows the two per call, labelled."""
    closing = DASHBOARDS["acl-recording"]["panels"][-1]
    names = [o["properties"][0]["value"] for o in closing["fieldConfig"]["overrides"]]
    assert names == ["Requests", "Redacted", "Rule checks p95 (live)", "AI checks p95 (live)"]
    assert "overhead" not in json.dumps(closing["targets"])
    rule, ai = (t["expr"] for t in closing["targets"][2:])
    for expr, present, absent in (
        (rule, ("sql_guard", "pii", "signatures"), ("prompt_injection", "intent_judge")),
        (ai, ("prompt_injection", "intent_judge", "output_policy"), ("sql_guard", "pii")),
    ):
        assert expr.startswith("quantile_over_time(0.95, ")
        assert all(f'latency_ms.controls.{c}"' in expr for c in present), expr
        assert not any(f'latency_ms.controls.{c}"' in expr for c in absent), expr
    assert '| ms != "0"' in ai  # only calls where an AI check ran
