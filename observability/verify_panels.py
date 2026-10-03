"""Checks every panel query of the provisioned dashboards against the running stack.

For each panel target of the four dashboards (read back from Grafana's API, so what is
checked is what Grafana provisioned):

1. the raw query goes to Prometheus (``/api/v1/query``) or Loki (``/loki/api/v1/query`` /
   ``query_range``) through Grafana's data source proxy, which proves the PromQL/LogQL parses
   and evaluates (Prometheus and Loki are not published outside ``ops``);
2. the panel's target goes through ``/api/ds/query``, the same path a browser uses, and the
   returned frames are counted.

    uv run python -m observability.verify_panels --password "$ACL_GRAFANA_ADMIN_PASSWORD"

``$session_id`` defaults to the session with the most audit lines in the window. Exit status
1 when any query errors; empty results are listed, not failures (a quiet stack has no blocks).
"""

import argparse
import os
import re
import sys
import time
from collections.abc import Sequence
from typing import Final, Literal, cast

import httpx
from pydantic import BaseModel, ConfigDict

DASHBOARD_UIDS: Final = ("acl-posture", "acl-threats", "acl-session-trace", "acl-performance")
WINDOW_S: Final = 15 * 60
BUILTINS: Final = {
    "$__rate_interval": "1m",
    "$__range": f"{WINDOW_S}s",
    "$__interval": "15s",
    "$__auto": "15s",
}
_RECENT_SESSION: Final = (
    'topk(1, sum by (session_id) (count_over_time({job="acl", decision=~".+"} '
    f'| json session_id="session_id" | session_id != "" [{WINDOW_S}s])))'
)

type JsonObject = dict[str, object]


def _object(value: object) -> JsonObject:
    return cast(JsonObject, value) if isinstance(value, dict) else {}


def _list(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


class PanelCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    dashboard: str
    panel: str
    ref_id: str
    kind: Literal["prometheus", "loki"]
    raw_error: str | None
    ds_error: str | None
    frames: int  # frames with at least one value
    values: int

    @property
    def failed(self) -> bool:
        return self.raw_error is not None or self.ds_error is not None


class Grafana:
    def __init__(self, url: str, user: str, password: str) -> None:
        self._client = httpx.Client(base_url=url, auth=(user, password), timeout=60)
        self.uids = {
            str(ds["type"]): str(ds["uid"])
            for ds in (_object(item) for item in _list(self._get("/api/datasources")))
        }

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str, **params: str) -> object:
        response = self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    def dashboard(self, uid: str) -> JsonObject:
        return _object(_object(self._get(f"/api/dashboards/uid/{uid}")).get("dashboard"))

    def raw(self, kind: str, expr: str, *, instant: bool) -> tuple[str | None, JsonObject]:
        """The query straight to Prometheus or Loki, through the data source proxy."""
        now = int(time.time())
        base = f"/api/datasources/proxy/uid/{self.uids[kind]}"
        if kind == "prometheus":
            path, params = (
                (f"{base}/api/v1/query", {"query": expr, "time": str(now)})
                if instant
                else (
                    f"{base}/api/v1/query_range",
                    {"query": expr, "start": str(now - WINDOW_S), "end": str(now), "step": "15"},
                )
            )
        else:
            path, params = (
                (f"{base}/loki/api/v1/query", {"query": expr, "time": f"{now}000000000"})
                if instant
                else (
                    f"{base}/loki/api/v1/query_range",
                    {
                        "query": expr,
                        "start": f"{now - WINDOW_S}000000000",
                        "end": f"{now}000000000",
                        "step": "15",
                        "limit": "100",
                    },
                )
            )
        response = self._client.get(path, params=params)
        body = _object(response.json()) if response.content else {}
        if not response.is_success or body.get("status") != "success":
            return f"{response.status_code}: {body.get('error') or response.text[:300]}", body
        return None, body

    def ds_query(self, target: JsonObject) -> tuple[str | None, int, int]:
        now_ms = int(time.time() * 1000)
        response = self._client.post(
            "/api/ds/query",
            json={
                "queries": [target],
                "from": str(now_ms - WINDOW_S * 1000),
                "to": str(now_ms),
            },
        )
        body = _object(response.json())
        results = _object(body.get("results"))
        frames_with_values, values = 0, 0
        for result in (_object(r) for r in results.values()):
            if error := result.get("error"):
                return str(error), 0, 0
            for frame in (_object(f) for f in _list(result.get("frames"))):
                columns = _list(_object(frame.get("data")).get("values"))
                count = max((len(_list(column)) for column in columns), default=0)
                values += count
                frames_with_values += 1 if count else 0
        if not response.is_success:
            return f"{response.status_code}: {response.text[:300]}", 0, 0
        return None, frames_with_values, values

    def busiest_session(self) -> str:
        error, body = self.raw("loki", _RECENT_SESSION, instant=True)
        result = _list(_object(body.get("data")).get("result"))
        if error or not result:
            return ""
        return str(_object(_object(result[0]).get("metric")).get("session_id", ""))


def substitute(expr: str, session_id: str, *, builtins: bool) -> str:
    expr = expr.replace("$session_id", session_id)
    if builtins:
        for name, value in BUILTINS.items():
            expr = expr.replace(name, value)
    if re.search(r"\$\w", expr) and builtins:
        msg = f"unresolved variable in {expr!r}"
        raise ValueError(msg)
    return expr


def check(grafana: Grafana, session_id: str) -> list[PanelCheck]:
    checks: list[PanelCheck] = []
    for uid in DASHBOARD_UIDS:
        dashboard = grafana.dashboard(uid)
        for panel in (_object(p) for p in _list(dashboard.get("panels"))):
            for target in (_object(t) for t in _list(panel.get("targets"))):
                kind = cast(
                    Literal["prometheus", "loki"], _object(target.get("datasource"))["type"]
                )
                expr = str(target["expr"])
                instant = bool(target.get("instant")) or target.get("queryType") == "instant"
                raw_error, _ = grafana.raw(
                    kind, substitute(expr, session_id, builtins=True), instant=instant
                )
                sent = {
                    **target,
                    "expr": substitute(expr, session_id, builtins=False),
                    "datasource": {"type": kind, "uid": grafana.uids[kind]},
                }
                ds_error, frames, values = grafana.ds_query(sent)
                checks.append(
                    PanelCheck(
                        dashboard=str(dashboard["title"]),
                        panel=str(panel["title"]),
                        ref_id=str(target["refId"]),
                        kind=kind,
                        raw_error=raw_error,
                        ds_error=ds_error,
                        frames=frames,
                        values=values,
                    )
                )
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    port = os.environ.get("ACL_GRAFANA_HOST_PORT", "3300")
    parser = argparse.ArgumentParser(description="Verify every dashboard panel query.")
    parser.add_argument("--url", default=f"http://127.0.0.1:{port}")
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default=os.environ.get("ACL_GRAFANA_ADMIN_PASSWORD"))
    parser.add_argument("--session-id", default=None, help="value for $session_id")
    args = parser.parse_args(argv)
    if not args.password:
        parser.error("--password or ACL_GRAFANA_ADMIN_PASSWORD is required")
    grafana = Grafana(args.url, args.user, args.password)
    try:
        session_id = args.session_id or grafana.busiest_session()
        print(f"$session_id = {session_id or '(none found)'}")
        checks = check(grafana, session_id)
    finally:
        grafana.close()
    for item in checks:
        if item.failed:
            status = f"ERROR raw={item.raw_error} ds={item.ds_error}"
        else:
            status = f"{item.frames} series, {item.values} values" if item.values else "empty"
        print(f"{item.dashboard:<14} {item.panel:<40} {item.ref_id} {item.kind:<10} {status}")
    failed = [item for item in checks if item.failed]
    empty = [item for item in checks if not item.failed and not item.values]
    print(f"\n{len(checks)} queries: {len(failed)} errors, {len(empty)} empty")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
