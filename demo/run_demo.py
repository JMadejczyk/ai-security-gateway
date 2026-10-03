"""The 3-minute demo (SPEC "Demo script"), run against the live stack with the demo overlay.

    make demo-up                                   # once: stack + demo overlay (demo-web)
    make demo                                      # all seven scenes
    uv run python -m demo.run_demo --scene 3       # one scene (repeat --scene for more)
    uv run python -m demo.run_demo --pause         # wait for Enter between scenes

Agent actions run inside the ``agent`` container (``edge`` network only); operator actions
(tokens, approvals, the policy edit) run here, against the operator API on 127.0.0.1. Every
scene asserts its own outcome: the exit status is 1 if any scene deviated, so ``make demo``
is also a check. ``--json PATH`` writes the per-scene results.
"""

import argparse
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from demo.orchestrator.models import DemoReport, SceneResult
from demo.orchestrator.narration import Narrator, format_summary, grafana_links
from demo.orchestrator.scenes import SCENES, Demo
from demo.orchestrator.stack import (
    REPO_ROOT,
    AgentRunner,
    AuditLog,
    Compose,
    Database,
    DemoError,
    Operator,
    PolicyFile,
)

REQUIRED_SERVICES: Final = frozenset({"gateway", "agent", "demo-web", "mcp-fetch", "ollama"})
POLICY: Final = REPO_ROOT / "config" / "policy.yaml"
POLICY_BACKUP: Final = REPO_ROOT / "reports" / ".demo-policy-backup.yaml"
EXIT_DEVIATED: Final = 1
EXIT_NOT_READY: Final = 2


def _env_port(env: Mapping[str, str], name: str, default: int) -> str:
    """A host port: the environment, else ``.env`` (compose reads it too), else the default."""
    if value := env.get(name):
        return value
    dotenv = REPO_ROOT / ".env"
    if dotenv.exists():
        for line in reversed(dotenv.read_text().splitlines()):
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return str(default)


def _parser(env: Mapping[str, str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_demo", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--scene",
        type=int,
        action="append",
        choices=range(1, len(SCENES) + 1),
        metavar="N",
        help="run only scene N (1-7); repeatable",
    )
    parser.add_argument("--pause", action="store_true", help="wait for Enter between scenes")
    parser.add_argument("--no-color", action="store_true", help="plain text transcript")
    parser.add_argument("--json", type=Path, default=None, help="write the results here")
    port = _env_port(env, "ACL_OPERATOR_HOST_PORT", 9090)
    parser.add_argument("--operator-url", default=f"http://127.0.0.1:{port}")
    grafana = _env_port(env, "ACL_GRAFANA_HOST_PORT", 3300)
    parser.add_argument("--grafana-url", default=f"http://127.0.0.1:{grafana}")
    return parser


def preflight(compose: Compose) -> str | None:
    """Why the stack cannot run the demo, or None when it can."""
    missing = REQUIRED_SERVICES - compose.running_services()
    if missing:
        return f"not running: {', '.join(sorted(missing))}. Start the stack with `make demo-up`."
    hosts = compose.run("exec", "-T", "gateway", "printenv", "ACL_EGRESS_DEMO_HOSTS", check=False)
    if "demo-web=" not in hosts.stdout:
        return (
            "the gateway runs without the demo overlay's demo-web binding "
            "(ACL_EGRESS_DEMO_HOSTS), so it refuses the demo page. Run `make demo-up`."
        )
    return None


def run(args: argparse.Namespace, narrator: Narrator) -> DemoReport:
    compose = Compose()
    reason = preflight(compose)
    if reason is not None:
        raise DemoError(reason)
    policy = PolicyFile(POLICY, POLICY_BACKUP)
    if policy.recover():
        narrator.line("restored config/policy.yaml left edited by an interrupted run", "warn")
    operator = Operator(args.operator_url)
    demo = Demo(
        compose=compose,
        agent=AgentRunner(compose),
        operator=operator,
        audit=AuditLog(compose),
        db=Database(compose),
        policy=policy,
        narrator=narrator,
        run_id=time.strftime("%H%M%S"),
    )
    chosen = [s for s in SCENES if not args.scene or s.number in args.scene]
    results: list[SceneResult] = []
    try:
        for index, scene in enumerate(chosen):
            if args.pause and index > 0:
                input(f"\n[Enter] scene {scene.number}: {scene.title} ")
            results.append(scene.run(demo))
    finally:
        operator.close()
    sessions = {label: sid for r in results for label, sid in r.sessions.items() if sid}
    return DemoReport(
        run_id=demo.run_id,
        scenes=tuple(results),
        grafana_links=tuple(grafana_links(args.grafana_url, sessions)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(os.environ).parse_args(argv)
    narrator = Narrator(colour=False if args.no_color else None)
    try:
        report = run(args, narrator)
    except DemoError as exc:
        narrator.line(f"demo cannot start: {exc}", "bad")
        return EXIT_NOT_READY
    narrator.lines(format_summary(report.scenes))
    narrator.line()
    narrator.line("Open in Grafana (admin / ACL_GRAFANA_ADMIN_PASSWORD from .env):")
    narrator.lines(f"  {link}" for link in report.grafana_links)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(report.model_dump_json(indent=2))
    return 0 if report.passed else EXIT_DEVIATED


if __name__ == "__main__":
    sys.exit(main())
