"""Stage one beat of the pitch video (docs/video.md) against the live stack.

    make record SCENE=2                          # reset, print the prompt, wait for Enter
    make record SCENE=3 RECORD_ARGS="--pace 3"   # the night job, slower
    uv run python -m demo.record --scene 5 --no-wait --pace 0   # rehearsal, no person needed

Never leaves config/policy.yaml modified: scene 5 edits it atomically and restores it on the
way out (Ctrl-C included), and a backup an interrupted run left behind is restored first.
Exit status 1 if the beat did not land, 2 if the stack is not ready.
"""

import argparse
import os
import sys
import time
from collections.abc import Sequence
from typing import Final

from demo.orchestrator.narration import Narrator, format_summary
from demo.orchestrator.recording import BEATS, Recorder, RecordOptions, beat
from demo.orchestrator.scenes import Demo
from demo.orchestrator.stack import (
    AgentRunner,
    AuditLog,
    Compose,
    Database,
    DemoError,
    Operator,
    PolicyFile,
)
from demo.run_demo import EXIT_DEVIATED, EXIT_NOT_READY, POLICY, POLICY_BACKUP, env_port, preflight

DEFAULT_PACE_S: Final = 2.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="record", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--scene", type=int, required=True, choices=[b.number for b in BEATS])
    parser.add_argument("--no-wait", action="store_true", help="no Enter; rehearse the beat")
    parser.add_argument(
        "--pace", type=float, default=DEFAULT_PACE_S, help="seconds between scripted steps"
    )
    parser.add_argument("--open", action="store_true", help="open the Grafana view (macOS)")
    parser.add_argument("--no-color", action="store_true")
    port = env_port(os.environ, "ACL_OPERATOR_HOST_PORT", 9090)
    parser.add_argument("--operator-url", default=f"http://127.0.0.1:{port}")
    grafana = env_port(os.environ, "ACL_GRAFANA_HOST_PORT", 3300)
    parser.add_argument("--grafana-url", default=f"http://127.0.0.1:{grafana}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    narrator = Narrator(colour=False if args.no_color else None)
    compose = Compose()
    reason = preflight(compose)
    if reason is not None:
        narrator.line(f"cannot record: {reason}", "bad")
        return EXIT_NOT_READY
    operator = Operator(args.operator_url)
    demo = Demo(
        compose=compose,
        agent=AgentRunner(compose),
        operator=operator,
        audit=AuditLog(compose),
        db=Database(compose),
        policy=PolicyFile(POLICY, POLICY_BACKUP),
        narrator=narrator,
        run_id="rec" + time.strftime("%H%M%S"),
    )
    options = RecordOptions(
        wait=not args.no_wait,
        pace_s=max(args.pace, 0.0),
        open_browser=args.open,
        grafana_url=args.grafana_url,
    )
    try:
        result = Recorder(demo, options).run(beat(args.scene))
    except DemoError as exc:
        narrator.line(f"cannot record: {exc}", "bad")
        return EXIT_NOT_READY
    finally:
        operator.close()
    narrator.lines(format_summary([result]))
    return 0 if result.passed else EXIT_DEVIATED


if __name__ == "__main__":
    sys.exit(main())
