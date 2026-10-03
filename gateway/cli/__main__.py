"""``python -m gateway.cli [--url URL] [--token TOKEN] [--json] <command> ...``

Talks to the operator listener only. ``--url`` defaults to ``ACL_OPERATOR_URL`` (else
``http://127.0.0.1:9090``), ``--token`` to ``ACL_OPERATOR_TOKEN``. Exit codes: 0 done,
1 refused by the API (reason code on stderr), 2 usage error.
"""

import argparse
import asyncio
import os
import sys
from collections.abc import Mapping, Sequence

import httpx

from gateway.cli.client import BEARER_ENV, DEFAULT_URL, URL_ENV, OperatorClient, OperatorError
from gateway.cli.registry import COMMANDS, Run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gateway.cli", description=__doc__)
    parser.add_argument("--url", default=None, help=f"operator API (env {URL_ENV})")
    parser.add_argument("--token", default=None, help=f"operator bearer token (env {BEARER_ENV})")
    parser.add_argument("--json", action="store_true", help="print JSON instead of tables")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for module in COMMANDS:
        module.register(subparsers)
    return parser


async def run(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    environ = os.environ if env is None else env
    args = build_parser().parse_args(argv)
    command: Run = args.run
    url = args.url or environ.get(URL_ENV, DEFAULT_URL)
    token = args.token or environ.get(BEARER_ENV)
    async with OperatorClient(url, token, transport=transport) as client:
        try:
            return await command(args, client)
        except OperatorError as exc:
            status = f" (HTTP {exc.status})" if exc.status is not None else ""
            print(f"error: {exc.code}{status}: {exc.message}", file=sys.stderr)
            return 1


def main() -> None:
    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
