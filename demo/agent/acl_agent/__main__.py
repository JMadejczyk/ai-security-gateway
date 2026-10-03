"""``python -m acl_agent <command>``: one agent action, its outcome as one JSON line on stdout.

    mcp <server> <tool> '<json arguments>' [--approval-id ID] [--wait-throttle]
    chat '<prompt>' [--max-tokens N]
    probe <host> <port>

The bearer token comes from ``ACL_TOKEN`` (never argv, so it stays out of ``ps``), the gateway
from ``ACL_GATEWAY_URL`` (compose sets ``http://gateway:8080``, the only address on ``edge``).
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from typing import Final, cast

import httpx

from .client import JsonObject, MCPSession, chat, unthrottled
from .probe import probe

TIMEOUT_S: Final = 150.0  # an LLM turn on CPU can wait ~60 s, plus the judges


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acl_agent", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mcp = commands.add_parser("mcp", help="one MCP tools/call through the gateway")
    mcp.add_argument("server")
    mcp.add_argument("tool")
    mcp.add_argument("arguments", help="tool arguments as a JSON object")
    mcp.add_argument("--approval-id", default=None, help="retry a held call with its approval")
    mcp.add_argument(
        "--wait-throttle", action="store_true", help="sleep through Retry-After and retry"
    )
    llm = commands.add_parser("chat", help="one user turn on /v1/chat/completions")
    llm.add_argument("prompt")
    llm.add_argument("--max-tokens", type=int, default=80)
    tcp = commands.add_parser("probe", help="resolve and TCP-connect, bypassing the gateway")
    tcp.add_argument("host")
    tcp.add_argument("port", type=int)
    return parser


def _client() -> httpx.Client:
    return httpx.Client(base_url=os.environ["ACL_GATEWAY_URL"], timeout=TIMEOUT_S)


def _mcp(args: argparse.Namespace) -> JsonObject:
    arguments = cast(JsonObject, json.loads(args.arguments))
    token = os.environ["ACL_TOKEN"]
    with _client() as client, MCPSession(client, token, args.server) as session:
        outcome = (
            unthrottled(lambda: session.call(args.tool, args.approval_id, **arguments))
            if args.wait_throttle
            else session.call(args.tool, args.approval_id, **arguments)
        )
    return {"kind": "mcp", "server": args.server, "tool": args.tool, **asdict(outcome)}


def _chat(args: argparse.Namespace) -> JsonObject:
    with _client() as client:
        outcome = chat(client, os.environ["ACL_TOKEN"], args.prompt, max_tokens=args.max_tokens)
    return {"kind": "chat", **asdict(outcome)}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    match args.command:
        case "mcp":
            reply = _mcp(args)
        case "chat":
            reply = _chat(args)
        case _:
            reply = {"kind": "probe", **asdict(probe(args.host, args.port))}
    sys.stdout.write(json.dumps(reply) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
