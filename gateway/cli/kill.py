"""``kill <agent> --reason ...`` and ``unkill <agent>``: the per-agent kill switch (admin)."""

import argparse
from typing import TYPE_CHECKING

from gateway.approvals.views import KillList, KillView
from gateway.cli.client import OperatorClient, parse
from gateway.cli.output import as_json, table

if TYPE_CHECKING:
    from gateway.cli.registry import Subcommands


def register(subparsers: "Subcommands") -> None:
    kill = subparsers.add_parser("kill", help="stop every call of an agent (admin)")
    kill.add_argument("agent", nargs="?", help="omit to list killed agents")
    kill.add_argument("--reason", default="", help="kept on the record (max 200 chars)")
    kill.set_defaults(run=kill_agent)

    unkill = subparsers.add_parser("unkill", help="lift an agent's kill switch (admin)")
    unkill.add_argument("agent")
    unkill.set_defaults(run=unkill_agent)


async def kill_agent(args: argparse.Namespace, client: OperatorClient) -> int:
    if args.agent is None:
        kills = parse(await client.get("/admin/kill"), KillList)
        if args.json:
            print(as_json(kills))
        else:
            rows = [[k.agent, k.reason, k.killed_by, k.killed_at] for k in kills.kills]
            headers = ("agent", "reason", "killed_by", "killed_at")
            print(table(headers, rows) if rows else "no agent is killed")
        return 0
    response = await client.post("/admin/kill", json={"agent": args.agent, "reason": args.reason})
    killed = parse(response, KillView)
    if args.json:
        print(as_json(killed))
    else:
        print(f"{killed.agent}: killed ({killed.revoked_approvals} unused approvals denied)")
    return 0


async def unkill_agent(args: argparse.Namespace, client: OperatorClient) -> int:
    unkilled = parse(await client.post("/admin/unkill", json={"agent": args.agent}), KillView)
    print(as_json(unkilled) if args.json else f"{unkilled.agent}: kill switch off")
    return 0
