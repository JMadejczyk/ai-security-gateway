"""``approvals list|show|approve|deny``: the approval queue (``/admin/approvals``)."""

import argparse
from typing import TYPE_CHECKING

from gateway.approvals.model import ApprovalState
from gateway.approvals.views import ApprovalList, ApprovalView
from gateway.cli.client import OperatorClient, parse
from gateway.cli.output import as_json, fields, table

if TYPE_CHECKING:
    from gateway.cli.registry import Subcommands

COLUMNS = ("id", "state", "agent", "principal", "tool", "resources", "reasons", "expires_at")


def register(subparsers: "Subcommands") -> None:
    parser = subparsers.add_parser("approvals", help="list and decide held operations")
    commands = parser.add_subparsers(dest="approvals_command", required=True)

    listing = commands.add_parser("list", help="approvals you may see (default: pending)")
    listing.add_argument(
        "--state",
        choices=[*(s.value for s in ApprovalState), "all"],
        default=ApprovalState.PENDING.value,
    )
    listing.add_argument("--limit", type=int, default=100)
    listing.set_defaults(run=list_approvals)

    show = commands.add_parser("show", help="one approval")
    show.add_argument("approval_id")
    show.set_defaults(run=show_approval)

    for verb, run in (("approve", approve), ("deny", deny)):
        decide = commands.add_parser(verb, help=f"{verb} a held operation")
        decide.add_argument("approval_id")
        decide.add_argument("--note", default=None, help="kept on the record (max 500 chars)")
        decide.set_defaults(run=run)


async def list_approvals(args: argparse.Namespace, client: OperatorClient) -> int:
    params: dict[str, str | int] = {"limit": args.limit}
    if args.state != "all":
        params["state"] = args.state
    listed = parse(await client.get("/admin/approvals", **params), ApprovalList)
    if args.json:
        print(as_json(listed))
    else:
        rows = [[getattr(a, column) for column in COLUMNS] for a in listed.approvals]
        print(table(COLUMNS, rows) if rows else "no approvals")
    return 0


async def show_approval(args: argparse.Namespace, client: OperatorClient) -> int:
    approval = parse(await client.get(f"/admin/approvals/{args.approval_id}"), ApprovalView)
    print(as_json(approval) if args.json else fields(approval))
    return 0


async def approve(args: argparse.Namespace, client: OperatorClient) -> int:
    return await _decide(args, client, "approve")


async def deny(args: argparse.Namespace, client: OperatorClient) -> int:
    return await _decide(args, client, "deny")


async def _decide(args: argparse.Namespace, client: OperatorClient, verb: str) -> int:
    body = {"note": args.note} if args.note is not None else None
    response = await client.post(f"/admin/approvals/{args.approval_id}/{verb}", json=body)
    decided = parse(response, ApprovalView)
    print(as_json(decided) if args.json else f"{decided.id}: {decided.state}")
    return 0
