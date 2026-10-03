"""``pin <server> [--write] [--pins-dir DIR]``: review and approve an MCP server's tool baseline.

Fetches what the gateway sees upstream (``GET /admin/mcp/{server}/tools``, admin role) as a
candidate pin file, prints a diff against ``<pins-dir>/<server>.json`` (tools added, removed,
and every changed field of a changed tool, as JSON pointers), and writes the candidate only
with ``--write``. Run it on the host, where ``pins/`` is a working-tree directory; the gateway
mounts it read-only and picks a new file up on the next call, no restart needed::

    python -m gateway.cli --token "$ADMIN_TOKEN" pin sales_db          # review
    python -m gateway.cli --token "$ADMIN_TOKEN" pin sales_db --write  # approve

Exit codes: 0 up to date or written, 3 differences found and not written (so a CI job can
detect drift), 1 refused by the API, 2 usage error.
"""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import ValidationError

from gateway.cli.client import OperatorClient, OperatorError, parse
from gateway.proxies.mcp.pins import MISSING, PinDiff, PinFile, ToolBaseline

if TYPE_CHECKING:
    from gateway.cli.registry import Subcommands

DEFAULT_PINS_DIR: Final = Path("pins")
DIFFERENCES: Final = 3
_VALUE_CHARS: Final = 160


def register(subparsers: "Subcommands") -> None:
    pin = subparsers.add_parser(
        "pin", help="diff an MCP server's tools against its pin file; --write approves (admin)"
    )
    pin.add_argument("server", help="MCP server name from policy.yaml (upstreams.mcp)")
    pin.add_argument("--write", action="store_true", help="write the reviewed baseline")
    pin.add_argument("--pins-dir", type=Path, default=DEFAULT_PINS_DIR, help="default: ./pins")
    pin.set_defaults(run=run)


async def run(args: argparse.Namespace, client: OperatorClient) -> int:
    server: str = args.server
    candidate = parse(await client.get(f"/admin/mcp/{server}/tools"), PinFile)
    if candidate.server != server:
        raise OperatorError("server_mismatch", "the gateway answered for another server")
    path: Path = args.pins_dir / f"{server}.json"
    current = _read(path)
    diff = PinDiff.between(current, candidate)
    if args.json:
        print(json.dumps(_diff_json(diff, path, written=args.write and not diff.empty)))
    else:
        print(render(diff, server, path, exists=current is not None))
    if diff.empty:
        return 0
    if not args.write:
        if not args.json:
            print(f"not written: review the diff, then re-run with --write to approve {path}")
        return DIFFERENCES
    _write(path, candidate)
    if not args.json:
        print(f"wrote {path}")
    return 0


def _write(path: Path, pin: PinFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pin.to_json())


def _read(path: Path) -> PinFile | None:
    try:
        return PinFile.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        return None
    except ValidationError:
        # An unreadable pin file fails closed in the gateway; approving a fresh one fixes it.
        print(f"warning: {path} is not a valid pin file; diffing against nothing")
        return None


def _value(value: object) -> str:
    if value is MISSING:
        return "(absent)"
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= _VALUE_CHARS else text[: _VALUE_CHARS - 3] + "..."


def _tool_lines(sign: str, tool: ToolBaseline) -> list[str]:
    lines = [f"{sign} tool {tool.name}"]
    lines.append(f"    description: {_value(tool.description)}")
    lines.append(f"    inputSchema: {_value(tool.input_schema)}")
    lines.append(f"    annotations: {_value(tool.annotations)}")
    return lines


def render(diff: PinDiff, server: str, path: Path, *, exists: bool) -> str:
    """The human-reviewable diff: ``+`` added, ``-`` removed, ``~`` changed tools."""
    if diff.empty:
        return f"{server}: {path} is up to date"
    target = str(path) if exists else f"{path} (new file)"
    head = (
        f"{server}: {len(diff.added)} added, {len(diff.removed)} removed, "
        f"{len(diff.changed)} changed against {target}"
    )
    lines = [head]
    if diff.protocol is not None:
        lines.append(f"~ protocol_version: {diff.protocol[0]} -> {diff.protocol[1]}")
    for tool in diff.added:
        lines += _tool_lines("+", tool)
    for tool in diff.removed:
        lines += _tool_lines("-", tool)
    for change in diff.changed:
        lines.append(f"~ tool {change.name}")
        lines += [
            f"    {field.pointer}: {_value(field.before)} -> {_value(field.after)}"
            for field in change.changes
        ]
    return "\n".join(lines)


def _diff_json(diff: PinDiff, path: Path, *, written: bool) -> dict[str, object]:
    def value(item: object) -> object:
        return None if item is MISSING else item

    return {
        "path": str(path),
        "written": written,
        "added": [tool.name for tool in diff.added],
        "removed": [tool.name for tool in diff.removed],
        "changed": {
            change.name: [
                {"pointer": f.pointer, "before": value(f.before), "after": value(f.after)}
                for f in change.changes
            ]
            for change in diff.changed
        },
        "protocol": list(diff.protocol) if diff.protocol is not None else None,
    }
