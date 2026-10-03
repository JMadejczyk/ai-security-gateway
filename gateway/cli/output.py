"""Plain-text tables and key/value blocks for the CLI (``--json`` prints the models instead)."""

from collections.abc import Sequence
from datetime import datetime
from typing import cast

from pydantic import BaseModel


def cell(value: object) -> str:
    match value:
        case None:
            return "-"
        case datetime():
            return value.isoformat(timespec="seconds").replace("+00:00", "Z")
        case tuple() | list():
            items = cast("Sequence[object]", value)
            return ",".join(cell(v) for v in items) or "-"
        case _:
            return str(value)


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [[cell(v) for v in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        widths = [max(w, len(c)) for w, c in zip(widths, row, strict=True)]
    lines = ["  ".join(h.upper().ljust(w) for h, w in zip(headers, widths, strict=True))]
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in cells]
    return "\n".join(line.rstrip() for line in lines)


def fields(model: BaseModel) -> str:
    data = model.model_dump()
    width = max((len(k) for k in data), default=0)
    return "\n".join(f"{key.ljust(width)}  {cell(getattr(model, key))}" for key in data)


def as_json(model: BaseModel) -> str:
    return model.model_dump_json(indent=2)
