"""A fake DNS for ``egress``: gateway suites never query real DNS.

`FakeResolver` answers from a table and gives every other host one public address
(``PUBLIC_IP``), so the MCP suites' ``https://example.com/`` keeps working offline.
A test makes a host resolve elsewhere, or not at all, with ``resolver.answers[host] = [...]``
(an empty list means unresolvable), or stalls every lookup with ``resolver.stall = True``.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from ipaddress import ip_address
from typing import Final

from gateway.controls.egress import IPAddress

PUBLIC_IP: Final = "93.184.215.14"  # a globally routable unicast address


@dataclass
class FakeResolver:
    answers: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    default: list[str] = field(default_factory=lambda: [PUBLIC_IP])
    stall: bool = False  # never answer: the control's timeout must fire
    calls: list[tuple[str, int]] = field(default_factory=list[tuple[str, int]])

    async def __call__(self, host: str, port: int) -> Sequence[IPAddress]:
        self.calls.append((host, port))
        if self.stall:
            await asyncio.Event().wait()
        return [ip_address(a) for a in self.answers.get(host, self.default)]
