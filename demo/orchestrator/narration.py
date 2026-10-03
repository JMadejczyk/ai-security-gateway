"""The transcript: what each scene shows, as plain lines (ANSI colour only on a terminal).

The ``describe_*`` and ``format_*`` functions are pure, so the wording is unit-tested; the
`Narrator` only adds colour and writes lines to its stream.
"""

import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, TextIO
from urllib.parse import urlencode

from demo.orchestrator.models import (
    AuditEntry,
    ChatReply,
    Check,
    ProbeReply,
    ReloadEvent,
    SceneResult,
    ToolReply,
)

WIDTH: Final = 92
MAX_RESULT_CHARS: Final = 70
GRAFANA_RANGE: Final = {"orgId": "1", "from": "now-30m", "to": "now"}

_STYLES: Final = {
    "title": "\033[1;36m",
    "actor": "\033[1m",
    "ok": "\033[32m",
    "bad": "\033[1;31m",
    "dim": "\033[2m",
    "warn": "\033[33m",
}
_RESET: Final = "\033[0m"


def _short(value: object, limit: int = MAX_RESULT_CHARS) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe_tool_reply(reply: ToolReply) -> str:
    """``allowed: <result>``, or the gateway's refusal with its reason code."""
    waits = (
        f" (throttled first: waited {', '.join(f'{w:.1f}' for w in reply.throttle_waits)} s"
        " on Retry-After)"
        if reply.throttle_waits
        else ""
    )
    if reply.ok:
        return f"ALLOWED -> {_short(reply.result)}{waits}"
    held = f", approval_id={reply.approval_id}" if reply.approval_id else ""
    return f"REFUSED reason_code={reply.reason}{held}{waits}"


def describe_chat_reply(reply: ChatReply) -> str:
    if reply.reason == "ok":
        return f"ANSWERED in {reply.elapsed_s:.1f} s: {_short(reply.answer, 120)!s}"
    return f"REFUSED HTTP {reply.status} reason_code={reply.reason} in {reply.elapsed_s:.1f} s"


def describe_probe(reply: ProbeReply) -> str:
    where = f"{reply.host}:{reply.port}"
    if reply.connected:
        return f"{where} CONNECTED ({', '.join(reply.resolved)})"
    if not reply.resolved:
        return f"{where} no connection: name does not resolve ({reply.error})"
    return f"{where} no connection: {reply.error}"


def describe_audit(entry: AuditEntry) -> str:
    """The audit fields a judge looks for, on one line (never payloads: the log has none)."""
    deciding = ", ".join(f"{v.control}/{v.stage}={v.reason_code}" for v in entry.deciding())
    parts = [
        f"audit: {entry.action} {entry.resource} -> {entry.decision} ({entry.reason_code})",
        f"risk={entry.risk:.2f}",
        f"taint={str(entry.taint).lower()}",
        f"rev={entry.policy_revision}",
    ]
    if deciding:
        parts.append(f"verdicts: {deciding}")
    return "  ".join(parts)


def describe_scope(entry: AuditEntry) -> str:
    return "effective scope: " + (", ".join(entry.effective_scope) or "(nothing)")


def describe_reload(event: ReloadEvent) -> str:
    return (
        f"audit: policy_reload {event.result} {event.previous_revision} -> {event.revision}"
        f" at {event.ts}"
    )


def format_check(check: Check) -> str:
    mark = "PASS" if check.passed else "FAIL"
    expected = check.expected[0] if len(check.expected) == 1 else " | ".join(check.expected)
    if check.passed:
        return f"[{mark}] {check.name}: {check.actual}"
    return f"[{mark}] {check.name}: got {check.actual}, expected {expected}"


def format_summary(results: Sequence[SceneResult]) -> list[str]:
    """One line per scene (status, time, checks), then the total."""
    lines = ["", "=" * WIDTH, "Summary", "-" * WIDTH]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        passed = sum(c.passed for c in result.checks)
        lines.append(
            f"{status}  scene {result.number}  {result.title:<46} {result.elapsed_s:6.1f} s"
            f"  {passed}/{len(result.checks)} checks"
        )
        lines.extend(f"        {format_check(c)}" for c in result.failed_checks)
        if result.error:
            lines.append(f"        error: {result.error}")
    total = sum(r.elapsed_s for r in results)
    ok = sum(r.passed for r in results)
    lines.append("-" * WIDTH)
    lines.append(f"{ok}/{len(results)} scenes as expected, {total:.1f} s in total")
    return lines


def grafana_links(base_url: str, sessions: Mapping[str, str]) -> list[str]:
    """Threats, then one Session trace link per labelled session."""
    base = base_url.rstrip("/")
    links = [f"Threats:        {base}/d/acl-threats?{urlencode(GRAFANA_RANGE)}"]
    for label, session_id in sessions.items():
        query = urlencode({**GRAFANA_RANGE, "var-session_id": session_id})
        links.append(f"Session trace:  {base}/d/acl-session-trace?{query}   ({label})")
    return links


def use_colour(stream: TextIO) -> bool:
    return stream.isatty() and "NO_COLOR" not in os.environ


class Narrator:
    """Writes the transcript. ``colour=None`` decides from the stream (a TTY, no NO_COLOR)."""

    def __init__(self, stream: TextIO | None = None, *, colour: bool | None = None) -> None:
        self._stream = stream or sys.stdout
        self._colour = use_colour(self._stream) if colour is None else colour

    def _style(self, text: str, style: str) -> str:
        return f"{_STYLES[style]}{text}{_RESET}" if self._colour else text

    def line(self, text: str = "", style: str | None = None) -> None:
        self._stream.write((self._style(text, style) if style else text) + "\n")
        self._stream.flush()

    def lines(self, texts: Iterable[str]) -> None:
        for text in texts:
            self.line(text)

    def scene(self, number: int, title: str, story: str) -> None:
        self.line()
        self.line("=" * WIDTH, "title")
        self.line(f"Scene {number}: {title}", "title")
        self.line(story, "dim")
        self.line("-" * WIDTH, "title")

    def act(self, who: str, what: str) -> None:
        self.line(f"{self._style(who, 'actor')}  {what}")

    def outcome(self, text: str, *, good: bool) -> None:
        self.line(f"    {self._style('->', 'ok' if good else 'warn')} {text}")

    def detail(self, text: str) -> None:
        self.line(f"       {self._style(text, 'dim')}")

    def check(self, check: Check) -> None:
        self.line(f"    {self._style(format_check(check), 'ok' if check.passed else 'bad')}")

    def scene_end(self, result: SceneResult) -> None:
        status = "as expected" if result.passed else "DEVIATED"
        self.line(
            f"Scene {result.number} {status} in {result.elapsed_s:.1f} s",
            "ok" if result.passed else "bad",
        )
        if result.error:
            self.line(f"error: {result.error}", "bad")
