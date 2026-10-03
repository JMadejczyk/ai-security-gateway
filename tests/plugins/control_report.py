"""``control(<id>, outcome)`` markers: what each test proves, the coverage check and the report.

A test declares the control behaviour it asserts::

    @pytest.mark.control("secrets", "deny")
    async def test_api_key_in_the_prompt_is_blocked(...): ...

    pytest.param(..., marks=pytest.mark.control("pii", "redact"))  # one case of a table

``outcome`` is ``allow`` (the control let a legitimate call through), ``deny`` (it blocked),
``redact``, ``require_approval`` or ``log_only`` (it recorded without enforcing). Every
outcome other than ``allow`` must be one the control's catalog entry supports (``deny`` is
the ``block`` mode), so ``intent_judge`` cannot claim ``deny`` and ``authn`` cannot claim
``redact``. Unknown ids, unsupported outcomes and malformed markers are a usage error at
collection time, before anything runs.

``--control-coverage`` (set by ``make test``) makes the run fail unless every catalog control
has at least one passing ``allow`` test and one passing deny-side test: ``deny``,
``redact`` or ``require_approval``, as far as the control supports them (``log_only`` never
counts). A subset run without the flag only reports. ``--control-report=DIR`` writes
``DIR/controls.md`` and ``DIR/controls.json``. With pytest-html, the same table is embedded
in the HTML report's summary.

Each claim is also recorded as a ``control`` property of its test case in the JUnit XML, so
``python tests/plugins/control_report.py --junit reports/junit.xml --out reports`` rebuilds
``controls.md``/``controls.json`` from the last run without running anything (``make report``).
"""

import argparse
import html
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

import pytest
from pydantic import BaseModel, ConfigDict, Field

from gateway.core.catalog import CONTROL_CATALOG, ControlSpec
from gateway.core.types import Channel, ControlMode, Stage

MARKER: Final = "control"
COVERAGE_OPTION: Final = "--control-coverage"
REPORT_OPTION: Final = "--control-report"
JUNIT_PROPERTY: Final = "control"
MAX_EXAMPLES: Final = 3
PLUGIN_NAME: Final = "control-coverage"


class Outcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REDACT = "redact"
    REQUIRE_APPROVAL = "require_approval"
    LOG_ONLY = "log_only"


OUTCOME_OF_MODE: Final[dict[ControlMode, Outcome]] = {
    ControlMode.BLOCK: Outcome.DENY,
    ControlMode.REDACT: Outcome.REDACT,
    ControlMode.REQUIRE_APPROVAL: Outcome.REQUIRE_APPROVAL,
    ControlMode.LOG_ONLY: Outcome.LOG_ONLY,
}


def supported_outcomes(spec: ControlSpec) -> tuple[Outcome, ...]:
    """``allow`` plus the outcome of every mode the control supports, in catalog order."""
    return (Outcome.ALLOW, *(OUTCOME_OF_MODE[mode] for mode in spec.modes))


def deny_side(spec: ControlSpec) -> tuple[Outcome, ...]:
    """The enforcing outcomes that satisfy the coverage check's deny side."""
    return tuple(o for o in supported_outcomes(spec) if o not in {Outcome.ALLOW, Outcome.LOG_ONLY})


class MarkerError(ValueError):
    """A ``control`` marker that names no catalog control or an unsupported outcome."""


class ControlClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    control: str
    outcome: Outcome

    @classmethod
    def parse(cls, args: Sequence[object], kwargs: dict[str, object]) -> Self:
        """The claim of one ``control(<id>, outcome)`` marker; `MarkerError` when invalid."""
        values = [*args, *(kwargs[k] for k in ("outcome",) if k in kwargs)]
        if set(kwargs) - {"outcome"} or len(values) != 2:
            msg = "use control(<id>, <outcome>)"
            raise MarkerError(msg)
        control, outcome = values
        if not isinstance(control, str) or control not in CONTROL_CATALOG:
            msg = f"unknown control id {control!r} (catalog: {', '.join(CONTROL_CATALOG)})"
            raise MarkerError(msg)
        spec = CONTROL_CATALOG[control]
        allowed = supported_outcomes(spec)
        if outcome not in {o.value for o in allowed}:
            msg = (
                f"control {control!r} cannot have outcome {outcome!r} "
                f"(supported: {', '.join(allowed)})"
            )
            raise MarkerError(msg)
        return cls(control=control, outcome=Outcome(str(outcome)))

    @classmethod
    def from_property(cls, value: str) -> Self:
        control, _, outcome = value.partition(":")
        return cls.parse((control, outcome), {})

    def as_property(self) -> str:
        return f"{self.control}:{self.outcome}"


class TestStatus(StrEnum):
    PASSED = "passed"
    SKIPPED = "skipped"
    FAILED = "failed"


_STATUS_RANK: Final = {TestStatus.PASSED: 0, TestStatus.SKIPPED: 1, TestStatus.FAILED: 2}


class ClaimedTest(BaseModel):
    """One test that ran, the claims its markers make and how it ended."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodeid: str
    claims: tuple[ControlClaim, ...]
    status: TestStatus


class Tally(BaseModel):
    """How the tests claiming one outcome of one control ended."""

    model_config = ConfigDict(extra="forbid")

    passed: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped

    def add(self, status: TestStatus) -> None:
        setattr(self, status.value, getattr(self, status.value) + 1)


class ControlRow(BaseModel):
    """One catalog control: its metadata, the claims made about it, and whether it is covered."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    stages: list[str]
    channels: list[str]
    modes: list[str]
    mandatory: bool
    supported: list[Outcome]
    deny_side: list[Outcome]
    outcomes: dict[Outcome, Tally]
    examples: dict[Outcome, list[str]] = Field(default_factory=dict[Outcome, list[str]])

    @classmethod
    def empty(cls, spec: ControlSpec) -> Self:
        supported = supported_outcomes(spec)
        return cls(
            id=spec.id,
            kind=spec.kind.value,
            stages=[s.value for s in Stage if s in spec.stages],  # pipeline order: pre, post
            channels=[c.value for c in Channel if c in spec.channels],
            modes=[m.value for m in spec.modes],
            mandatory=spec.mandatory,
            supported=list(supported),
            deny_side=list(deny_side(spec)),
            outcomes={o: Tally() for o in supported},
        )

    def record(self, outcome: Outcome, nodeid: str, status: TestStatus) -> None:
        self.outcomes[outcome].add(status)
        if status is TestStatus.PASSED:
            examples = self.examples.setdefault(outcome, [])
            if len(examples) < MAX_EXAMPLES:
                examples.append(nodeid)

    @property
    def has_allow(self) -> bool:
        return self.outcomes[Outcome.ALLOW].passed > 0

    @property
    def has_deny(self) -> bool:
        return any(self.outcomes[o].passed > 0 for o in self.deny_side)

    @property
    def gaps(self) -> list[str]:
        missing: list[str] = []
        if not self.has_allow:
            missing.append("no passing allow test")
        if not self.has_deny:
            missing.append(f"no passing {'/'.join(self.deny_side)} test")
        return missing

    def example(self, outcomes: Iterable[Outcome]) -> str | None:
        for outcome in outcomes:
            if found := self.examples.get(outcome):
                return found[0]
        return None


class ControlsReport(BaseModel):
    """Per-control coverage of one run: what ``controls.md``/``controls.json`` hold."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    source: str  # "pytest run" or the JUnit file it was rebuilt from
    enforced: bool  # whether gaps failed the run (--control-coverage)
    claimed_tests: int
    rows: list[ControlRow]

    @classmethod
    def build(cls, tests: Iterable[ClaimedTest], *, source: str, enforced: bool) -> Self:
        rows = {control_id: ControlRow.empty(spec) for control_id, spec in CONTROL_CATALOG.items()}
        claimed = 0
        for test in tests:
            claimed += 1
            for claim in test.claims:
                rows[claim.control].record(claim.outcome, test.nodeid, test.status)
        return cls(
            generated_at=datetime.now(UTC),
            source=source,
            enforced=enforced,
            claimed_tests=claimed,
            rows=list(rows.values()),
        )

    @property
    def gaps(self) -> dict[str, list[str]]:
        return {row.id: row.gaps for row in self.rows if row.gaps}

    @property
    def covered(self) -> int:
        return sum(1 for row in self.rows if not row.gaps)

    def headline(self) -> str:
        total = len(self.rows)
        if not self.gaps:
            return (
                f"all {total} catalog controls have a passing allow and deny-side test "
                f"({self.claimed_tests} tests carry control markers)"
            )
        return f"{self.covered}/{total} catalog controls covered; gaps: " + "; ".join(
            f"{control}: {', '.join(missing)}" for control, missing in self.gaps.items()
        )


# ----------------------------------------------------------------------------- writers


_COLUMNS: Final = tuple(Outcome)


def _cell(row: ControlRow, outcome: Outcome) -> str:
    """``passed`` when every claiming test passed, else ``passed/total``; n/a if unsupported."""
    if outcome not in row.outcomes:
        return "n/a"
    tally = row.outcomes[outcome]
    return str(tally.passed) if tally.passed == tally.total else f"{tally.passed}/{tally.total}"


def _status(row: ControlRow) -> str:
    return "covered" if not row.gaps else "GAP: " + ", ".join(row.gaps)


class MarkdownReport:
    def __init__(self, report: ControlsReport) -> None:
        self._report = report

    def render(self) -> str:
        report = self._report
        check = "enforced (--control-coverage)" if report.enforced else "not enforced (subset run)"
        lines = [
            "# Control coverage",
            "",
            f"Generated {report.generated_at:%Y-%m-%d %H:%M:%S} UTC from {report.source}. "
            f"Coverage check: {check}.",
            "",
            f"**{report.headline()}.**",
            "",
            "Cells count passing tests per outcome (`passed/total` when some did not pass; "
            "`n/a`: the control has no such mode). The deny side is `deny`, `redact` or "
            "`require_approval`, as the control supports them.",
            "",
            "| Control | Kind | Stage | Channel | Modes | Mandatory | "
            + " | ".join(o.value for o in _COLUMNS)
            + " | Status |",
            "| --- | --- | --- | --- | --- | --- | "
            + " | ".join("---:" for _ in _COLUMNS)
            + " | --- |",
        ]
        for row in report.rows:
            cells = " | ".join(_cell(row, o) for o in _COLUMNS)
            lines.append(
                f"| `{row.id}` | {row.kind} | {'+'.join(row.stages)} | {', '.join(row.channels)} "
                f"| {', '.join(row.modes)} | {'yes' if row.mandatory else 'no'} | {cells} "
                f"| {_status(row)} |"
            )
        lines += [
            "",
            "## Example tests",
            "",
            "| Control | Allow | Deny side |",
            "| --- | --- | --- |",
        ]
        for row in report.rows:
            allow = row.example([Outcome.ALLOW])
            deny = row.example(row.deny_side)
            lines.append(f"| `{row.id}` | {_code(allow)} | {_code(deny)} |")
        return "\n".join(lines) + "\n"


def _code(nodeid: str | None) -> str:
    return f"`{nodeid}`" if nodeid else "none"


class HtmlReport:
    """The table for pytest-html's summary section (inline styles: the report is one file)."""

    def __init__(self, report: ControlsReport) -> None:
        self._report = report

    def render(self) -> str:
        report = self._report
        style = "border:1px solid #ccc;padding:2px 6px;"
        head = "".join(
            f'<th style="{style}">{html.escape(h)}</th>'
            for h in (
                *("Control", "Kind", "Stage", "Channel", "Modes", "Mandatory"),
                *(o.value for o in _COLUMNS),
                *("Status", "Example (allow / deny side)"),
            )
        )
        body: list[str] = []
        for row in report.rows:
            colour = "#e6f4ea" if not row.gaps else "#fce8e6"
            examples = " / ".join(
                html.escape(e or "none")
                for e in (row.example([Outcome.ALLOW]), row.example(row.deny_side))
            )
            cells = [
                f"<code>{html.escape(row.id)}</code>",
                html.escape(row.kind),
                html.escape("+".join(row.stages)),
                html.escape(", ".join(row.channels)),
                html.escape(", ".join(row.modes)),
                "yes" if row.mandatory else "no",
                *(html.escape(_cell(row, o)) for o in _COLUMNS),
                html.escape(_status(row)),
                f'<span style="font-size:smaller">{examples}</span>',
            ]
            body.append(
                f'<tr style="background:{colour}">'
                + "".join(f'<td style="{style}">{c}</td>' for c in cells)
                + "</tr>"
            )
        return (
            '<div class="control-coverage"><h2>Control coverage</h2>'
            f"<p><strong>{html.escape(report.headline())}</strong></p>"
            f'<table style="border-collapse:collapse;font-size:12px"><tr>{head}</tr>'
            + "".join(body)
            + "</table></div>"
        )


def write_reports(report: ControlsReport, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    markdown, data = directory / "controls.md", directory / "controls.json"
    markdown.write_text(MarkdownReport(report).render())
    data.write_text(report.model_dump_json(indent=2) + "\n")
    return markdown, data


# ----------------------------------------------------------------------------- pytest


class ControlCoverage:
    """Collects the claims of the tests that run and turns them into the report."""

    def __init__(self, config: pytest.Config) -> None:
        self._config = config
        self._claims: dict[str, tuple[ControlClaim, ...]] = {}
        self._status: dict[str, TestStatus] = {}
        self.report: ControlsReport | None = None

    @property
    def enforced(self) -> bool:
        return bool(self._config.getoption(COVERAGE_OPTION))

    @pytest.hookimpl(tryfirst=True)  # before -m/-k deselection: every marker is validated
    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        errors: list[str] = []
        for item in items:
            claims: list[ControlClaim] = []
            for marker in item.iter_markers(MARKER):
                try:
                    claims.append(ControlClaim.parse(marker.args, dict(marker.kwargs)))
                except MarkerError as exc:
                    errors.append(f"{item.nodeid}: {exc}")
            if claims:
                unique = tuple(dict.fromkeys(claims))
                self._claims[item.nodeid] = unique
                item.user_properties.extend((JUNIT_PROPERTY, c.as_property()) for c in unique)
        if errors:
            raise pytest.UsageError("invalid control markers:\n  " + "\n  ".join(errors))

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.nodeid not in self._claims:
            return
        if report.failed:
            status = TestStatus.FAILED
        elif report.skipped:
            status = TestStatus.SKIPPED
        elif report.when == "call":
            status = TestStatus.PASSED
        else:
            return
        previous = self._status.get(report.nodeid)
        if previous is None or _STATUS_RANK[status] > _STATUS_RANK[previous]:
            self._status[report.nodeid] = status

    def build(self) -> ControlsReport:
        tests = (
            ClaimedTest(nodeid=nodeid, claims=self._claims[nodeid], status=status)
            for nodeid, status in self._status.items()
        )
        return ControlsReport.build(tests, source="pytest run", enforced=self.enforced)

    @pytest.hookimpl(tryfirst=True)  # before pytest-html renders its report
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if session.config.getoption("collectonly"):
            return
        self.report = report = self.build()
        if directory := self._config.getoption(REPORT_OPTION):
            write_reports(report, Path(directory))
        if self.enforced and report.gaps and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        report = self.report
        if report is None or not (self.enforced or report.claimed_tests):
            return
        terminalreporter.section("control coverage")
        if not report.gaps:
            terminalreporter.write_line(report.headline(), green=True)
            return
        summary = (
            f"{report.covered}/{len(report.rows)} catalog controls have a passing allow and "
            "deny-side test"
        )
        if not self.enforced:  # a subset run: report, never fail
            terminalreporter.write_line(f"{summary} (not enforced without {COVERAGE_OPTION})")
            return
        for control, missing in report.gaps.items():
            terminalreporter.write_line(f"{control}: {', '.join(missing)}", red=True)
        terminalreporter.write_line(f"control coverage FAILED: {summary}", red=True, bold=True)

    @pytest.hookimpl(optionalhook=True)  # pytest-html
    def pytest_html_results_summary(self, prefix: list[str]) -> None:
        report = self.report if self.report is not None else self.build()
        prefix.append(HtmlReport(report).render())


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("control coverage")
    group.addoption(
        COVERAGE_OPTION,
        action="store_true",
        default=False,
        help="fail the run unless every catalog control has a passing allow and deny-side test",
    )
    group.addoption(
        REPORT_OPTION,
        metavar="DIR",
        default=None,
        help="write DIR/controls.md and DIR/controls.json (per-control coverage)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER}(id, outcome): the test proves catalog control `id` reaches `outcome` "
        f"({', '.join(Outcome)}); drives the per-control report and coverage check",
    )
    config.pluginmanager.register(ControlCoverage(config), PLUGIN_NAME)


# ------------------------------------------------------------------- rebuild from JUnit


def tests_from_junit(path: Path) -> list[ClaimedTest]:
    """The claimed tests of a JUnit file this plugin's run wrote (``control`` properties)."""
    tests: list[ClaimedTest] = []
    for case in ET.parse(path).getroot().iter("testcase"):  # noqa: S314 -- our own file
        claims = tuple(
            ControlClaim.from_property(prop.get("value", ""))
            for prop in case.iter("property")
            if prop.get("name") == JUNIT_PROPERTY
        )
        if not claims:
            continue
        if case.find("failure") is not None or case.find("error") is not None:
            status = TestStatus.FAILED
        elif case.find("skipped") is not None:
            status = TestStatus.SKIPPED
        else:
            status = TestStatus.PASSED
        nodeid = f"{case.get('classname', '')}::{case.get('name', '')}"
        tests.append(ClaimedTest(nodeid=nodeid, claims=claims, status=status))
    return tests


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild controls.md/.json from a JUnit file.")
    parser.add_argument("--junit", type=Path, default=Path("reports/junit.xml"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    if not args.junit.is_file():
        print(f"{args.junit}: not found; run `make test` first", file=sys.stderr)
        return 2
    report = ControlsReport.build(
        tests_from_junit(args.junit), source=str(args.junit), enforced=False
    )
    markdown, data = write_reports(report, args.out)
    print(f"{report.headline()}\nwrote {markdown} and {data}")
    return 1 if report.gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
