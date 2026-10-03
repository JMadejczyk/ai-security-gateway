"""The perf report: latency samples, percentile summaries, and ``reports/perf.{json,md}``.

`LatencySample` is one measured call. `ScenarioResult.of` summarizes a scenario's samples into
percentiles per measure and per control and checks the SPEC target (deterministic controls,
combined, p95 < 5 ms). `PerfReport` is the whole run with its environment; `PerfReportWriter`
writes it as JSON (round-trips through `PerfReport.model_validate_json`) and as Markdown.
"""

import math
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field

from gateway.core.catalog import control_spec
from gateway.core.types import ControlKind

TARGET_P95_MS: Final = 5.0  # SPEC "Control catalog": deterministic controls, combined
TOP_CONTRIBUTORS: Final = 3
AUDIT_AGREEMENT: Final = 0.01  # relative tolerance between audit and histogram sums
FORMAL_SCENARIO: Final = "llm_typical"  # the SPEC target is stated for the typical prompt


class Configuration(StrEnum):
    """What runs beside the deterministic controls."""

    DETERMINISTIC = "deterministic"  # semantic controls removed from the registry
    FAKE_SEMANTIC = "fake_semantic"  # semantic controls on a marker classifier and a fake judge
    REAL_MODEL = "real_model"  # semantic controls on the pinned ONNX classifier, fake judge


class Verdict(StrEnum):
    WITHIN = "PASS"  # under the target
    OVER = "FAIL"


def is_deterministic(control_id: str) -> bool:
    return control_spec(control_id).kind is ControlKind.DETERMINISTIC


def percentile(ordered: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default); ``ordered`` is sorted."""
    if not ordered:
        return math.nan
    rank = (len(ordered) - 1) * q
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LatencyStats(_Model):
    """Milliseconds."""

    n: int
    p50: float
    p95: float
    p99: float
    mean: float
    max: float

    @classmethod
    def of(cls, samples: Iterable[float]) -> Self:
        ordered = sorted(samples)
        n = len(ordered)
        return cls(
            n=n,
            p50=percentile(ordered, 0.50),
            p95=percentile(ordered, 0.95),
            p99=percentile(ordered, 0.99),
            mean=sum(ordered) / n if n else math.nan,
            max=ordered[-1] if n else math.nan,
        )


@dataclass(frozen=True, slots=True)
class LatencySample:
    """One measured call, in milliseconds.

    ``overhead_ms`` is what ``acl_overhead_seconds{channel}`` observed for the call (gateway
    time minus upstream); ``audit_overhead_ms`` is ``latency_ms.total - latency_ms.upstream``
    from its audit entry. Calls outside the pipeline (``tools/list``) have neither.
    ``controls`` is the time per control from ``acl_control_latency_seconds``, every stage
    (pre, post, sealing) summed. ``audit_deterministic_ms`` is the same sum taken over the
    audit entry's ``latency_ms.controls`` (every stage summed there too), as a cross-check.
    """

    e2e_ms: float
    upstream_ms: float
    decision: str
    overhead_ms: float | None = None
    audit_overhead_ms: float | None = None
    audit_deterministic_ms: float | None = None
    controls: Mapping[str, float] = field(default_factory=dict[str, float])

    @property
    def client_overhead_ms(self) -> float:
        """What the agent waited for beyond the upstream: HTTP, routing, pipeline."""
        return self.e2e_ms - self.upstream_ms

    @property
    def deterministic_ms(self) -> float:
        return sum(ms for c, ms in self.controls.items() if is_deterministic(c))

    @property
    def semantic_ms(self) -> float:
        return sum(ms for c, ms in self.controls.items() if not is_deterministic(c))


class TargetCheck(_Model):
    """SPEC target: deterministic controls combined p95 < 5 ms."""

    target_p95_ms: float = TARGET_P95_MS
    measured_p95_ms: float
    verdict: Verdict
    top_contributors: tuple[tuple[str, float], ...]  # (control, p95 ms), largest first


class ScenarioResult(_Model):
    scenario: str
    configuration: Configuration
    channel: str
    description: str
    iterations: int
    warmup: int
    decisions: dict[str, int]
    e2e: LatencyStats
    upstream: LatencyStats
    client_overhead: LatencyStats
    overhead: LatencyStats | None  # acl_overhead_seconds; None outside the pipeline
    deterministic: LatencyStats
    # The same sum as the audit entries report it (`latency_ms.controls`): a cross-check.
    audit_deterministic: LatencyStats | None
    semantic: LatencyStats
    controls: dict[str, LatencyStats]
    target: TargetCheck | None  # None for calls the controls never see (tools/list)
    # Largest |histogram overhead - audit overhead| over the samples: the two sources agree.
    overhead_source_max_diff_ms: float | None

    @classmethod
    def of(
        cls,
        *,
        scenario: str,
        configuration: Configuration,
        channel: str,
        description: str,
        warmup: int,
        samples: Sequence[LatencySample],
    ) -> Self:
        decisions: dict[str, int] = {}
        for sample in samples:
            decisions[sample.decision] = decisions.get(sample.decision, 0) + 1
        control_ids = sorted({c for s in samples for c in s.controls})
        controls = {
            c: LatencyStats.of(s.controls.get(c, 0.0) for s in samples) for c in control_ids
        }
        overheads = [s.overhead_ms for s in samples if s.overhead_ms is not None]
        diffs = [
            abs(s.overhead_ms - s.audit_overhead_ms)
            for s in samples
            if s.overhead_ms is not None and s.audit_overhead_ms is not None
        ]
        deterministic = LatencyStats.of(s.deterministic_ms for s in samples)
        audited = [
            s.audit_deterministic_ms for s in samples if s.audit_deterministic_ms is not None
        ]
        return cls(
            scenario=scenario,
            configuration=configuration,
            channel=channel,
            description=description,
            iterations=len(samples),
            warmup=warmup,
            decisions=decisions,
            e2e=LatencyStats.of(s.e2e_ms for s in samples),
            upstream=LatencyStats.of(s.upstream_ms for s in samples),
            client_overhead=LatencyStats.of(s.client_overhead_ms for s in samples),
            overhead=LatencyStats.of(overheads) if overheads else None,
            deterministic=deterministic,
            audit_deterministic=LatencyStats.of(audited) if audited else None,
            semantic=LatencyStats.of(s.semantic_ms for s in samples),
            controls=controls,
            target=_target(deterministic, controls) if controls else None,
            overhead_source_max_diff_ms=max(diffs) if diffs else None,
        )


def _target(deterministic: LatencyStats, controls: Mapping[str, LatencyStats]) -> TargetCheck:
    ranked = sorted(
        ((c, stats.p95) for c, stats in controls.items() if is_deterministic(c)),
        key=lambda item: item[1],
        reverse=True,
    )
    return TargetCheck(
        measured_p95_ms=deterministic.p95,
        verdict=Verdict.WITHIN if deterministic.p95 < TARGET_P95_MS else Verdict.OVER,
        top_contributors=tuple(ranked[:TOP_CONTRIBUTORS]),
    )


class ConcurrencyResult(_Model):
    """Many sessions at once on one event loop: throughput, and latency under that load."""

    scenario: str
    configuration: Configuration
    sessions: int
    calls_per_session: int
    wall_s: float
    throughput_rps: float
    decisions: dict[str, int]
    e2e: LatencyStats
    mean_overhead_ms: float  # acl_overhead_seconds sum / count over the run


class Environment(_Model):
    timestamp: datetime
    git_commit: str
    git_dirty: bool
    python: str
    platform: str
    cpu_model: str
    cpu_count: int | None
    # 1/5/15-minute load average when the run started: other work on the box skews the numbers.
    load_average: tuple[float, float, float] | None

    @classmethod
    def capture(cls, repo: Path) -> Self:
        status = _run(["git", "status", "--porcelain"], repo)
        return cls(
            timestamp=datetime.now(UTC),
            git_commit=_run(["git", "rev-parse", "HEAD"], repo) or "unknown",
            git_dirty=bool(status),
            python=sys.version.split()[0],
            platform=platform.platform(),
            cpu_model=_cpu_model(),
            cpu_count=os.cpu_count(),
            load_average=_load_average(),
        )


def _run(command: list[str], cwd: Path) -> str:
    try:
        done = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            command, cwd=cwd, capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _load_average() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except OSError:
        return None


def _cpu_model() -> str:
    if sys.platform == "darwin":
        if brand := _run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], Path.cwd()):
            return brand
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


class RunSettings(_Model):
    iterations: int
    warmup: int
    model_iterations: int
    concurrency_sessions: int
    concurrency_calls: int
    real_model: str  # "measured", or why it was not


class PerfReport(_Model):
    environment: Environment
    settings: RunSettings
    target_p95_ms: float = TARGET_P95_MS
    results: list[ScenarioResult] = Field(default_factory=list[ScenarioResult])
    concurrency: list[ConcurrencyResult] = Field(default_factory=list[ConcurrencyResult])

    def result(self, scenario: str, configuration: Configuration) -> ScenarioResult:
        for result in self.results:
            if result.scenario == scenario and result.configuration is configuration:
                return result
        msg = f"no result for {scenario} / {configuration}"
        raise KeyError(msg)


# ------------------------------------------------------------------------------ writers

_NOTES: Final = """\
## How to read this

- **e2e**: the agent's request through the agent app (in-process ASGI transport, no sockets),
  until the full response body is read. **upstream**: wall time inside the instant upstream
  fakes, LLM and MCP alike. **client overhead** = e2e - upstream: everything the gateway added.
- **overhead** is `acl_overhead_seconds{channel}` for the call (pipeline time minus the
  upstream call), cross-checked per call against the audit entry's
  `latency_ms.total - latency_ms.upstream`; the largest difference is in perf.json
  (`overhead_source_max_diff_ms`). `tools/list` does not run the pipeline, so it has neither.
- **det. controls** / **sem. controls**: per call, the sum over deterministic / semantic
  controls of `acl_control_latency_seconds`, every stage (pre, post, sealing) included.
  `authz` is base authorization plus session restrictions, timed per interaction.
  `sql_guard` includes its `explain` round trip to the (instant) SQL server: a short-lived MCP
  session of its own (initialize, notification, call, delete) per statement.
- Configurations: **deterministic** removes every semantic control from the registry (the
  `tools/list` screens still run, on the marker classifier); **fake_semantic** runs them on
  the marker classifier and the fake judge (the plumbing: text extraction, prose filtering,
  windows, judge request building, without inference); **real_model** runs the pinned ONNX
  classifier with the fake judge. Every call carries new text, so the classifier's score
  cache never answers for it: the real-model numbers are the cold, worst case.
- A control's time is wall time around its `evaluate`, so an `await` inside it (budget
  reservation, `sql_guard`'s planner call, Presidio off-loop for long text) also bills whatever
  else the event loop or the machine ran meanwhile. Spikes on `budget` in the real-model rows
  are that, not budget work; read those rows on a quiet machine (see the load average).
- Presidio runs without an NLP engine (regex and checksum recognizers only), so `pii` is
  inside the deterministic budget here.
- Sequential calls, one session per scenario; warm-up calls discarded. Budgets are raised in
  the perf policy copy so a long run is not cut off by `per_session.tool_calls`; the budget
  control still reserves and settles on every call.
"""


def _ms(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.2f}"


def _stats(stats: LatencyStats | None) -> str:
    return "n/a" if stats is None else f"{_ms(stats.p50)} / {_ms(stats.p95)} / {_ms(stats.p99)}"


def _decisions(decisions: Mapping[str, int]) -> str:
    return ", ".join(f"{d} {n}" for d, n in sorted(decisions.items()))


class PerfReportWriter:
    """Writes one `PerfReport` as ``perf.json`` and ``perf.md`` into a directory."""

    def __init__(self, report: PerfReport) -> None:
        self._report = report

    def write(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        json_path, md_path = directory / "perf.json", directory / "perf.md"
        json_path.write_text(self._report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        md_path.write_text(self.markdown(), encoding="utf-8")
        return json_path, md_path

    def markdown(self) -> str:
        report = self._report
        lines = [
            "# Gateway performance",
            "",
            *self._environment(),
            "",
            "Written by `tests/perf/` (`make perf`). Milliseconds, as p50 / p95 / p99.",
            "",
            *self._target_section(),
            "",
            *self._interpretation(),
            "",
            *self._scenario_table(),
            "",
            *self._control_tables(),
        ]
        if report.concurrency:
            lines += ["", *self._concurrency_table()]
        lines += ["", _NOTES]
        return "\n".join(lines)

    def _environment(self) -> list[str]:
        env, settings = self._report.environment, self._report.settings
        dirty = " (uncommitted changes)" if env.git_dirty else ""
        return [
            f"- Run: {env.timestamp.isoformat(timespec='seconds')}, commit "
            f"`{env.git_commit[:12]}`{dirty}",
            f"- Machine: {env.cpu_model}, {env.cpu_count} CPUs, {env.platform}, "
            f"Python {env.python}",
            "- Load average at start (1/5/15 min): "
            + (" / ".join(f"{x:.1f}" for x in env.load_average) if env.load_average else "n/a")
            + " (other work on the machine inflates every number, the real model's most)",
            f"- Iterations: {settings.iterations} per scenario after {settings.warmup} warm-up "
            f"calls; real model: {settings.model_iterations} or fewer for large inputs "
            f"({settings.real_model})",
        ]

    def _target_section(self) -> list[str]:
        lines = [
            f"## SPEC target: deterministic controls combined, p95 < {_ms(TARGET_P95_MS)} ms",
            "",
            "| Scenario | Configuration | det. controls p95 | Verdict | Top contributors (p95) "
            "| det. p95 as audited |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in self._report.results:
            if r.target is None:
                continue
            top = ", ".join(f"{c} {_ms(ms)}" for c, ms in r.target.top_contributors)
            audited = _ms(r.audit_deterministic.p95) if r.audit_deterministic else "n/a"
            lines.append(
                f"| {r.scenario} | {r.configuration} | {_ms(r.target.measured_p95_ms)} "
                f"| **{r.target.verdict}** | {top} | {audited} |"
            )
        lines += [
            "",
            "The formal target is the typical prompt (`llm_typical`); the other rows apply the "
            "same bar to every scenario. *det. p95 as audited* sums the audit entry's "
            "`latency_ms.controls` instead of the histogram: the two should match.",
        ]
        return lines

    def _find(self, scenario: str, configuration: Configuration) -> ScenarioResult | None:
        try:
            return self._report.result(scenario, configuration)
        except KeyError:
            return None

    def _interpretation(self) -> list[str]:
        lines = ["## Interpretation", ""]
        results = self._report.results
        formal = self._find(FORMAL_SCENARIO, Configuration.DETERMINISTIC)
        if formal is not None and formal.target is not None:
            lines.append(
                f"- SPEC target on the typical prompt (`{FORMAL_SCENARIO}`, deterministic "
                f"controls only): p95 {_ms(formal.target.measured_p95_ms)} ms against "
                f"{_ms(TARGET_P95_MS)} ms, **{formal.target.verdict}**."
            )
        failed = [r for r in results if r.target is not None and r.target.verdict is Verdict.OVER]
        for r in failed:
            assert r.target is not None
            top = ", ".join(f"`{c}` {_ms(ms)} ms" for c, ms in r.target.top_contributors)
            lines.append(
                f"- Over the bar: `{r.scenario}` ({r.configuration}), deterministic p95 "
                f"{_ms(r.target.measured_p95_ms)} ms; largest: {top}."
            )
        if not failed:
            lines.append("- Every scenario keeps its deterministic controls under the bar.")
        plumbing = [
            (name, semantic.overhead.p50 - plain.overhead.p50)
            for name in dict.fromkeys(r.scenario for r in results)
            if (plain := self._find(name, Configuration.DETERMINISTIC)) is not None
            and (semantic := self._find(name, Configuration.FAKE_SEMANTIC)) is not None
            and plain.overhead is not None
            and semantic.overhead is not None
        ]
        if plumbing:
            deltas = ", ".join(f"`{name}` {delta:+.2f}" for name, delta in plumbing)
            lines.append(
                "- Semantic plumbing without inference (fake_semantic minus deterministic, "
                f"overhead p50, ms): {deltas}."
            )
        real = [r for r in results if r.configuration is Configuration.REAL_MODEL]
        shares = [
            f"`{r.scenario}` {_ms(r.controls['prompt_injection'].p50)} ms "
            f"({100 * r.controls['prompt_injection'].p50 / r.overhead.p50:.0f}%)"
            for r in real
            if "prompt_injection" in r.controls and r.overhead is not None and r.overhead.p50 > 0
        ]
        if shares:
            lines.append(
                "- With the real ONNX classifier, `prompt_injection` dominates (p50, and its "
                f"share of the overhead p50): {', '.join(shares)}. It is cold here (new text "
                "every call); re-sent chat history is answered from its score cache."
            )
        elif not real:
            lines.append(f"- Real classifier: {self._report.settings.real_model}.")
        if formal is not None and formal.overhead is not None:
            outside = formal.client_overhead.p50 - formal.overhead.p50
            lines.append(
                f"- Outside the pipeline (`{FORMAL_SCENARIO}`, client overhead minus overhead, "
                f"p50): {_ms(outside)} ms of HTTP handling, routing and response building."
            )
            stream = self._find(f"{FORMAL_SCENARIO}_stream", Configuration.DETERMINISTIC)
            if stream is not None:
                extra = stream.client_overhead.p50 - formal.client_overhead.p50
                lines.append(f"- SSE re-emission (`stream: true`) adds {_ms(extra)} ms at p50.")
        if formal is not None and formal.audit_deterministic is not None:
            audited, observed = formal.audit_deterministic.p50, formal.deterministic.p50
            agree = math.isclose(audited, observed, rel_tol=AUDIT_AGREEMENT)
            lines.append(
                f"- Audit vs metrics (`{FORMAL_SCENARIO}`, deterministic controls, p50): "
                f"`latency_ms.controls` sums to {_ms(audited)} ms, "
                f"`acl_control_latency_seconds` to {_ms(observed)} ms"
                + (": they agree." if agree else ": **they disagree**.")
            )
        return lines

    def _scenario_table(self) -> list[str]:
        lines = [
            "## Latency per scenario",
            "",
            "| Scenario | Configuration | n | e2e | upstream | client overhead | overhead "
            "(acl_overhead_seconds) | det. controls | sem. controls | Decisions |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {r.scenario} | {r.configuration} | {r.iterations} | {_stats(r.e2e)} "
            f"| {_stats(r.upstream)} | {_stats(r.client_overhead)} | {_stats(r.overhead)} "
            f"| {_stats(r.deterministic)} | {_stats(r.semantic)} | {_decisions(r.decisions)} |"
            for r in self._report.results
        ]
        lines += ["", "Scenarios:", ""]
        seen: set[str] = set()
        for r in self._report.results:
            if r.scenario not in seen:
                seen.add(r.scenario)
                lines.append(f"- `{r.scenario}`: {r.description}")
        return lines

    def _control_tables(self) -> list[str]:
        lines = ["## Per control (p50 / p95 / p99)", ""]
        for configuration in Configuration:
            results = [r for r in self._report.results if r.configuration is configuration]
            control_ids = sorted({c for r in results for c in r.controls})
            if not control_ids:
                continue
            lines += [
                f"### {configuration}",
                "",
                "| Scenario | " + " | ".join(control_ids) + " |",
                "| --- " * (len(control_ids) + 1) + "|",
            ]
            for r in results:
                if not r.controls:
                    continue
                cells = [_stats(r.controls.get(c)) if c in r.controls else "-" for c in control_ids]
                lines.append(f"| {r.scenario} | " + " | ".join(cells) + " |")
            lines.append("")
        return lines

    def _concurrency_table(self) -> list[str]:
        lines = [
            "## Throughput (concurrent sessions, one event loop)",
            "",
            "| Scenario | Configuration | Sessions x calls | Wall s | Calls/s | e2e "
            "| mean overhead | Decisions |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {c.scenario} | {c.configuration} | {c.sessions} x {c.calls_per_session} "
            f"| {c.wall_s:.2f} | {c.throughput_rps:.1f} | {_stats(c.e2e)} "
            f"| {_ms(c.mean_overhead_ms)} | {_decisions(c.decisions)} |"
            for c in self._report.concurrency
        ]
        return lines
