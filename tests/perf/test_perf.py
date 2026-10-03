"""Gateway overhead with instantly answering upstreams (SPEC "Test suite": ``tests/perf/``).

`test_perf_report` (marker ``perf``, run by ``make perf``) measures every scenario in every
configuration and writes ``reports/perf.json`` and ``reports/perf.md``. Iterations come from
``ACL_PERF_ITERATIONS`` (default 200), ``ACL_PERF_WARMUP`` (20), ``ACL_PERF_MODEL_ITERATIONS``
(20, for the real classifier; fewer for inputs over 1 KB), ``ACL_PERF_SESSIONS`` (16) and
``ACL_PERF_SESSION_CALLS`` (25) for the throughput run. The real ONNX classifier is measured
when ``models/cache`` holds the pinned model (``make models``); the report says when it did not.

`test_perf_smoke` runs in the default suite: a few iterations of two scenarios, written to a
temporary directory. It proves the report is produced and that the deterministic controls are
not wildly over budget (p95 < 50 ms, ten times the SPEC target, so a loaded CI box does not
flake); the SPEC target itself is judged by the full run.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from injection_kit import MODELS_DIR, REPO_ROOT, real_model_present
from perf_bench import SCENARIOS, measure, measure_concurrent, perf_gateway
from perf_report import (
    Configuration,
    Environment,
    PerfReport,
    PerfReportWriter,
    RunSettings,
)

from gateway.injection.classifier import InjectionClassifier, OnnxInjectionClassifier
from gateway.injection.manifest import ModelVerificationError

SMOKE_BUDGET_MS = 50.0
SOURCES_AGREE_MS = 0.01  # histogram vs audit overhead: the same two clock readings


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class RunPlan:
    scenarios: tuple[str, ...]
    configurations: tuple[Configuration, ...]
    concurrent: tuple[tuple[str, Configuration], ...]
    iterations: int
    warmup: int
    model_iterations: int
    sessions: int
    session_calls: int

    def iterations_for(self, scenario: str, configuration: Configuration) -> tuple[int, int]:
        """(iterations, warm-up): the real model runs at a few KB/s, so it gets fewer, and
        fewer still the more text a call carries."""
        if configuration is not Configuration.REAL_MODEL:
            return self.iterations, self.warmup
        size = max(SCENARIOS[scenario].classified_bytes, 1024)
        return max(3, self.model_iterations * 1024 // size), 1


async def run(
    plan: RunPlan,
    tmp_path: Path,
    classifier: InjectionClassifier | None,
    real_model: str,
) -> PerfReport:
    report = PerfReport(
        environment=Environment.capture(REPO_ROOT),
        settings=RunSettings(
            iterations=plan.iterations,
            warmup=plan.warmup,
            model_iterations=plan.model_iterations,
            concurrency_sessions=plan.sessions,
            concurrency_calls=plan.session_calls,
            real_model=real_model,
        ),
    )
    for configuration in plan.configurations:
        model = classifier if configuration is Configuration.REAL_MODEL else None
        if configuration is Configuration.REAL_MODEL and model is None:
            continue
        async with perf_gateway(tmp_path, configuration, model) as gateway:
            for name in plan.scenarios:
                iterations, warmup = plan.iterations_for(name, configuration)
                report.results.append(
                    await measure(gateway, SCENARIOS[name], iterations=iterations, warmup=warmup)
                )
            for name, wanted in plan.concurrent:
                if wanted is configuration:
                    report.concurrency.append(
                        await measure_concurrent(
                            gateway,
                            SCENARIOS[name],
                            sessions=plan.sessions,
                            calls=plan.session_calls,
                        )
                    )
    return report


def _real_classifier() -> tuple[InjectionClassifier | None, str]:
    if not real_model_present():
        return None, "not measured: the pinned model is not in models/cache (make models)"
    try:
        return OnnxInjectionClassifier.from_models_dir(MODELS_DIR), "measured"
    except ModelVerificationError as exc:
        return None, f"not measured: {exc}"


def _check_integrity(report: PerfReport) -> None:
    """Every measured call took the allow path, and the audit entries agree with the metrics
    on both the overhead and the deterministic controls' time."""
    for result in report.results:
        if result.configuration is not Configuration.REAL_MODEL:
            assert set(result.decisions) == {"allow"}, (result.scenario, result.decisions)
        if result.overhead_source_max_diff_ms is not None:
            assert result.overhead_source_max_diff_ms < SOURCES_AGREE_MS, result.scenario
        if result.audit_deterministic is not None:
            audited, observed = result.audit_deterministic.p50, result.deterministic.p50
            assert audited == pytest.approx(observed, rel=1e-6), result.scenario
    for result in report.concurrency:
        assert set(result.decisions) == {"allow"}, (result.scenario, result.decisions)


async def test_perf_smoke(tmp_path: Path):
    plan = RunPlan(
        scenarios=("llm_typical", "mcp_sql_query"),
        configurations=(Configuration.DETERMINISTIC, Configuration.FAKE_SEMANTIC),
        concurrent=(("llm_typical", Configuration.DETERMINISTIC),),
        iterations=8,
        warmup=2,
        model_iterations=0,
        sessions=4,
        session_calls=2,
    )
    report = await run(plan, tmp_path, None, "not measured: smoke run")
    json_path, md_path = PerfReportWriter(report).write(tmp_path / "reports")

    assert PerfReport.model_validate_json(json_path.read_text()) == report
    assert "## SPEC target" in md_path.read_text()
    _check_integrity(report)
    typical = report.result("llm_typical", Configuration.DETERMINISTIC)
    assert typical.target is not None
    assert typical.target.measured_p95_ms < SMOKE_BUDGET_MS, typical.target
    assert {"pii", "secrets", "signatures", "budget"} <= set(typical.controls)
    assert "prompt_injection" not in typical.controls  # removed in the deterministic run
    semantic = report.result("llm_typical", Configuration.FAKE_SEMANTIC)
    assert "prompt_injection" in semantic.controls
    sql = report.result("mcp_sql_query", Configuration.DETERMINISTIC)
    assert "sql_guard" in sql.controls


@pytest.mark.perf
async def test_perf_report(tmp_path: Path):
    classifier, real_model = _real_classifier()
    plan = RunPlan(
        scenarios=tuple(SCENARIOS),
        configurations=tuple(Configuration),
        concurrent=(
            ("llm_typical", Configuration.DETERMINISTIC),
            ("llm_typical", Configuration.FAKE_SEMANTIC),
            ("mcp_write_report", Configuration.DETERMINISTIC),
            ("mcp_write_report", Configuration.FAKE_SEMANTIC),
        ),
        iterations=_env_int("ACL_PERF_ITERATIONS", 200),
        warmup=_env_int("ACL_PERF_WARMUP", 20),
        model_iterations=_env_int("ACL_PERF_MODEL_ITERATIONS", 20),
        sessions=_env_int("ACL_PERF_SESSIONS", 16),
        session_calls=_env_int("ACL_PERF_SESSION_CALLS", 25),
    )
    report = await run(plan, tmp_path, classifier, real_model)
    directory = Path(os.environ.get("ACL_PERF_REPORT_DIR", REPO_ROOT / "reports"))
    json_path, md_path = PerfReportWriter(report).write(directory)

    assert PerfReport.model_validate_json(json_path.read_text()) == report
    assert md_path.is_file()
    _check_integrity(report)
