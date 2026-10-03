"""The ``control(<id>, outcome)`` plugin (tests/plugins/control_report.py), run with pytester."""

import json
from pathlib import Path

import pytest
from plugins.control_report import ControlsReport, deny_side, main

from gateway.core.catalog import CONTROL_CATALOG

PLUGIN = 'pytest_plugins = ["plugins.control_report"]\n'


@pytest.fixture
def tester(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(PLUGIN)
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    return pytester


def full_coverage(*, broken: str | None = None) -> str:
    """A test module with one allow and one deny-side test per catalog control; the deny-side
    test of ``broken`` fails."""
    lines = ["import pytest", ""]
    for control_id, spec in CONTROL_CATALOG.items():
        deny = deny_side(spec)[0]
        failing = "assert False" if control_id == broken else "pass"
        lines += [
            f'@pytest.mark.control("{control_id}", "allow")',
            f"def test_{control_id}_allow(): pass",
            "",
            f'@pytest.mark.control("{control_id}", "{deny}")',
            f"def test_{control_id}_deny(): {failing}",
            "",
        ]
    return "\n".join(lines)


def report_of(directory: Path) -> dict:
    return json.loads((directory / "controls.json").read_text())


def row(report: dict, control_id: str) -> dict:
    [found] = [r for r in report["rows"] if r["id"] == control_id]
    return found


def test_an_unknown_control_id_is_a_usage_error_at_collection(tester: pytest.Pytester):
    tester.makepyfile(
        """
        import pytest

        @pytest.mark.control("secret", "deny")
        def test_typo(): pass
        """
    )
    result = tester.runpytest("-p", "no:cacheprovider")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*invalid control markers*", "*unknown control id 'secret'*"])
    assert "1 passed" not in result.stdout.str()  # nothing ran


@pytest.mark.parametrize(
    ("marker", "message"),
    [
        ('"intent_judge", "deny"', "*cannot have outcome 'deny'*require_approval*"),
        ('"authn", "redact"', "*cannot have outcome 'redact'*"),
        ('"secrets", "log_only"', "*cannot have outcome 'log_only'*"),
        ('"secrets"', "*use control(<id>, <outcome>)*"),
        ('"secrets", "deny", "extra"', "*use control(<id>, <outcome>)*"),
    ],
)
def test_unsupported_outcomes_and_malformed_markers_are_refused(
    tester: pytest.Pytester, marker: str, message: str
):
    tester.makepyfile(f"import pytest\n\n@pytest.mark.control({marker})\ndef test_x(): pass\n")
    result = tester.runpytest("-p", "no:cacheprovider")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([message])


def test_markers_are_validated_even_on_deselected_tests(tester: pytest.Pytester):
    tester.makepyfile(
        """
        import pytest

        @pytest.mark.slow
        @pytest.mark.control("nope", "deny")
        def test_hidden(): pass
        """
    )
    result = tester.runpytest("-p", "no:cacheprovider", "-m", "not slow", "-W", "ignore")
    assert result.ret == pytest.ExitCode.USAGE_ERROR


def test_a_coverage_gap_fails_the_run_with_a_clear_message(tester: pytest.Pytester):
    tester.makepyfile(
        """
        import pytest

        @pytest.mark.control("secrets", "allow")
        def test_clean(): pass

        @pytest.mark.control("secrets", "redact")
        def test_masked(): pass

        @pytest.mark.control("pii", "log_only")
        def test_recorded(): pass
        """
    )
    result = tester.runpytest("-p", "no:cacheprovider", "--control-coverage")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    out = result.stdout.str()
    assert "3 passed" in out
    assert "secrets:" not in out  # covered: allow + redact
    result.stdout.fnmatch_lines(
        [
            "pii: no passing allow test, no passing deny/redact test",  # log_only: not deny side
            "*intent_judge: no passing allow test, no passing require_approval test*",
            "*control coverage FAILED: 1/15 catalog controls*",
        ]
    )


def test_full_coverage_passes(tester: pytest.Pytester):
    tester.makepyfile(full_coverage())
    result = tester.runpytest("-p", "no:cacheprovider", "--control-coverage")
    assert result.ret == pytest.ExitCode.OK
    result.stdout.fnmatch_lines(["*all 15 catalog controls have a passing allow and deny-side*"])


def test_a_failing_test_does_not_count_as_coverage(tester: pytest.Pytester, tmp_path: Path):
    tester.makepyfile(full_coverage(broken="authn"))
    out = tmp_path / "reports"
    result = tester.runpytest(
        "-p", "no:cacheprovider", "--control-coverage", f"--control-report={out}"
    )
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*authn: no passing deny test*"])
    authn = row(report_of(out), "authn")
    assert authn["outcomes"]["deny"] == {"passed": 0, "failed": 1, "skipped": 0}


def test_a_subset_run_without_the_flag_only_reports(tester: pytest.Pytester):
    tester.makepyfile(
        """
        import pytest

        @pytest.mark.control("egress", "deny")
        def test_one(): pass
        """
    )
    result = tester.runpytest("-p", "no:cacheprovider")
    assert result.ret == pytest.ExitCode.OK
    result.stdout.fnmatch_lines(
        ["*0/15 catalog controls*(not enforced without --control-coverage)"]
    )


def test_report_files_count_outcomes_per_control(tester: pytest.Pytester, tmp_path: Path):
    tester.makepyfile(
        """
        import pytest

        REDACT = pytest.mark.control("pii", "redact")

        @pytest.mark.parametrize(
            "case",
            [
                pytest.param(1, marks=REDACT),
                pytest.param(2, marks=REDACT),
                pytest.param(3, marks=pytest.mark.control("pii", "allow")),
            ],
        )
        def test_table(case): pass

        @REDACT
        def test_skipped(): pytest.skip("no model")

        @pytest.mark.control("pii", "deny")
        @pytest.mark.control("secrets", "deny")
        def test_two_claims(): pass

        def test_unmarked(): pass
        """
    )
    out = tmp_path / "reports"
    result = tester.runpytest("-p", "no:cacheprovider", f"--control-report={out}")
    assert result.ret == pytest.ExitCode.OK
    report = report_of(out)
    assert report["claimed_tests"] == 5
    assert report["enforced"] is False
    pii = row(report, "pii")
    assert pii["outcomes"]["redact"] == {"passed": 2, "failed": 0, "skipped": 1}
    assert pii["outcomes"]["allow"]["passed"] == 1
    assert pii["outcomes"]["deny"]["passed"] == 1
    assert pii["examples"]["allow"] == [
        "test_report_files_count_outcomes_per_control.py::test_table[3]"
    ]
    assert row(report, "secrets")["outcomes"]["deny"]["passed"] == 1
    assert pii["supported"] == ["allow", "deny", "redact", "log_only"]
    assert pii["deny_side"] == ["deny", "redact"]
    markdown = (out / "controls.md").read_text()
    assert (
        "| `pii` | deterministic | pre+post | llm, mcp, a2a | block, redact, log_only | no "
        in markdown
    )
    assert "| 1 | 1 | 2/3 | n/a | 0 | covered |" in markdown
    assert "| `intent_judge` |" in markdown
    assert "GAP: no passing allow test" in markdown
    ControlsReport.model_validate(report)  # the JSON round-trips into the model


def test_the_html_report_embeds_the_table(tester: pytest.Pytester, tmp_path: Path):
    tester.makepyfile(full_coverage())
    html = tmp_path / "report.html"
    result = tester.runpytest(
        "-p", "no:cacheprovider", f"--html={html}", "--self-contained-html", "--control-coverage"
    )
    assert result.ret == pytest.ExitCode.OK
    page = html.read_text()
    assert "Control coverage" in page
    assert "<code>tool_poisoning</code>" in page


def test_make_report_rebuilds_the_table_from_junit(tester: pytest.Pytester, tmp_path: Path):
    tester.makepyfile(full_coverage(broken="budget"))
    live, rebuilt, junit = tmp_path / "live", tmp_path / "rebuilt", tmp_path / "junit.xml"
    tester.runpytest("-p", "no:cacheprovider", f"--junitxml={junit}", f"--control-report={live}")
    assert main(["--junit", str(junit), "--out", str(rebuilt)]) == 1  # budget has a gap
    for left, right in zip(report_of(live)["rows"], report_of(rebuilt)["rows"], strict=True):
        assert left["outcomes"] == right["outcomes"]
    assert main(["--junit", str(tmp_path / "missing.xml"), "--out", str(rebuilt)]) == 2
