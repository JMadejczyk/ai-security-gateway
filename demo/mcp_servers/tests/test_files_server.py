from __future__ import annotations

from pathlib import Path

import pytest

from acl_demo_mcp.files_server import FilesSettings, ReportPathError, ReportStore


@pytest.fixture
def root(tmp_path: Path) -> Path:
    reports = tmp_path / "reports"
    reports.mkdir()
    return reports


@pytest.fixture
def store(root: Path) -> ReportStore:
    return ReportStore(FilesSettings(root=root, max_bytes=64))


def test_writes_plain_name_inside_root(store: ReportStore, root: Path) -> None:
    assert store.write("q3-summary.md", "hello") == 5
    assert (root / "q3-summary.md").read_text() == "hello"


@pytest.mark.parametrize(
    "name",
    [
        "",
        "..",
        ".",
        "../escape.txt",
        "sub/report.txt",
        "/etc/passwd",
        "..\\escape.txt",
        ".hidden",
        "report\x00.txt",
        "raport-żółw.txt",
        "a" * 129,
    ],
)
def test_rejects_names_that_are_not_plain_files(store: ReportStore, name: str) -> None:
    with pytest.raises(ReportPathError):
        store.write(name, "x")


def test_never_follows_a_planted_symlink(store: ReportStore, root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    (root / "link.txt").symlink_to(outside)
    with pytest.raises(ReportPathError):
        store.write("link.txt", "pwned")
    assert not outside.exists()


def test_never_overwrites_an_existing_report(store: ReportStore, root: Path) -> None:
    store.write("once.txt", "first")
    with pytest.raises(ReportPathError):
        store.write("once.txt", "second")
    assert (root / "once.txt").read_text() == "first"


def test_rejects_oversized_content(store: ReportStore, root: Path) -> None:
    with pytest.raises(ReportPathError):
        store.write("big.txt", "x" * 65)
    assert not (root / "big.txt").exists()
