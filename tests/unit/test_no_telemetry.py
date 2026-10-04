"""ONNX Runtime's telemetry uploader never starts in the gateway (no phoning home).

ORT 1.30's POSIX 1DS uploader starts when ``onnxruntime`` is imported: it persists a device id
and an offline event queue under ``$XDG_CACHE_HOME``/``~/.cache`` (Linux) or ``~/Library/
Application Support`` (macOS), then posts to mobile.events.data.microsoft.com. A fresh
interpreter with a scratch home shows whether it started, without any network: those files
exist only when the uploader initialized. (Measured 2026-10-04: a bare ``import onnxruntime``
creates ``.onnxruntime/deviceid`` and ``onnxruntime.db`` there; in the gateway container a
recording proxy saw three ``CONNECT mobile.events.data.microsoft.com:443``.)
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE = """
import os, sys
import gateway.injection.classifier  # imports onnxruntime through gateway.injection
import onnxruntime
onnxruntime.get_available_providers()
print(os.environ["ORT_DISABLE_TELEMETRY"], os.environ["HF_HUB_OFFLINE"],
      os.environ["HF_HUB_DISABLE_TELEMETRY"])
"""


def run_probe(home: Path, code: str, **env: str) -> subprocess.CompletedProcess[str]:
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("ORT_", "HF_HUB_"))}
    clean.update(HOME=str(home), XDG_CACHE_HOME=str(home / "cache"), PYTHONPATH=str(REPO_ROOT))
    clean.update(env)
    return subprocess.run(  # noqa: S603 -- our own interpreter and a fixed script
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=clean,
        cwd=REPO_ROOT,
        timeout=120,
    )


def uploader_traces(home: Path) -> list[str]:
    return sorted(
        str(p.relative_to(home))
        for p in home.rglob("*")
        if ".onnxruntime" in p.parts or p.name == "deviceid"
    )


@pytest.mark.parametrize("inherited", [None, "0"], ids=["unset", "inherited-0"])
def test_importing_the_classifier_never_starts_the_ort_uploader(tmp_path, inherited):
    env = {"ORT_DISABLE_TELEMETRY": inherited} if inherited is not None else {}
    probe = run_probe(tmp_path, PROBE, **env)
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.split() == ["1", "1", "1"]
    assert uploader_traces(tmp_path) == []
    assert "telemetry" not in probe.stderr.lower()


def test_onnxruntime_imported_first_is_refused(tmp_path):
    """Imported before the gateway could set the switch, ORT would already be uploading."""
    probe = run_probe(tmp_path, "import onnxruntime\nimport gateway.injection\n", CI="1")
    assert probe.returncode != 0
    assert "ORT_DISABLE_TELEMETRY" in probe.stderr
