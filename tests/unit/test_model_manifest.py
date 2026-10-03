"""The pinned classifier model: manifest, hash verification, fetching, startup refusal."""

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from gateway_testkit import INTERNAL_KEY, JWT_SECRET, ROOT_POLICY
from pydantic import ValidationError

from gateway import __main__ as entrypoint
from gateway.injection.classifier import UnavailableClassifier, load_classifier
from gateway.injection.fetch import fetch_file
from gateway.injection.manifest import ModelManifest, ModelVerificationError

CONFIG = json.dumps({"id2label": {"0": "SAFE", "1": "INJECTION"}}).encode()
FILES = {
    "onnx/model.onnx": b"\x08\x01onnx-bytes",
    "onnx/tokenizer.json": b"{}",
    "onnx/config.json": CONFIG,
}


def entry(path: str, data: bytes) -> dict[str, object]:
    return {"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def manifest_for(contents: dict[str, bytes] = FILES, **overrides: object) -> ModelManifest:
    roles = dict(zip(("model", "tokenizer", "config"), contents.items(), strict=True))
    document: dict[str, object] = {
        "repo_id": "acme/injection-model",
        "revision": "a" * 40,
        "license": "Apache-2.0",
        "injection_label": "INJECTION",
        "max_tokens": 512,
        "files": {role: entry(path, data) for role, (path, data) in roles.items()},
    }
    document.update(overrides)
    return ModelManifest.model_validate(document)


def install(manifest: ModelManifest, root: Path, files: dict[str, bytes] = FILES) -> None:
    for file in manifest.all_files():
        target = manifest.local_path(root, file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[file.path])


def test_the_shipped_manifest_pins_a_full_revision_and_every_file():
    manifest = ModelManifest.load()
    assert manifest.repo_id == "protectai/deberta-v3-base-prompt-injection-v2"
    assert len(manifest.revision) == 40
    assert manifest.license == "Apache-2.0"
    assert manifest.files.model.path.endswith(".onnx")  # the ONNX export only, never pickle
    assert {f.path for f in manifest.all_files()} == {
        "onnx/model.onnx",
        "onnx/tokenizer.json",
        "onnx/config.json",
    }


def test_verified_files_load(tmp_path):
    manifest = manifest_for()
    install(manifest, tmp_path)
    verified = manifest.verify(tmp_path)
    assert verified.injection_index == 1
    assert (
        verified.model_path == tmp_path / "acme--injection-model" / ("a" * 40) / "onnx/model.onnx"
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda p: p.unlink(), "missing"),
        (lambda p: p.write_bytes(p.read_bytes() + b"!"), "bytes"),
        (lambda p: p.write_bytes(b"X" + p.read_bytes()[1:]), "SHA-256"),
    ],
    ids=["missing", "wrong-size", "wrong-digest"],
)
def test_any_difference_refuses_the_model(tmp_path, tamper, message):
    manifest = manifest_for()
    install(manifest, tmp_path)
    tamper(manifest.local_path(tmp_path, manifest.files.model))
    with pytest.raises(ModelVerificationError, match=message):
        manifest.verify(tmp_path)


@pytest.mark.parametrize(
    "config",
    [
        b"not json",
        b'{"id2label": {"0": "SAFE"}}',
        b'{"id2label": {"0": "INJECTION", "1": "INJECTION"}}',
    ],
    ids=["not-json", "no-injection-label", "two-injection-labels"],
)
def test_the_config_must_name_one_injection_class(tmp_path, config):
    files = {**FILES, "onnx/config.json": config}
    manifest = manifest_for(files)
    install(manifest, tmp_path, files)
    with pytest.raises(ModelVerificationError, match="id2label"):
        manifest.verify(tmp_path)


@pytest.mark.parametrize(
    "override",
    [
        {"revision": "main"},  # a branch, not a commit
        {"repo_id": "../etc"},
        {
            "files": {
                "model": entry("../model.onnx", b"x"),
                "tokenizer": entry("t", b"x"),
                "config": entry("c", b"x"),
            }
        },
    ],
    ids=["branch-revision", "bad-repo-id", "path-traversal"],
)
def test_the_manifest_rejects_unpinned_or_escaping_entries(override):
    with pytest.raises(ValidationError):
        manifest_for(**override)


def test_load_classifier_refuses_a_missing_model(tmp_path):
    with pytest.raises(ModelVerificationError, match="missing"):
        load_classifier(tmp_path)


def test_load_classifier_disabled_returns_the_fail_closed_classifier(tmp_path):
    assert isinstance(load_classifier(tmp_path, enabled=False), UnavailableClassifier)


def test_the_gateway_refuses_to_start_without_a_verified_model(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("ACL_JWT_SECRET", JWT_SECRET)
    monkeypatch.setenv("ACL_INTERNAL_KEY", INTERNAL_KEY)
    monkeypatch.setenv("ACL_POLICY_PATH", str(ROOT_POLICY))
    monkeypatch.setenv("ACL_MODELS_DIR", str(tmp_path))  # empty: nothing verifies
    monkeypatch.delenv("ACL_INJECTION_CLASSIFIER", raising=False)
    assert entrypoint.main() == 2
    assert "refusing to start" in caplog.text
    assert "missing" in caplog.text


# ------------------------------------------------------------------------- fetch


def serving(files: dict[str, bytes], requests: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        name = request.url.path.split(f"/resolve/{'a' * 40}/", 1)[1]
        return httpx.Response(200, content=files[name])

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_downloads_the_pinned_revision_and_verifies_it(tmp_path):
    manifest, requests = manifest_for(), list[str]()
    with serving(FILES, requests) as client:
        for file in manifest.all_files():
            fetch_file(client, manifest, file, tmp_path, "https://hub.test")
    manifest.verify(tmp_path)
    assert requests[0] == f"/acme/injection-model/resolve/{'a' * 40}/onnx/model.onnx"


def test_fetch_refuses_altered_bytes_and_leaves_nothing_behind(tmp_path):
    manifest, requests = manifest_for(), list[str]()
    altered = {**FILES, "onnx/model.onnx": b"\x08\x01evil-bytes"}
    with serving(altered, requests) as client, pytest.raises(ModelVerificationError):
        fetch_file(client, manifest, manifest.files.model, tmp_path, "https://hub.test")
    target = manifest.local_path(tmp_path, manifest.files.model)
    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()


def test_fetch_skips_files_already_verified(tmp_path):
    manifest, requests = manifest_for(), list[str]()
    install(manifest, tmp_path)
    with serving(FILES, requests) as client:
        for file in manifest.all_files():
            fetch_file(client, manifest, file, tmp_path, "https://hub.test")
    assert requests == []
