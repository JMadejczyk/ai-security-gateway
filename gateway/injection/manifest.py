"""The pinned classifier model and the check that what is on disk is exactly that model.

`ModelManifest` (``model_manifest.json`` next to this module, reviewed with the code) names a
Hugging Face repository, a full revision commit and, for every file the gateway reads, its
path in the repository, its size and its SHA-256. Files live under
``<models_dir>/<repo_id with / as -->/<revision>/<path>``, so a manifest bump never verifies
against the files of the previous revision.

`ModelManifest.verify` hashes every file before anything parses it: a missing file, a size
or digest mismatch, or an unreadable file raises `ModelVerificationError` and the gateway
refuses to start (SPEC "Stack": no ``prompt_injection`` without a verified model). Only the
ONNX export is ever loaded: no pickle, no ``trust_remote_code``.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Self, cast

from pydantic import Field, StringConstraints, field_validator

from gateway.core.envelope import FrozenModel

MANIFEST_PATH: Final = Path(__file__).with_name("model_manifest.json")
_READ_CHUNK: Final = 1 << 20

type Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
type CommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
type RepoId = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
]


class ModelVerificationError(Exception):
    """The model files on disk are not the pinned ones (missing, wrong size or digest)."""


class ManifestFile(FrozenModel):
    path: str = Field(min_length=1)  # relative POSIX path inside the repository
    size: int = Field(gt=0)
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def _relative(cls, value: str) -> str:
        parts = PurePosixPath(value).parts
        if value.startswith("/") or not parts or any(p in {"..", "."} for p in parts):
            msg = f"manifest path {value!r} must be relative, without '.' or '..'"
            raise ValueError(msg)
        return value


class ModelFiles(FrozenModel):
    model: ManifestFile  # the ONNX graph with its weights
    tokenizer: ManifestFile  # tokenizer.json (a `tokenizers` serialization)
    config: ManifestFile  # config.json: the id2label map names the injection class


class ModelManifest(FrozenModel):
    repo_id: RepoId
    revision: CommitSha
    license: str = Field(min_length=1)
    injection_label: str = Field(min_length=1)  # the id2label value of the positive class
    max_tokens: int = Field(gt=2, le=8192)  # the model's window, special tokens included
    files: ModelFiles

    @classmethod
    def load(cls, path: Path = MANIFEST_PATH) -> Self:
        return cls.model_validate_json(path.read_bytes())

    @property
    def directory(self) -> PurePosixPath:
        """Where this revision's files live, relative to the models directory."""
        return PurePosixPath(self.repo_id.replace("/", "--"), self.revision)

    def all_files(self) -> tuple[ManifestFile, ...]:
        return (self.files.model, self.files.tokenizer, self.files.config)

    def local_path(self, models_dir: Path, file: ManifestFile) -> Path:
        return models_dir / self.directory / file.path

    def download_url(self, file: ManifestFile, endpoint: str = "https://huggingface.co") -> str:
        return f"{endpoint}/{self.repo_id}/resolve/{self.revision}/{file.path}"

    def verify(self, models_dir: Path) -> "VerifiedModel":
        """Hash every pinned file; raises `ModelVerificationError` on any difference."""
        for file in self.all_files():
            check_file(self.local_path(models_dir, file), file)
        tokenizer = self.local_path(models_dir, self.files.tokenizer).read_text("utf-8")
        config = self.local_path(models_dir, self.files.config).read_bytes()
        return VerifiedModel(
            manifest=self,
            model_path=self.local_path(models_dir, self.files.model),
            tokenizer_json=tokenizer,
            injection_index=_label_index(config, self.injection_label),
        )


@dataclass(frozen=True, slots=True)
class VerifiedModel:
    """A model whose files matched the manifest when they were hashed."""

    manifest: ModelManifest
    model_path: Path
    tokenizer_json: str
    injection_index: int  # the logit column of the injection class


def file_digest(path: Path) -> tuple[int, str]:
    """(size, SHA-256 hex) of a file, read in chunks."""
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def check_file(path: Path, expected: ManifestFile) -> None:
    """Raises `ModelVerificationError` unless ``path`` has the pinned size and digest."""
    try:
        size, sha256 = file_digest(path)
    except FileNotFoundError:
        msg = f"model file {expected.path} is missing at {path}"
        raise ModelVerificationError(msg) from None
    except OSError as exc:
        msg = f"model file {expected.path} cannot be read: {type(exc).__name__}"
        raise ModelVerificationError(msg) from None
    if size != expected.size:
        msg = f"model file {expected.path} has {size} bytes, the manifest pins {expected.size}"
        raise ModelVerificationError(msg)
    if sha256 != expected.sha256:
        msg = (
            f"model file {expected.path} has SHA-256 {sha256}, the manifest pins {expected.sha256}"
        )
        raise ModelVerificationError(msg)


def _label_index(config: bytes, label: str) -> int:
    try:
        document: object = json.loads(config)
    except ValueError:
        document = None
    mapping = cast("dict[str, object]", document) if isinstance(document, dict) else {}
    id2label = mapping.get("id2label")
    entries = cast("dict[object, object]", id2label).items() if isinstance(id2label, dict) else ()
    matches = [str(index) for index, name in entries if name == label]
    if len(matches) != 1 or not matches[0].isdigit():
        msg = f"model config does not name exactly one {label!r} class in id2label"
        raise ModelVerificationError(msg)
    return int(matches[0])
