"""Fake injection classifiers and judges for tests that must not load the real model.

`MarkerClassifier` scores a text by the markers it contains, so a test states its intent in
the text itself: ``INJECT_MARKER`` scores above the policy threshold, ``DOUBT_MARKER`` inside the
judge band, anything else 0. The gateway harness (`gateway_testkit.running_gateway`) uses it
by default, so every existing suite runs with ``prompt_injection`` active and clean.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from pydantic import BaseModel

from gateway.injection.classifier import InjectionScore
from gateway.injection.manifest import ModelManifest
from gateway.judges.client import JudgeResult, JudgeUnavailableError

INJECT_MARKER: Final = "<<acl-test-injection>>"
DOUBT_MARKER: Final = "<<acl-test-doubtful>>"
DEFAULT_SCORES: Final = {INJECT_MARKER: 0.99, DOUBT_MARKER: 0.6}

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
MODELS_DIR: Final = REPO_ROOT / "models" / "cache"


def real_model_present() -> bool:
    """True when the pinned model files are in ``models/cache`` (sizes only: cheap)."""
    manifest = ModelManifest.load()
    return all(
        manifest.local_path(MODELS_DIR, f).is_file()
        and manifest.local_path(MODELS_DIR, f).stat().st_size == f.size
        for f in manifest.all_files()
    )


@dataclass
class MarkerClassifier:
    """Scores each text by the highest marker it contains; records every call."""

    scores: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORES))
    calls: list[list[str]] = field(default_factory=list[list[str]])

    def __call__(self, texts: Sequence[str]) -> list[InjectionScore]:
        self.calls.append(list(texts))
        return [self._score(text) for text in texts]

    def _score(self, text: str) -> InjectionScore:
        best = InjectionScore(0.0, 0, len(text))
        for marker, score in self.scores.items():
            if (at := text.find(marker)) != -1 and score > best.score:
                best = InjectionScore(score, max(at - 40, 0), min(at + len(marker) + 40, len(text)))
        return best

    @property
    def classified(self) -> list[str]:
        return [text for call in self.calls for text in call]


@dataclass
class ScriptedJudge:
    """A judge answering ``answer`` (a model instance) or raising unavailable when None."""

    answer: BaseModel | None = None
    reason: JudgeResult = JudgeResult.TIMEOUT
    contents: list[str] = field(default_factory=list[str])

    async def judge[T: BaseModel](
        self, *, control_id: str, instructions: str, content: str, response_model: type[T]
    ) -> T:
        del control_id, instructions
        self.contents.append(content)
        if self.answer is None:
            raise JudgeUnavailableError(self.reason)
        return response_model.model_validate(self.answer.model_dump())
