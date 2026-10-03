"""A deterministic `JudgeClient` stand-in: the testkit's default for every gateway it builds.

The root policy configures ``judges:``, so without a stand-in every test that copies it would
need a scripted judge upstream or see the judge controls fail closed. `FakeJudgeClient`
answers by response model: aligned, no violations, not an injection. A test overrides one
answer with ``harness.container.judges.answers[<model name>] = <dict | callable | error>``
(a callable gets ``(control_id, content)``). It keeps the real client's contract: no
``judges`` section → not configured; a lenient response model → `TypeError`; content over
the cap → unavailable; the answer validated strictly against the model.

Suites that exercise the real client over a scripted upstream pass ``judge_factory=JudgeClient``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from gateway.judges.client import JudgeClient, JudgeResult, JudgeUnavailableError, forbids_extra

type Answer = dict[str, Any] | JudgeUnavailableError | Callable[[str, str], Any]

DEFAULT_ANSWERS: dict[str, Answer] = {
    "IntentAssessment": {"aligned": True, "confidence": 1.0, "rationale": "fake judge"},
    "OutputAssessment": {"violations": []},
    "InjectionJudgement": {"is_injection": False, "confidence": 1.0, "rationale": "fake judge"},
}


@dataclass(frozen=True)
class JudgeCall:
    control_id: str
    model: str
    content: str


class FakeJudgeClient(JudgeClient):
    answers: dict[str, Answer]
    calls: list[JudgeCall]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.answers = dict(DEFAULT_ANSWERS)
        self.calls = []

    async def judge(self, *, control_id, instructions, content, response_model):  # type: ignore[override]
        del instructions
        if not forbids_extra(response_model.model_json_schema()):
            msg = f"judge response model {response_model.__name__} must set extra='forbid'"
            raise TypeError(msg)
        self.calls.append(JudgeCall(control_id, response_model.__name__, content))
        settings = self._current_snapshot().policy.judges
        if settings is None:
            raise JudgeUnavailableError(JudgeResult.NOT_CONFIGURED)
        if len(content) > settings.max_content_chars:
            raise JudgeUnavailableError(JudgeResult.CONTENT_TOO_LARGE)
        answer: Any = self.answers.get(response_model.__name__)
        if callable(answer):
            answer = answer(control_id, content)
        if isinstance(answer, JudgeUnavailableError):
            raise answer
        if answer is None:
            raise JudgeUnavailableError(JudgeResult.SCHEMA_MISMATCH)
        try:
            validated: BaseModel = response_model.model_validate(answer, strict=True)
        except ValidationError:  # like the real client: a garbled verdict is no verdict
            raise JudgeUnavailableError(JudgeResult.SCHEMA_MISMATCH) from None
        return validated
