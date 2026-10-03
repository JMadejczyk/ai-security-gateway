"""The attack corpus (``corpus.yaml``) through the real apps: one parametrized runner.

Each case carries the ``control`` markers its ``proves`` field names, so the corpus feeds the
per-control report. ``test_attack`` runs every case with the deterministic classifier
(``fake_injection`` substrings score as injections); ``test_attack_real_model`` runs the cases
that set ``real_model`` again with the pinned ONNX classifier (marked ``model``; skipped when
``models/cache`` lacks it).
"""

from pathlib import Path

import pytest
from attack_kit import AttackCase, check, load_corpus, run
from injection_kit import MODELS_DIR, real_model_present

from gateway.injection.classifier import OnnxInjectionClassifier

CORPUS = load_corpus()


def params(cases: list[AttackCase], *, real: bool = False) -> list[object]:
    """One param per case, carrying its control markers. With the real classifier a case
    proves its controls only where the model is expected to reach the same decision."""
    return [
        pytest.param(
            case,
            id=case.id,
            marks=case.marks() if not real or case.real_model == case.expect.decision else (),
        )
        for case in cases
    ]


@pytest.mark.parametrize("case", params(CORPUS))
async def test_attack(case: AttackCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen = await run(tmp_path, monkeypatch, case, case.classifier())
    check(case, seen)


@pytest.fixture(scope="module")
def real_classifier() -> OnnxInjectionClassifier:
    return OnnxInjectionClassifier.from_models_dir(MODELS_DIR)


@pytest.mark.model
@pytest.mark.skipif(not real_model_present(), reason="pinned model not in models/cache")
@pytest.mark.parametrize("case", params([c for c in CORPUS if c.real_model is not None], real=True))
async def test_attack_real_model(
    case: AttackCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_classifier: OnnxInjectionClassifier,
):
    seen = await run(tmp_path, monkeypatch, case, real_classifier)
    check(case, seen, decision=case.real_model)


def test_the_corpus_covers_every_attack_category():
    from attack_kit import Category  # noqa: PLC0415 -- this test is about the corpus schema

    assert {c.category for c in CORPUS} == set(Category)
