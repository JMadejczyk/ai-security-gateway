"""``pii``: personal data in prompts, answers, tool arguments and tool results.

Presidio with our recognizers (`gateway.controls.pii_recognizers`): ``PL_PESEL``, ``PL_NIP``,
``IBAN_CODE``, ``EMAIL_ADDRESS``, ``PHONE_NUMBER``, filtered by the policy's ``entities`` and
``threshold``. Modes: ``redact`` (spans labelled with the entity ID), ``block``, ``log_only``.
"""

import asyncio
from typing import ClassVar

from gateway.controls.detection import detection_verdict
from gateway.controls.pii_recognizers import PresidioPiiAnalyzer, default_analyzer
from gateway.controls.text import TextExtractor, TextSegment
from gateway.core.envelope import Interaction, Span, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import ControlKind, Stage
from gateway.policy.schema import PiiConfig

PII_DETECTED = "pii_detected"
NO_PII = "no_pii"
# Below this much text the scan (~0.1 ms per KiB) is cheaper than a hop to a worker thread.
INLINE_SCAN_CHARS = 4096


class PiiControl(Control):
    id: ClassVar[str] = "pii"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE, Stage.POST})
    kind: ClassVar[ControlKind] = ControlKind.DETERMINISTIC

    def __init__(
        self,
        analyzer: PresidioPiiAnalyzer | None = None,
        extractor: TextExtractor | None = None,
    ) -> None:
        self._analyzer = analyzer if analyzer is not None else default_analyzer()
        self._extractor = extractor if extractor is not None else TextExtractor()

    async def evaluate(self, interaction: Interaction, stage: Stage, cfg: ControlConfig) -> Verdict:
        settings = cfg if isinstance(cfg, PiiConfig) else PiiConfig(**cfg.model_dump())
        segments = self._extractor.segments(interaction, stage)
        if sum(len(segment.text) for segment in segments) <= INLINE_SCAN_CHARS:
            spans = self.scan(segments, settings)
        else:  # Presidio is synchronous CPU work: keep long scans off the event loop
            spans = await asyncio.to_thread(self.scan, segments, settings)
        return detection_verdict(self.id, settings, spans, detected=PII_DETECTED, clean=NO_PII)

    def scan(self, segments: list[TextSegment], settings: PiiConfig) -> list[Span]:
        """Spans of every configured entity scoring at least ``settings.threshold``."""
        return [
            Span(path=segment.pointer, start=f.start, end=f.end, label=f.entity.value)
            for segment in segments
            for f in self._analyzer.find(segment.text, settings.entities, settings.threshold)
        ]
