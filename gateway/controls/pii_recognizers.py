"""Presidio, configured for the ``pii`` control: pattern and checksum recognizers, no NLP model.

Every entity the policy can name (`PiiEntity`) is found by a pattern plus a validator: a
checksum (PESEL, NIP, IBAN), the public suffix list (e-mail) or a numbering plan (phone).
None needs named-entity recognition, so the analyzer runs on Presidio's `NoOpNlpEngine`:
no spaCy model is loaded or downloaded, and nothing the image needs comes from the network
at runtime. The cost is Presidio's context enhancement, which reads tokens and lemmas from
an NLP pass; the phone recognizer does its own (cheap, regex) context check instead.

Offsets in every result are Python string indices, i.e. code points, as spans require.
"""

import functools
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import ClassVar, Final, cast, override

import tldextract
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer import RecognizerResult as PresidioResult
from presidio_analyzer.nlp_engine import NlpArtifacts, NoOpNlpEngine
from presidio_analyzer.predefined_recognizers import (
    EmailRecognizer,
    IbanRecognizer,
    PhoneRecognizer,
)

from gateway.policy.schema import PiiEntity

LANGUAGE: Final = "en"  # Presidio's language key; every recognizer here is language-neutral
_DIGITS: Final = frozenset("0123456789")


@dataclass(frozen=True, slots=True)
class PiiFinding:
    """One detection in one string: entity ID, code-point offsets and Presidio's score."""

    entity: PiiEntity
    start: int
    end: int
    score: float


# ------------------------------------------------------------------------ validators


_PESEL_WEIGHTS: Final = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
# The month field carries the century: +80 → 1800s, +0 → 1900s, +20 → 2000s, ...
_PESEL_CENTURIES: Final = {0: 1900, 1: 2000, 2: 2100, 3: 2200, 4: 1800}
_NIP_WEIGHTS: Final = (6, 5, 7, 2, 3, 4, 5, 6, 7)
_NIP_MODULUS: Final = 11
_PESEL_LENGTH: Final = 11
_NIP_LENGTH: Final = 10


def _ascii_digits(value: str, length: int) -> list[int] | None:
    """The digits of ``value`` when it is exactly ``length`` ASCII digits (``str.isdigit``
    also accepts other scripts' digits, which a PESEL or NIP never contains)."""
    if len(value) != length or not _DIGITS.issuperset(value):
        return None
    return [int(ch) for ch in value]


def is_valid_pesel(value: str) -> bool:
    """11 digits, a real birth date (century encoded in the month) and the check digit."""
    digits = _ascii_digits(value, _PESEL_LENGTH)
    if digits is None:
        return False
    *body, check = digits
    if (10 - sum(w * d for w, d in zip(_PESEL_WEIGHTS, body, strict=True)) % 10) % 10 != check:
        return False
    century, month = divmod(int(value[2:4]), 20)
    base = _PESEL_CENTURIES.get(century)
    if base is None:
        return False
    try:
        date(base + int(value[0:2]), month, int(value[4:6]))
    except ValueError:
        return False
    return True


def is_valid_nip(value: str) -> bool:
    """10 digits (dashes ignored) whose weighted sum mod 11 equals the last digit."""
    digits = _ascii_digits(value.replace("-", ""), _NIP_LENGTH)
    if digits is None:
        return False
    *body, check = digits
    remainder = sum(w * d for w, d in zip(_NIP_WEIGHTS, body, strict=True)) % _NIP_MODULUS
    return remainder == check  # a remainder of 10 is never a valid NIP: no digit equals it


# ----------------------------------------------------------------------- recognizers


class PeselRecognizer(PatternRecognizer):
    """``PL_PESEL``: the Polish national identification number."""

    PATTERNS: ClassVar[list[Pattern]] = [
        Pattern("PESEL", r"(?<![0-9])[0-9]{11}(?![0-9])", 0.3),
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity=PiiEntity.PL_PESEL.value,
            patterns=self.PATTERNS,
            supported_language=LANGUAGE,
        )

    @override
    def validate_result(self, pattern_text: str) -> bool:
        return is_valid_pesel(pattern_text)


class NipRecognizer(PatternRecognizer):
    """``PL_NIP``: the Polish tax ID, bare (``1234563218``) or dashed (``123-456-32-18``,
    ``123-45-63-218``). A ``PL`` VAT prefix is left outside the span."""

    PATTERNS: ClassVar[list[Pattern]] = [
        Pattern("NIP", r"(?<![0-9-])[0-9]{3}-?[0-9]{3}-?[0-9]{2}-?[0-9]{2}(?![0-9-])", 0.3),
        Pattern("NIP (3-2-2-3)", r"(?<![0-9-])[0-9]{3}-[0-9]{2}-[0-9]{2}-[0-9]{3}(?![0-9-])", 0.3),
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity=PiiEntity.PL_NIP.value,
            patterns=self.PATTERNS,
            supported_language=LANGUAGE,
        )

    @override
    def validate_result(self, pattern_text: str) -> bool:
        return is_valid_nip(pattern_text)


def _merged(windows: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted, non-overlapping ``(start, end)`` ranges covering every given window."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


class OfflineEmailRecognizer(EmailRecognizer):
    """Presidio's e-mail recognizer, offline and run only around an ``@``.

    - The stock recognizer calls `tldextract.extract`, which downloads the public suffix list
      on first use: a network call (and a stall on the gateway's internal-only networks) in
      the request path. This one reads the snapshot bundled with tldextract.
    - Presidio's pattern tries every word of the text as a local part (about 5 ms per 20 KB
      once the text holds a single ``@``). An address is at most 64 characters before its
      ``@`` and 255 after it, so the pattern only reads those windows.
    """

    LOCAL_MAX: ClassVar[int] = 64
    DOMAIN_MAX: ClassVar[int] = 255
    _AT: ClassVar[re.Pattern[str]] = re.compile("@")
    _SUFFIXES: ClassVar[tldextract.TLDExtract] = tldextract.TLDExtract(
        cache_dir=None, suffix_list_urls=()
    )

    def __init__(self) -> None:
        super().__init__(supported_language=LANGUAGE)

    @override
    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
        regex_flags: int | None = None,
    ) -> list[PresidioResult]:
        ats = (match.start() for match in self._AT.finditer(text))
        windows = _merged((max(0, i - self.LOCAL_MAX), i + 1 + self.DOMAIN_MAX) for i in ats)
        results: list[PresidioResult] = []
        for start, end in windows:
            for found in super().analyze(text[start:end], entities, nlp_artifacts, regex_flags):
                found.start += start
                found.end += start
                results.append(found)
        return results

    @override
    def validate_result(self, pattern_text: str) -> bool:
        return self._SUFFIXES(pattern_text).fqdn != ""


class ContextPhoneRecognizer(PhoneRecognizer):
    """Presidio's phone recognizer (python-phonenumbers, validated against the numbering
    plan), scored up when the number is unambiguous, and run only where a number can be.

    Presidio gives every match 0.4 and relies on NLP context words to raise it; without an
    NLP pass a bare nine-digit order number would score the same as ``+48 600 700 800``. A
    number written with a country code, or preceded by a phone word, scores `CONFIDENT`.
    National-format numbers are matched for ``regions`` (international ones in any case).

    The matcher's own candidate search runs one very large regex over the whole text (about
    8 ms per 20 KB of digit-heavy prose). A cheap pre-scan finds runs of digits and phone
    punctuation with at least `MIN_DIGITS` digits (a Polish number has nine, and so does
    nearly every number with a country code) and the matcher only reads those, plus one
    character either side for its boundary checks. Dates and amounts never reach it.
    """

    CONFIDENT: ClassVar[float] = 0.75
    CONTEXT_WINDOW: ClassVar[int] = 32
    MIN_DIGITS: ClassVar[int] = 9
    _CONTEXT: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:tel|telefon\w{0,3}|komórk\w{0,4}|kom|zadzwoń|phone|mobile|cell|call)\b",
        re.IGNORECASE,
    )
    _CANDIDATE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<![0-9])[+(]?[0-9][0-9 ().\-/]{5,24}[0-9](?![0-9])"
    )

    def __init__(self, regions: Sequence[str] = ("PL",)) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType] -- presidio leaves supported_regions unannotated
            supported_language=LANGUAGE, supported_regions=tuple(regions)
        )

    @override
    def analyze(
        self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None
    ) -> list[PresidioResult]:
        results: list[PresidioResult] = []
        for candidate in self._CANDIDATE.finditer(text):
            if sum(ch.isdigit() for ch in candidate[0]) < self.MIN_DIGITS:
                continue
            offset = max(0, candidate.start() - 1)
            window = text[offset : candidate.end() + 1]
            # Presidio annotates the parameter as NlpArtifacts but defaults it to None itself.
            for found in super().analyze(window, entities, cast("NlpArtifacts", nlp_artifacts)):
                found.start += offset
                found.end += offset
                if self._unambiguous(text, found.start):
                    found.score = max(found.score, self.CONFIDENT)
                results.append(found)
        return results

    def _unambiguous(self, text: str, start: int) -> bool:
        if text.startswith(("+", "00"), start):
            return True
        window = text[max(0, start - self.CONTEXT_WINDOW) : start]
        return self._CONTEXT.search(window) is not None


# -------------------------------------------------------------------------- analyzer


class PresidioPiiAnalyzer:
    """A Presidio `AnalyzerEngine` over our recognizers. Build once: it is not cheap."""

    def __init__(self, phone_regions: Sequence[str] = ("PL",)) -> None:
        registry = RecognizerRegistry(supported_languages=[LANGUAGE])
        for recognizer in (
            PeselRecognizer(),
            NipRecognizer(),
            IbanRecognizer(supported_language=LANGUAGE),
            OfflineEmailRecognizer(),
            ContextPhoneRecognizer(phone_regions),
        ):
            registry.add_recognizer(recognizer)
        self._engine = AnalyzerEngine(
            registry=registry,
            nlp_engine=NoOpNlpEngine(models=[{"lang_code": LANGUAGE, "model_name": "none"}]),
            supported_languages=[LANGUAGE],
        )
        # Compile every pattern now rather than on the first request.
        self.find("warm-up a@example.com +48 600 700 800 44051401359", tuple(PiiEntity), 0.0)

    def find(self, text: str, entities: Iterable[PiiEntity], threshold: float) -> list[PiiFinding]:
        """Detections of ``entities`` scoring at least ``threshold``, in text order."""
        wanted = [entity.value for entity in entities]
        if not wanted or not text:
            return []
        results = self._engine.analyze(
            text=text, language=LANGUAGE, entities=wanted, score_threshold=threshold
        )
        findings = [
            PiiFinding(PiiEntity(r.entity_type), r.start, r.end, r.score)
            for r in results
            if r.end > r.start
        ]
        return sorted(findings, key=lambda f: (f.start, f.end, f.entity))


@functools.cache
def default_analyzer() -> PresidioPiiAnalyzer:
    """The process-wide analyzer: recognizers hold no per-request state, so one is shared."""
    return PresidioPiiAnalyzer()
