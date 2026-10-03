"""Running a text detector over an interaction's segments so that disguise does not help.

A detector finds ``(start, end, label)`` in one string. `scan` runs it so that two cheap
evasions fail, and maps every finding back to the original code points of every segment it
touches:

- **Disguised characters.** Each segment is scanned in a normalized view: every code point
  NFKC-folded on its own (full-width digits become ASCII digits, ligatures their letters) and
  format characters (Unicode category Cf: zero-width space and joiners, soft hyphen, BOM, bidi
  marks) removed. Folding per code point, rather than NFKC over the whole string, keeps the
  index map trivial: every view character comes from exactly one original code point. A
  finding covers the full original range, invisible characters inside it included.
- **Split values.** A secret cut across content parts, messages or argument fields is
  rebuilt by scanning the joined views around every boundary between consecutive segments
  (`WINDOW` characters each side, so the cost stays linear; protocol fields such as ``role``
  are left out of the join, see `TextSegment.joinable`). Only findings that cross a
  boundary are taken from the joined scan; each contributing fragment gets its own hit.
"""

import re
import sys
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Self

from gateway.controls.text import TextSegment

type Detector = Callable[[str], Iterable[tuple[int, int, str]]]

WINDOW: Final = 4096  # characters of joined context scanned on each side of a boundary


def _format_characters() -> re.Pattern[str]:
    """Every Unicode Cf code point, as one character class (built once, ~0.1 s)."""
    points = [c for c in range(sys.maxunicode + 1) if unicodedata.category(chr(c)) == "Cf"]
    ranges: list[tuple[int, int]] = []
    for point in points:
        if ranges and point == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], point)
        else:
            ranges.append((point, point))
    body = "".join(
        re.escape(chr(a)) if a == b else f"{re.escape(chr(a))}-{re.escape(chr(b))}"
        for a, b in ranges
    )
    return re.compile(f"[{body}]")


_FORMAT: Final = _format_characters()


@dataclass(frozen=True, slots=True)
class NormalizedView:
    """The text a detector sees, and where each of its characters came from."""

    text: str
    origin: tuple[int, ...] | None  # original index of each view character; None = identity

    @classmethod
    def of(cls, text: str) -> Self:
        if text.isascii() or (
            unicodedata.is_normalized("NFKC", text) and _FORMAT.search(text) is None
        ):
            return cls(text, None)  # the common case: nothing to fold, nothing hidden
        chars: list[str] = []
        origin: list[int] = []
        for index, char in enumerate(text):
            if unicodedata.category(char) == "Cf":
                continue
            folded = unicodedata.normalize("NFKC", char)
            chars.append(folded)
            origin.extend([index] * len(folded))
        return cls("".join(chars), tuple(origin))

    def original(self, start: int, end: int) -> tuple[int, int]:
        """The original code-point range of view characters ``[start, end)``."""
        if self.origin is None:
            return start, end
        return self.origin[start], self.origin[end - 1] + 1


@dataclass(frozen=True, slots=True)
class Hit:
    """A finding in one segment, in original code points of ``segment.text``."""

    segment: TextSegment
    start: int
    end: int
    label: str


def scan(segments: Sequence[TextSegment], detector: Detector) -> list[Hit]:
    """Every finding of ``detector`` in ``segments`` (see the module docstring)."""
    views = [NormalizedView.of(segment.text) for segment in segments]
    hits: list[Hit] = []
    for segment, view in zip(segments, views, strict=True):
        for start, end, label in detector(view.text):
            if end > start:
                hits.append(Hit(segment, *view.original(start, end), label))
    joinable = [(s, v) for s, v in zip(segments, views, strict=True) if s.joinable]
    if len(joinable) > 1:
        hits.extend(
            _across_boundaries([s for s, _ in joinable], [v for _, v in joinable], detector)
        )
    return hits


class _Joined:
    """The views of consecutive segments as one text, and the way back to each segment."""

    def __init__(self, segments: Sequence[TextSegment], views: Sequence[NormalizedView]) -> None:
        self._parts = list(zip(segments, views, strict=True))
        self.starts: list[int] = []  # offset of each view in `text`
        total = 0
        for view in views:
            self.starts.append(total)
            total += len(view.text)
        self.text = "".join(view.text for view in views)

    def windows(self) -> list[tuple[int, int]]:
        """Merged ranges of `WINDOW` characters around every boundary between segments."""
        windows: list[tuple[int, int]] = []
        for boundary in self.starts[1:]:
            low, high = max(0, boundary - WINDOW), min(len(self.text), boundary + WINDOW)
            if windows and low <= windows[-1][1]:
                windows[-1] = (windows[-1][0], high)
            else:
                windows.append((low, high))
        return windows

    def split(self, start: int, end: int, label: str) -> list[Hit]:
        """One hit per segment the joined range covers; none if it lies in one segment."""
        pieces: list[Hit] = []
        for (segment, view), offset in zip(self._parts, self.starts, strict=True):
            low, high = max(start, offset), min(end, offset + len(view.text))
            if low < high:
                pieces.append(Hit(segment, *view.original(low - offset, high - offset), label))
        return pieces if len(pieces) > 1 else []


def _across_boundaries(
    segments: Sequence[TextSegment], views: Sequence[NormalizedView], detector: Detector
) -> list[Hit]:
    joined = _Joined(segments, views)
    hits: list[Hit] = []
    for low, high in joined.windows():
        for start, end, label in detector(joined.text[low:high]):
            hits.extend(joined.split(low + start, low + end, label))
    return hits
