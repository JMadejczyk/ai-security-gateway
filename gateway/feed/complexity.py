"""Static bounds on feed regexes, checked before anything is compiled.

Matching runs with a timeout, but compiling does not: ``regex`` expands counted repeats when
it compiles, so a 160-byte feed with ``a{1000000}`` costs hundreds of megabytes before any
timeout could apply. Every feed pattern therefore passes a conservative scan first:

- counted repeats (``{m}``, ``{m,n}``) are bounded by `MAX_REPEAT_COUNT`, and a ``{`` that
  does not open one must be escaped (``\\{``): in ``regex`` a bare ``{e<=1}`` is fuzzy matching;
- the pattern's *weight* (atoms times the finite repeat counts around them, the size of what
  the compiler expands) is bounded per pattern and, in `FeedBudget`, per feed;
- an unbounded quantifier directly over a group that itself contains one (``(a+)+``, star
  height 2, the classic catastrophic-backtracking shape) is refused;
- backreferences, lookbehind, conditionals, recursion and inline flags beyond ``aimsu`` are
  refused: none is needed for a signature, and each makes matching cost hard to bound;
- verbose mode (``x``, inline, scoped or combined with other flags) is refused because it
  changes the tokens themselves: whitespace and ``#`` comments stop being atoms, so
  ``(?x)(a{100}) {300}`` repeats the group while this scan would read a literal space. Parsing
  verbose syntax too would mean a second tokenizer that must agree with ``regex`` on every
  edge case; refusing it costs signature authors nothing (write ``\\s`` or a literal space).
  The flags that remain only change what an atom *matches* (``i`` case, ``m``/``s`` anchors and
  ``.``, ``a``/``u`` character classes), never where one starts or ends, so the scan stays in
  step with the compiler.

The scan is deliberately stricter than ``regex``'s grammar: a pattern it cannot classify is
refused, never waved through.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

MAX_REPEAT_COUNT: Final = 1000
MAX_PATTERN_WEIGHT: Final = 20_000
MAX_FEED_WEIGHT: Final = 200_000  # ~275 B peak per unit while compiling (measured)
MAX_NESTING: Final = 32

_ALLOWED_FLAGS: Final = frozenset("aimsu-")  # never x: see the module docstring
_BRACED_ESCAPES: Final = frozenset("pPNxuU")  # \p{L}, \N{name}, \x{41}: skip to the brace


class Refusal(StrEnum):
    REPEAT_TOO_LARGE = f"counted repeat above {MAX_REPEAT_COUNT}"
    REPEAT_INVERTED = "counted repeat with max below min"
    BARE_BRACE = "a literal { must be escaped as \\{"
    NOTHING_TO_REPEAT = "quantifier without an atom (or a quantifier on a quantifier)"
    NESTED_UNBOUNDED = "nested unbounded quantifiers"
    PATTERN_TOO_HEAVY = f"weight above {MAX_PATTERN_WEIGHT}"
    FEED_TOO_HEAVY = f"the feed's patterns weigh more than {MAX_FEED_WEIGHT} in total"
    TOO_DEEP = "groups nested too deeply"
    UNBALANCED = "unbalanced parenthesis"
    LOOKBEHIND = "lookbehind"
    BACKREFERENCE = "backreference"
    GROUP_SYNTAX = "backreference, conditional, recursion or unsupported group syntax"
    VERBOSE_MODE = "verbose mode (the x flag)"
    UNSUPPORTED_ESCAPE = "\\g, \\k, \\Q or \\E escape"
    UNTERMINATED = "unterminated escape, group name or character class"
    NESTED_CLASS = "nested character class"


class PatternTooComplexError(ValueError):
    """The pattern falls outside the statically bounded subset."""

    def __init__(self, refusal: Refusal) -> None:
        super().__init__(f"pattern too complex or unsupported: {refusal.value}")
        self.refusal = refusal


@dataclass(slots=True)
class _Group:
    """One open group of the scan: its weight so far and what a quantifier would repeat."""

    weight: int = 0  # the current alternative
    alternatives: int = 0  # every finished alternative, summed
    atom: int = 0  # weight of the last atom, which a following quantifier repeats
    atom_unbounded: bool = False  # the last atom contains an unbounded quantifier
    unbounded: bool = False  # an unbounded quantifier anywhere in the group
    quantifiable: bool = False  # the last token is an atom (not a quantifier or a start)

    def add_atom(self, weight: int, *, unbounded: bool = False) -> None:
        self.weight += weight
        self.atom, self.atom_unbounded, self.quantifiable = weight, unbounded, True
        self.unbounded |= unbounded

    def alternate(self) -> None:
        self.alternatives += self.weight
        self.weight, self.quantifiable = 0, False

    def repeat(self, low: int, high: int | None) -> None:
        if not self.quantifiable:
            raise PatternTooComplexError(Refusal.NOTHING_TO_REPEAT)
        if high is None and self.atom_unbounded:
            raise PatternTooComplexError(Refusal.NESTED_UNBOUNDED)
        self.weight += self.atom * (max(low, high or 1) - 1)
        self.unbounded |= high is None
        self.quantifiable = False

    def total(self) -> int:
        return self.alternatives + self.weight


def _counted_repeat(pattern: str, start: int) -> tuple[int, int | None, int] | None:
    """Parse ``{m}``, ``{m,}``, ``{,n}`` or ``{m,n}`` at ``start``: (min, max, end) or None."""
    end = pattern.find("}", start)
    if end < 0:
        return None
    body = pattern[start + 1 : end]
    low, comma, high = body.partition(",")
    if not (low or high) or not (low.isdigit() or not low) or not (high.isdigit() or not high):
        return None
    if not comma:
        high = low
    return int(low or "0"), (int(high) if high else None), end + 1


def _group_open(pattern: str, i: int) -> int:
    """Index after a group opener at ``i`` (``(``, ``(?:``, ``(?P<n>``, ``(?i)``...)."""
    if not pattern.startswith("(?", i):
        return i + 1
    rest = pattern[i + 2 :]
    if rest.startswith((":", "=", "!", ">")):
        return i + 3
    if rest.startswith(("<=", "<!")):
        raise PatternTooComplexError(Refusal.LOOKBEHIND)
    if rest.startswith(("P<", "<")):
        close = pattern.find(">", i)
        if close < 0:
            raise PatternTooComplexError(Refusal.UNTERMINATED)
        return close + 1
    flags = ""
    for ch in rest:
        if ch in {")", ":"}:
            break
        flags += ch
    if "x" in flags:
        raise PatternTooComplexError(Refusal.VERBOSE_MODE)
    if not flags or not set(flags) <= _ALLOWED_FLAGS:
        raise PatternTooComplexError(Refusal.GROUP_SYNTAX)
    return i + 2 + len(flags)  # the ")" or ":" is handled by the caller's loop


def _escape_end(pattern: str, i: int) -> int:
    """Index after the escape at ``i``; backreferences and literal quoting are refused."""
    if i + 1 >= len(pattern):
        raise PatternTooComplexError(Refusal.UNTERMINATED)
    ch = pattern[i + 1]
    if ch.isdigit() and ch != "0":
        raise PatternTooComplexError(Refusal.BACKREFERENCE)
    if ch in {"g", "k", "Q", "E"}:
        raise PatternTooComplexError(Refusal.UNSUPPORTED_ESCAPE)
    if ch in _BRACED_ESCAPES and pattern.startswith("{", i + 2):
        close = pattern.find("}", i + 2)
        if close < 0:
            raise PatternTooComplexError(Refusal.UNTERMINATED)
        return close + 1
    return i + 2


def _class_end(pattern: str, i: int) -> int:
    """Index after the character class opening at ``i``."""
    j = i + 1
    if pattern.startswith("^", j):
        j += 1
    if pattern.startswith("]", j):  # a leading ] is literal
        j += 1
    while j < len(pattern):
        if pattern[j] == "\\":
            j = _escape_end(pattern, j)
            continue
        if pattern[j] == "[":
            raise PatternTooComplexError(Refusal.NESTED_CLASS)
        if pattern[j] == "]":
            return j + 1
        j += 1
    raise PatternTooComplexError(Refusal.UNTERMINATED)


def _atom_end(pattern: str, i: int) -> int:
    """Index after the single-width atom at ``i``: an escape, a class or one character."""
    if pattern[i] == "\\":
        return _escape_end(pattern, i)
    if pattern[i] == "[":
        return _class_end(pattern, i)
    return i + 1


def _quantifier(pattern: str, i: int) -> tuple[int, int | None, int]:
    """(min, max or None for unbounded, index after) of the quantifier at ``i``."""
    ch = pattern[i]
    if ch == "{":
        parsed = _counted_repeat(pattern, i)
        if parsed is None:
            raise PatternTooComplexError(Refusal.BARE_BRACE)
        low, high, end = parsed
        if max(low, high or 0) > MAX_REPEAT_COUNT:
            raise PatternTooComplexError(Refusal.REPEAT_TOO_LARGE)
        if high is not None and high < low:
            raise PatternTooComplexError(Refusal.REPEAT_INVERTED)
    else:
        low, high, end = (1 if ch == "+" else 0), (1 if ch == "?" else None), i + 1
    if pattern.startswith(("?", "+"), end):  # lazy or possessive
        end += 1
    return low, high, end


def _open(pattern: str, i: int, stack: list[_Group]) -> int:
    """Handle ``(`` at ``i``: push a group, or skip inline flags; index after the opener."""
    j = _group_open(pattern, i)
    if pattern.startswith("(?", i) and pattern.startswith(")", j):  # (?i): flags only
        return j + 1
    if len(stack) >= MAX_NESTING:
        raise PatternTooComplexError(Refusal.TOO_DEEP)
    stack.append(_Group())
    return j + 1 if pattern.startswith(":", j) else j


def pattern_weight(pattern: str) -> int:
    """Statically bounded size of ``pattern``; raises `PatternTooComplexError`."""
    stack = [_Group()]
    i = 0
    while i < len(pattern):
        ch, group = pattern[i], stack[-1]
        if ch in "*+?{":
            low, high, i = _quantifier(pattern, i)
            group.repeat(low, high)
        elif ch == "(":
            i = _open(pattern, i, stack)
        elif ch == ")":
            if len(stack) == 1:
                raise PatternTooComplexError(Refusal.UNBALANCED)
            closed = stack.pop()
            stack[-1].add_atom(closed.total(), unbounded=closed.unbounded)
            i += 1
        elif ch == "|":
            group.alternate()
            i += 1
        else:
            i = _atom_end(pattern, i)
            group.add_atom(1)
        if sum(g.total() for g in stack) > MAX_PATTERN_WEIGHT:
            raise PatternTooComplexError(Refusal.PATTERN_TOO_HEAVY)
    if len(stack) != 1:
        raise PatternTooComplexError(Refusal.UNBALANCED)
    return stack[0].total()


@dataclass(slots=True)
class FeedBudget:
    """Total weight of every pattern in one feed, so many medium patterns cannot add up."""

    used: int = 0

    def spend(self, weight: int) -> None:
        self.used += weight
        if self.used > MAX_FEED_WEIGHT:
            raise PatternTooComplexError(Refusal.FEED_TOO_HEAVY)
