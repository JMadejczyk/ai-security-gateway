"""``secrets``: credentials in prompts, answers, tool arguments and tool results (mandatory).

Our own reviewed rule set, no third-party rule files. Each `SecretRule` is one regex with
bounded repetition and no nested quantifiers (linear time on any input), the capture group
that is the secret itself, and an optional validator that rejects look-alikes:

- provider keys with a fixed prefix: AWS access key IDs, GitHub, OpenAI, Anthropic, Slack,
  Stripe live keys, Google API keys;
- an AWS secret access key, only next to a name that says so (40 base64 characters alone
  are indistinguishable from a hash);
- JWTs, whose header must decode to a JSON object with ``alg``;
- PEM private key blocks, including a truncated one (no ``END`` line);
- the password of a connection string (``postgres://user:pass@host``);
- ``password=`` / ``api_key:`` / ``secret=`` style assignments with a non-trivial value
  (a quoted value is taken whole, up to its closing quote, escapes respected);
- any value whose JSON key names a credential (``{"password": "..."}``, ``"api_key"``,
  ``"authorization"``...), in tool arguments, tool results and decoded tool-call arguments.

Every segment is scanned through `gateway.controls.scanning.scan`: normalized (full-width
and zero-width disguises) and across segment boundaries (a key split over two content
parts), with each fragment masked. A segment the extractor could not decode reliably
(`SegmentKind.UNSCANNABLE`) is refused outright: this control is mandatory and cannot vouch
for content it could not read.

False positives are kept low on purpose: provider tokens must mix character classes
(``sk-learn`` and ``sk-some-long-lowercase-words`` are prose), placeholders are not secrets
when the whole value is one (``<password>``, ``${DB_PASSWORD}``, ``os.environ``, ``changeme``,
``****``; a password merely starting with ``%`` or containing ``getenv`` is still a secret, and
a connection string's password is percent-decoded first), and AWS's documented
example credentials (key IDs ending in ``EXAMPLE``, the secret key ending in ``EXAMPLEKEY``)
are allowed: they are published in AWS documentation, can never authenticate, and appear in
every tutorial a user might paste. A real key ID ends that way with probability 32^-7.

Modes: ``block`` or ``redact``; mandatory, so never ``log_only`` (the policy schema refuses
it). Redaction masks only the secret, e.g. the password inside a connection string.
"""

import base64
import binascii
import json
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Final
from urllib.parse import unquote

from gateway.controls.detection import detection_verdict
from gateway.controls.scanning import Hit, scan
from gateway.controls.text import SegmentKind, TextExtractor, TextSegment
from gateway.core.envelope import Interaction, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import ControlKind, Decision, Stage

DETECTED = "secret_detected"  # reason codes
CLEAN = "no_secrets"
UNSCANNABLE = "unscannable_content"


class SecretKind(StrEnum):
    """Span labels: what kind of credential was found."""

    API_KEY = "API_KEY"
    ACCESS_TOKEN = "ACCESS_TOKEN"  # noqa: S105 -- a span label, not a credential
    PRIVATE_KEY = "PRIVATE_KEY"
    CONNECTION_STRING = "CONNECTION_STRING"
    PASSWORD = "PASSWORD"  # noqa: S105 -- a span label, not a credential
    SECRET = "SECRET"  # noqa: S105 -- a span label, not a credential


type Validator = Callable[[str], bool]
type Classifier = Callable[[re.Match[str]], SecretKind]


@dataclass(frozen=True, slots=True)
class SecretFinding:
    rule: str
    kind: SecretKind
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SecretRule:
    """One credential family: ``pattern``'s group ``group`` is the secret (the first of the
    groups that took part in the match, when ``group`` names alternatives).

    ``validate`` rejects look-alikes by the secret's text; ``classify`` picks the label from
    the whole match when it depends on context (``db_password=`` vs ``api_key=``). ``hints``
    are lower-case literals every match contains: a text holding none of them is skipped
    without running the regex (a substring test is far cheaper than a regex scan).
    """

    id: str
    kind: SecretKind
    pattern: re.Pattern[str]
    group: int | tuple[int, ...] = 0
    validate: Validator | None = None
    classify: Classifier | None = None
    hints: tuple[str, ...] = ()

    def find(self, text: str) -> Iterator[SecretFinding]:
        groups = (self.group,) if isinstance(self.group, int) else self.group
        for match in self.pattern.finditer(text):
            group = next((g for g in groups if match.start(g) >= 0), groups[0])
            start, end = match.span(group)
            if end <= start:
                continue
            if self.validate is not None and not self.validate(match[group]):
                continue
            kind = self.classify(match) if self.classify is not None else self.kind
            yield SecretFinding(self.id, kind, start, end)


# ----------------------------------------------------------------------- validators

_MIN_ASSIGNED_SECRET: Final = 8
_LONG_ASSIGNED_SECRET: Final = 16
_MIN_CHARACTER_CLASSES: Final = 2
_PLACEHOLDER_WORDS: Final = frozenset(
    {
        "password", "passwd", "secret", "token", "apikey", "api_key", "changeme", "example",
        "redacted", "placeholder", "username", "none", "null", "undefined", "xxxxxxxx",
    }
)  # fmt: skip
# Complete placeholder expressions: a value is a placeholder only when it is one of these whole.
_PLACEHOLDER: Final = re.compile(
    r"\$\{[A-Za-z_][A-Za-z0-9_]{0,63}\}"  # ${DB_PASSWORD}
    r"|\$[A-Z_][A-Z0-9_]{0,63}"  # $DB_PASSWORD
    r"|%[A-Za-z_][A-Za-z0-9_]{0,63}%"  # %DB_PASSWORD%
    r"|%\([A-Za-z_][A-Za-z0-9_]{0,63}\)s"  # %(password)s
    r"|\{\{ {0,4}[A-Za-z_][\w.]{0,63} {0,4}\}\}"  # {{ password }}
    r"|\{[A-Za-z_][A-Za-z0-9_]{0,63}\}"  # {password}
    r"|<[A-Za-z][\w .-]{0,63}>"  # <your-password>
    r"|(?:os\.)?(?:environ|getenv)(?:\.get)?|process\.env(?:\.[A-Za-z_]\w{0,63})?"  # references
    r"|(?:your|my)[_-]?(?:db[_-]?)?(?:password|passwd|secret|token|api[_-]?key|key)"
)


def _character_classes(value: str) -> int:
    return sum(
        (
            any(c.islower() for c in value),
            any(c.isupper() for c in value),
            any(c.isdigit() for c in value),
            any(not c.isalnum() for c in value),
        )
    )


def _mixed_token(value: str) -> bool:
    """Random-looking: lower case, upper case and digits all present."""
    return (
        any(c.islower() for c in value)
        and any(c.isupper() for c in value)
        and any(c.isdigit() for c in value)
    )


def _has_digit(value: str) -> bool:
    return any(c.isdigit() for c in value)


def _is_placeholder(value: str) -> bool:
    """The whole value is a placeholder word, expression, or one repeated character."""
    stripped = value.strip("'\"")
    return (
        stripped.lower() in _PLACEHOLDER_WORDS
        or len(set(stripped)) <= 1  # ****, xxxxxxxx
        or _PLACEHOLDER.fullmatch(stripped) is not None
    )


def _real_password(value: str) -> bool:
    """Not a placeholder, and either long or mixing character classes."""
    if len(value) < _MIN_ASSIGNED_SECRET or _is_placeholder(value):
        return False
    return len(value) >= _LONG_ASSIGNED_SECRET or _character_classes(value) >= (
        _MIN_CHARACTER_CLASSES
    )


def _real_uri_password(value: str) -> bool:
    """A connection string's password, judged after percent-decoding (``%24%7BX%7D``)."""
    return not _is_placeholder(unquote(value))


def _not_aws_example(suffix: str) -> Validator:
    def check(value: str) -> bool:
        return not value.endswith(suffix)

    return check


def _jwt_header(token: str) -> bool:
    """The first segment is base64url JSON naming an ``alg``, as every JWS/JWE header does."""
    header = token.split(".", 1)[0]
    try:
        decoded: object = json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))
    except (binascii.Error, ValueError):
        return False
    return isinstance(decoded, dict) and isinstance(decoded.get("alg"), str)  # pyright: ignore[reportUnknownMemberType] -- JSON object


# ---------------------------------------------------------------------------- rules

_BOUNDARY_BEFORE: Final = r"(?<![A-Za-z0-9_-])"
_BOUNDARY_AFTER: Final = r"(?![A-Za-z0-9_-])"
_ASSIGNED_NAME: Final = (
    r"(?<![A-Za-z0-9])"  # db_password and my-api-key match at the keyword itself
    r"(password|passwd|pwd|api[_-]?key|apikey|secret(?:[_-]?key)?|client[_-]?secret"
    r"|access[_-]?token|auth[_-]?token|private[_-]?key)"
)
_SCHEMES: Final = (
    r"(?:postgres(?:ql)?|mysql|mariadb|mssql|sqlserver|mongodb(?:\+srv)?|rediss?|amqps?)"
)


_ASSIGNED_KINDS: Final = (
    (("pass", "pwd"), SecretKind.PASSWORD),
    (("token",), SecretKind.ACCESS_TOKEN),
    (("api",), SecretKind.API_KEY),
)


def _assigned_kind(match: re.Match[str]) -> SecretKind:
    """The label an assignment's name implies: ``db_password`` → PASSWORD, ..."""
    lowered = match[1].lower()
    for needles, kind in _ASSIGNED_KINDS:
        if any(needle in lowered for needle in needles):
            return kind
    return SecretKind.SECRET


RULES: Final[tuple[SecretRule, ...]] = (
    SecretRule(
        "aws_access_key_id",
        SecretKind.API_KEY,
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16}(?![A-Z0-9])"),
        validate=_not_aws_example("EXAMPLE"),
        hints=("akia", "asia", "abia", "acca"),
    ),
    SecretRule(
        "aws_secret_access_key",
        SecretKind.API_KEY,
        re.compile(
            r"(?i:aws[_-]?secret(?:[_-]?access)?[_-]?key|secret[_-]?access[_-]?key)"
            r"[\"']?\s{0,4}[:=]{1,2}\s{0,4}[\"']?([A-Za-z0-9/+]{40})(?![A-Za-z0-9/+=])"
        ),
        group=1,
        validate=_not_aws_example("EXAMPLEKEY"),
        hints=("secret",),
    ),
    SecretRule(
        "github_token",
        SecretKind.ACCESS_TOKEN,
        re.compile(
            rf"{_BOUNDARY_BEFORE}(?:gh[pousr]_[A-Za-z0-9]{{36,251}}|github_pat_[A-Za-z0-9_]{{82,240}})"
            rf"{_BOUNDARY_AFTER}"
        ),
        validate=_has_digit,
        hints=("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"),
    ),
    SecretRule(
        "anthropic_key",
        SecretKind.API_KEY,
        re.compile(
            rf"{_BOUNDARY_BEFORE}sk-ant-[a-z]{{2,10}}[0-9]{{0,4}}-[A-Za-z0-9_-]{{32,250}}"
            rf"{_BOUNDARY_AFTER}"
        ),
        validate=_mixed_token,
        hints=("sk-ant-",),
    ),
    SecretRule(
        "openai_key",
        SecretKind.API_KEY,
        re.compile(
            rf"{_BOUNDARY_BEFORE}sk-(?:(?:proj|svcacct|admin)-[A-Za-z0-9_-]{{40,250}}"
            rf"|[A-Za-z0-9]{{32,200}}){_BOUNDARY_AFTER}"
        ),
        validate=_mixed_token,
        hints=("sk-",),
    ),
    SecretRule(
        "slack_token",
        SecretKind.ACCESS_TOKEN,
        re.compile(rf"{_BOUNDARY_BEFORE}xox[abposr]-[A-Za-z0-9-]{{10,250}}{_BOUNDARY_AFTER}"),
        validate=_has_digit,
        hints=("xox",),
    ),
    SecretRule(
        "stripe_live_key",
        SecretKind.API_KEY,
        re.compile(rf"{_BOUNDARY_BEFORE}(?:sk|rk)_live_[A-Za-z0-9]{{20,250}}{_BOUNDARY_AFTER}"),
        hints=("_live_",),
    ),
    SecretRule(
        "google_api_key",
        SecretKind.API_KEY,
        re.compile(rf"{_BOUNDARY_BEFORE}AIza[A-Za-z0-9_-]{{35}}{_BOUNDARY_AFTER}"),
        hints=("aiza",),
    ),
    SecretRule(
        "jwt",
        SecretKind.ACCESS_TOKEN,
        re.compile(
            rf"{_BOUNDARY_BEFORE}eyJ[A-Za-z0-9_-]{{6,4096}}\.[A-Za-z0-9_-]{{2,16384}}"
            rf"\.[A-Za-z0-9_-]{{0,4096}}{_BOUNDARY_AFTER}"
        ),
        validate=_jwt_header,
        hints=("eyj",),
    ),
    SecretRule(
        "private_key",
        SecretKind.PRIVATE_KEY,
        re.compile(
            # A whole block, or a header followed by key material when the END line is missing.
            r"-----BEGIN (?:[A-Z0-9]{1,16} ){0,3}PRIVATE KEY(?: BLOCK)?-----"
            r"(?:[A-Za-z0-9+/=\s\\:,.-]{0,16384}?-----END (?:[A-Z0-9]{1,16} ){0,3}PRIVATE KEY"
            r"(?: BLOCK)?-----|[A-Za-z0-9+/=\s\\:,.-]{0,16384})"
        ),
        hints=("private key",),
    ),
    SecretRule(
        "connection_string_password",
        SecretKind.CONNECTION_STRING,
        re.compile(rf"(?i:{_SCHEMES})://[^\s:/@'\"]{{0,128}}:([^\s/@'\"]{{1,256}})@"),
        group=1,
        validate=_real_uri_password,
        hints=("://",),
    ),
    SecretRule(
        "assigned_secret",
        SecretKind.SECRET,
        re.compile(
            rf"(?i:{_ASSIGNED_NAME})(?![A-Za-z0-9])[\"']?\s{{0,4}}[:=]{{1,2}}\s{{0,4}}"
            # A quoted value runs to its closing quote (escapes skipped), or 256 characters.
            r"(?:\"((?:[^\"\\\n]|\\.){1,256})|'((?:[^'\\\n]|\\.){1,256})"
            # An unquoted one to a space or separator. `&` is kept: masking the rest of a
            # query string is safer than leaving part of a password behind.
            r"|([^\s\"'`,;<>(){}\[\]]{8,256}))"
        ),
        group=(2, 3, 4),
        validate=_real_password,
        classify=_assigned_kind,
        hints=("pass", "pwd", "key", "secret", "token"),
    ),
)


class SecretScanner:
    """Runs every rule over a string; overlapping findings keep the earliest, longest one."""

    def __init__(self, rules: Sequence[SecretRule] = RULES) -> None:
        self._rules = tuple(rules)

    def find(self, text: str) -> list[SecretFinding]:
        lowered = text.lower()  # for the hint tests only; offsets always come from `text`
        found = [
            finding
            for rule in self._rules
            if not rule.hints or any(hint in lowered for hint in rule.hints)
            for finding in rule.find(text)
        ]
        kept: list[SecretFinding] = []
        for finding in sorted(found, key=lambda f: (f.start, -f.end)):
            if kept and finding.start < kept[-1].end:
                continue
            kept.append(finding)
        return kept


# A JSON key whose (lower-cased, alphanumeric) name ends with one of these names a credential.
_SENSITIVE_KEYS: Final = (
    (("password", "passwd", "pwd", "passphrase"), SecretKind.PASSWORD),
    (("privatekey",), SecretKind.PRIVATE_KEY),
    (("apikey", "accesskey"), SecretKind.API_KEY),
    (("token", "authorization"), SecretKind.ACCESS_TOKEN),
    (("secret", "secretkey", "credential", "credentials"), SecretKind.SECRET),
)
_MIN_KEYED_SECRET: Final = 6


def sensitive_key(key: str | None) -> SecretKind | None:
    """The credential a JSON key names (``db_password``, ``X-Api-Key``), or None."""
    if not key:
        return None
    name = re.sub(r"[^a-z0-9]", "", key.lower())
    for suffixes, kind in _SENSITIVE_KEYS:
        if name.endswith(suffixes):
            return kind
    return None


def keyed_secrets(segments: Iterable[TextSegment]) -> list[Hit]:
    """Whole values stored under a credential's key, unless they are placeholders."""
    hits: list[Hit] = []
    for segment in segments:
        kind = sensitive_key(segment.key)
        value = segment.text.strip()
        if kind is None or len(value) < _MIN_KEYED_SECRET or _is_placeholder(value):
            continue
        hits.append(Hit(segment, 0, len(segment.text), kind.value))
    return hits


class SecretsControl(Control):
    id: ClassVar[str] = "secrets"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE, Stage.POST})
    kind: ClassVar[ControlKind] = ControlKind.DETERMINISTIC
    mandatory: ClassVar[bool] = True

    def __init__(
        self, scanner: SecretScanner | None = None, extractor: TextExtractor | None = None
    ) -> None:
        self._scanner = scanner if scanner is not None else SecretScanner()
        self._extractor = extractor if extractor is not None else TextExtractor()

    async def evaluate(self, interaction: Interaction, stage: Stage, cfg: ControlConfig) -> Verdict:
        segments = self._extractor.segments(interaction, stage)
        if any(segment.kind is SegmentKind.UNSCANNABLE for segment in segments):
            return Verdict(
                decision=Decision.BLOCK,
                control_id=self.id,
                reason_code=UNSCANNABLE,
                reason="content could not be decoded for scanning",
            )
        hits = scan(segments, self._detect) + keyed_secrets(segments)
        return detection_verdict(self.id, cfg, hits, detected=DETECTED, clean=CLEAN)

    def _detect(self, text: str) -> list[tuple[int, int, str]]:
        return [(f.start, f.end, f.kind.value) for f in self._scanner.find(text)]
