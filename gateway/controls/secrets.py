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
- ``password=`` / ``api_key:`` / ``secret=`` style assignments with a non-trivial value.

False positives are kept low on purpose: provider tokens must mix character classes
(``sk-learn`` and ``sk-some-long-lowercase-words`` are prose), placeholders (``<password>``,
``${DB_PASSWORD}``, ``changeme``-style words, ``****``) are not secrets, and AWS's documented
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
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Final

from gateway.controls.detection import detection_verdict
from gateway.controls.text import TextExtractor
from gateway.core.envelope import Interaction, Span, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import ControlKind, Stage

DETECTED = "secret_detected"  # reason codes
CLEAN = "no_secrets"


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
    """One credential family: ``pattern``'s group ``group`` is the secret.

    ``validate`` rejects look-alikes by the secret's text; ``classify`` picks the label from
    the whole match when it depends on context (``db_password=`` vs ``api_key=``). ``hints``
    are lower-case literals every match contains: a text holding none of them is skipped
    without running the regex (a substring test is far cheaper than a regex scan).
    """

    id: str
    kind: SecretKind
    pattern: re.Pattern[str]
    group: int = 0
    validate: Validator | None = None
    classify: Classifier | None = None
    hints: tuple[str, ...] = ()

    def find(self, text: str) -> Iterator[SecretFinding]:
        for match in self.pattern.finditer(text):
            start, end = match.span(self.group)
            if end <= start:
                continue
            if self.validate is not None and not self.validate(match[self.group]):
                continue
            kind = self.classify(match) if self.classify is not None else self.kind
            yield SecretFinding(self.id, kind, start, end)


# ----------------------------------------------------------------------- validators

_MIN_ASSIGNED_SECRET: Final = 8
_LONG_ASSIGNED_SECRET: Final = 16
_MIN_CHARACTER_CLASSES: Final = 2
_PLACEHOLDER_WORDS: Final = frozenset(
    {
        "password", "passwd", "pass", "pwd", "secret", "token", "apikey", "api_key",
        "changeme", "example", "redacted", "placeholder", "username", "user", "none", "null",
        "your_password", "yourpassword", "mypassword", "your_api_key", "your_token",
        "your_secret", "xxxxxxxx",
    }
)  # fmt: skip
_TEMPLATE: Final = re.compile(r"^(?:\$|%|\{|<|\[)|(?:\}|>|\])$|environ|getenv|process\.env")


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
    lowered = value.lower().strip("'\"")
    return (
        lowered in _PLACEHOLDER_WORDS
        or len(set(lowered)) == 1  # ****, xxxxxxxx
        or _TEMPLATE.search(lowered) is not None
        or lowered.startswith(("your", "<", "example"))
    )


def _real_password(value: str) -> bool:
    """Not a placeholder, and either long or mixing character classes."""
    if len(value) < _MIN_ASSIGNED_SECRET or _is_placeholder(value):
        return False
    return len(value) >= _LONG_ASSIGNED_SECRET or _character_classes(value) >= (
        _MIN_CHARACTER_CLASSES
    )


def _not_placeholder(value: str) -> bool:
    return not _is_placeholder(value)


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
        validate=_not_placeholder,
        hints=("://",),
    ),
    SecretRule(
        "assigned_secret",
        SecretKind.SECRET,
        re.compile(
            rf"(?i:{_ASSIGNED_NAME})(?![A-Za-z0-9])[\"']?\s{{0,4}}[:=]{{1,2}}\s{{0,4}}[\"']?"
            # The value runs to a quote, space or separator. `&` is kept: masking the rest
            # of a query string is safer than leaving part of a password behind.
            r"([^\s\"'`,;<>(){}\[\]]{8,256})"
        ),
        group=2,
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
        spans = [
            Span(path=segment.pointer, start=f.start, end=f.end, label=f.kind.value)
            for segment in self._extractor.segments(interaction, stage)
            for f in self._scanner.find(segment.text)
        ]
        return detection_verdict(self.id, cfg, spans, detected=DETECTED, clean=CLEAN)
