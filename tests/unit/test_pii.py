"""``pii``: Polish IDs by checksum, Presidio's IBAN/e-mail/phone, threshold, spans, modes."""

import asyncio
from typing import Any

import pytest

from gateway.controls import pii as pii_module
from gateway.controls.pii import NO_PII, PII_DETECTED, PiiControl
from gateway.controls.pii_recognizers import default_analyzer, is_valid_nip, is_valid_pesel
from gateway.core.envelope import Interaction, Span
from gateway.core.interfaces import ControlConfig
from gateway.core.types import Action, Channel, ControlMode, Decision, Stage
from gateway.policy.schema import PiiConfig, PiiEntity

CONTROL = PiiControl()
ALL = tuple(PiiEntity)
ALLOW = pytest.mark.control("pii", "allow")
DENY = pytest.mark.control("pii", "deny")
REDACT = pytest.mark.control("pii", "redact")
LOG_ONLY = pytest.mark.control("pii", "log_only")

PESEL = "44051401359"  # 1944-05-14
IBAN = "PL61 1090 1014 0000 0712 1981 2874"


def chat(make_ctx, content: str) -> Interaction:
    return Interaction(
        session_id="s-test",
        principal="anna@demo",
        actor="databot",
        mode=make_ctx().mode,
        channel=Channel.LLM,
        action=Action.GENERATE,
        resource="model:qwen3:8b",
        payload={"model": "qwen3:8b", "messages": [{"role": "user", "content": content}]},
        context=make_ctx(),
    )


def found(text: str, threshold: float = 0.6, entities: tuple[PiiEntity, ...] = ALL):
    return [
        (f.entity.value, text[f.start : f.end])
        for f in default_analyzer().find(text, entities, threshold)
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # PESEL: 11 digits, check digit, a real date with the century in the month
        (f"PESEL {PESEL}.", [("PL_PESEL", PESEL)]),
        ("born 2002: 02270803624", [("PL_PESEL", "02270803624")]),
        ("leap day 2000: 00222900016", [("PL_PESEL", "00222900016")]),
        ("born 1884: 84810101010", [("PL_PESEL", "84810101010")]),
        ("bad check digit 44051401358", []),
        ("checksum fine, 1999-02-29 is no date: 99022900014", []),
        ("twelve digits 440514013591", []),
        # NIP: bare, 3-3-2-2 and 3-2-2-3 dashes, weights 6,5,7,2,3,4,5,6,7
        ("NIP 1234563218", [("PL_NIP", "1234563218")]),
        ("NIP 123-456-32-18", [("PL_NIP", "123-456-32-18")]),
        ("NIP 123-45-63-218", [("PL_NIP", "123-45-63-218")]),
        ("VAT PL5260001246", [("PL_NIP", "5260001246")]),
        ("bad checksum 1234563219", []),
        ("bad checksum 123-456-32-19", []),
        # IBAN: Presidio's recognizer validates the mod-97 checksum
        (f"konto {IBAN}", [("IBAN_CODE", IBAN)]),
        ("konto PL61109010140000071219812874", [("IBAN_CODE", "PL61109010140000071219812874")]),
        ("one digit off PL61 1090 1014 0000 0712 1981 2875", []),
        # e-mail: a known public suffix, read from the bundled list (no network)
        ("mail jan.kowalski@example.com", [("EMAIL_ADDRESS", "jan.kowalski@example.com")]),
        ("mail jan@firma.pl", [("EMAIL_ADDRESS", "jan@firma.pl")]),
        ("not an address: jan@localhost", []),
        # phone: python-phonenumbers, confident with a country code or a phone word nearby
        ("Zadzwoń +48 600 700 800", [("PHONE_NUMBER", "+48 600 700 800")]),
        ("call 0048 600 700 800", [("PHONE_NUMBER", "0048 600 700 800")]),
        ("tel. 600 700 800", [("PHONE_NUMBER", "600 700 800")]),
        ("order 600700800 shipped", []),  # a bare nine-digit number scores below 0.6
        # prose
        ("How many customers do we have in Kraków?", []),
    ],
)
def test_detection_table(text, expected):
    assert found(text) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (PESEL, True),
        ("4405140135", False),  # too short
        ("4405140135a", False),
        ("٤٤٠٥١٤٠١٣٥٩", False),  # Arabic-Indic digits pass str.isdigit, never a PESEL
        ("44053201359", False),  # month 32: no century encodes it
    ],
)
def test_pesel_validator(value, expected):
    assert is_valid_pesel(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1234563218", True), ("123-456-32-18", True), ("1234563210", False), ("123", False)],
)
def test_nip_validator(value, expected):
    assert is_valid_nip(value) is expected


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (0.3, [("PHONE_NUMBER", "600700800"), ("PHONE_NUMBER", "+48 600 700 800")]),
        (0.6, [("PHONE_NUMBER", "+48 600 700 800")]),
        (0.8, []),  # above the confident phone score; checksum entities score 1.0
        (1.0, []),
    ],
)
def test_threshold_filters_by_score(threshold, expected):
    assert found("order 600700800, or +48 600 700 800", threshold) == expected


def test_windowed_recognizers_keep_offsets_in_long_text():
    """E-mail and phone matching only read windows around candidates; offsets stay global."""
    filler = "Zażółć gęślą jaźń. " * 600  # ~11 KiB, non-ASCII, digit-free
    text = f"a@firma.pl,b@firma.pl {filler}tel. +48 600 700 800 {filler}koniec c@firma.pl"
    assert found(text) == [
        ("EMAIL_ADDRESS", "a@firma.pl"),
        ("EMAIL_ADDRESS", "b@firma.pl"),  # overlapping windows are merged, not doubled
        ("PHONE_NUMBER", "+48 600 700 800"),
        ("EMAIL_ADDRESS", "c@firma.pl"),
    ]


def test_dates_and_amounts_are_not_phone_numbers():
    assert found("termin 2026-09-14, kwota 1 234,56 zł, tel. (12) 345-67-89", 0.3) == [
        ("PHONE_NUMBER", "(12) 345-67-89")
    ]


@REDACT
@pytest.mark.parametrize(("repeat", "threaded"), [(10, False), (400, True)])
async def test_only_long_text_is_scanned_off_the_event_loop(
    make_ctx, monkeypatch, repeat, threaded
):
    """Over INLINE_SCAN_CHARS the scan runs in a worker thread; the verdict is the same."""
    hops: list[object] = []
    to_thread = asyncio.to_thread

    async def recording(func, /, *args):
        hops.append(func)
        return await to_thread(func, *args)

    monkeypatch.setattr(pii_module.asyncio, "to_thread", recording)
    content = "Zażółć gęślą jaźń. " * repeat + f"PESEL {PESEL}"
    verdict = await CONTROL.evaluate(
        chat(make_ctx, content), Stage.PRE, PiiConfig(mode=ControlMode.REDACT)
    )
    start = content.index(PESEL)
    assert [(s.start, s.end, s.label) for s in verdict.redactions] == [
        (start, start + 11, "PL_PESEL")
    ]
    assert bool(hops) is threaded


def test_threshold_keeps_checksum_entities_at_one():
    assert found(f"{PESEL} {IBAN}", 1.0) == [("PL_PESEL", PESEL), ("IBAN_CODE", IBAN)]


def test_only_configured_entities_are_reported():
    text = f"{PESEL} jan@firma.pl"
    assert found(text, entities=(PiiEntity.EMAIL_ADDRESS,)) == [("EMAIL_ADDRESS", "jan@firma.pl")]
    assert found(text, entities=()) == []


@REDACT
async def test_spans_are_json_pointers_with_code_point_offsets(make_ctx):
    content = f"Zażółć gęślą jaźń: {PESEL}, mail jan@firma.pl"
    verdict = await CONTROL.evaluate(
        chat(make_ctx, content), Stage.PRE, PiiConfig(mode=ControlMode.REDACT)
    )
    start = content.index(PESEL)
    assert start != content.encode().index(PESEL.encode())  # bytes and code points differ
    email = content.index("jan@")
    assert verdict.redactions == (
        Span(path="/messages/0/content", start=start, end=start + 11, label="PL_PESEL"),
        Span(path="/messages/0/content", start=email, end=email + 12, label="EMAIL_ADDRESS"),
    )


@ALLOW
async def test_post_stage_scans_the_answer_not_the_prompt(make_ctx):
    item = chat(make_ctx, f"my PESEL is {PESEL}").model_copy(
        update={"result": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}}
    )
    verdict = await CONTROL.evaluate(item, Stage.POST, PiiConfig(mode=ControlMode.REDACT))
    assert (verdict.decision, verdict.reason_code) == (Decision.ALLOW, NO_PII)


@pytest.mark.parametrize(
    ("mode", "content", "decision", "enforced", "reason_code", "risk"),
    [
        pytest.param(ControlMode.REDACT, f"PESEL {PESEL}", Decision.REDACT, True, PII_DETECTED,
                     0.1, marks=REDACT),
        pytest.param(ControlMode.REDACT, "no personal data", Decision.ALLOW, True, NO_PII, 0.0,
                     marks=ALLOW),
        pytest.param(ControlMode.BLOCK, f"PESEL {PESEL}", Decision.BLOCK, True, PII_DETECTED,
                     0.1, marks=DENY),
        pytest.param(ControlMode.BLOCK, "no personal data", Decision.ALLOW, True, NO_PII, 0.0,
                     marks=ALLOW),
        pytest.param(ControlMode.LOG_ONLY, f"PESEL {PESEL}", Decision.REDACT, False,
                     PII_DETECTED, 0.1, marks=LOG_ONLY),
        pytest.param(ControlMode.LOG_ONLY, "no personal data", Decision.ALLOW, True, NO_PII,
                     0.0, marks=ALLOW),
    ],
)  # fmt: skip
async def test_modes(make_ctx, mode, content, decision, enforced, reason_code, risk):
    verdict = await CONTROL.evaluate(chat(make_ctx, content), Stage.PRE, PiiConfig(mode=mode))
    assert (verdict.decision, verdict.enforced, verdict.reason_code) == (
        decision,
        enforced,
        reason_code,
    )
    assert verdict.risk_delta == pytest.approx(risk)  # the catalog default when unset


@DENY
async def test_reason_names_entities_never_values_and_risk_comes_from_config(make_ctx):
    content = f"{PESEL}, {IBAN}, jan@firma.pl"
    cfg = PiiConfig(mode=ControlMode.BLOCK, risk_delta=0.25)
    verdict = await CONTROL.evaluate(chat(make_ctx, content), Stage.PRE, cfg)
    assert verdict.reason == "EMAIL_ADDRESS, IBAN_CODE, PL_PESEL"
    assert verdict.risk_delta == 0.25
    dumped: dict[str, Any] = verdict.model_dump(exclude={"redactions"})
    assert not any(v in str(dumped) for v in (PESEL, "1090", "jan@firma.pl"))


@REDACT
async def test_a_plain_control_config_gets_pii_defaults(make_ctx):
    verdict = await CONTROL.evaluate(
        chat(make_ctx, f"PESEL {PESEL}"), Stage.PRE, ControlConfig(mode=ControlMode.REDACT)
    )
    assert verdict.decision is Decision.REDACT


async def test_an_unsupported_mode_fails_loudly(make_ctx):
    with pytest.raises(ValueError, match="does not support"):
        await CONTROL.evaluate(
            chat(make_ctx, PESEL), Stage.PRE, PiiConfig(mode=ControlMode.REQUIRE_APPROVAL)
        )


def test_policy_entities_are_validated():
    assert PiiConfig().entities == ALL
    assert PiiConfig(entities=("PL_PESEL",)).entities == (PiiEntity.PL_PESEL,)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="PL_PASSPORT"):
        PiiConfig(entities=("PL_PASSPORT",))  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="duplicate"):
        PiiConfig(entities=(PiiEntity.PL_NIP, PiiEntity.PL_NIP))
