"""The attack corpus (``corpus.yaml``): its schema, and one runner per channel.

Every case is sent through the real apps (`gateway_testkit.running_gateway`): LLM cases
against a mocked upstream (respx), MCP cases against the in-process MCP servers of
``tests/mcp/upstreams.py``. The runner observes what an attacker would see (the decision and
reason code of the refusal), what reached the upstream, the audit entries and the session's
taint, and `check` compares that with the case's ``expect`` block.

Payload strings may name a fake credential as ``{{NAME}}`` (see `FAKES`): the values are
assembled at import, so neither the corpus nor this file carries a scanner-shaped secret.

``fake_injection`` lists substrings that the deterministic `MarkerClassifier` scores as an
injection (0.99); ``fake_doubt`` substrings score 0.6, inside the judge band, where the fake LLM
judge decides (``setup.tool_judge`` scripts its ``tool_poisoning`` answer,
``setup.prompt_judge`` its answer about a hit on the user's own prompt). The same payload
runs again with the real pinned classifier (``test_attack_real_model``, marked ``model``) when
the case sets ``real_model``.
"""

import importlib
import json
import re
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal

import httpx
import jwt
import pytest
import respx
import yaml
from gateway_testkit import (
    LLM_BASE,
    Harness,
    bearer,
    chat,
    claims,
    completion,
    echo_completion,
    running_gateway,
    sign,
    unsigned,
)
from injection_kit import MarkerClassifier
from judge_kit import DEFAULT_ANSWERS
from plugins.control_report import ControlClaim
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gateway.injection.classifier import InjectionClassifier
from gateway.judges.client import JudgeResult, JudgeUnavailableError
from gateway.sessions import SessionUpdate

# tests/mcp is not on this directory's import path (and not a package): load its harness by
# name, as tests/approvals does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp"))
mcp_harness = importlib.import_module("mcp_harness")
upstreams = importlib.import_module("upstreams")
pin_kit = importlib.import_module("pin_kit")

CORPUS_PATH: Final = Path(__file__).with_name("corpus.yaml")
CHAT: Final = "/v1/chat/completions"
META: Final = "ai-control-layer/approval_id"
HOSTS: Final = {"sales_db": "mcp-postgres", "web": "mcp-fetch", "reports": "mcp-files"}
OPERATOR: Final = "olga@demo"  # ops-team: approves nightly_etl and databot calls

FAKES: Final[dict[str, str]] = {
    "OPENAI_KEY": "sk-proj-" + "Ab3De5Fg7Hi9Jk1Lm3No5Pq7Rs9Tu1Vw3Xy5Za7Bc9De",
    "OPENAI_KEY_HEAD": "sk-proj-" + "Ab3De5Fg7Hi9Jk1Lm3No",
    "OPENAI_KEY_TAIL": "5Pq7Rs9Tu1Vw3Xy5Za7Bc9De",
    "STRIPE_KEY": "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc",
    "STRIPE_KEY_ESCAPED": "\\u0073k_live_" + "4eC39HqLyjWDarjtT1zdp7dc",  # JSON-escaped "s"
    "GITHUB_TOKEN": "ghp_" + "aB3dE5fG7hI9jK1lM3nO5pQ7rS9tU1vW3xY5",
    "PG_URL": "postgresql://etl_admin:" + "Wq7rT2vX9kLm4Pz" + "@db.internal:5432/sales",
    "PG_PASSWORD": "Wq7rT2vX9kLm4Pz",
    "RSA_HEADER": "-----BEGIN " + "RSA PRIVATE KEY-----",
    "PESEL": "44051401359",
    "IBAN": "PL61109010140000071219812874",
}
_FAKE: Final = re.compile(r"\{\{([A-Z_]+)\}\}")


def expand(value: Any) -> Any:
    """``value`` with every ``{{NAME}}`` replaced by its fake (recursively)."""
    if isinstance(value, str):
        return _FAKE.sub(lambda m: FAKES[m[1]], value)
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


# ------------------------------------------------------------------------------ schema


class Category(StrEnum):
    DIRECT_INJECTION = "direct_prompt_injection"
    INDIRECT_INJECTION = "indirect_injection"
    CREDENTIAL_LEAKAGE = "secret_leakage"
    PII_EXFILTRATION = "pii_exfiltration"
    TOOL_POISONING = "tool_poisoning"
    SQL_ATTACK = "sql_attack"
    PATH_TRAVERSAL = "path_traversal"
    SSRF = "ssrf"
    RESOURCE_ABUSE = "resource_abuse"
    APPROVAL_ABUSE = "approval_abuse"
    IDENTITY_ABUSE = "identity_abuse"
    BENIGN = "benign"  # look-alikes that must pass: the corpus's false-positive guard


type Decision = Literal["allow", "redact", "block", "require_approval"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectedVerdict(_Strict):
    control: str
    decision: Decision
    reason_code: str | None = None


class Expect(_Strict):
    decision: Decision
    reason: list[str] = Field(default_factory=list)  # any of these codes (block/approval)
    verdicts: list[ExpectedVerdict] = Field(default_factory=list)  # in some audit entry
    upstream_called: bool | None = None  # did the protected upstream (model or tool) run
    taint: bool | None = None
    absent: list[str] = Field(default_factory=list)  # never in the agent's answer or audit
    absent_upstream: list[str] = Field(default_factory=list)  # never reached the upstream
    present_upstream: list[str] = Field(default_factory=list)  # what the upstream got instead

    @field_validator("reason", mode="before")
    @classmethod
    def _one_or_many(cls, value: object) -> object:
        return [value] if isinstance(value, str) else value


class Describe(_Strict):
    """Change a tool's description on its server, before the operator pins it (poisoned at
    registration) or after (a rug pull)."""

    server: str
    tool: str
    description: str
    when: Literal["before_pin", "after_pin"] = "before_pin"


class Setup(_Strict):
    fetch_page: str | None = None  # what the web server's fetch answers
    dns: dict[str, list[str]] = Field(default_factory=dict)  # egress resolver answers
    plan_cost: float | None = None  # what sales_db's planner answers
    describe: Describe | None = None
    policy: dict[str, Any] = Field(default_factory=dict)  # deep-merged into policy.yaml
    risk: float | None = None  # session risk raised before the attack call
    repeat: int = 1  # send the call this many times; the last one is observed
    vary: str | None = None  # suffix this argument with the attempt number
    # What the (fake) LLM judge answers tool_poisoning for a definition in its judge band.
    tool_judge: Literal["poisoned", "clean"] | None = None
    # What it answers prompt_injection about a hit on the user's own prompt (``timeout``: no
    # answer in time, so the prompt is allowed and the session tainted).
    prompt_judge: Literal["injection", "clean", "timeout"] | None = None


Channel = Literal["llm_prompt", "llm_response", "mcp_call", "mcp_listing", "scenario"]


class AttackCase(_Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    category: Category
    title: str
    channel: Channel
    sub: str = "anna@demo"
    server: str | None = None
    tool: str | None = None
    scenario: str | None = None
    payload: Any = None
    fake_injection: list[str] = Field(default_factory=list)
    fake_doubt: list[str] = Field(default_factory=list)  # scored 0.6: inside the judge band
    setup: Setup = Setup()
    expect: Expect
    proves: list[str] = Field(min_length=1)  # "<control>:<outcome>": the case's markers
    real_model: Decision | None = None  # expected decision with the real classifier
    see: list[str] = Field(default_factory=list)  # the deep tests behind the case

    def claims(self) -> list[ControlClaim]:
        return [ControlClaim.from_property(p) for p in self.proves]

    def marks(self) -> list[pytest.MarkDecorator]:
        return [pytest.mark.control(c.control, c.outcome.value) for c in self.claims()]

    def classifier(self) -> MarkerClassifier:
        scores = dict.fromkeys(expand(self.fake_doubt), 0.6)
        scores.update(dict.fromkeys(expand(self.fake_injection), 0.99))
        return MarkerClassifier(scores=scores)


def load_corpus(path: Path = CORPUS_PATH) -> list[AttackCase]:
    cases = [AttackCase.model_validate(c) for c in yaml.safe_load(path.read_text())["cases"]]
    ids = [c.id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        msg = f"duplicate corpus ids: {sorted(duplicates)}"
        raise ValueError(msg)
    return cases


# -------------------------------------------------------------------------- observing


class Observed(BaseModel):
    decision: Decision
    reason: str | None
    agent_saw: str  # the response body or tool result as the agent got it
    upstream_got: str  # every request body the protected upstream received
    upstream_called: bool
    audit_raw: str
    audit: list[dict[str, Any]]
    taint: bool | None


def _first_word(text: str) -> str:
    return re.split(r"[\s:;,]", text.strip(), maxsplit=1)[0]


def _audit_decision(gateway: Harness) -> Decision:
    entries = gateway.audit_entries()
    return "redact" if entries and entries[-1].get("decision") == "redact" else "allow"


async def _taint(gateway: Harness) -> bool | None:
    entries = gateway.audit_entries()
    if not entries or not entries[-1].get("session_id"):
        return None
    session = await gateway.container.sessions.get(entries[-1]["session_id"])
    return None if session is None else session.taint


async def _observe_http(
    gateway: Harness, response: httpx.Response, upstream_got: Sequence[str]
) -> Observed:
    if response.status_code == 200:
        decision, reason = _audit_decision(gateway), None
    else:
        error = response.json().get("error", {})
        code = error.get("code")  # HTTP API: a string; MCP JSON-RPC: an int plus data
        reason = code if isinstance(code, str) else error.get("data", {}).get("reason_code")
        decision = "require_approval" if reason == "approval_required" else "block"
    return Observed(
        decision=decision,
        reason=reason,
        agent_saw=response.text,
        upstream_got="\n".join(upstream_got),
        upstream_called=bool(upstream_got),
        audit_raw=gateway.audit.getvalue(),
        audit=gateway.audit_entries(),
        taint=await _taint(gateway),
    )


async def _observe_tool(stack: Any, result: dict[str, Any], tool: str) -> Observed:
    gateway: Harness = stack.gateway
    if result.get("isError"):
        text = result["content"][0]["text"]
        reason = _first_word(text)
        decision: Decision = "require_approval" if reason == "approval_required" else "block"
    else:
        decision, reason = _audit_decision(gateway), None
    calls = stack.log.of(tool)
    return Observed(
        decision=decision,
        reason=reason,
        agent_saw=json.dumps(result),
        upstream_got="\n".join(json.dumps(c.arguments) for c in calls),
        upstream_called=bool(calls),
        audit_raw=gateway.audit.getvalue(),
        audit=gateway.audit_entries(),
        taint=await _taint(gateway),
    )


def check(case: AttackCase, seen: Observed, *, decision: Decision | None = None) -> None:
    """Assert ``seen`` matches ``case.expect`` (``decision`` overrides the expected one)."""
    expect = expand(case.expect.model_dump())
    wanted = decision or expect["decision"]
    assert seen.decision == wanted, (
        f"{case.id}: expected {wanted}, got {seen.decision} ({seen.reason})"
    )
    if expect["reason"] and wanted == expect["decision"]:
        assert seen.reason in expect["reason"], f"{case.id}: reason {seen.reason}"
    if wanted != expect["decision"]:
        return  # a real-model divergence is reported by decision only
    verdicts = [v for entry in seen.audit for v in entry.get("verdicts", ())]
    for want in expect["verdicts"]:
        assert any(
            v["control"] == want["control"]
            and v["decision"] == want["decision"]
            and (want["reason_code"] is None or v["reason_code"] == want["reason_code"])
            for v in verdicts
        ), f"{case.id}: no {want} among {verdicts}"
    if expect["upstream_called"] is not None:
        assert seen.upstream_called is expect["upstream_called"], f"{case.id}: upstream"
    if expect["taint"] is not None:
        assert seen.taint is expect["taint"], f"{case.id}: taint {seen.taint}"
    for value in expect["absent"]:
        assert value not in seen.agent_saw, f"{case.id}: leaked to the agent"
        assert value not in seen.audit_raw, f"{case.id}: leaked to the audit log"
    for value in expect["absent_upstream"]:
        assert value not in seen.upstream_got, f"{case.id}: reached the upstream"
    for value in expect["present_upstream"]:
        assert value in seen.upstream_got, f"{case.id}: {value!r} did not reach the upstream"


# ---------------------------------------------------------------------------- stacks


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def apply_policy(gateway: Harness, patch: dict[str, Any]) -> None:
    if not patch:
        return
    document = yaml.safe_load(gateway.policy_path.read_text())
    _deep_merge(document, patch)
    gateway.policy_path.write_text(yaml.safe_dump(document, sort_keys=False))
    assert gateway.container.policy_store.reload().error is None


async def raise_risk(gateway: Harness, token: str, risk: float | None) -> None:
    if risk is None:
        return
    session_id = jwt.decode(token, options={"verify_signature": False})["session_id"]
    await gateway.container.sessions.apply(
        session_id, SessionUpdate(risk_delta=risk), half_life_s=600
    )


def script_judge(gateway: Harness, setup: Setup) -> None:
    """Make the fake judge answer ``tool_poisoning`` (``setup.tool_judge``) and
    ``prompt_injection`` (``setup.prompt_judge``) as the case says; anything unscripted keeps
    the default answer (`judge_kit.DEFAULT_ANSWERS`)."""
    if setup.tool_judge is None and setup.prompt_judge is None:
        return
    judges: Any = gateway.container.judges  # the harness's FakeJudgeClient
    default = DEFAULT_ANSWERS["InjectionJudgement"]
    scripted: dict[str, object] = {}
    if setup.tool_judge is not None:
        scripted["tool_poisoning"] = _verdict(setup.tool_judge == "poisoned")
    if setup.prompt_judge == "timeout":
        scripted["prompt_injection"] = JudgeUnavailableError(JudgeResult.TIMEOUT)
    elif setup.prompt_judge is not None:
        scripted["prompt_injection"] = _verdict(setup.prompt_judge == "injection")

    def judge(control_id: str, content: str) -> object:
        del content
        answer = scripted.get(control_id, default)
        if isinstance(answer, JudgeUnavailableError):
            raise answer
        return answer

    judges.answers["InjectionJudgement"] = judge


def _verdict(is_injection: bool) -> dict[str, object]:
    return {"is_injection": is_injection, "confidence": 0.9, "rationale": "scripted"}


def _describe(servers: dict[str, Any], describe: Describe) -> None:
    tool = servers[HOSTS[describe.server]]._tool_manager.get_tool(describe.tool)
    assert tool is not None, describe
    tool.description = describe.description


@asynccontextmanager
async def mcp_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: AttackCase,
    classifier: InjectionClassifier,
) -> AsyncIterator[Any]:
    """The MCP stack of ``tests/mcp`` with the case's setup applied around the pinning."""
    servers: dict[str, Any] = {}

    def recording(host: str, build: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def build_and_record(log: Any) -> Any:
            servers[host] = build(log)
            return servers[host]

        return build_and_record

    monkeypatch.setattr(
        upstreams,
        "SERVERS",
        {host: recording(host, build) for host, build in upstreams.SERVERS.items()},
    )
    setup = case.setup
    async with upstreams.running_upstreams() as (transport, log):
        if setup.describe is not None and setup.describe.when == "before_pin":
            _describe(servers, setup.describe)
        pin_kit.write_pins(tmp_path / "pins", await pin_kit.capture_pins(transport))
        if setup.describe is not None and setup.describe.when == "after_pin":
            _describe(servers, setup.describe)
        if setup.fetch_page is not None:
            log.fetch_page = expand(setup.fetch_page)
        if setup.plan_cost is not None:
            log.plan_cost = setup.plan_cost
        async with running_gateway(tmp_path, transport=transport, classifier=classifier) as gw:
            gw.resolver.answers.update(setup.dns)
            script_judge(gw, setup)
            apply_policy(gw, setup.policy)
            yield mcp_harness.MCPStack(gw, transport, log)


@asynccontextmanager
async def llm_gateway(
    tmp_path: Path, case: AttackCase, classifier: InjectionClassifier
) -> AsyncIterator[tuple[Harness, respx.MockRouter]]:
    async with running_gateway(tmp_path, classifier=classifier) as gateway:
        apply_policy(gateway, case.setup.policy)
        script_judge(gateway, case.setup)
        with respx.mock(base_url=LLM_BASE, assert_all_called=False) as router:
            yield gateway, router


# --------------------------------------------------------------------------- runners


def _messages(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, str):
        return [{"role": "user", "content": payload}]
    return payload


async def run_llm_prompt(tmp_path: Path, case: AttackCase, classifier: Any) -> Observed:
    async with llm_gateway(tmp_path, case, classifier) as (gateway, router):
        route = router.post("/chat/completions").mock(side_effect=echo_completion)
        token = await gateway.token(case.sub)
        await raise_risk(gateway, token, case.setup.risk)
        body = chat(messages=_messages(expand(case.payload)))
        response = await gateway.agent.post(CHAT, json=body, headers=bearer(token))
        sent = [call.request.content.decode() for call in route.calls]
        return await _observe_http(gateway, response, sent)


async def run_llm_response(tmp_path: Path, case: AttackCase, classifier: Any) -> Observed:
    """The model's answer is the attack: ``payload`` is its text, or ``{content, tool_calls}``."""
    payload = expand(case.payload)
    if isinstance(payload, str):
        answer = completion(payload)
    else:
        answer = completion(payload.get("content"), tool_calls=payload.get("tool_calls"))
    async with llm_gateway(tmp_path, case, classifier) as (gateway, router):
        route = router.post("/chat/completions").mock(return_value=httpx.Response(200, json=answer))
        token = await gateway.token(case.sub)
        response = await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))
        observed = await _observe_http(gateway, response, [])
        return observed.model_copy(update={"upstream_called": route.called})


async def run_mcp_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: AttackCase, classifier: Any
) -> Observed:
    assert case.server is not None, case.id
    assert case.tool is not None, case.id
    async with mcp_stack(tmp_path, monkeypatch, case, classifier) as stack:
        client = await mcp_harness.connect(stack, case.sub, case.server)
        await raise_risk(stack.gateway, client.token, case.setup.risk)
        arguments: dict[str, Any] = expand(case.payload) or {}
        result: dict[str, Any] = {}
        for attempt in range(case.setup.repeat):
            sent = dict(arguments)
            if case.setup.vary is not None:
                sent[case.setup.vary] = f"{attempt}-{sent[case.setup.vary]}"
            result = await client.call(case.tool, **sent)
        return await _observe_tool(stack, result, case.tool)


async def run_mcp_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: AttackCase, classifier: Any
) -> Observed:
    """``tools/list`` must hide the tool, and calling it by name anyway must be refused."""
    assert case.server is not None, case.id
    assert case.tool is not None, case.id
    async with mcp_stack(tmp_path, monkeypatch, case, classifier) as stack:
        client = await mcp_harness.connect(stack, case.sub, case.server)
        listed = case.tool in await client.tools()
        result = await client.call(case.tool, **(expand(case.payload) or {}))
        observed = await _observe_tool(stack, result, case.tool)
        hidden = observed.decision != "allow"
        assert listed is not hidden, f"{case.id}: listed={listed} but decision {observed.decision}"
        return observed


# ------------------------------------------------------------------- scenarios


async def _tainted_etl_reports(stack: Any) -> Any:
    """nightly_etl's reports client in a session the untrusted web tool has tainted."""
    web, reports = await mcp_harness.connect_all(stack, "svc:nightly_etl", "web", "reports")
    await web.call("fetch", url="https://example.com/outlook")
    return reports


async def _held(reports: Any, arguments: dict[str, Any]) -> str:
    result = await reports.call("write_report", **arguments)
    assert result["isError"], result
    assert result["content"][0]["text"].startswith("approval_required"), result
    return result["_meta"][META]


async def _retry(reports: Any, approval_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    params = {"name": "write_report", "arguments": arguments, "_meta": {META: approval_id}}
    response = await reports.request("tools/call", params)
    assert response.status_code == 200, response.text
    return response.json()["result"]


async def _approve(stack: Any, approval_id: str, operator: str = OPERATOR) -> httpx.Response:
    token = await stack.gateway.operator_token(operator)
    return await stack.gateway.operator.post(
        f"/admin/approvals/{approval_id}/approve", headers=bearer(token)
    )


async def scenario_approval_replay(stack: Any, payload: dict[str, Any]) -> Observed:
    """Approve one held write, run it, then present the same approval again."""
    reports = await _tainted_etl_reports(stack)
    approval_id = await _held(reports, payload)
    assert (await _approve(stack, approval_id)).status_code == 200
    assert (await _retry(reports, approval_id, payload))["isError"] is False
    replay = await _retry(reports, approval_id, payload)
    observed = await _observe_tool(stack, replay, "write_report")
    return observed.model_copy(update={"upstream_called": len(stack.log.of("write_report")) > 1})


async def scenario_approval_tamper(stack: Any, payload: dict[str, Any]) -> Observed:
    """Approve one held write, then retry it with other arguments under the same id."""
    reports = await _tainted_etl_reports(stack)
    approval_id = await _held(reports, payload["approved"])
    assert (await _approve(stack, approval_id)).status_code == 200
    tampered = await _retry(reports, approval_id, payload["sent"])
    return await _observe_tool(stack, tampered, "write_report")


async def scenario_approval_other_session(stack: Any, payload: dict[str, Any]) -> Observed:
    """An approval granted to one session, presented by another session of the same agent."""
    reports = await _tainted_etl_reports(stack)
    approval_id = await _held(reports, payload)
    assert (await _approve(stack, approval_id)).status_code == 200
    intruder = await _tainted_etl_reports(stack)
    return await _observe_tool(stack, await _retry(intruder, approval_id, payload), "write_report")


async def scenario_self_approval(stack: Any, payload: dict[str, Any]) -> Observed:
    """root's own databot write is held (risk > 0.5); root, an admin, approves it himself."""
    (reports,) = await mcp_harness.connect_all(stack, "root@demo", "reports")
    await raise_risk(stack.gateway, reports.token, 0.6)
    approval_id = await _held(reports, payload)
    response = await _approve(stack, approval_id, "root@demo")
    observed = await _observe_http(stack.gateway, response, [])
    return observed.model_copy(update={"upstream_called": bool(stack.log.of("write_report"))})


async def scenario_session_revival(stack: Any, payload: dict[str, Any]) -> Observed:
    """End a session, then present a validly signed token naming it again."""
    del payload
    gateway: Harness = stack.gateway
    token = await gateway.token("anna@demo")
    ended = await gateway.agent.delete("/v1/session", headers=bearer(token))
    assert ended.status_code == 200, ended.text
    revived = sign(claims(gateway.clock, session_id=ended.json()["session_id"]))
    client = mcp_harness.MCPClient(gateway.agent, revived, "reports")
    return await _observe_http(gateway, await client.initialize(), [])


async def scenario_session_rebinding(stack: Any, payload: dict[str, Any]) -> Observed:
    """A second principal presents a token naming another principal's live session."""
    del payload
    gateway: Harness = stack.gateway
    anna = sign(claims(gateway.clock, session_id="s-shared"))
    assert (await mcp_harness.MCPClient(gateway.agent, anna, "web").initialize()).status_code == 200
    bartek = sign(claims(gateway.clock, sub="bartek@demo", roles=["intern"], session_id="s-shared"))
    client = mcp_harness.MCPClient(gateway.agent, bartek, "web")
    return await _observe_http(gateway, await client.initialize(), [])


async def scenario_token(stack: Any, payload: dict[str, Any]) -> Observed:
    """One crafted agent token against the agent API: ``alg_none``, ``unregistered_agent``,
    ``wrong_mode`` or ``operator`` (an operator token, another audience)."""
    gateway: Harness = stack.gateway
    match payload["token"]:
        case "alg_none":
            token = unsigned(claims(gateway.clock), {"alg": "none", "typ": "JWT"})
        case "unregistered_agent":
            token = sign(claims(gateway.clock, act={"sub": "shadow_agent"}))
        case "wrong_mode":
            token = sign(claims(gateway.clock, mode="autonomous"))
        case "operator":
            token = await gateway.operator_token("root@demo")
        case other:
            raise AssertionError(other)
    client = mcp_harness.MCPClient(gateway.agent, token, "reports")
    return await _observe_http(gateway, await client.initialize(), [])


async def scenario_agent_token_on_admin(stack: Any, payload: dict[str, Any]) -> Observed:
    """An admin's agent token (not an operator token) against the operator API."""
    del payload
    gateway: Harness = stack.gateway
    token = await gateway.token("root@demo")
    return await _observe_http(
        gateway, await gateway.operator.get("/admin/approvals", headers=bearer(token)), []
    )


async def scenario_taint_then_write(stack: Any, payload: dict[str, Any]) -> Observed:
    """Read a page carrying a hidden injection, then write a report in the same session."""
    web, reports = await mcp_harness.connect_all(stack, payload["sub"], "web", "reports")
    await web.call("fetch", url="https://supplier.example/prices")
    result = await reports.call("write_report", name="q3.md", content="Q3 summary")
    return await _observe_tool(stack, result, "write_report")


SCENARIOS: Final[dict[str, Callable[[Any, dict[str, Any]], Any]]] = {
    "approval_replay": scenario_approval_replay,
    "approval_tamper": scenario_approval_tamper,
    "approval_other_session": scenario_approval_other_session,
    "self_approval": scenario_self_approval,
    "session_revival": scenario_session_revival,
    "session_rebinding": scenario_session_rebinding,
    "token": scenario_token,
    "agent_token_on_admin": scenario_agent_token_on_admin,
    "taint_then_write": scenario_taint_then_write,
}


async def run_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: AttackCase, classifier: Any
) -> Observed:
    assert case.scenario in SCENARIOS, case.id
    async with mcp_stack(tmp_path, monkeypatch, case, classifier) as stack:
        return await SCENARIOS[case.scenario](stack, expand(case.payload) or {})


async def run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: AttackCase, classifier: Any
) -> Observed:
    match case.channel:
        case "llm_prompt":
            return await run_llm_prompt(tmp_path, case, classifier)
        case "llm_response":
            return await run_llm_response(tmp_path, case, classifier)
        case "mcp_call":
            return await run_mcp_call(tmp_path, monkeypatch, case, classifier)
        case "mcp_listing":
            return await run_mcp_listing(tmp_path, monkeypatch, case, classifier)
        case "scenario":
            return await run_scenario(tmp_path, monkeypatch, case, classifier)
