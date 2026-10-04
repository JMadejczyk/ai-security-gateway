"""`make record SCENE=n`: one beat of the pitch video (docs/video.md), staged for the camera.

A beat is either typed in opencode by a person (scenes 1, 2, 4) or scripted from this terminal
(3: the night job, which has no person at a keyboard; 5: the live policy edit), plus the end
shot (6). For each beat the recorder:

1. resets exactly what the beat needs: the demo overlay is up and bound, the policy file is
   the committed one (a backup an interrupted run left behind is restored), stale pending
   approvals from earlier takes are denied, and opencode beats are told to relaunch opencode,
   which mints a fresh token and so a fresh session with no taint;
2. prints the caption, the prompts to type and the Grafana view to frame (``--open`` opens it);
3. waits for Enter, then runs the scripted part paced for filming (``--pace`` seconds between
   steps, Enter at the approval and while the policy is raised), or, for an opencode beat,
   reads the audit log for what the take produced and checks the expected reason codes.

``--no-wait`` rehearses a beat without a person: no Enter, and an opencode beat is played by
the demo agent (the matching ``make demo`` scene) so the reset is proven to leave the stack in
a state where the beat lands. The scripted parts reuse ``demo.orchestrator.scenes``.
"""

import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Annotated, Final
from urllib.parse import urlencode

from pydantic import AfterValidator, BaseModel, ConfigDict

from demo.orchestrator.models import Check, SceneResult
from demo.orchestrator.narration import Narrator, describe_audit
from demo.orchestrator.scenes import (
    ANNA,
    BARTEK,
    ETL,
    INJECTION_PAGE,
    OLGA,
    PESEL,
    SCENES,
    Demo,
    Scene,
)

MAX_CAPTION_WORDS: Final = 6
CLOCK_SKEW_S: Final = 2.0  # the gateway's clock (Docker VM) may trail the host's
GRAFANA_RANGE: Final = {"orgId": "1", "from": "now-5m", "to": "now", "refresh": "5s"}


def _short_caption(caption: str) -> str:
    if not 0 < len(caption.split()) <= MAX_CAPTION_WORDS:
        msg = f"a caption has 1 to {MAX_CAPTION_WORDS} words: {caption!r}"
        raise ValueError(msg)
    return caption


class Prompt(BaseModel):
    """What a person types, and where."""

    model_config = ConfigDict(frozen=True)

    who: str  # the opencode window: anna@demo or bartek@demo
    text: str


class GrafanaView(BaseModel):
    model_config = ConfigDict(frozen=True)

    uid: str
    label: str
    panel_hint: str  # what to keep in frame
    flags: tuple[str, ...] = ("kiosk",)  # value-less URL switches

    def url(self, base: str, session_id: str | None = None) -> str:
        query = {**GRAFANA_RANGE, "theme": "dark"}
        if session_id:
            query["var-session_id"] = session_id
        flags = "".join(f"&{flag}" for flag in self.flags)
        return f"{base.rstrip('/')}/d/{self.uid}?{urlencode(query)}{flags}"


RECORDING = "acl-recording"  # the 40%-column dashboard for the video (grafana/dashboards)
POSTURE = "acl-posture"
SESSION_TRACE = "acl-session-trace"
_BARE = ("kiosk", "_dash.hideTimePicker", "_dash.hideVariables", "_dash.hideLinks")


def recording(panel_hint: str) -> GrafanaView:
    """The recording dashboard; with no session id its panels cover every session."""
    return GrafanaView(uid=RECORDING, label="Recording", panel_hint=panel_hint, flags=_BARE)


class Beat(BaseModel):
    """One shot of the storyboard. ``stand_in`` is the ``make demo`` scene that plays it when no
    person types (``--no-wait``), or runs it on camera for a scripted beat."""

    model_config = ConfigDict(frozen=True)

    number: int
    timecode: str
    title: str
    caption: Annotated[str, AfterValidator(_short_caption)]
    scripted: bool
    stand_in: int | None
    principals: tuple[str, ...] = ()  # whose audit entries the take produces (opencode beats)
    opencode: tuple[str, ...] = ()  # launch commands, one per window
    prompts: tuple[Prompt, ...] = ()
    expect: tuple[str, ...] = ()  # reason codes the take must produce (opencode beats)
    views: tuple[GrafanaView, ...] = ()


BEATS: Final[tuple[Beat, ...]] = (
    Beat(
        number=1,
        timecode="0:15-0:40",
        title="Same assistant, same question, two people",
        caption="Same question. 40 vs 7.",
        scripted=False,
        stand_in=1,
        principals=(ANNA, BARTEK),
        opencode=("make opencode AS=anna", "make opencode AS=bartek"),
        prompts=(
            Prompt(who=ANNA, text="How many customers do we have?"),
            Prompt(who=BARTEK, text="How many customers do we have?"),
        ),
        expect=("allowed",),
        views=(recording("Last decisions: two allow rows on db:sales.customers"),),
    ),
    Beat(
        number=2,
        timecode="0:40-1:20",
        title="A hidden instruction compromises the session",
        caption="One bad page. Session loses writes.",
        scripted=False,
        stand_in=2,
        principals=(ANNA,),
        opencode=("make opencode AS=anna",),
        prompts=(
            Prompt(who=ANNA, text=f"Read {INJECTION_PAGE} and summarise it in two sentences."),
            Prompt(
                who=ANNA,
                text="Save a short report named q3-notes.md saying the market notes page was "
                "blocked.",
            ),
        ),
        expect=("prompt_injection_detected", "action_removed_by_session_risk"),
        views=(
            recording(
                "risk gauge and 'Session marked compromised'; Last decisions: the fetch block, "
                "then the write blocked (after the take: this session's link below)"
            ),
        ),
    ),
    Beat(
        number=3,
        timecode="1:20-1:50",
        title="The night job pauses and asks a human",
        caption="Paused for approval. Runs once.",
        scripted=True,
        stand_in=3,
        principals=(ETL,),
        views=(
            recording(
                "Approvals waiting (orange at 1, back to 0); Last decisions: the orange approval "
                "row, then the allow row"
            ),
        ),
    ),
    Beat(
        number=4,
        timecode="1:50-2:15",
        title="Secrets never leave; personal data is masked",
        caption="Secrets blocked. PESEL masked.",
        scripted=False,
        stand_in=5,
        principals=(ANNA,),
        opencode=("make opencode AS=anna",),
        prompts=(
            Prompt(
                who=ANNA,
                text="My AWS access key is AKIAQ3EGRVW6XKZT4M7N. Which region is it for?",
            ),
            Prompt(
                who=ANNA,
                text=f"Convert this ticket line to upper case: caller PESEL {PESEL}, "
                "invoice resend.",
            ),
        ),
        expect=("secret_detected", "pii_detected"),
        views=(
            recording("Last decisions: the secret_detected block and the redact row; Redacted"),
        ),
    ),
    Beat(
        number=5,
        timecode="2:15-2:35",
        title="Change one line of policy, live",
        caption="One line changed. New verdict.",
        scripted=True,
        stand_in=7,
        views=(
            recording(
                "Risk over time: the blue policy-reload marker (5-10 s after the reload), "
                "a second one on restore"
            ),
        ),
    ),
    Beat(
        number=6,
        timecode="2:35-3:00",
        title="One gateway, any model, any tool",
        caption="One gateway. Runs on your machine.",
        scripted=False,
        stand_in=None,
        views=(
            recording("the bottom row: Requests, Redacted, Rule checks p95, AI checks p95"),
            GrafanaView(
                uid=POSTURE, label="Posture (full width)", panel_hint="the top stats, slow pan"
            ),
        ),
    ),
)


def beat(number: int) -> Beat:
    for candidate in BEATS:
        if candidate.number == number:
            return candidate
    msg = f"no beat {number}; beats are 1-{len(BEATS)}"
    raise ValueError(msg)


def scene_for(number: int | None) -> Scene | None:
    return next((s for s in SCENES if s.number == number), None) if number else None


def audit_ts(moment: datetime) -> str:
    """``moment`` in the audit log's own format (``2026-10-04T10:12:03.123Z``)."""
    return (
        moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"
    )


def expected_codes_seen(expect: Sequence[str], seen: Sequence[str]) -> tuple[Check, ...]:
    """One check per expected reason code: it appeared in the take's audit entries."""
    return tuple(
        Check(
            name=f"take produced {code}",
            expected=("seen",),
            actual="seen" if code in seen else "missing",
        )
        for code in expect
    )


class RecordOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    wait: bool = True  # False: --no-wait (no Enter; opencode beats played by the demo agent)
    pace_s: float = 2.0
    open_browser: bool = False
    grafana_url: str = "http://127.0.0.1:3300"


class Recorder:
    """Stages one beat. ``ask`` reads Enter (stdin), ``sleep`` paces, ``opener`` opens a URL."""

    def __init__(  # noqa: PLR0913 -- every collaborator is a seam the tests replace
        self,
        demo: Demo,
        options: RecordOptions,
        *,
        ask: Callable[[str], str] = input,
        sleep: Callable[[float], None] = time.sleep,
        opener: Callable[[str], None] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.demo = demo
        self.options = options
        self._ask = ask
        self._sleep = sleep
        self._opener = opener or _open_url
        self._now = now

    @property
    def narrator(self) -> Narrator:
        return self.demo.narrator

    # ---------------------------------------------------------------- reset

    def reset(self, chosen: Beat) -> list[Check]:
        """The state this beat starts from. Raises `DemoError` when the stack is not ready."""
        n = self.narrator
        checks: list[Check] = []
        if self.demo.policy.recover():
            n.line("restored config/policy.yaml left edited by an interrupted run", "warn")
        expected = self.demo.policy.revision()
        revision = self.demo.operator.wait_for_revision(lambda rev: rev == expected)
        n.detail(f"policy revision {revision}: the gateway runs config/policy.yaml as it is")
        checks.append(Check(name="policy file in force", expected=(expected,), actual=revision))
        stale = self.demo.operator.pending_approvals(OLGA)
        for approval in stale:
            self.demo.operator.deny(OLGA, approval.id)
        n.detail(
            f"approval queue: denied {len(stale)} stale pending approval(s) from earlier takes"
        )
        left = self.demo.operator.pending_approvals(OLGA)
        checks.append(Check(name="approval queue empty", expected=("0",), actual=str(len(left))))
        for command in chosen.opencode:
            who = command.rsplit("=", 1)[-1]
            n.detail(
                f"quit opencode and relaunch for a fresh session (no taint): {command}"
                f"   (empty chat history: rm -rf demo/opencode/.home/{who}/)"
            )
        for check in checks:
            n.check(check)
        return checks

    # ---------------------------------------------------------------- staging

    def show(self, chosen: Beat, session_id: str | None = None) -> None:
        n = self.narrator
        n.scene(chosen.number, f"[{chosen.timecode}] {chosen.title}", f"caption: {chosen.caption}")
        for prompt in chosen.prompts:
            n.act(f"type in opencode ({prompt.who})", prompt.text)
        if chosen.scripted:
            n.act("this terminal", "plays the scripted part after Enter")
        for view in chosen.views:
            url = view.url(self.options.grafana_url, session_id)
            n.act(f"Grafana: {view.label}", url)
            n.detail(f"frame: {view.panel_hint}")
            if self.options.open_browser:
                self._opener(url)

    def _pace(self, step: str) -> None:
        self.narrator.detail(f"... {step}")
        if self.options.pace_s > 0:
            self._sleep(self.options.pace_s)

    def _hold(self, message: str) -> None:
        if self.options.wait:
            self._ask(f"\n{message} ")

    # ---------------------------------------------------------------- the beat

    def run(self, chosen: Beat) -> SceneResult:
        started = time.monotonic()
        checks = self.reset(chosen)
        self.show(chosen)
        take_start = self._now() - timedelta(seconds=CLOCK_SKEW_S)
        if self.options.wait:
            prompt = "to roll the scripted part" if chosen.scripted else "when the take is done"
            self._ask(f"\n[Enter] {prompt} ")
        played = self._play(chosen, take_start)
        checks.extend(played.checks)
        for label, session_id in played.sessions.items():
            for view in (recording(""), GrafanaView(uid=SESSION_TRACE, label="", panel_hint="")):
                name = "Recording" if view.uid == RECORDING else "Session trace"
                url = view.url(self.options.grafana_url, session_id)
                self.narrator.act(f"Grafana: {name} ({label})", url)
        return SceneResult(
            number=chosen.number,
            title=chosen.title,
            checks=tuple(checks),
            sessions=played.sessions,
            elapsed_s=round(time.monotonic() - started, 1),
            error=played.error,
        )

    def _play(self, chosen: Beat, take_start: datetime) -> SceneResult:
        scene = scene_for(chosen.stand_in)
        if chosen.scripted or (scene is not None and not self.options.wait):
            if scene is None:  # pragma: no cover -- every scripted beat names its scene
                msg = f"beat {chosen.number} has no scene to play"
                raise ValueError(msg)
            if not chosen.scripted:
                self.narrator.line(
                    f"rehearsal: the demo agent plays this beat (make demo scene {scene.number})",
                    "dim",
                )
            self.demo.pace, self.demo.hold = self._pace, self._hold
            return scene.run(self.demo)
        if not chosen.expect:
            return SceneResult(
                number=chosen.number,
                title=chosen.title,
                checks=(Check(name="nothing to check", expected=("ok",), actual="ok"),),
            )
        return self._observe(chosen, audit_ts(take_start))

    def _observe(self, chosen: Beat, since: str) -> SceneResult:
        """What the opencode take produced, from the audit log."""
        n = self.narrator
        seen: list[str] = []
        sessions: dict[str, str] = {}
        for principal in chosen.principals:
            for entry in self.demo.audit.since(principal, since):
                n.detail(describe_audit(entry))
                seen.extend([entry.reason_code, *(v.reason_code for v in entry.verdicts)])
                sessions[f"{principal} take"] = entry.session_id
        checks = expected_codes_seen(chosen.expect, seen)
        for check in checks:
            n.check(check)
        return SceneResult(
            number=chosen.number, title=chosen.title, checks=checks, sessions=sessions
        )


def _open_url(url: str) -> None:
    """Best effort: macOS ``open``, else nothing (the URL is printed anyway)."""
    if sys.platform == "darwin":
        subprocess.run(["/usr/bin/open", url], check=False)  # noqa: S603 -- fixed binary
