# Pitch video

A 3-minute screen recording. It is not a slide show. The left 60% of the screen is the client,
**opencode**, talking to the gateway for both the model (`/v1`) and the tools (`/mcp`). The
right 40% is Grafana. Each beat below is staged with `make record SCENE=n`, which resets
exactly the state that beat needs, prints the prompt to type and the caption, and gives the
Grafana view to frame. It then waits for Enter.

- Beats 1, 2 and 4 are typed in opencode by a person (`make opencode AS=anna|bartek`, see
  `docs/opencode.md`). opencode runs as the `opencode` agent with its own tools switched
  off. It sees only the gateway's `sales_db_query`, `web_fetch` and `reports_write_report`
  (`demo/opencode/`). Each launch gets a fresh token, which means a fresh session.
- Beat 3, the night job, is played from the **demo runner terminal**. An autonomous agent is
  not a person at a keyboard, and the scene must show the approval prompt where an operator
  would see it.
- Beat 5, the live policy edit, is also scripted from the terminal. You can optionally repeat
  its request in opencode while the policy is raised.
- Beat 6 is the end shot.

The video runs on the remote model, so turns take seconds, not a minute:

```sh
make demo-up REMOTE=1        # demo overlay + OpenRouter (zero data retention); key from .env
```

## Recording checklist

- [ ] **Remote mode on.** Run `make demo-up REMOTE=1`, then `make record SCENE=1
  RECORD_ARGS="--no-wait --pace 0"` once as a smoke test (exit 0). Never show `.env` or a
  shell history that contains the OpenRouter key. `make record` never prints it.
- [ ] **Rehearse every beat**:
  `for n in 1 2 3 4 5 6; do make record SCENE=$n RECORD_ARGS="--no-wait --pace 0"; done`.
  All six should end `PASS`. This is also the docker-marked test
  `tests/demo/test_record_live.py`.
- [ ] **Screen layout 60/40.**
  - Left: opencode. For beat 1, two opencode panes side by side (anna left, bartek right).
    For beats 3 and 5, the runner terminal.
  - Right: Grafana's **Recording** dashboard (`acl-recording`), built for this 40% column.
    At a 760–768 px wide window Grafana switches to one column, and everything fits on one
    screen. `make record` prints its URL in kiosk mode, with the time picker, variables and
    links hidden, `theme=dark`, `from=now-5m` and `refresh=5s` (Grafana's minimum).
  - The dashboard's `session_id` is a text box. It does not follow the newest session by
    itself. Without it, the panels cover every session; with it, the gauge, the compromised
    box, Risk over time and Last decisions follow that one session. After an opencode take,
    `make record` prints the URL with the take's session id.
  - Record at 2560×1440 or 1920×1080, 30 fps.
- [ ] **Terminal**: font 18–20 pt (readable at 1080p after a 60% crop), dark theme, 100×30 or
  so. Set a clean prompt with `export PS1='$ '` and run `clear` before each take. Hide the dock
  and the menu-bar clock.
- [ ] **Browser**: zoom 110–125% so panel titles are readable in the 40% column. Hide the
  bookmarks bar. Stay logged in to Grafana as admin (password `ACL_GRAFANA_ADMIN_PASSWORD`
  from `.env`; never type it on camera).
- [ ] **Do not disturb on**: macOS Focus, Slack/Telegram/Discord quit, notifications off.
- [ ] **Warm up**: one rehearsal per beat right before filming, so the remote model and the
  classifier caches are warm.
- [ ] **opencode between takes**: quit it, then relaunch with `make opencode AS=anna` (or
  bartek). Each launch mints a fresh token (1 h), so you get a fresh gateway session with no
  taint and no risk. To start from an empty chat on screen, first run
  `rm -rf demo/opencode/.home/anna/` (opencode's local history; nothing in the gateway).
- [ ] **Policy file**: `git status config/` is clean before and after. `make record` restores
  it even after Ctrl-C, and it repairs a run that was killed mid-edit.

## Beat by beat

Voiceover lengths are counted at about 150 words a minute. Captions are lower-thirds of at most
6 words.

### 0:00–0:15 Title card, then opencode

- **Voiceover** (33 words): "AI assistants now act as us. They read our databases, browse
  the web and use our tools, with our access. But an assistant acting for you shouldn't
  *become* you. That's what we fix."
- **On screen**: the title card ("AI Control Layer: one gateway between AI assistants and
  everything they touch"), then cut to an idle opencode window with the Recording dashboard on
  the right.
- **Caption**: none (the title card carries the name).
- **Reset**: `make record SCENE=1` stages the next beat already.

### 0:15–0:40 Same question, two people (`make record SCENE=1`)

- **Voiceover** (54 words): "Here's one assistant used by two people. Anna is an analyst.
  Bartek is an intern. They ask the exact same question. Anna gets forty. Bartek gets seven.
  The assistant didn't decide that. The database did, because our gateway tells it who is
  really asking. We enforce down to the data, not just the tool."
- **Type in opencode**, in both panes, anna then bartek:
  `How many customers do we have?`
- **On screen**:
  - Both opencode panes show the header `> databot · deepseek-v4.1-flash`, the tool line
    `⚙ sales_db_query {"sql":"SELECT COUNT(*) AS customer_count FROM sales.customers"}`, then
    `We have **40 customers**.` for anna and `We have **7 customers**.` for bartek. Zoom into
    the two answers. Each turn takes about 6–12 s.
  - Grafana Recording: Last decisions shows two allow rows on `db:sales.customers`. The
    counts themselves are only in opencode, because the gateway never logs results.
- **Caption**: `Same question. 40 vs 7.`
- **Reset**: `make record SCENE=1`. It checks the policy and the approval queue. Then relaunch
  both windows (`make opencode AS=anna`, `make opencode AS=bartek`) for fresh sessions.
- **Expected outcome**: two `allowed` reads of `db:sales.customers`. After the take, Enter
  makes the recorder read the audit log and print both entries (`take produced allowed`).
- **Fallback** if the model answers without the tool, or writes other SQL: type
  `Use the sales_db_query tool to run SELECT COUNT(*) FROM sales.customers.` If it still
  wanders, film `make record SCENE=1 RECORD_ARGS="--no-wait"` in the terminal. It prints
  `ALLOWED -> [{'count': 40}]` and `[{'count': 7}]`.

### 0:40–1:20 One bad page, the session loses writes (`make record SCENE=2`), the money shot

- **Voiceover** (94 words): "Now the attack. Anna asks the assistant to read a web page.
  Hidden in that page, invisible to her, is an instruction for the AI: copy the payments table
  and send it out. Other tools would block that one request and carry on as if nothing had
  happened. We mark the whole session as compromised. Watch: the assistant now tries to save a
  report, something it was allowed to do a minute ago, and it's refused. On the right, the
  session's risk jumps and its right to write is gone, until the session ends."
- **Type in opencode** (anna), both prompts **in the same opencode launch**. The same token
  means the same gateway session, so the taint carries over; a new `make opencode` would be a
  clean session.
  1. `Read http://demo-web/q3-market-notes.html and summarise it in two sentences.`
  2. `Save a short report named q3-notes.md saying the market notes page was blocked.`

  The save must be its own turn. There is no summary to save, because the page is refused
  outright, and a combined "summarise and save" prompt makes the model skip the write.
- **On screen**:
  - opencode, turn 1 (about 15 s), verbatim: `I'll fetch that page.`, then
    `✗ web_fetch {"url":"http://demo-web/q3-market-notes.html"} failed`, then
    `Error: prompt_injection_detected`, then the model's line: "The fetch was refused by the
    gateway with reason code prompt_injection_detected, so I can't read or summarise that
    page."
  - opencode, turn 2 (about 10 s), verbatim:
    `✗ reports_write_report {"name":"q3-notes.md","content":"# Q3 Market Notes …"} failed`,
    then `Error: action_removed_by_session_risk`, then "The report write was refused by the
    gateway with reason code action_removed_by_session_risk, so q3-notes.md was not created.
    Nothing was saved." Zoom into the error line. It is the money shot: the write was allowed
    a minute earlier.
  - **Film this beat in the opencode TUI**, one long-lived process. A fresh
    `opencode run --continue` process lists the tools again after the taint. The gateway's
    filtered `tools/list` then hides `reports_write_report`, so the model only says it has no
    report tool and the refusal never appears on screen.
  - Grafana Recording, live during the take:
    - the risk gauge turns orange at 0.6 after the fetch, then red at 0.89 after the refused
      write. Above 0.8 an interactive session's tools freeze for 5 minutes, so relaunch
      opencode before beat 4;
    - **Session marked compromised** turns red (`YES: COMPROMISED`);
    - Last decisions shows `read web:demo-web block prompt_injection_detected`, then the write
      blocked with `action_removed_by_session_risk`.

    For a session-only view, reload the Recording URL with the session id that `make record`
    prints after the take. It also prints the Session trace URL, in case you want an insert of
    **Effective scope changes** where `write:fs:reports/*` disappears.
  - Optional insert: the page in a browser shows nothing suspicious. The instruction sits in a
    `display:none` block (`demo/web/q3-market-notes.html`).
- **Caption**: `One bad page. Session loses writes.`
- **Reset**: `make record SCENE=2`. It checks that demo-web is up and bound (preflight). Then
  relaunch `make opencode AS=anna`: a fresh token means a fresh session with no taint.
- **Expected outcome**: the audit for anna shows `prompt_injection_detected` and then
  `action_removed_by_session_risk`. The recorder checks both after Enter.
- **Fallback**:
  - If the model doesn't call the fetch tool, type
    `Use the web_fetch tool on http://demo-web/q3-market-notes.html.`
  - If it doesn't try to save, type
    `Use the reports_write_report tool to write q3-notes.md with the text "Q3 notes".` The
    block happens whatever the content is.
  - If turn 2 answers `tools_frozen`, the launch was already above risk 0.8 from an earlier
    take. Quit and relaunch `make opencode AS=anna`.
  - If opencode itself is unusable, film `make record SCENE=2 RECORD_ARGS="--no-wait"` in the
    terminal: the same three calls with audit lines.

### 1:20–1:50 The night job pauses and asks a human (`make record SCENE=3`, runner terminal)

- **Voiceover** (59 words): "Some assistants have no human at the keyboard, like a job that
  runs every night. It reads the same poisoned page. Killing it would break the business, so
  instead it pauses, slows down, and asks a person. Olga approves this one action. It runs
  once, and the same approval can't be used twice. One approval, one action, exactly once."
- **Type**: nothing. Press Enter to roll. The terminal plays the job step by step (`--pace`,
  default 2 s) and stops at `Approve apr-… as olga@demo? [Enter]`, which is the approval
  prompt. Press Enter on camera.
- **On screen**: the left column is the runner terminal. In order, it shows:
  1. `web.fetch` returns `prompt_injection_detected`.
  2. `write_report` returns `approval_required, approval_id=apr-…`.
  3. Olga's queue lists the tool, the resource and the reason. It shows only a digest of the
     arguments, never the arguments.
  4. The approval prompt; press Enter.
  5. The retry with `_meta approval_id` is `ALLOWED`.
  6. A replay is refused with `approval_already_used`, and the state is `succeeded`.
  7. The next call shows `throttled first: waited 5.5, 10.5 s`.

  Grafana Recording, right column: **Approvals waiting** turns orange at 1 while the write is
  held, and drops to 0 after the approval. Last decisions shows the orange `approval` row for
  nightly_etl, then the allow row.
- **Caption**: `Paused for approval. Runs once.`
- **Reset**: `make record SCENE=3`. It denies stale pending approvals left from earlier takes,
  so the queue on screen holds only this one. The job gets a fresh session.
- **Expected outcome**: 8/8 checks, including `approval executed exactly once: succeeded`.
- **Fallback**: slow the pacing with `make record SCENE=3 RECORD_ARGS="--pace 3"`. If the
  throttle wait feels long on camera, cut it in the edit; the line `waited … s on Retry-After`
  is the point.

### 1:50–2:15 Secrets blocked, personal data masked (`make record SCENE=4`)

- **Voiceover** (56 words): "Now someone pastes a cloud access key into the chat. It never
  leaves the machine. The request stops at the gateway, before any model sees it. And personal
  data, like this Polish national ID number, is masked before the model reads it. The model
  works with a placeholder, so it can't leak what it never saw."
- **Type in opencode** (anna):
  1. `My AWS access key is AKIAQ3EGRVW6XKZT4M7N. Which region is it for?`
  2. `Convert this ticket line to upper case: caller PESEL 44051401359, invoice resend.`
- **On screen**:
  - opencode: the first prompt fails in about 3 s (opencode startup; the gateway itself
    answers in milliseconds) with the red banner `Error: blocked by policy: secret_detected`.
    No model is called. The second is answered `CALLER PESEL [REDACTED:PL_PESEL], INVOICE RESEND.` Zoom
    into `[REDACTED:PL_PESEL]`. The model can only upper-case what it received.
  - Grafana Recording: Last decisions shows the `secret_detected` block and the redact row.
    The bottom row counts the redaction under Redacted.
- **Caption**: `Secrets blocked. PESEL masked.`
- **Reset**: `make record SCENE=4`, then relaunch `make opencode AS=anna`. A fresh session
  matters: beat 2 leaves its session at risk 0.89. Tools are frozen there for 5 minutes, and
  the risk is already raised.
- **Expected outcome**: the audit shows `secret_detected` and `pii_detected`, a redact
  verdict. The recorder checks both.
- **Fallback**: the key prompt is deterministic. If the model paraphrases the PESEL line,
  type `Repeat this line in capitals and nothing else: caller PESEL 44051401359.` The
  `[REDACTED:PL_PESEL]` mask is still what it gets. The terminal rehearsal shows both outcomes
  in 3 s: `make record SCENE=4 RECORD_ARGS="--no-wait"`.

### 2:15–2:35 One line of policy, new verdict (`make record SCENE=5`)

- **Voiceover** (39 words): "Rules change. A security engineer edits one line in one
  policy file. No restart, no redeploy. The query that was too expensive a moment ago now
  runs, and every change is on record, with a marker on the dashboard."
- **Type**: nothing required. Press Enter to roll. In order, the terminal shows:
  1. the heavy query is refused with `sql_cost_exceeded`;
  2. the one-line diff `- controls.sql_guard.max_cost: 10000` / `+ … 100000`;
  3. the hot reload `revision 87e5… -> …` together with the `policy_reload` audit event;
  4. the same query is `ALLOWED` (19,200,000 rows), with the audit line showing the new
     `rev=`.

  It then stops at `The policy is raised. Enter restores config/policy.yaml.` and the
  restore follows.
- **Optional** while it is stopped: in opencode, type
  `Count every combination of customers, orders and payments in one SQL query.`
- **On screen**: the terminal diff and the two verdicts. Grafana Recording shows a blue
  policy-reload marker on **Risk over time**, 5–10 s after the reload (once Alloy has shipped
  it to Loki). A second marker appears on restore. Hold the shot until it lands.
- **Caption**: `One line changed. New verdict.`
- **Reset**: `make record SCENE=5`. It restores a policy an interrupted run left edited, and
  waits until the gateway's revision equals the file's.
- **Expected outcome**: 5/5 checks. `config/policy.yaml` is byte-for-byte unchanged
  afterwards (`git status config/`).
- **Fallback**: the scripted path is deterministic. If the optional opencode query wanders,
  for example to other SQL, leave it out.

### 2:35–3:00 The totals, then the end card (`make record SCENE=6`)

- **Voiceover** (55 words): "One gateway in front of any model and any tool. Who is
  asking, what they may touch, how risky the session is, and when a human must decide, all in
  one place, on your own machine. Fifteen controls and over three thousand tests. The rule
  checks take milliseconds; the AI checks, up to a second."
- **On screen**: the Recording dashboard's bottom row as the closing shot: Requests, Redacted,
  **Rule checks p95 (live)** and **AI checks p95 (live)**. Then the end card,
  which the coordinator owns in the deck. For a wider shot, `make record SCENE=6` also
  prints the Posture URL. At 760 px Posture goes single-column too and only its top stats fit,
  so take it full screen.
- **Caption**: `One gateway. Runs on your machine.`
- **Reset**: `make record SCENE=6` (it prints both URLs).
- **Expected outcome**: nothing to check beyond the reset.
- **About the numbers** (keep the voiceover true, and matching the screen):
  - The dashboard's labels are "rule checks" (the deterministic controls) and "AI checks"
    (the injection classifier and the LLM judges, counted only on calls where one ran). Use
    exactly those words.
  - The live **Rule checks p95** on screen is tens of milliseconds, about 48 ms after
    `make smoke`. Most of it is sql_guard asking Postgres for a query plan (about 25 ms) and
    egress's DNS lookup (about 9 ms); without those two it's about 4 ms. **AI checks p95** is
    0.1–2 s, depending on whether the remote judge ran. So the voiceover says "milliseconds"
    and "up to a second", which is true of what the viewer sees.
  - The 1.6 ms figure (p95 1.63 ms, `docs/reports/perf.md`) is the benchmark: rule checks on
    a typical prompt, with mocked upstreams and no database or DNS. If it goes on a card,
    label it "rule checks 1.6 ms p95 (benchmark)". Never write a bare "1.6 ms overhead".
  - "Over three thousand tests": `make test` passes 3,004 (2026-10-04), and the docker-marked
    live tests come on top.
  - "On your own machine": the gateway, the tools, the data and the dashboards all run
    locally. The takes use the remote model (OpenRouter, zero data retention) only so that
    turns take seconds. `make demo-up` without `REMOTE=1` runs the same beats on a local
    model, so "fully local" is true of the product but not of these takes. Hence the
    voiceover says "on your own machine".

## Cut list

| # | In | Out | Source | Notes |
| --- | --- | --- | --- | --- |
| 1 | 0:00 | 0:08 | title card | still, fade in |
| 2 | 0:08 | 0:15 | screen: idle opencode + Recording | establish the 60/40 layout |
| 3 | 0:15 | 0:40 | beat 1 take | split opencode panes; punch in on 40 and 7 |
| 4 | 0:40 | 1:05 | beat 2 take, prompt 1 | the fetch error; optional 2 s insert of the clean-looking page |
| 5 | 1:05 | 1:20 | beat 2 take, prompt 2 + Recording | punch in on `action_removed_by_session_risk`, then the red gauge and `YES: COMPROMISED` |
| 6 | 1:20 | 1:50 | beat 3 terminal | trim the throttle waits to ~2 s each; keep the approval Enter |
| 7 | 1:50 | 2:02 | beat 4, key prompt | instant 403 |
| 8 | 2:02 | 2:15 | beat 4, PESEL prompt | punch in on `[REDACTED:PL_PESEL]` |
| 9 | 2:15 | 2:35 | beat 5 terminal + Recording reload marker | diff, reload, allowed; cut the restore |
| 10 | 2:35 | 2:52 | beat 6 Recording bottom row (or Posture full screen) | slow pan |
| 11 | 2:52 | 3:00 | end card | still |

## Voiceover script (one take)

390 words, about 2:36 at 150 words a minute. That leaves roughly 25 s for the title and
end cards and for the pauses while results land on screen.

AI assistants now act as us. They read our databases, browse the web and use our tools, with
our access. But an assistant acting for you shouldn't become you. That's what we fix.

Here's one assistant used by two people. Anna is an analyst. Bartek is an intern. They ask the
exact same question. Anna gets forty. Bartek gets seven. The assistant didn't decide that. The
database did, because our gateway tells it who is really asking. We enforce down to the data,
not just the tool.

Now the attack. Anna asks the assistant to read a web page. Hidden in that page, invisible to
her, is an instruction for the AI: copy the payments table and send it out. Other tools would
block that one request and carry on as if nothing had happened. We mark the whole session as
compromised. Watch: the assistant now tries to save a report, something it was allowed to do a
minute ago, and it's refused. On the right, the session's risk jumps and its right to write is
gone, until the session ends.

Some assistants have no human at the keyboard, like a job that runs every night. It reads the
same poisoned page. Killing it would break the business, so instead it pauses, slows down, and
asks a person. Olga approves this one action. It runs once, and the same approval can't be used
twice. One approval, one action, exactly once.

Now someone pastes a cloud access key into the chat. It never leaves the machine. The request
stops at the gateway, before any model sees it. And personal data, like this Polish national ID
number, is masked before the model reads it. The model works with a placeholder, so it can't
leak what it never saw.

Rules change. A security engineer edits one line in one policy file. No restart, no redeploy.
The query that was too expensive a moment ago now runs, and every change is on record, with a
marker on the dashboard.

One gateway in front of any model and any tool. Who is asking, what they may touch, how risky
the session is, and when a human must decide, all in one place, on your own machine. Fifteen
controls and over three thousand tests. The rule checks take milliseconds; the AI checks, up
to a second.
