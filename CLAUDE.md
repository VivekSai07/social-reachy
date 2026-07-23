# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HRI (human-robot interaction) prototyping on [Reachy Mini](https://huggingface.co/docs/reachy_mini),
developed against the MuJoCo simulator (`--sim`) before any physical robot is
involved. Two apps, built in phases (the daemon runs **one at a time** — see
"Architecture" below):

- **Phase 1 — `apps/social_app` (implemented, verified working)**:
  gaze/attention — webcam-based face tracking (`social_app/perception.py`)
  drives the simulated head via a smoothed, hysteresis-gated control loop
  (`social_app/gaze.py`, `social_app/main.py`). See its `plan.md`.
- **Phase 2 — `apps/companion` (implemented, on branch `feature/ai-brain-memory`,
  awaiting a human sim test before merge — see the Workflow merge-gate rule
  below)**: friendly conversation with persistent, temporally-aware memory
  across sessions, on the `reachy-mini-app-assistant --template conversation`
  scaffold (Hugging Face Realtime backend) — phase 1's gaze code ported in
  as a `GazeMove`. See its `plan.md` for what was built, and
  `docs/ai_brain_notes.md` for the upstream template's architecture
  (including scaffolding-tool bugs found — see Environment below).

## Environment

Native Windows Python via `uv` — no Docker. Reachy Mini's own docs never
mention Docker, the simulator needs a real GUI window, and Windows is a
natively supported platform for the SDK, so a container would only add
friction here.

- Python is pinned to **3.12** (`.python-version`), Reachy Mini supports
  3.10–3.12. venv lives at `.venv/`.
- Setup: `uv venv .venv --python 3.12 && uv pip install --python .venv "reachy-mini[mujoco]"`
- Each app is installed editable: `uv pip install --python .venv -e apps/social_app`
  and `uv pip install --python .venv -e apps/companion`. `apps/companion`'s
  `pyproject.toml` requires `reachy-mini>=1.10.0rc2`, which bumps the shared
  venv above `apps/social_app`'s originally-installed 1.9.0 — re-verify
  `apps/social_app` (`reachy-mini-app-assistant check`) after any dependency
  change here, since both apps share one venv.
- **`reachy-mini-app-assistant create --template conversation` is broken in
  the installed SDK** (as of `reachy-mini` 1.9.0/1.10.0rc2) and needed 3
  local patches to `.venv/Lib/site-packages/reachy_mini/apps/fork_conversation.py`
  before `apps/companion` could be scaffolded (wrong hardcoded git branch,
  a Windows read-only-file cleanup bug, and a cp1252/UTF-8 decode crash —
  worked around with `$env:PYTHONUTF8="1"`). These patches live only in the
  gitignored `.venv/`, so **re-scaffolding a new `conversation`-template app
  on a fresh clone will hit the same bugs again** — full patch content and a
  4th (structural, non-Windows-specific) profile-location bug are documented
  in `apps/companion/plan.md`'s "Known upstream bugs" section; check there
  before re-patching blind, and check whether upstream has fixed any of
  these first.

## Commands

```powershell
# Start the simulated robot (MuJoCo viewer window)
.\scripts\run_sim.ps1                  # --scene empty (default)
.\scripts\run_sim.ps1 -Scene minimal   # table + objects

# Run an app directly against a running daemon (only one app at a time)
.venv\Scripts\python.exe -m social_app.main
.venv\Scripts\python.exe -m companion.main --gradio   # companion needs mic/speakers too

# Validate an app's structure/entry points (does a real install/uninstall test)
.venv\Scripts\reachy-mini-app-assistant.exe check apps\social_app
.venv\Scripts\reachy-mini-app-assistant.exe check apps\companion

# companion's own test suite (pytest/pytest-asyncio installed separately, not a project-wide dependency)
cd apps\companion; ..\..\.venv\Scripts\python.exe -m pytest
```

Daemon REST/WebSocket API + docs: `http://localhost:8000/docs` once
`run_sim.ps1` is running.

There is no test suite or linter configured yet — `reachy-mini-app-assistant
check` (above) is currently the only automated validation available, and it
only checks app packaging/structure, not behavior.

## Git conventions

- Never add a `Co-Authored-By: Claude` (or similar) trailer to commit
  messages. If one slips into a commit, rewrite that commit rather than
  papering over it with a follow-up commit.

## Workflow

- When implementing a new idea or feature, invoke the
  `/feature-dev:feature-dev` skill and do the work on a separate branch —
  never directly on `main`. Standard flow: branch → implement → verify →
  merge/PR, so `main` stays untouched until the work is ready.
- Never merge a feature branch into `main` unilaterally. The user tests the
  implementation locally (e.g. on their laptop) first; only merge into
  `main` — with a clear merge message — after they explicitly confirm that
  test passed.
- After finishing a development task (before considering it done), check
  whether this file is still accurate — architecture, commands, phase
  status — and update it as part of wrapping up the work, not as an
  afterthought.

## Architecture

- `apps/<name>/` — one Reachy Mini **app** per directory. The daemon runs
  **one app at a time**, as a subprocess (`python -u -m <name>.main`). An app
  is a class extending `ReachyMiniApp` with a `run(reachy_mini, stop_event)`
  method (see `apps/social_app/social_app/main.py`); `wrapped_run()` handles
  the connection lifecycle and is called from `__main__`.
  - `goto_target(...)` for smooth gestures ≥0.5s; `set_target(...)` for
    real-time/high-frequency control (tracking, games, 10Hz+ loops).
  - Motion safety limits are clamped by the SDK: head pitch/roll ±40°, head
    yaw ±180°, body yaw ±160°, head-vs-body yaw delta ≤65°.
  - **Never hand-create or hand-edit an app's folder structure/entry
    points.** Always scaffold with
    `reachy-mini-app-assistant create <name> <path> [--template default|conversation]`
    — it generates the `pyproject.toml` entry point
    (`reachy_mini_apps`), HF Space metadata, and package layout that the
    daemon and `check`/`publish` depend on. `--publish` immediately creates a
    **public** Hugging Face Space — never run it without the user explicitly
    asking for it.
  - `default` template = blank/minimal skeleton (what `social_app` was
    scaffolded from, since extended with phase-1 gaze tracking).
    `conversation` template forks the reference conversation app (VAD + LLM +
    TTS + movement fusion already wired) — what `apps/companion` (phase 2)
    is built on. See the Environment section above before re-scaffolding
    with this template — the CLI has known bugs that need patching first.
- `apps/social_app/social_app/` — three files, one responsibility each, so
  there is exactly one `set_target()` call site in the app (a hard rule from
  the upstream SDK's `control-loops.md`): `perception.py` (webcam capture +
  face detection on a background thread, reusing `reachy_mini.vision.*`
  internals), `gaze.py` (pure, I/O-free pose smoothing/hysteresis), `main.py`
  (the fixed-rate control loop and sole `set_target()` call). This split is
  deliberate so a future phase-2 control loop can reuse `gaze.py`/
  `perception.py` unchanged — confirmed by `apps/companion`, which does
  exactly that.
- `apps/companion/src/companion/` — the conversation template's own
  structure (much larger than `social_app`'s, see its own `plan.md` and
  `docs/ai_brain_notes.md` for the full architecture), extended with:
  `perception.py`/`gaze.py` (ported verbatim from `social_app`), `gaze_move.py`
  (`GazeMove`, a continuous `Move` modeled on the template's own
  `BreathingMove` — the template's `MovementManager` in `moves.py` still
  owns the single `set_target()` call site; `GazeMove` just competes for the
  same move-queue slot as breathing/dance/emotion moves, evicted the same
  way when a real move plays), `sessions.py` (SQLite session log for
  temporal awareness, sibling to the template's existing `memory.py` fact
  store), and `tools/gaze_tracking.py` (an LLM-callable tool toggling gaze,
  replacing the template's daemon-side `head_tracking` tool — useless in sim
  for the same reason `social_app`'s webcam tracker exists).
- `apps/social_app/plan.md` — requirements/approach doc for that app,
  written before real behavior logic goes into `main.py` (convention from the
  upstream SDK's `AGENTS.md`: scaffold → plan → get sign-off → implement).
  Check for a `plan.md` in an app directory before writing behavior code into
  it, and keep it updated as design decisions are made.
- `docs/reachy_mini_notes.md` — condensed reference notes on the SDK (daemon,
  `ReachyMini` client, motion API, CLI) pulled from the official docs, kept
  local so they don't need re-fetching each session.
- `scripts/run_sim.ps1` — convenience wrapper around
  `reachy-mini-daemon --sim`.

## Adding a new app/prototype

Use the CLI from the venv, not manual folders:

```powershell
.venv\Scripts\reachy-mini-app-assistant.exe create <name> apps
```

Then write/update `apps/<name>/plan.md` with the HRI question being
prototyped before implementing `main.py`.
