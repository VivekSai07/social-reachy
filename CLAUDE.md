# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HRI (human-robot interaction) prototyping on [Reachy Mini](https://huggingface.co/docs/reachy_mini),
developed against the MuJoCo simulator (`--sim`) before any physical robot is
involved. `apps/social_app` is the project's one app, built in phases (see
its `plan.md` for the full rationale):

- **Phase 1 (implemented)**: gaze/attention — webcam-based face tracking
  (`social_app/perception.py`) drives the simulated head via a smoothed,
  hysteresis-gated control loop (`social_app/gaze.py`,
  `social_app/main.py`).
- **Phase 2 (not started, architecture researched)**: friendly, human-like
  conversation with persistent, temporally-aware memory across sessions,
  layered on the `reachy-mini-app-assistant --template conversation`
  scaffold (Hugging Face Realtime backend) rather than replacing phase 1.
  Before starting this phase, read `docs/ai_brain_notes.md` (conversation
  template architecture) and the Phase 2 section of `apps/social_app/plan.md`
  (planned extension + two open decisions) — both already researched, don't
  re-derive them.

## Environment

Native Windows Python via `uv` — no Docker. Reachy Mini's own docs never
mention Docker, the simulator needs a real GUI window, and Windows is a
natively supported platform for the SDK, so a container would only add
friction here.

- Python is pinned to **3.12** (`.python-version`), Reachy Mini supports
  3.10–3.12. venv lives at `.venv/`.
- Setup: `uv venv .venv --python 3.12 && uv pip install --python .venv "reachy-mini[mujoco]"`
- The app is installed editable: `uv pip install --python .venv -e apps/social_app`

## Commands

```powershell
# Start the simulated robot (MuJoCo viewer window)
.\scripts\run_sim.ps1                  # --scene empty (default)
.\scripts\run_sim.ps1 -Scene minimal   # table + objects

# Run the app directly against a running daemon
.venv\Scripts\python.exe -m social_app.main

# Validate an app's structure/entry points (does a real install/uninstall test)
.venv\Scripts\reachy-mini-app-assistant.exe check apps\social_app
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
    TTS + movement fusion already wired) — likely what phase 2 will migrate
    to, rather than hand-rolling that plumbing into `social_app` as-is.
- `apps/social_app/social_app/` — three files, one responsibility each, so
  there is exactly one `set_target()` call site in the app (a hard rule from
  the upstream SDK's `control-loops.md`): `perception.py` (webcam capture +
  face detection on a background thread, reusing `reachy_mini.vision.*`
  internals), `gaze.py` (pure, I/O-free pose smoothing/hysteresis), `main.py`
  (the fixed-rate control loop and sole `set_target()` call). This split is
  deliberate so a future phase-2 control loop can reuse `gaze.py`/
  `perception.py` unchanged.
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
