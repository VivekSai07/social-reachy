# social-reachy

Human-robot interaction (HRI) prototyping on [Reachy Mini](https://huggingface.co/docs/reachy_mini),
developed and tested against the MuJoCo simulator. Roadmap: gaze/attention
tracking (phase 1, implemented) → friendly, human-like conversation (phase 2).

## Setup

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv "reachy-mini[mujoco]"
uv pip install --python .venv -e apps/social_app
```

## Run

```powershell
# Terminal 1 — start the simulated robot (MuJoCo viewer)
.\scripts\run_sim.ps1            # or: .\scripts\run_sim.ps1 -Scene minimal

# Terminal 2 — run the app directly for quick iteration
.venv\Scripts\python.exe -m social_app.main
```

The daemon also exposes a dashboard/API at http://localhost:8000 (docs at
`/docs`) where the app can be started/stopped once installed.

## Layout

- `apps/social_app/` — the Reachy Mini app (scaffolded via
  `reachy-mini-app-assistant`, see its `plan.md` before extending `main.py`).
- `docs/reachy_mini_notes.md` — condensed SDK reference notes.
- `scripts/` — dev convenience scripts.

See [CLAUDE.md](CLAUDE.md) for details geared at AI coding agents working in
this repo.
