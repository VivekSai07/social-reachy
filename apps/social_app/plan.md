# plan.md — social_app

Written before implementing real behavior logic, per Reachy Mini's agent
guidelines (AGENTS.md): scaffold first, plan next, implement only after
sign-off below.

## Understanding so far

- Goal: prototype socially-aware HRI behaviors on Reachy Mini, developed and
  tested against the MuJoCo simulator. Sim only — no physical hardware.
- Phase 1 (implemented): gaze/attention — track a human face via the
  laptop's own webcam and turn the simulated head to follow it.
- Phase 2 (future, not started, architecture researched and documented
  below): friendly, human-like conversational behavior with persistent,
  temporally-aware memory across sessions — an "AI brain" that knows time
  has passed between conversations and keeps a consistent buddy tone.

## Answers (from user, resolving the prior open questions)

1. First social signal: gaze/attention (head tracking a person).
2. Vision input: laptop's own integrated webcam.
3. Yes, a friendly LLM-driven conversation is planned for later — responses
   should read as human-like and warm, not robotic. Not built yet.
4. Sim only; no physical robot planned.
5. Beyond "it works": legibility (smooth, readable attention) and general
   robustness (no flicker on bad frames, no freeze when no one's present)
   are the success criteria.

## Technical approach (phase 1 — implemented)

Three files in `social_app/`, one responsibility each, so there's exactly
one `set_target()` call site in the whole app (per the SDK's own
`control-loops.md` rule):

- `perception.py` — `WebcamFaceTracker`: background thread, `cv2.VideoCapture`
  (laptop webcam) → `reachy_mini.vision.face_detector.FaceDetector` (YuNet
  ONNX, reused from the SDK's own internals) → `reachy_mini.vision.
  face_tracking.Tracker` (selection/hysteresis, also reused) → non-blocking
  `latest()` snapshot read, mirroring the SDK's own `FaceTracker` thread
  architecture.
- `gaze.py` — `GazeController`: pure, I/O-free. Turns the latest
  `FaceObservation` into smoothed `(yaw_deg, pitch_deg)` via a two-layer
  hysteresis (grace period for single dropped frames, longer timeout for a
  dead tracker) plus an EMA smoother, with a slow idle sway when no face has
  been seen recently. No `set_target()` call — deliberately reusable by a
  future conversation-phase control loop.
- `main.py` — the control loop itself (50Hz, `time.monotonic()`-timed, no
  drift): reads `tracker.latest()` → `gaze.update(...)` →
  `create_head_pose(yaw=..., pitch=...)` → the one `set_target()` call.

Full design rationale, constants table, and the "why" behind each choice:
see the plan this was implemented from at
`C:\Users\Vivek Sai\.claude\plans\answers-to-your-questions-synchronous-dewdrop.md`.

Known trade-off accepted: `reachy_mini.vision.*` is undocumented internal
API, not part of the SDK's public contract — a future `reachy-mini` version
bump could move/rename it. Acceptable for a sim-only prototype; if an
upgrade breaks the import, that's the first place to look.

Unresolved until tested in the sim viewer (not analytically solvable —
there's no calibration between an arbitrary laptop webcam and the robot's
coordinate frame): the sign of `yaw_sign`/`pitch_sign` in `GazeConfig`
(`gaze.py`), and whether the default gain constants feel right. Expected
tuning step, not a bug — see verification section of the plan file above.

## Technical approach (phase 2 — "AI brain," researched, not implemented)

Full research and rationale:
`C:\Users\Vivek Sai\.claude\plans\answers-to-your-questions-synchronous-dewdrop.md`
(the plan-mode session that produced this section) and
`docs/ai_brain_notes.md` (condensed reference on the conversation
template's actual architecture — read that before starting phase 2 work).

**Backend decision (confirmed with user)**: build on top of
`reachy-mini-app-assistant create --template conversation` as-is — Hugging
Face's hosted Realtime speech-to-speech backend (no API key needed in
`deployed` mode) — rather than rebuilding STT/LLM/TTS/tool-calling around a
different provider.

**Key finding**: the conversation template already ships a persistent
memory system (`memory.py` JSON fact-store + `remember`/`forget` tools +
a static persona file at `profiles/default/instructions.txt` that's already
close to the "friendly buddy" tone wanted). Phase 2 is an *extension* of
that, not a cold build. The actual gap is timestamps: nothing in the
existing system tracks *when*, so it can't express "it's been 2 days" or
recap what was discussed last time.

**Planned extension**: one additional local SQLite file, `sessions.db`
(stdlib `sqlite3`, zero new dependencies), alongside — not replacing — the
existing fact-store:
- `sessions` table: `session_id, start_ts, end_ts, summary`. `summary` is
  one LLM call at session end, triggered from the existing
  `tools/go_to_sleep.py` code path (the natural "conversation is ending"
  hook).
- At next session start: load the last row's `end_ts` + `summary`, compute
  the time delta, inject both into the system prompt at the same point
  `format_memory_for_prompt()` already uses in `prompts.py`.
- Graduate to embedding-based retrieval (`sqlite-vec`, an additive SQLite
  extension) only once flat summaries stop scaling (~30-50 sessions) — not
  built now.

**Two open decisions to resolve before implementing** (deliberately left
open by the research pass, not analytically resolvable without hands-on
experimentation):
1. How phase 1's `gaze.py`/`perception.py` control loop merges with the
   conversation template's own tool-call → queue → control-loop pattern —
   both must ultimately share the one `set_target()` call site the SDK
   requires. (`gaze.py`/`perception.py` were deliberately built I/O-decoupled
   in phase 1 specifically so this merge is additive, not a rewrite.)
2. Which app directory phase 2 lives in — re-scaffolding `social_app`
   itself with `--template conversation` (needs care not to destroy the
   phase-1 code) vs. scaffolding a fresh app and porting `gaze.py`/
   `perception.py` into it.

## Status

Phase 1: implemented, passing `reachy-mini-app-assistant check` and a
standalone sanity check of the smoothing/hysteresis logic, **and now
visually confirmed working** — gaze tracking runs smoothly against a real
face in the MuJoCo viewer (tested on a second machine after cloning the
repo).

Phase 2: architecture researched and documented above; not implemented.
Per CLAUDE.md's Workflow rule, implementation happens on a separate branch
via `/feature-dev:feature-dev` — resolve the two open decisions above
first.
