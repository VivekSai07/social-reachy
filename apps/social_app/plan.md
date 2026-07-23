# plan.md — social_app

Written before implementing real behavior logic, per Reachy Mini's agent
guidelines (AGENTS.md): scaffold first, plan next, implement only after
sign-off below.

## Understanding so far

- Goal: prototype socially-aware HRI behaviors on Reachy Mini, developed and
  tested against the MuJoCo simulator. Sim only — no physical hardware.
- Phase 1 (implemented): gaze/attention — track a human face via the
  laptop's own webcam and turn the simulated head to follow it.
- Phase 2 (future, not started): friendly, human-like conversational
  behavior (LLM + speech), structured to build on phase 1 rather than
  replace it.

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

## Status

Phase 1 implemented and passing `reachy-mini-app-assistant check` plus a
standalone sanity check of the smoothing/hysteresis logic. **Not yet
visually verified against a real face in the MuJoCo viewer** — that step
needs a human watching the sim window while moving in front of the webcam
(see verification steps in the plan file above), including the expected
sign/gain tuning pass.

Phase 2 (conversation) is intentionally not started — revisit this file
before beginning it.
