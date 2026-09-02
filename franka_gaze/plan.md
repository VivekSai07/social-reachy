# plan.md — franka_gaze

Written before implementing behavior logic, following this repo's existing
convention (`apps/social_app/plan.md`).

## Goal

Prove that Reachy Mini's gaze/attention architecture generalizes beyond
Reachy: a Franka Panda arm's end-effector visibly, smoothly orients toward
a detected human face, using the exact same perception/smoothing code
(`social_app.perception.WebcamFaceTracker`, `social_app.gaze.GazeController`)
unchanged, with only a new IK stage added at the end of the pipeline.

## Design

Full spec: `docs/superpowers/specs/2026-09-01-franka-gaze-design.md`.
Full research this follows from: `docs/franka_realsense_gaze_findings.md`
(see its §5 for why this is a fixed-camera design, not eye-in-hand).

## Status

**Implemented and manually verified (2026-09-02).** All 5 tasks from
`docs/superpowers/plans/2026-09-01-franka-gaze-prototype.md` complete,
each individually task-reviewed. Ran `python -m franka_gaze.main` and
confirmed the end-effector visibly tracks a face detected via the laptop
webcam, in the MuJoCo viewer, with no code changes needed to `YAW_SIGN`/
`PITCH_SIGN`/`smoothing_alpha` — user's assessment: works, "not what I
expected at the end to be, but for a prototype it's perfect." No sign-flip
or gain retuning was required for this pass; open follow-up is to capture
what specifically diverged from expectation before scoping further work
(see conversation, not yet written up in a doc).

Known limitations carried over from the design spec, not yet addressed:
no true eye-in-hand (camera stays world-fixed, per §5 of the research
doc), no numeric tracking-accuracy measurement, idle-sway inherited
from `gaze.py` unchanged (arm will sway when no face is seen).
