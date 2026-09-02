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
or gain retuning was required for this pass.

**Fixed (2026-09-02) — yaw axis is now genuine look-at pointing.**
The previous `yaw_pitch_to_target` composed local Euler rotations on
`home_mat`, which rotated the hand about its own approach axis for yaw — a
no-op for pointing direction (see git history, commit `352c364`, for the
original bug). Replaced with a real look-at construction: a fixed
`REFERENCE_DIRECTION`/`REFERENCE_DISTANCE_M` define a nominal "person zone"
point in front of the arm's base (not the table), yaw/pitch pan/tilt around
that point, and the target orientation is built via a genuine look-at basis
(`_look_at_rotation`) aiming the approach axis at the resulting point, plus
a small fixed `MAX_LEAN_M` (10cm) lean-in toward the look direction.
Regression-tested (`test_nonzero_yaw_changes_target_orientation` would have
failed under the old code). Manually verified in the sim viewer — user's
assessment: "looks good at me, I'm impressed" — no `REFERENCE_DIRECTION`
sign-flip was needed; the initial guess (world `+X`) was correct on the
first try. Full design/implementation: `docs/plans/2026-09-02-franka-gaze-lookat-fix.md`.

Known limitations carried over from the design spec, not yet addressed:
no true eye-in-hand (camera stays world-fixed, per §5 of the research
doc), no numeric tracking-accuracy measurement, idle-sway inherited
from `gaze.py` unchanged (arm will sway when no face is seen).
