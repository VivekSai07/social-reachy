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

In progress — see `docs/superpowers/plans/2026-09-01-franka-gaze-prototype.md`
for the task-by-task implementation plan.
