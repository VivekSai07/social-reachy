# franka_gaze

A Franka Panda gaze/attention prototype, simulated in MuJoCo. Reuses
Reachy Mini's webcam face-tracking (`social_app.perception`,
`social_app.gaze`) unchanged; the only new logic is converting the
resulting yaw/pitch target into Franka joint angles via IK.

See `plan.md` for design rationale and
`docs/superpowers/specs/2026-09-01-franka-gaze-design.md` for the full
design spec.

Run: `.venv\Scripts\python.exe -m franka_gaze.main`
