This document traces Reachy Mini's working webcam-driven gaze pipeline
(`apps/social_app`) from source — perception, smoothing, and control-loop
integration — and researches, without writing any code, whether and how the
same architecture could port to a Franka Panda arm with a wrist-mounted
RealSense camera (eye-in-hand), simulated in MuJoCo. It is a documentation
spike only: nothing under `apps/` is modified. See the plan this doc was
produced from —
`docs/superpowers/plans/2026-09-01-franka-realsense-gaze-research.md` — on
branch `research/franka-realsense-gaze`, for the task breakdown and
self-review notes behind sections 1-4 below.

## 1. Perception layer

`apps/social_app/social_app/perception.py`'s `WebcamFaceTracker` runs webcam
capture and face detection on a background thread, decoupled from the
50Hz control loop that consumes its output.

**Why a background thread.** `cv2.VideoCapture.read()` blocks for the
duration of a frame grab, which is incompatible with a fixed-rate control
loop that also has to call `set_target()` on schedule. Running capture +
detect on its own thread, paced to `target_fps` independently of the
control loop's rate, keeps the control loop free to run at its own cadence
without waiting on camera I/O.

**Why `SimpleQueue` + `latest()`.** The background thread publishes each
tick's result (a `FaceObservation`) into a `queue.SimpleQueue`. The control
loop reads via `latest()`, which drains the entire queue and returns only
the most recent item (or `None` if nothing new arrived since the last
call). This is a non-blocking, most-recent-wins snapshot read: the control
loop never blocks waiting for a new observation, and it never falls behind
processing stale backlog if it happens to poll less often than the
detector produces frames.

**Reused SDK internals.** Two pieces are reused from `reachy_mini.vision.*`
rather than reimplemented, despite being undocumented, private-ish
internals of the SDK's own camera pipeline:

- `reachy_mini.vision.face_detector.FaceDetector` — a YuNet ONNX face
  detector (model auto-downloads from the HF Hub on first construction,
  then caches locally; pinned to 1 CPU thread internally).
- `reachy_mini.vision.face_tracking.Tracker` — hysteresis-based single-face
  selection across frames via `Tracker.select(faces, width, height)`.

The one piece intentionally *not* reused is the private `_center` helper;
`_normalized_center` reimplements that math locally so the app doesn't
depend on an SDK private symbol across upgrades.

**Normalization math.** `_normalized_center(face, width, height)` maps the
detected face's nose keypoint to a `[-1, 1]` range independent of capture
resolution:

```python
(
    face.nose[0] / max(width - 1, 1) * 2 - 1,
    face.nose[1] / max(height - 1, 1) * 2 - 1,
)
```

### Sequence diagram

Verified against `apps/social_app/social_app/perception.py`'s `_run`
(lines 103–149): each tick calls `cap.read()`, then
`detector.detect(frame_bgr)`, then `tracker.select(faces, width, height)`,
then (if a face was selected) `_normalized_center(...)`, then
`self._observations.put(FaceObservation(...))`. The control loop's
`latest()` call happens independently, on its own cadence, and simply
drains whatever is queued.

```mermaid
sequenceDiagram
    participant Cap as cv2.VideoCapture
    participant Det as FaceDetector (YuNet)
    participant Trk as Tracker (hysteresis)
    participant Q as SimpleQueue
    participant Loop as Control loop (50Hz)

    loop every 1/target_fps s (background thread)
        Cap->>Det: frame_bgr
        Det->>Trk: list[Face]
        Trk->>Trk: select() best face (area/jump/misses gating)
        Trk-->>Q: FaceObservation(center, timestamp)
    end
    Loop->>Q: latest() [non-blocking, drains backlog]
    Q-->>Loop: FaceObservation | None
```

### Tunable constants

| Constant | Guards against |
|---|---|
| `camera_index` | Selecting the wrong physical camera when multiple are attached. |
| `target_fps` | Runaway CPU use / unnecessary detector load — paces the background thread's capture+detect rate independently of the control loop's rate. |
| `min_area_frac` | Spurious detections of faces too small/far away to be a meaningful gaze target. |
| `max_jump` | Implausible frame-to-frame teleports (e.g. detector jumping to a different face or a false positive) being accepted as continuous tracking. |
| `max_misses` | Dropping the tracked face too eagerly on a handful of missed detections (blinks, brief occlusion, a bad frame) — tolerates up to N consecutive misses before giving up. |

## 2. Gaze smoothing (pure function, I/O-free)

`apps/social_app/social_app/gaze.py`'s `GazeController.update()`
(`gaze.py:48-87`) turns the latest `FaceObservation` into a smoothed
`(yaw_deg, pitch_deg)` head-pose target, once per control-loop tick.

**Why no I/O dependencies.** `GazeController` takes no camera, queue, or
robot handle — just a `FaceObservation | None` and a timestamp in, a pose
tuple out. This is deliberate: it lets a future or different control loop
call the same `update()` without extracting or rewriting the logic. That
reuse already happened once — `apps/companion/gaze_move.py`'s `GazeMove`
consumes `GazeController` directly rather than reimplementing smoothing, per
`apps/social_app/plan.md` lines 105-113 (phase 1's `gaze.py`/`perception.py`
were "deliberately built I/O-decoupled... specifically so this merge is
additive, not a rewrite") and lines 126-127 (confirming the resolution:
"Open decision 1 above (gaze/conversation-loop merge) is resolved by
`apps/companion/gaze_move.py`'s `GazeMove`").

**Two independent timeout tiers.** `update()` tracks two separate
timestamps — `_last_observation_at` (any tick from the tracker thread,
face found or not) and `_last_face_seen_at` (last tick where a face was
actually found) — and gates on two separate durations:

- `face_grace_period_s = 0.4` — coast through a single dropped detector
  frame (a blink, a bad frame) without snapping the target to idle sway.
- `observation_timeout_s = 1.0` — a longer, coarser check for the tracker
  *thread* itself being stalled or dead (no ticks at all, not just no
  face).

These are different failure modes on purpose: a momentary "face not found
this frame" and "the background thread stopped producing observations
entirely" need different tolerances, so they get independent timers rather
than one shared one.

**Idle-sway fallback.** When either timer condition fails —
`tracker_alive` is false or `face_recently_seen` is false — the target
becomes a slow `sin()` wave (`idle_amplitude_deg`, `idle_period_s`) instead
of a fixed pose. This is what stops the head locking into a static position
when no one is present; per `plan.md`, `apps/social_app`'s Environment/
Status notes independently confirm this behavior was visually verified
against a real face in the MuJoCo viewer.

**EMA runs unconditionally, after target selection.** Verified directly
against `gaze.py:56-87`: target selection (real face target vs. idle sway
target) happens first and is branched — `if tracker_alive and
face_recently_seen: target_yaw, target_pitch = ...; else: target_yaw =
idle sin(...), target_pitch = 0.0`. But the EMA update —

```python
self._yaw_deg += cfg.smoothing_alpha * (target_yaw - self._yaw_deg)
self._pitch_deg += cfg.smoothing_alpha * (target_pitch - self._pitch_deg)
```

— sits *after* that `if`/`else` block, at the same indentation level, and
runs every single call regardless of which branch produced `target_yaw`/
`target_pitch`. `smoothing_alpha = 0.25` is applied uniformly whether the
target came from a real face or from idle sway, which is exactly why
tracking-to-idle transitions don't visibly jump — the same low-pass filter
smooths across the state boundary, not just within a state.

**Portability warning — flagged prominently, not buried.** `yaw_sign` and
`pitch_sign` in `GazeConfig` are explicitly *not analytically solvable*:
per `apps/social_app/plan.md` lines 60-64, "there's no calibration between
an arbitrary laptop webcam and the robot's coordinate frame" — the correct
sign depends on the physical/simulated camera and joint coordinate
conventions and can only be determined by observing which way the head
turns in the sim viewer and flipping the sign if it's backwards. **This is
the single most important portability note for a Franka/RealSense port**:
expect to re-derive `yaw_sign`/`pitch_sign` (and re-tune `yaw_gain_deg`/
`pitch_gain_deg`) empirically against the new camera/arm frame — there is
no formula to carry over, only the tuning procedure.

### State diagram

Verified against `gaze.py:68-86`: `tracker_alive` (derived from
`_last_observation_at` vs. `observation_timeout_s`) and `face_recently_seen`
(derived from `_last_face_seen_at` vs. `face_grace_period_s`) are exactly
the two conditions gating tracked-vs-idle target selection, combined with
`and`. The EMA line runs unconditionally after target selection, not inside
either branch — confirmed by re-reading `gaze.py:77-86` above.

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Tracking: face observation with center != None
    Tracking --> Tracking: face seen within face_grace_period_s
    Tracking --> Coasting: no new observation, but < face_grace_period_s since last face
    Coasting --> Tracking: face reappears
    Coasting --> Idle: face_grace_period_s exceeded
    Tracking --> Idle: observation_timeout_s exceeded (tracker stalled)
    Idle --> Idle: sin() sway target, EMA-smoothed toward it
    note right of Tracking
      target = clamp(sign * axis * gain, limit)
      output += alpha * (target - output)   # EMA, every state
    end note
```

## 3. Control loop integration

`apps/social_app/social_app/main.py:52-88` runs a fixed-rate 50Hz loop
(`LOOP_HZ = 50.0`) that fuses gaze and antenna animation state into a
single per-tick call to `reachy_mini.set_target()`.

**Drift-free timing.** The loop tracks a `next_tick` accumulator
(`next_tick = time.monotonic()` before the loop, `next_tick +=
LOOP_PERIOD_S` after each tick) rather than sleeping a naive `1/hz` each
iteration. `sleep_for = next_tick - time.monotonic()` is computed fresh
every tick, so any time spent doing work (tracker read, gaze update, the
`set_target` call itself) is subtracted from the next sleep rather than
silently accumulating drift; if the loop falls behind entirely (`sleep_for
<= 0`), `next_tick` is reset to `time.monotonic()` rather than trying to
catch up by busy-looping through missed ticks.

**Per-tick sequence, verified directly against `main.py:59-79`:**

1. `tracker.latest()` — non-blocking read of the most recent
   `FaceObservation` from the background perception thread (section 1).
2. `gaze.update(obs, now)` — pure smoothing function (section 2), returns
   `(yaw_deg, pitch_deg)`.
3. `create_head_pose(yaw=yaw_deg, pitch=pitch_deg, degrees=True)` →
   `head_pose`.
4. Independently, antenna animation state is computed each tick from a
   `sin()` wave (or zeros if disabled) → `antennas_deg` → `antennas_rad`.
5. Both `head_pose` (step 3) and `antennas_rad` (step 4) are fully computed
   *before* the single call: `reachy_mini.set_target(head=head_pose,
   antennas=antennas_rad)` at `main.py:76-79`.

**Verification note.** Re-checked `main.py:59-79` directly: there is
exactly one `set_target()` call in the loop body (line 76), and both of its
arguments (`head_pose`, computed at line 60; `antennas_rad`, computed at
line 74) are assigned before that call — not two separate calls, and
nothing after the call feeds back into it that tick. The Mermaid flowchart
below matches this.

```mermaid
flowchart LR
    A[tracker.latest\nnon-blocking] --> B[gaze.update\nyaw_deg, pitch_deg]
    B --> C[create_head_pose\nyaw, pitch, degrees=True]
    D[antenna sin animation] --> E
    C --> E[reachy_mini.set_target\nhead=..., antennas=...]
    E -->|SDK clamps to safety limits| F[(Motors / MuJoCo sim)]
    style E fill:#f96,stroke:#333
```

**The upstream "one `set_target()` call site" rule.** Per
`docs/reachy_mini_notes.md:34-39`: `goto_target(...)` is for smooth,
≥0.5s interpolated moves; `set_target(...)` is for real-time/high-frequency
control (tracking, games, 10Hz+ loops), controlling `head` (6DOF Stewart
platform pose), `antennas` (2 motors), and `body_yaw`. The architectural
consequence — stated identically in `apps/social_app/plan.md:31-33` ("so
there's exactly one `set_target()` call site in the whole app (per the
SDK's own `control-loops.md` rule)") — is that **an app should have exactly
one `set_target()` call site**, so that multiple behavior sources (gaze,
idle animation, and in a future phase, conversation-driven moves) don't
fight over the motor target on the same tick. They must instead be fused
into a single pose/antenna-array pair before that one call, which is
exactly why `gaze.py`'s `GazeController.update()` is a pure function
returning a value rather than owning any I/O or calling `set_target`
itself (section 2) — it composes into whatever single call site the host
loop provides, rather than competing for one of its own.

**Safety clamps (SDK-enforced, independent of app logic).** Per
`docs/reachy_mini_notes.md:38`: head pitch/roll ±40°, head yaw ±180°, body
yaw ±160°, head-vs-body yaw delta ≤65°. These are applied by the daemon/SDK
itself regardless of what `main.py` computes and sends — the app cannot
bypass them by construction. This matters for a future Franka/RealSense
port: Franka's arm has 7 revolute joints and no equivalent "head" pose
abstraction, so these specific numbers don't transfer at all. What *does*
transfer is the concept — a hard, SDK/driver-enforced joint-limit clamp
applied after whatever the control loop computes, independent of and
downstream from app-level smoothing logic — which will need to be
re-derived from Franka's own joint limits (e.g. via its controller's
safety/limit configuration) rather than assumed away.

## 4. Franka + RealSense (eye-in-hand) in MuJoCo — feasibility research

This section is a documentation/research spike, not an implementation —
nothing in `apps/` changes. It maps each concept from sections 1-3 (Reachy
Mini's working webcam-gaze pipeline) onto a hypothetical Franka Panda arm
with a wrist-mounted RealSense camera, simulated in MuJoCo, and flags what
is confirmed vs. what still needs a hands-on spike.

**1. Franka MuJoCo model source.** MuJoCo Menagerie
(`github.com/google-deepmind/mujoco_menagerie`, maintained by Google
DeepMind) ships a `franka_emika_panda/` directory containing `panda.xml` —
confirmed via the repo's GitHub listing and its own README. The README
states the MJCF is a simplified robot description derived from the
publicly available Franka URDF (DAE meshes converted to OBJ via Blender,
then `obj2mjcf`; a convex decomposition of link5's collision mesh via
V-HACD; the URDF loaded into MuJoCo and re-saved as MJCF), requires MuJoCo
≥2.3.3, and separately ships a `scene.xml` that adds a textured
groundplane/skybox/haze around the robot. The model includes gripper/fingertip-related assets around `link7`/`link8`
(not independently re-verified against the README's own wording for this
pass), but **does not itself document an explicit flange/end-effector body
name or a camera-attachment example** — that requires opening the vendored `panda.xml` directly.
**Not yet verified against the vendored XML**: the exact body name to
parent a `<camera>` under (commonly a wrist/flange/`attachment_site`-style
body in Menagerie models, but this repo does not yet vendor the model, so
the name must be confirmed once `franka_emika_panda/panda.xml` is actually
pulled in — do not guess it from memory, per this doc's own instructions
in sections 1-3's style).

**2. Camera-in-hand attachment.** MuJoCo's MJCF format supports a
`<camera>` element as a child of any `<body>` element; a body-child camera
is attached to that body's local frame and moves/rotates with the body
throughout simulation, with `pos`/`quat` (or `euler`) attributes giving its
offset and orientation relative to the parent body's frame (confirmed via
`mujoco.readthedocs.io`'s XML reference for the `camera` element). Parenting
a `<camera>` under the Panda's end-effector/flange body therefore gives a
kinematically-following eye-in-hand camera "for free" — no separate mount
rigid body or joint needs modeling. This directly substitutes for a real
RealSense's extrinsics: the fixed transform from the camera's optical
frame to the gripper/flange mounting point, which on real hardware is
either measured or taken from Intel's RealSense CAD/datasheet for the
specific model, then hand-coded as the `<camera pos="..." quat="...">`
offset in the MJCF. **Not yet verified**: the exact numeric offset for any
specific RealSense model — that number comes from Intel's own mechanical
drawings/CAD for the chosen camera and a chosen mount bracket, not from
this research pass.

**3. Face/attention-target detection with a simulated camera.** MuJoCo's
Python bindings (`mujoco.Renderer`) can render an offscreen RGB frame from
any named camera in the model, including a body-attached one, once per
`mj_step` or on its own cadence. That rendered frame is just an image
array — nothing in `reachy_mini.vision.face_detector.FaceDetector` (a
general-purpose YuNet ONNX face detector, per section 1) requires the
frame to have come from `cv2.VideoCapture`; it should be reusable unchanged
against a MuJoCo-rendered frame, with the new work being only a frame-source
adapter (MuJoCo render → same `FaceObservation`-shaped output), not new
detection logic. This is a meaningful design fork from `social_app`,
though: `perception.py`'s docstring notes there's no human in the MuJoCo
scene for Reachy's setup because the webcam sees the real room, not the
sim — but a Franka eye-in-hand camera renders the MuJoCo world itself, so
a Franka simulation *can* place a synthetic human/face asset in-scene for
the detector to find, which Reachy's design deliberately does not do.

**4. Control-loop equivalent.** Franka has no "head pose" abstraction —
its actuation target is end-effector Cartesian pose or joint angles, not
yaw/pitch of a 2-DOF-ish head. `gaze.py`'s *shape* (a pure, I/O-free
smoothing function — EMA + two-tier hysteresis + idle fallback — that
takes an observation and returns a target) is straightforwardly portable;
what changes is what the *target* represents (a wrist orientation, or a
look-at point that still needs conversion to a joint/Cartesian command via
IK, rather than head yaw/pitch degrees) and what *applies* it (a MuJoCo
`mj_step` position/velocity actuator command loop, rather than the SDK's
`set_target()`). The "exactly one control-apply call site" discipline
from section 3 (`main.py`'s sole `set_target()` call) is general good
control-loop practice, not Reachy-specific, and is worth carrying over
verbatim into any Franka port.

**5. Safety-limit equivalent.** Real Franka Panda joints have documented
angle/velocity/torque limits, and independently MuJoCo's own
`<joint range="...">` attribute clamps joint motion at the physics-engine
level — enforced directly by the solver, unlike Reachy's SDK-side software
clamp described in section 3. Conceptually this is the same idea as the
±40°/±180°/±160°/≤65° numbers in section 3 (bound the commanded target
before/at the point of physical actuation), just a different mechanism
(engine-level joint-range clamp vs. SDK-level pose clamp) and necessarily
different numbers. **Not yet verified**: the actual `range` values in the
vendored Menagerie `panda.xml` — these must be read from that file once
vendored, not guessed or assumed to match Reachy's numbers in any way.

```mermaid
flowchart LR
    A[MuJoCo camera\nchild of flange body] --> B[mujoco.Renderer\nrgb frame]
    B --> C[FaceDetector\nreused from reachy_mini.vision]
    C --> D[Tracker.select\nreused]
    D --> E[GazeController-equivalent\nEMA + hysteresis, ported shape]
    E --> F{Target type differs:\nlook-at point / wrist orientation}
    F --> G[IK or Cartesian controller]
    G --> H[Single control-apply call\nper physics step]
    H -->|MuJoCo joint range clamps| I[(mj_step / Panda arm)]
    style H fill:#f96,stroke:#333
```

**Comparison table** (Reachy vs. Franka+RealSense-in-hand):

| Concept | Reachy Mini (`social_app`) | Franka + RealSense eye-in-hand |
|---|---|---|
| Camera source | Real laptop webcam (`cv2.VideoCapture`) — no human in MuJoCo scene | MuJoCo-rendered camera child of end-effector body — scene *can* contain a synthetic face |
| Detection | `reachy_mini.vision.face_detector.FaceDetector` (YuNet) | Same library reusable as-is; new frame-source adapter only |
| Target representation | Head yaw/pitch (deg) | End-effector Cartesian pose or wrist orientation → needs IK |
| Smoothing | EMA + two-tier hysteresis (`gaze.py`) | Same *shape* portable; target type changes |
| Actuation call | `reachy_mini.set_target()`, exactly one call site | Position/velocity actuator command via `mj_step`, same one-call-site discipline recommended |
| Safety clamping | SDK-side software clamp (fixed degrees) | MuJoCo `<joint range>` engine-level clamp, values from vendored MJCF |
| Sign/gain tuning | Empirical, platform-specific (`yaw_sign`/`pitch_sign`), not analytically solvable | Expect the same category of empirical tuning step for the look-at→IK mapping |

**Open questions** (explicitly unresolved — flagged for a hands-on spike, not answered here):

- Exact Menagerie model/filename and flange body name to attach the camera to (needs the model vendored into this repo and the XML actually opened, not guessed).
- Whether `mujoco.Renderer`-based per-tick rendering can sustain the detector's needed frame rate without starving the physics step loop.
- Which RealSense model's real extrinsics/FOV to mirror (D405 vs. D435i) for realism if this ever needs to match real hardware later — D405 (7cm–~50cm ideal depth range, ~1.5m max usable) is the common wrist-mount choice for its short minimum depth range, which matters for close-range manipulation; D435i has a much longer range (~0.2m–10m per Intel's published specs) but is less suited to close-in eye-in-hand work. Exact current datasheet numbers should be re-confirmed against Intel's own product/spec pages at spike time rather than trusted from this pass alone.
- IK solver choice for converting a look-at target into a joint/Cartesian command (out of scope for this research doc — a separate spike).

## Summary

**Directly reusable, largely as-is:**

- The face detector library, `reachy_mini.vision.face_detector.FaceDetector`
  (YuNet ONNX) plus `Tracker.select()` — a general-purpose detector that
  doesn't care whether frames come from `cv2.VideoCapture` or a MuJoCo
  render (§1, §4 point 3).
- The smoothing/hysteresis *shape* in `gaze.py` — EMA + two-tier timeout
  (grace period vs. tracker-alive) + idle-sway fallback, as a pure,
  I/O-free function taking an observation and returning a target (§2, §4
  point 4).
- The one-control-apply-call-site discipline (`main.py`'s sole
  `set_target()`) — general good control-loop practice worth carrying over
  verbatim, independent of Reachy specifics (§3, §4 point 4).
- The "hard, driver/engine-enforced clamp downstream of app logic" concept —
  the *mechanism* differs (SDK software clamp vs. MuJoCo `<joint range>`)
  but the architectural pattern transfers (§3, §4 point 5).

**Must be rebuilt, not ported:**

- Target representation and IK: Franka has no head-pose abstraction, so a
  yaw/pitch target must become an end-effector Cartesian pose or wrist
  orientation, requiring an IK (or Cartesian controller) step that doesn't
  exist in the Reachy pipeline at all (§3, §4 point 4).
- The actuation call itself: `reachy_mini.set_target()` has no Franka/MuJoCo
  equivalent — a position/velocity actuator command loop driven by
  `mj_step` must be written from scratch (§4 point 4).
- Safety clamp values: Reachy's ±40°/±180°/±160°/≤65° numbers are
  Reachy-specific and don't transfer; Franka's own joint `range` limits
  must be read from the vendored MJCF once it exists (§3, §4 point 5).
- Sign/gain tuning (`yaw_sign`/`pitch_sign`/gains): explicitly not
  analytically solvable for Reachy and won't be for a Franka look-at→IK
  mapping either — expect a fresh empirical tuning pass against the new
  camera/arm frame, not a transferable formula (§2, §4 comparison table).

**Genuinely open (flagged for a hands-on spike, not resolved here):**

- The exact Menagerie model filename, flange/end-effector body name, and
  joint `range` values — none of this repo vendors `franka_emika_panda/`
  yet, so these must be read from the actual XML once pulled in, not
  guessed (§4 points 1, 5).
- Frame-source adapter performance: whether `mujoco.Renderer`-based
  per-tick rendering can sustain the detector's needed frame rate without
  starving the physics step loop (§4 point 3, open questions).
- Which RealSense model (D405 vs. D435i) to mirror for realistic
  extrinsics/FOV, and the camera's exact mounting offset — both require
  vendor datasheet/CAD data not gathered in this pass (§4 points 2, open
  questions).
- IK solver choice for the look-at-to-joint/Cartesian conversion — explicitly
  out of scope for this doc (§4, open questions).
