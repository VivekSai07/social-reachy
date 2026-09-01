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
