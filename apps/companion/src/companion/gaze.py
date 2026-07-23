"""Pure pose-computation: turns the latest face observation into a smoothed
head-pose target. No I/O, no threading, no set_target() call -- designed to
be invoked once per control-loop tick so a future conversation-phase loop
can call the same GazeController.update() without extracting or rewriting
this logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from companion.perception import FaceObservation


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass
class GazeConfig:
    # --- Starting points only: tune empirically in the sim viewer. ---
    yaw_gain_deg: float = 45.0  # deg of head yaw per unit of normalized x (-1..1)
    pitch_gain_deg: float = 25.0  # deg of head pitch per unit of normalized y (-1..1)
    yaw_sign: float = -1.0  # unknown a priori -- flip if head turns the wrong way in sim
    pitch_sign: float = 1.0  # unknown a priori -- flip if head tilts the wrong way in sim
    yaw_limit_deg: float = 45.0  # sub-range of hardware +-180 deg
    pitch_limit_deg: float = 20.0  # sub-range of hardware +-40 deg
    smoothing_alpha: float = 0.25  # EMA coefficient per tick (higher = snappier, more jitter)
    face_grace_period_s: float = 0.4  # coast through detector gaps shorter than this
    observation_timeout_s: float = 1.0  # tracker thread considered stalled/dead beyond this
    idle_amplitude_deg: float = 8.0  # slow idle sway amplitude (yaw) when no face
    idle_period_s: float = 8.0  # idle sway period


class GazeController:
    """Stateful smoother: call update() once per control-loop tick."""

    def __init__(self, config: GazeConfig | None = None) -> None:
        self._cfg = config or GazeConfig()
        self._yaw_deg = 0.0
        self._pitch_deg = 0.0
        self._target_yaw_deg = 0.0
        self._target_pitch_deg = 0.0
        self._last_face_seen_at: float | None = None
        self._last_observation_at: float | None = None

    def update(
        self, observation: FaceObservation | None, now: float
    ) -> tuple[float, float]:
        """Returns (yaw_deg, pitch_deg). `observation` may be None (no new
        detector tick since the last call) or have `center=None` (detector
        ran, found nothing this frame)."""
        cfg = self._cfg

        if observation is not None:
            self._last_observation_at = observation.timestamp
            if observation.center is not None:
                x, y = observation.center
                self._target_yaw_deg = _clamp(
                    cfg.yaw_sign * x * cfg.yaw_gain_deg, cfg.yaw_limit_deg
                )
                self._target_pitch_deg = _clamp(
                    cfg.pitch_sign * y * cfg.pitch_gain_deg, cfg.pitch_limit_deg
                )
                self._last_face_seen_at = observation.timestamp

        tracker_alive = (
            self._last_observation_at is not None
            and (now - self._last_observation_at) <= cfg.observation_timeout_s
        )
        face_recently_seen = (
            self._last_face_seen_at is not None
            and (now - self._last_face_seen_at) <= cfg.face_grace_period_s
        )

        if tracker_alive and face_recently_seen:
            target_yaw, target_pitch = self._target_yaw_deg, self._target_pitch_deg
        else:
            target_yaw = cfg.idle_amplitude_deg * math.sin(
                2 * math.pi * now / cfg.idle_period_s
            )
            target_pitch = 0.0

        self._yaw_deg += cfg.smoothing_alpha * (target_yaw - self._yaw_deg)
        self._pitch_deg += cfg.smoothing_alpha * (target_pitch - self._pitch_deg)
        return self._yaw_deg, self._pitch_deg
