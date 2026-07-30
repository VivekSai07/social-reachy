"""Continuous head-tracking move driven by the webcam-based gaze controller.

Modeled on moves.BreathingMove: infinite duration, evicted from the move
queue by the same idle-management logic (see moves.py's _manage_breathing).

GazeController/WebcamFaceTracker are NOT owned by this class -- they are
long-lived, owned by MovementManager, and passed in by reference each time a
GazeMove is (re)constructed. Constructing fresh instances per activation
would reset the controller's EMA smoothing state (visible snap-to-center
every time gaze regains the idle slot) and break its stale-tracker timeout
logic, which compares against absolute time.monotonic() timestamps carried
on FaceObservation from the tracker thread -- not against this move's local
`t`, which restarts at 0 on every activation.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import linear_pose_interpolation

from companion.gaze import GazeController
from companion.perception import WebcamFaceTracker


class GazeMove(Move):  # type: ignore
    """Continuously look toward the tracked face; blends in from wherever the head currently is."""

    def __init__(
        self,
        gaze_controller: GazeController,
        webcam_tracker: WebcamFaceTracker,
        interpolation_start_pose: NDArray[np.float32],
        interpolation_start_antennas: Tuple[float, float],
        interpolation_duration: float = 0.5,
    ):
        """Initialize the gaze move.

        Args:
            gaze_controller: shared, long-lived GazeController (owned by MovementManager).
            webcam_tracker: shared, long-lived WebcamFaceTracker (owned by MovementManager).
            interpolation_start_pose: 4x4 matrix of current head pose to blend from.
            interpolation_start_antennas: current antenna positions to blend from.
            interpolation_duration: duration of the initial blend into tracking (seconds).

        """
        self._gaze_controller = gaze_controller
        self._webcam_tracker = webcam_tracker
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration

        self.neutral_antennas = np.array([-0.1745, 0.1745])  # ~10° offset to reduce shaking

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float("inf")  # Continuous tracking (never ends naturally)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate the gaze move at time t.

        `t` is used only for the local startup blend below. The gaze
        controller itself is always driven by time.monotonic() directly, not
        `t` -- see the class docstring for why.
        """
        yaw_deg, pitch_deg = self._gaze_controller.update(self._webcam_tracker.latest(), time.monotonic())
        target_head_pose = create_head_pose(yaw=yaw_deg, pitch=pitch_deg, degrees=True)

        if t < self.interpolation_duration:
            interpolation_t = t / self.interpolation_duration
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose,
                target_head_pose,
                interpolation_t,
            )
            antennas_interp = (
                1 - interpolation_t
            ) * self.interpolation_start_antennas + interpolation_t * self.neutral_antennas
            antennas = antennas_interp.astype(np.float64)
        else:
            head_pose = target_head_pose
            antennas = self.neutral_antennas.astype(np.float64)

        return (head_pose, antennas, 0.0)
