"""Webcam-based face tracking: capture + detect on a background thread.

Mirrors reachy_mini.vision.face_tracking.FaceTracker's architecture
(background thread does capture+detect+select, publishes to a
queue.SimpleQueue, latest() drains it for a non-blocking snapshot read) but
reads from the laptop's own webcam via OpenCV instead of the daemon's
GStreamer camera pipe -- there is no human inside the MuJoCo scene for the
robot's own simulated camera to see.
"""

from __future__ import annotations

import logging
import platform
import queue
import threading
import time
from dataclasses import dataclass

import cv2

from reachy_mini.vision.face_detector import Face, FaceDetector
from reachy_mini.vision.face_tracking import Tracker

logger = logging.getLogger(__name__)


@dataclass
class FaceObservation:
    """One detection tick's result."""

    center: tuple[float, float] | None
    timestamp: float


def _normalized_center(face: Face, width: int, height: int) -> tuple[float, float]:
    """Nose position normalized to [-1, 1].

    Reimplemented locally rather than importing
    reachy_mini.vision.face_tracking._center, which is a private helper we
    shouldn't depend on across SDK upgrades.
    """
    return (
        face.nose[0] / max(width - 1, 1) * 2 - 1,
        face.nose[1] / max(height - 1, 1) * 2 - 1,
    )


class WebcamFaceTracker:
    """Runs webcam capture + face detection on a background thread; exposes
    the latest observation via a non-blocking latest() snapshot read."""

    def __init__(
        self,
        camera_index: int = 0,
        capture_width: int = 640,
        capture_height: int = 480,
        target_fps: float = 20.0,
        min_area_frac: float = 0.02,
        max_jump: float = 0.5,
        max_misses: int = 15,
    ) -> None:
        self._camera_index = camera_index
        self._capture_width = capture_width
        self._capture_height = capture_height
        self._frame_interval = 1.0 / target_fps
        self._min_area_frac = min_area_frac
        self._max_jump = max_jump
        self._max_misses = max_misses

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._observations: "queue.SimpleQueue[FaceObservation]" = queue.SimpleQueue()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._observations = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="webcam-face-tracker"
        )
        self._thread.start()

    def latest(self) -> FaceObservation | None:
        """Return the most recent observation since the last call, draining
        any backlog. Returns None if no new observation has arrived."""
        obs: FaceObservation | None = None
        while not self._observations.empty():
            obs = self._observations.get_nowait()
        return obs

    def stop(self) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning("Webcam face tracker thread did not stop in time.")
        else:
            self._thread = None

    def _run(self) -> None:
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._camera_index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._capture_height)
        if not cap.isOpened():
            logger.warning(
                "Webcam unavailable (index=%d); gaze tracking disabled, "
                "idle behavior only.",
                self._camera_index,
            )
            return
        try:
            # Model auto-downloads from HF Hub on first construction (needs
            # network once, then cached); pinned to 1 CPU thread internally.
            detector = FaceDetector()
            tracker = Tracker(
                min_area_frac=self._min_area_frac,
                max_jump=self._max_jump,
                max_misses=self._max_misses,
            )
            next_tick = time.monotonic()
            while not self._stop.is_set():
                ok, frame_bgr = cap.read()
                if not ok:
                    self._stop.wait(0.05)
                    continue
                height, width = frame_bgr.shape[:2]
                face = tracker.select(detector.detect(frame_bgr), width, height)
                center = (
                    _normalized_center(face, width, height)
                    if face is not None
                    else None
                )
                self._observations.put(
                    FaceObservation(center=center, timestamp=time.monotonic())
                )
                next_tick += self._frame_interval
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    self._stop.wait(sleep_for)
                else:
                    next_tick = time.monotonic()
        except Exception:
            logger.exception("Webcam face tracker crashed.")
        finally:
            cap.release()
