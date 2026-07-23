import time
import threading
from unittest.mock import MagicMock, call
from collections.abc import Callable

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset
from companion.moves import BreathingMove, MovementManager
from companion.gaze_move import GazeMove
from companion.dance_emotion_moves import EmotionQueueMove


class _FakeMove:
    """Minimal non-emotion Move stub returning a fixed head pose."""

    def __init__(self, head: np.ndarray) -> None:
        self._head = head
        self.duration = 10.0

    def evaluate(self, t: float):
        return (self._head, np.array([0.0, 0.0]), 0.0)


def _wait_for(predicate: Callable[[], bool], timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_stop_can_skip_neutral_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep shutdown should stop the movement loop without undoing the sleep pose."""
    robot = MagicMock()
    manager = MovementManager(robot)
    started = threading.Event()

    def fake_working_loop() -> None:
        started.set()
        while not manager._stop_event.is_set():
            time.sleep(0.001)

    monkeypatch.setattr(manager, "working_loop", fake_working_loop)

    manager.start()
    assert started.wait(timeout=1.0)

    manager.stop(reset_to_neutral=False)

    assert manager._thread is None
    robot.goto_target.assert_not_called()


def test_head_tracking_follows_speaking() -> None:
    """Once enabled, tracking owns the head when idle and releases it while the assistant speaks."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    manager = MovementManager(robot)
    manager.start()
    try:
        # The head_tracking tool enables tracking with full weight.
        manager.set_head_tracking(True)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)

        # Speaking with a locked face captures the anchor and releases the head.
        manager.set_speaking(True)
        assert _wait_for(lambda: call(weight=0.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is not None)

        # Done speaking hands the head back to tracking.
        robot.start_head_tracking.reset_mock()
        manager.set_speaking(False)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is None)
    finally:
        manager.stop(reset_to_neutral=False)

    robot.stop_head_tracking.assert_called_once()


def test_idle_move_is_gaze_when_gaze_tracking_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """When gaze tracking is enabled, the idle move should be a GazeMove, not BreathingMove."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    manager = MovementManager(robot)
    monkeypatch.setattr(manager._webcam_tracker, "start", lambda: None)
    monkeypatch.setattr(manager._webcam_tracker, "stop", lambda: None)
    manager.idle_inactivity_delay = 0.05
    manager.start()
    try:
        manager.set_gaze_tracking(True)
        assert _wait_for(lambda: isinstance(manager.state.current_move, GazeMove), timeout=2.0)
        assert not isinstance(manager.state.current_move, BreathingMove)
    finally:
        manager.stop(reset_to_neutral=False)


def test_toggling_gaze_tracking_evicts_current_idle_move(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flipping gaze tracking should switch the idle move immediately, not wait for an unrelated move to evict it.

    Both BreathingMove and GazeMove have an infinite duration, so without the
    proactive eviction in the set_gaze_tracking command handler, the switch
    would sit inert until an unrelated real move happened to interrupt
    whichever one is currently running.
    """
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    manager = MovementManager(robot)
    monkeypatch.setattr(manager._webcam_tracker, "start", lambda: None)
    monkeypatch.setattr(manager._webcam_tracker, "stop", lambda: None)
    manager.idle_inactivity_delay = 0.05
    manager.start()
    try:
        assert _wait_for(lambda: isinstance(manager.state.current_move, BreathingMove), timeout=2.0)

        manager.set_gaze_tracking(True)
        assert _wait_for(lambda: isinstance(manager.state.current_move, GazeMove), timeout=2.0)

        manager.set_gaze_tracking(False)
        assert _wait_for(lambda: isinstance(manager.state.current_move, BreathingMove), timeout=2.0)
    finally:
        manager.stop(reset_to_neutral=False)


def test_speaking_anchor_composes_emotions_and_holds_dances_from_neutral() -> None:
    """While speaking: hold the anchor, compose emotions onto it, play dances from neutral."""
    robot = MagicMock()
    manager = MovementManager(robot)
    anchor = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True)
    manager._track_anchor = anchor

    # No move: the head holds the captured look-at anchor.
    manager.state.current_move = None
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, anchor)

    # Emotion: composed onto the anchor exactly like the daemon wobble.
    emotion_head = create_head_pose(0, 0, 0, 0, 0, 15, degrees=True)
    recorded = MagicMock()
    recorded.get.return_value = _FakeMove(emotion_head)
    manager.state.current_move = EmotionQueueMove("happy", recorded)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, compose_world_offset(anchor, emotion_head))

    # Any other move (e.g. a dance) plays from its own neutral base, ignoring the anchor.
    dance_head = create_head_pose(0, 0, 0, 0, 25, 0, degrees=True)
    manager.state.current_move = _FakeMove(dance_head)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, dance_head)
