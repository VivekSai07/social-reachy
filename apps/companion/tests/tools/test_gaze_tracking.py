from unittest.mock import MagicMock

import pytest

from companion.tools.core_tools import ToolDependencies
from companion.tools.gaze_tracking import GazeTracking


@pytest.mark.asyncio
async def test_gaze_tracking_enables_and_disables() -> None:
    """The tool forwards the toggle to the movement manager."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    result = await GazeTracking()(deps, enabled=True)
    deps.movement_manager.set_gaze_tracking.assert_called_with(True)
    assert result == {"status": "following"}

    result = await GazeTracking()(deps, enabled=False)
    deps.movement_manager.set_gaze_tracking.assert_called_with(False)
    assert result == {"status": "stopped following"}
