import logging
from typing import Any, Dict

from companion.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class GazeTracking(Tool):
    """Enable or disable following the user's face with the head, using the app's own webcam-based tracker.

    Distinct from the daemon's built-in head tracking (which uses the
    robot's own camera and is useless in the MuJoCo simulator, since there's
    no human in the simulated scene for it to see): this uses the laptop's
    webcam directly, via companion.perception.WebcamFaceTracker.
    """

    name = "gaze_tracking"
    description = (
        "Enable or disable following the user's face with the head. "
        "Use when asked to follow, keep looking at, or stop following the user."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "True to start following the user's face, false to stop.",
            },
        },
        "required": ["enabled"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Toggle webcam-based gaze tracking."""
        enabled = bool(kwargs.get("enabled", True))
        logger.info("Tool call: gaze_tracking enabled=%s", enabled)
        deps.movement_manager.set_gaze_tracking(enabled)
        return {"status": "following" if enabled else "stopped following"}
