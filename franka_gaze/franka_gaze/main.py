"""Control loop: webcam face -> smoothed yaw/pitch (social_app's existing
code, unchanged) -> Franka joint targets (this package's ik.py) -> MuJoCo
sim. One control-application call site per tick, mirroring
apps/social_app/social_app/main.py's single set_target() rule.
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from social_app.gaze import GazeConfig, GazeController
from social_app.perception import WebcamFaceTracker

from franka_gaze.ik import HOME_QPOS, compute_home_pose, solve, yaw_pitch_to_target

MODEL_PATH = Path(__file__).parent.parent / "models" / "franka_emika_panda" / "scene.xml"
LOOP_HZ = 30.0
LOOP_PERIOD_S = 1.0 / LOOP_HZ

# Unknown a priori, like social_app.gaze.GazeConfig's own yaw_sign/pitch_sign
# -- tune empirically in the sim viewer if the arm turns the wrong way.
YAW_SIGN = 1.0
PITCH_SIGN = 1.0


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    arm_actuator_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") for i in range(1, 8)
    ]

    home_pos, home_mat = compute_home_pose(model)
    q = np.array(HOME_QPOS, dtype=float)
    data.qpos[: len(q)] = q
    mujoco.mj_forward(model, data)

    tracker = WebcamFaceTracker()
    gaze = GazeController(GazeConfig())
    tracker.start()

    t0 = time.monotonic()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            next_tick = time.monotonic()
            while viewer.is_running():
                now = time.monotonic()

                yaw_deg, pitch_deg = gaze.update(tracker.latest(), now - t0)
                target_pos, target_mat = yaw_pitch_to_target(
                    home_pos, home_mat, YAW_SIGN * yaw_deg, PITCH_SIGN * pitch_deg
                )

                new_q = solve(model, data, q, target_pos, target_mat)
                if np.all(np.isfinite(new_q)):
                    q = new_q
                # else: hold last good q (IK failure -- coast, don't snap,
                # matching gaze.py's own grace-period philosophy)

                # The one control-application call site for this loop.
                data.ctrl[arm_actuator_ids] = q

                mujoco.mj_step(model, data)
                viewer.sync()

                next_tick += LOOP_PERIOD_S
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
