"""Pure-function IK: converts a smoothed (yaw_deg, pitch_deg) target from
social_app.gaze.GazeController into Franka joint angles. No I/O, no
mujoco.viewer dependency -- takes an already-loaded model/data, mirroring
social_app.gaze.GazeController's own I/O-free design so this stays testable
in isolation and reusable from a different control loop later.
"""

from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
HAND_BODY_NAME = "hand"

# Franka's standard published "ready" pose (radians) -- the common default
# starting configuration used across Franka examples/tutorials.
HOME_QPOS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


def _arm_joint_ids(model: mujoco.MjModel) -> list[int]:
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES]


def _arm_qpos_adrs(model: mujoco.MjModel) -> list[int]:
    return [model.jnt_qposadr[jid] for jid in _arm_joint_ids(model)]


def _joint_limits(model: mujoco.MjModel) -> np.ndarray:
    """Returns a (7, 2) array of (lo, hi) radians, read from the model."""
    return np.array([model.jnt_range[jid] for jid in _arm_joint_ids(model)])


def _hand_body_id(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY_NAME)


def compute_home_pose(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Returns (home_pos, home_mat): the hand body's world position and
    3x3 rotation matrix at HOME_QPOS."""
    data = mujoco.MjData(model)
    qpos_adrs = _arm_qpos_adrs(model)
    data.qpos[qpos_adrs] = HOME_QPOS
    mujoco.mj_forward(model, data)
    hand_id = _hand_body_id(model)
    home_pos = data.xpos[hand_id].copy()
    home_mat = data.xmat[hand_id].reshape(3, 3).copy()
    return home_pos, home_mat


def _rotz(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotx(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def yaw_pitch_to_target(
    home_pos: np.ndarray,
    home_mat: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Orientation-only target: position stays fixed at home_pos (no depth
    signal from a monocular webcam -- same angular-only approach
    social_app.gaze uses for Reachy's head). Orientation is home_mat
    rotated by yaw about the hand's local Z, then pitch about its local X.

    yaw_sign/pitch_sign are intentionally NOT included here (unlike
    social_app.gaze.GazeConfig) -- sign convention for this arm/camera pair
    is unknown a priori and must be tuned empirically in the sim viewer,
    same as Reachy's own GazeConfig.yaw_sign/pitch_sign. Task 4's main.py
    is where that tuning constant lives.
    """
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    target_mat = home_mat @ _rotz(yaw_rad) @ _rotx(pitch_rad)
    return home_pos.copy(), target_mat


def _mat_to_rotvec(mat: np.ndarray) -> np.ndarray:
    """Small-angle axis-angle extraction from a rotation matrix, used for
    the orientation error term. Good enough for the small per-tick errors
    this solver operates on (damped least squares, not a global solve)."""
    rotvec = np.array(
        [
            mat[2, 1] - mat[1, 2],
            mat[0, 2] - mat[2, 0],
            mat[1, 0] - mat[0, 1],
        ]
    )
    return 0.5 * rotvec


def solve(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_init: np.ndarray,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    max_iters: int = 20,
    tol: float = 1e-3,
    damping: float = 1e-4,
) -> np.ndarray:
    """Damped-least-squares differential IK for the 7-DOF arm. Holds
    end-effector position at target_pos while driving orientation toward
    target_mat. Returns a length-7 qpos array, clipped to the model's own
    joint ranges.
    """
    qpos_adrs = _arm_qpos_adrs(model)
    hand_id = _hand_body_id(model)
    limits = _joint_limits(model)

    q = np.array(qpos_init, dtype=float).copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(max_iters):
        data.qpos[qpos_adrs] = q
        mujoco.mj_forward(model, data)

        cur_pos = data.xpos[hand_id]
        cur_mat = data.xmat[hand_id].reshape(3, 3)

        pos_err = target_pos - cur_pos
        rot_err = _mat_to_rotvec(target_mat @ cur_mat.T)
        err = np.concatenate([pos_err, rot_err])

        if np.linalg.norm(err) < tol:
            break

        mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
        dof_adrs = [model.jnt_dofadr[jid] for jid in _arm_joint_ids(model)]
        j_pos = jacp[:, dof_adrs]
        j_rot = jacr[:, dof_adrs]
        jac = np.vstack([j_pos, j_rot])  # (6, 7)

        dq = np.linalg.solve(jac.T @ jac + damping * np.eye(7), jac.T @ err)
        q = q + dq
        q = np.clip(q, limits[:, 0], limits[:, 1])

    return q
