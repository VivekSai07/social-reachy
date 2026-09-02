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


# Fixed reference geometry for the look-at construction. Unknown a priori,
# same as social_app.gaze.GazeConfig's own yaw_sign/pitch_sign -- these are
# starting guesses to be tuned empirically in the sim viewer, not derived
# analytically. See franka_gaze/plan.md for the tuning procedure.
REFERENCE_DISTANCE_M = 0.6  # nominal distance to the "person zone"
REFERENCE_DIRECTION = np.array([1.0, 0.0, 0.0])  # world +X: starting guess for "where a person stands"
MAX_LEAN_M = 0.10  # fixed lean-in distance toward the look direction


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (right, up) unit vectors forming an orthonormal basis with
    `direction` (already unit length), for panning/tilting within the plane
    perpendicular to it."""
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, direction)
    right = right / np.linalg.norm(right)
    up = np.cross(direction, right)
    up = up / np.linalg.norm(up)
    return right, up


def _look_at_rotation(forward: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Builds a 3x3 rotation matrix whose local +Z axis (the Panda hand's
    approach axis) points along `forward` (already unit length), using
    `up_hint` to fix the remaining rotation about that axis. Standard
    camera-look-at basis construction.
    """
    right = np.cross(up_hint, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        raise ValueError(
            "_look_at_rotation: forward and up_hint are nearly parallel "
            f"(forward={forward!r}, up_hint={up_hint!r}); cannot construct "
            "a look-at basis from a degenerate cross product."
        )
    right = right / right_norm
    up = np.cross(forward, right)
    return np.column_stack([right, up, forward])


def yaw_pitch_to_target(
    home_pos: np.ndarray,
    home_mat: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Look-at target: aims the hand's approach axis (local Z) at a virtual
    point representing the tracked person, replacing the old approach of
    composing local Euler rotations on home_mat (which rotated the hand
    about its OWN approach axis for yaw -- a no-op for pointing direction;
    see franka_gaze/plan.md's "Known issue" note this fixes).

    At yaw=pitch=0 the target points at REFERENCE_POINT, a fixed nominal
    "person zone" in front of the arm's base -- NOT home_mat's own approach
    direction (which points at the table). yaw/pitch pan/tilt around that
    point within the plane perpendicular to REFERENCE_DIRECTION. Position
    leans a small fixed MAX_LEAN_M toward the look direction.

    home_mat is intentionally unused for orientation (kept as a parameter
    only for interface stability with main.py/solve(), which pass it
    unconditionally every tick).
    """
    del home_mat  # unused -- see docstring
    reference_point = home_pos + REFERENCE_DISTANCE_M * REFERENCE_DIRECTION
    right, up = _orthonormal_basis(REFERENCE_DIRECTION)

    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    offset = REFERENCE_DISTANCE_M * (np.tan(yaw_rad) * right + np.tan(pitch_rad) * up)
    target_point = reference_point + offset

    direction = target_point - home_pos
    direction = direction / np.linalg.norm(direction)

    target_mat = _look_at_rotation(direction, up)
    target_pos = home_pos + MAX_LEAN_M * direction
    return target_pos, target_mat


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

    # Scratch MjData, private to this solve -- must never touch the
    # caller's live simulation state (main.py passes the same MjData the
    # viewer renders and mj_step integrates from). Mirrors the pattern
    # already used by compute_home_pose() above.
    scratch = mujoco.MjData(model)
    scratch.qpos[qpos_adrs] = qpos_init

    q = np.array(qpos_init, dtype=float).copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(max_iters):
        scratch.qpos[qpos_adrs] = q
        mujoco.mj_forward(model, scratch)

        cur_pos = scratch.xpos[hand_id]
        cur_mat = scratch.xmat[hand_id].reshape(3, 3)

        pos_err = target_pos - cur_pos
        rot_err = _mat_to_rotvec(target_mat @ cur_mat.T)
        err = np.concatenate([pos_err, rot_err])

        if np.linalg.norm(err) < tol:
            break

        mujoco.mj_jacBody(model, scratch, jacp, jacr, hand_id)
        dof_adrs = [model.jnt_dofadr[jid] for jid in _arm_joint_ids(model)]
        j_pos = jacp[:, dof_adrs]
        j_rot = jacr[:, dof_adrs]
        jac = np.vstack([j_pos, j_rot])  # (6, 7)

        dq = np.linalg.solve(jac.T @ jac + damping * np.eye(7), jac.T @ err)
        q = q + dq
        q = np.clip(q, limits[:, 0], limits[:, 1])

    return q
