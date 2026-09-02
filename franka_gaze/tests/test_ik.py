from pathlib import Path

import mujoco
import numpy as np
import pytest

from franka_gaze.ik import (
    HOME_QPOS,
    _orthonormal_basis,
    compute_home_pose,
    solve,
    yaw_pitch_to_target,
)

MODEL_PATH = Path(__file__).parent.parent / "models" / "franka_emika_panda" / "scene.xml"


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


@pytest.fixture
def data(model):
    return mujoco.MjData(model)


def test_compute_home_pose_returns_finite_values(model):
    home_pos, home_mat = compute_home_pose(model)
    assert home_pos.shape == (3,)
    assert home_mat.shape == (3, 3)
    assert np.all(np.isfinite(home_pos))
    assert np.all(np.isfinite(home_mat))


def test_yaw_pitch_zero_points_at_reference(model):
    """At yaw=pitch=0 the target should aim at the fixed reference point
    (REFERENCE_DIRECTION), NOT at home_mat's own approach direction (which
    points at the table) -- this is the core behavioral fix."""
    from franka_gaze.ik import MAX_LEAN_M, REFERENCE_DIRECTION

    home_pos, home_mat = compute_home_pose(model)
    target_pos, target_mat = yaw_pitch_to_target(home_pos, home_mat, 0.0, 0.0)

    # target_mat's local Z column (the approach axis) should point along
    # REFERENCE_DIRECTION, not along home_mat's own local Z.
    approach_axis = target_mat[:, 2]
    np.testing.assert_allclose(approach_axis, REFERENCE_DIRECTION, atol=1e-6)

    # Position should lean exactly MAX_LEAN_M toward that direction, not
    # stay unchanged at home_pos.
    lean_distance = np.linalg.norm(target_pos - home_pos)
    np.testing.assert_allclose(lean_distance, MAX_LEAN_M, atol=1e-9)
    assert not np.allclose(target_mat, home_mat), (
        "target_mat must differ from home_mat -- home_mat points at the table, "
        "the reference point does not"
    )


def test_nonzero_yaw_changes_target_orientation(model):
    """Regression test for the bug fixed here: under the old Euler-composition
    code, yaw rotated the hand about its own approach axis, which is a no-op
    for pointing direction -- this test would have FAILED under that code."""
    home_pos, home_mat = compute_home_pose(model)
    _, target_mat_zero = yaw_pitch_to_target(home_pos, home_mat, 0.0, 0.0)
    _, target_mat_yawed = yaw_pitch_to_target(home_pos, home_mat, 30.0, 0.0)

    approach_zero = target_mat_zero[:, 2]
    approach_yawed = target_mat_yawed[:, 2]
    assert not np.allclose(approach_zero, approach_yawed, atol=1e-3), (
        "yaw must change the approach axis direction (genuine aiming), "
        "not just roll the wrist in place"
    )


def test_lean_magnitude_is_constant(model):
    """Position should lean by exactly MAX_LEAN_M toward the look direction
    for any yaw/pitch, not scaled by angle -- this is the fixed-magnitude
    lean design decision, not a variable one."""
    from franka_gaze.ik import MAX_LEAN_M

    home_pos, home_mat = compute_home_pose(model)
    for yaw_deg, pitch_deg in [(0.0, 0.0), (20.0, -10.0), (-35.0, 15.0)]:
        target_pos, _ = yaw_pitch_to_target(home_pos, home_mat, yaw_deg, pitch_deg)
        lean_distance = np.linalg.norm(target_pos - home_pos)
        np.testing.assert_allclose(lean_distance, MAX_LEAN_M, atol=1e-9)


def test_yaw_pitch_target_orientation_is_orthonormal(model):
    """target_mat must remain a valid rotation matrix (orthonormal columns,
    determinant +1) -- a sanity check on the new look-at construction."""
    home_pos, home_mat = compute_home_pose(model)
    _, target_mat = yaw_pitch_to_target(home_pos, home_mat, 25.0, -12.0)
    np.testing.assert_allclose(target_mat @ target_mat.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(target_mat), 1.0, atol=1e-6)


def test_solve_converges_for_home_target(model, data):
    """Solving for the home target from the home qpos should return
    (approximately) the home qpos itself -- the simplest possible
    convergence check."""
    home_pos, home_mat = compute_home_pose(model)
    qpos_init = np.array(HOME_QPOS)
    result = solve(model, data, qpos_init, home_pos, home_mat)
    np.testing.assert_allclose(result, HOME_QPOS, atol=1e-2)


def test_solve_respects_joint_limits(model, data):
    """An extreme yaw/pitch target should still return a qpos within the
    model's own joint ranges, even if the IK can't fully converge."""
    home_pos, home_mat = compute_home_pose(model)
    target_pos, target_mat = yaw_pitch_to_target(home_pos, home_mat, 60.0, 40.0)
    qpos_init = np.array(HOME_QPOS)
    result = solve(model, data, qpos_init, target_pos, target_mat)
    for i in range(7):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}")
        qpos_adr = model.jnt_qposadr[joint_id]
        lo, hi = model.jnt_range[joint_id]
        assert lo - 1e-6 <= result[i] <= hi + 1e-6, f"joint{i + 1} out of range: {result[i]}"


@pytest.mark.parametrize(
    "direction",
    [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])],
)
def test_orthonormal_basis_near_vertical_direction(direction):
    """direction nearly parallel to world_up ([0,0,1]) triggers the
    world_up fallback to [0,1,0] -- must still return a valid orthonormal
    basis, not NaNs from a near-zero cross product."""
    right, up = _orthonormal_basis(direction)

    np.testing.assert_allclose(np.linalg.norm(right), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(up), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.dot(right, direction), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.dot(up, direction), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.dot(right, up), 0.0, atol=1e-9)


def test_solve_no_nans(model, data):
    home_pos, home_mat = compute_home_pose(model)
    target_pos, target_mat = yaw_pitch_to_target(home_pos, home_mat, 20.0, -15.0)
    qpos_init = np.array(HOME_QPOS)
    result = solve(model, data, qpos_init, target_pos, target_mat)
    assert np.all(np.isfinite(result))
