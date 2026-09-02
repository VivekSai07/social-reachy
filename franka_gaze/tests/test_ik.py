from pathlib import Path

import mujoco
import numpy as np
import pytest

from franka_gaze.ik import (
    HOME_QPOS,
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


def test_yaw_pitch_zero_returns_home_unchanged(model):
    home_pos, home_mat = compute_home_pose(model)
    target_pos, target_mat = yaw_pitch_to_target(home_pos, home_mat, 0.0, 0.0)
    np.testing.assert_allclose(target_pos, home_pos, atol=1e-9)
    np.testing.assert_allclose(target_mat, home_mat, atol=1e-9)


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


def test_solve_no_nans(model, data):
    home_pos, home_mat = compute_home_pose(model)
    target_pos, target_mat = yaw_pitch_to_target(home_pos, home_mat, 20.0, -15.0)
    qpos_init = np.array(HOME_QPOS)
    result = solve(model, data, qpos_init, target_pos, target_mat)
    assert np.all(np.isfinite(result))
