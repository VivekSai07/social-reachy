from pathlib import Path

import mujoco

MODEL_PATH = Path(__file__).parent.parent / "models" / "franka_emika_panda" / "scene.xml"

EXPECTED_BODIES = [
    "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
    "hand", "left_finger", "right_finger",
]
EXPECTED_ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]


def test_model_loads():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    assert model is not None


def test_expected_bodies_present():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    for name in EXPECTED_BODIES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert body_id != -1, f"body {name!r} not found in model"


def test_expected_arm_joints_present():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    for name in EXPECTED_ARM_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id != -1, f"joint {name!r} not found in model"


def test_hand_body_has_no_site():
    """Documents the Global Constraint this plan depends on: IK targets the
    hand body's origin directly (mj_jacBody), not a site, because none
    exists on this model as vendored."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    site_names_on_hand = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(model.nsite)
        if model.site_bodyid[i] == hand_id
    ]
    assert site_names_on_hand == []
