# Franka Gaze Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Franka Panda arm, simulated in MuJoCo, whose end-effector visibly and smoothly orients toward a human face detected by the laptop's webcam — reusing Reachy Mini's existing perception/smoothing code unchanged.

**Architecture:** A new standalone package `franka_gaze/`, sibling to `apps/`, installed editable into the shared `.venv`. It imports `social_app.perception.WebcamFaceTracker` and `social_app.gaze.GazeController` directly (no code duplication), adds one new pure-function module (`ik.py`) that converts the existing `(yaw_deg, pitch_deg)` output into Franka joint targets via damped-least-squares differential IK, and a control-loop script (`main.py`) that drives `mujoco.viewer.launch_passive()` with a single "apply joint command" call site per tick — mirroring `apps/social_app/social_app/main.py`'s one-`set_target()`-call discipline.

**Tech Stack:** Python 3.12, `mujoco` (official Python bindings), `numpy`, vendored `franka_emika_panda` MJCF from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0), `pytest` for `ik.py`'s unit tests.

**Spec:** `docs/superpowers/specs/2026-09-01-franka-gaze-design.md`

## Global Constraints

- No physical hardware — MuJoCo simulation only.
- Reuse `social_app.perception.WebcamFaceTracker` and `social_app.gaze.GazeController` unchanged — no forking/copying their code.
- Camera stays world-fixed (the laptop webcam) — never a MuJoCo-rendered camera feed. No synthetic face asset in the scene.
- `franka_gaze/` is NOT scaffolded via `reachy-mini-app-assistant` — that tool is Reachy-daemon-specific.
- One control-application call site per tick in `main.py` (mirrors `apps/social_app`'s `set_target()` rule).
- IK failure (non-convergence) holds the last good joint target rather than snapping to an undefined pose.
- Success is visual/qualitative (smooth tracking in the MuJoCo viewer, no jitter/wild motion) — no numeric tracking-accuracy benchmark required for this pass.
- Confirmed Menagerie `franka_emika_panda/panda.xml` facts (verified via direct fetch, 2026-09-01 — re-verify if Menagerie has since changed): body names `link0`...`link7`, `hand`, `left_finger`, `right_finger`; joint names `joint1`...`joint7` (arm), `finger_joint1`/`finger_joint2` (gripper); actuators `actuator1`...`actuator7` (arm, position-servo `general` class), `actuator8` (gripper); **no `<site>` defined on the `hand` body** — IK must target the `hand` body's origin directly via `mj_jacBody`, not a site. `scene.xml` includes `panda.xml` plus a ground plane, skybox, and light — load `scene.xml` for the simulation, not `panda.xml` alone.

---

### Task 1: Vendor the Franka model and verify it loads

**Files:**
- Create: `franka_gaze/models/franka_emika_panda/panda.xml` (fetched from Menagerie)
- Create: `franka_gaze/models/franka_emika_panda/scene.xml` (fetched from Menagerie)
- Create: `franka_gaze/models/franka_emika_panda/assets/` (mesh/texture files the MJCF references — fetch whatever `panda.xml`'s `<mesh>`/`<texture>` elements point to)
- Create: `franka_gaze/models/franka_emika_panda/LICENSE` (Menagerie's own Apache-2.0 license, copied verbatim per its terms)
- Create: `franka_gaze/tests/test_model_loads.py`

**Interfaces:**
- Produces: a loadable MuJoCo model at `franka_gaze/models/franka_emika_panda/scene.xml`, confirmed (by this task's test) to expose body `"hand"` and joints `"joint1"`...`"joint7"` — later tasks depend on these exact names.

- [ ] **Step 1: Fetch the model files**

Clone or download the `franka_emika_panda` directory from
`https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda`
into `franka_gaze/models/franka_emika_panda/` — this includes `panda.xml`,
`scene.xml`, the `assets/` directory (STL/OBJ meshes referenced by
`<mesh>` elements) and the top-level `LICENSE` file. The simplest approach:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie
cd /tmp/menagerie
git sparse-checkout set franka_emika_panda LICENSE
```

Then copy `franka_emika_panda/*` into
`D:\Projects\social-reachy\franka_gaze\models\franka_emika_panda\`, and
copy the top-level `LICENSE` alongside it.

- [ ] **Step 2: Write the verification test**

```python
# franka_gaze/tests/test_model_loads.py
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
```

- [ ] **Step 3: Run the test to verify the vendored model matches expectations**

Run (from repo root, using the shared venv):
`.venv\Scripts\python.exe -m pytest franka_gaze/tests/test_model_loads.py -v`

Expected: all 4 tests PASS. If any body/joint name doesn't match, Menagerie's
model changed since this plan was written (2026-09-01) — stop and update
this plan's Global Constraints section with the actual names before
continuing to Task 3/4, since they hardcode these names.

- [ ] **Step 4: Commit**

```bash
git add franka_gaze/models/ franka_gaze/tests/test_model_loads.py
git commit -m "Vendor franka_emika_panda MuJoCo model from Menagerie"
```

---

### Task 2: Scaffold the franka_gaze package

**Files:**
- Create: `franka_gaze/pyproject.toml`
- Create: `franka_gaze/franka_gaze/__init__.py`
- Create: `franka_gaze/README.md`
- Create: `franka_gaze/plan.md`
- Modify: none

**Interfaces:**
- Produces: an editable-installed Python package `franka_gaze` in the shared `.venv`, importable as `import franka_gaze`, with `social_app` and `mujoco` available as dependencies.

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "franka_gaze"
version = "0.1.0"
description = "Franka Panda gaze/attention prototype in MuJoCo, reusing Reachy Mini's webcam face-tracking."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "social_app",
    "mujoco>=3.0",
    "numpy",
]

[tool.setuptools]
package-dir = { "" = "." }
include-package-data = true

[tool.setuptools.packages.find]
where = ["."]
include = ["franka_gaze*"]

[tool.setuptools.package-data]
franka_gaze = ["../models/**/*"]
```

- [ ] **Step 2: Write the package `__init__.py`**

```python
# franka_gaze/franka_gaze/__init__.py
```

(empty — package marker only)

- [ ] **Step 3: Write a minimal README**

```markdown
# franka_gaze

A Franka Panda gaze/attention prototype, simulated in MuJoCo. Reuses
Reachy Mini's webcam face-tracking (`social_app.perception`,
`social_app.gaze`) unchanged; the only new logic is converting the
resulting yaw/pitch target into Franka joint angles via IK.

See `plan.md` for design rationale and
`docs/superpowers/specs/2026-09-01-franka-gaze-design.md` for the full
design spec.

Run: `.venv\Scripts\python.exe -m franka_gaze.main`
```

- [ ] **Step 4: Write plan.md**

```markdown
# plan.md — franka_gaze

Written before implementing behavior logic, following this repo's existing
convention (`apps/social_app/plan.md`).

## Goal

Prove that Reachy Mini's gaze/attention architecture generalizes beyond
Reachy: a Franka Panda arm's end-effector visibly, smoothly orients toward
a detected human face, using the exact same perception/smoothing code
(`social_app.perception.WebcamFaceTracker`, `social_app.gaze.GazeController`)
unchanged, with only a new IK stage added at the end of the pipeline.

## Design

Full spec: `docs/superpowers/specs/2026-09-01-franka-gaze-design.md`.
Full research this follows from: `docs/franka_realsense_gaze_findings.md`
(see its §5 for why this is a fixed-camera design, not eye-in-hand).

## Status

In progress — see `docs/superpowers/plans/2026-09-01-franka-gaze-prototype.md`
for the task-by-task implementation plan.
```

- [ ] **Step 5: Install editable into the shared venv**

Run: `.venv\Scripts\python.exe -m pip install -e franka_gaze[test]` — if
this errors on missing `mujoco`, run
`.venv\Scripts\python.exe -m pip install mujoco>=3.0` first, then retry.

- [ ] **Step 6: Verify the import works**

Run: `.venv\Scripts\python.exe -c "import franka_gaze; from social_app.perception import WebcamFaceTracker; from social_app.gaze import GazeController; print('ok')"`

Expected output: `ok`

- [ ] **Step 7: Commit**

```bash
git add franka_gaze/pyproject.toml franka_gaze/franka_gaze/__init__.py franka_gaze/README.md franka_gaze/plan.md
git commit -m "Scaffold franka_gaze package"
```

---

### Task 3: IK — pure function, TDD

**Files:**
- Create: `franka_gaze/franka_gaze/ik.py`
- Test: `franka_gaze/tests/test_ik.py`

**Interfaces:**
- Consumes: the vendored model at `franka_gaze/models/franka_emika_panda/scene.xml` (Task 1), body name `"hand"`, joint names `"joint1"`...`"joint7"`.
- Produces:
  - `HOME_QPOS: list[float]` — length-7 constant, the standard Franka "ready" pose in radians: `[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]`.
  - `JOINT_LIMITS: list[tuple[float, float]]` — length-7 list of `(lo, hi)` in radians, read from the model at runtime, not hardcoded (`model.jnt_range` indexed by each joint's `qposadr`).
  - `compute_home_pose(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]` — returns `(home_pos: (3,) ndarray, home_mat: (3,3) ndarray)`, the `hand` body's world position and rotation matrix at `HOME_QPOS`.
  - `yaw_pitch_to_target(home_pos: np.ndarray, home_mat: np.ndarray, yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray]` — returns `(target_pos, target_mat)`; position is unchanged from `home_pos` (this is an orientation-only target — see spec's "no depth signal" rationale), orientation is `home_mat` rotated by yaw (about the hand's local Z) then pitch (about the hand's local X).
  - `solve(model: mujoco.MjModel, data: mujoco.MjData, qpos_init: np.ndarray, target_pos: np.ndarray, target_mat: np.ndarray, max_iters: int = 20, tol: float = 1e-3, damping: float = 1e-4) -> np.ndarray` — returns a length-7 `qpos` array for `joint1`...`joint7`, clipped to `JOINT_LIMITS`. This is the function `main.py` (Task 4) calls every control-loop tick.

- [ ] **Step 1: Write the failing tests**

```python
# franka_gaze/tests/test_ik.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest franka_gaze/tests/test_ik.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'franka_gaze.ik'`

- [ ] **Step 3: Write the implementation**

```python
# franka_gaze/franka_gaze/ik.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest franka_gaze/tests/test_ik.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add franka_gaze/franka_gaze/ik.py franka_gaze/tests/test_ik.py
git commit -m "Add IK: yaw/pitch to Franka joint angles via damped-least-squares"
```

---

### Task 4: Control loop — main.py

**Files:**
- Create: `franka_gaze/franka_gaze/main.py`

**Interfaces:**
- Consumes: `social_app.perception.WebcamFaceTracker` (`.start()`, `.latest() -> FaceObservation | None`, `.stop()`), `social_app.gaze.GazeController`/`GazeConfig` (`.update(observation, now) -> tuple[float, float]`), `franka_gaze.ik.{HOME_QPOS, compute_home_pose, yaw_pitch_to_target, solve}` (Task 3), `franka_gaze.models.franka_emika_panda.scene.xml` path (Task 1).
- Produces: an executable `python -m franka_gaze.main` entry point — no other task consumes this one's output, it's the top of the pipeline.

- [ ] **Step 1: Write main.py**

```python
# franka_gaze/franka_gaze/main.py
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
```

- [ ] **Step 2: Smoke-test that the script starts without crashing**

Run: `.venv\Scripts\python.exe -c "from franka_gaze.main import main; print('import ok')"`
Expected output: `import ok` (this only checks imports resolve — the MuJoCo
viewer window itself needs a human to run and look at, covered in Task 5).

- [ ] **Step 3: Commit**

```bash
git add franka_gaze/franka_gaze/main.py
git commit -m "Add franka_gaze control loop"
```

---

### Task 5: Manual sim verification and status update

**Files:**
- Modify: `franka_gaze/plan.md` (Status section)

**Interfaces:**
- Consumes: the complete pipeline from Tasks 1-4.
- Produces: nothing further tasks depend on — this is the terminal task.

- [ ] **Step 1: Run the full system**

Run: `.venv\Scripts\python.exe -m franka_gaze.main`

A MuJoCo viewer window should open showing the Franka Panda arm. Sit in
front of the laptop webcam and move your head/position; observe whether
the arm's end-effector visibly orients toward you, smoothly, without
jitter or wild swings — the same bar `apps/social_app`'s phase 1 was held
to (`apps/social_app/plan.md`'s "visually confirmed working").

- [ ] **Step 2: Tune sign/gain if needed**

If the arm turns the wrong way, flip `YAW_SIGN`/`PITCH_SIGN` in
`franka_gaze/franka_gaze/main.py` (same empirical-tuning step
`apps/social_app/social_app/gaze.py`'s own `yaw_sign`/`pitch_sign` needed
— not analytically solvable, per the design spec). If motion is too
jittery or too sluggish, adjust `GazeConfig(smoothing_alpha=...)` passed
into `GazeController` in `main.py`.

- [ ] **Step 3: Update plan.md status**

Edit `franka_gaze/plan.md`'s `## Status` section to record the outcome:
whether tracking was visually confirmed working, any sign/gain values that
needed tuning, and any observed issues (e.g. IK oscillation near workspace
edges, jitter at particular angles) for future reference.

- [ ] **Step 4: Commit**

```bash
git add franka_gaze/plan.md
git commit -m "Record manual verification results in franka_gaze/plan.md"
```

## Self-Review Notes

- **Spec coverage:** all of the design spec's components (`models/`, `ik.py`, `main.py`, `pyproject.toml`, `plan.md`) map to a task above (Tasks 1, 3, 4, 2, 2/5 respectively). The spec's two testing tiers (automated `ik.py` unit tests, manual whole-system check) map to Task 3 and Task 5.
- **Placeholder scan:** no TBD/TODO; every code step has real, complete code. Task 1's model-fetch step depends on an external clone rather than inline code because vendoring binary mesh/texture assets isn't expressible as a plan code block — this is the one step that isn't literal copy-paste-able code, flagged explicitly rather than hidden.
- **Type consistency:** `ik.solve`'s signature (`model, data, qpos_init, target_pos, target_mat, max_iters, tol, damping`) is used identically in Task 3's own tests and in Task 4's `main.py`. `yaw_pitch_to_target`'s `(home_pos, home_mat, yaw_deg, pitch_deg) -> (target_pos, target_mat)` signature likewise matches between Task 3 and Task 4. `HOME_QPOS` is defined once in `ik.py` (Task 3) and only ever imported, never redefined, in Task 4.
- Body/joint/actuator names (`hand`, `joint1..7`, `actuator1..7`) are used consistently across Tasks 1, 3, and 4 — all sourced from the same direct-fetch verification recorded in Global Constraints, not re-guessed per task.
