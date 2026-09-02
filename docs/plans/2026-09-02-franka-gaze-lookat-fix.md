# Franka Gaze Look-At Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use godmode:task-runner to implement this plan task-by-task.

**Goal:** Replace `franka_gaze/franka_gaze/ik.py`'s broken Euler-composition target math (yaw currently just rolls the gripper's own wrist axis, a no-op for pointing) with a real look-at/aim construction, so the Franka end-effector genuinely orients toward — and leans slightly toward — the tracked face.

**Architecture:** `yaw_pitch_to_target(home_pos, home_mat, yaw_deg, pitch_deg) -> (target_pos, target_mat)` keeps its exact signature (no changes needed in `main.py` or `solve()`), but its internals change completely: instead of rotating `home_mat` about its own local axes, it now computes a 3D "virtual person" point in front of the robot's base (a fixed reference point, panned/tilted by yaw/pitch), builds a genuine look-at rotation matrix aiming the hand's approach axis at that point, and offsets the target position by a small fixed lean (10cm) toward the look direction.

**Tech Stack:** Python 3.12, `mujoco`, `numpy`, `pytest` — same as the existing `franka_gaze` package, no new dependencies.

---

### Task 1: Look-at math in ik.py, with TDD

**Files:**
- Modify: `franka_gaze/franka_gaze/ik.py:51-81` (replace `_rotz`/`_rotx`/`yaw_pitch_to_target`, add new constants and helpers)
- Modify: `franka_gaze/tests/test_ik.py:35-39` (rewrite the now-invalid `test_yaw_pitch_zero_returns_home_unchanged`), add new tests

**Step 1: Write the failing tests**

Replace `franka_gaze/tests/test_ik.py:35-39`'s `test_yaw_pitch_zero_returns_home_unchanged` and add three new tests, so the file's imports/fixtures section (lines 1-25) stays as-is and only the test functions below are added/changed:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest franka_gaze/tests/test_ik.py -v`
Expected: the 4 new/changed tests FAIL — `test_yaw_pitch_zero_points_at_reference` and the others will fail with `ImportError: cannot import name 'MAX_LEAN_M'` or `AttributeError`/assertion failures against the old implementation, since `MAX_LEAN_M`/`REFERENCE_DIRECTION` don't exist yet. The 5 pre-existing tests (`test_compute_home_pose_returns_finite_values`, `test_solve_converges_for_home_target`, `test_solve_respects_joint_limits`, `test_solve_no_nans`) should still PASS unchanged — they don't depend on `yaw_pitch_to_target`'s specific geometry, only that it returns a valid position/orientation pair.

**Step 3: Write the implementation**

Replace `franka_gaze/franka_gaze/ik.py` lines 51-81 (the `_rotz`, `_rotx`, and `yaw_pitch_to_target` functions) with:

```python
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
    right = right / np.linalg.norm(right)
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
```

**Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest franka_gaze/tests/test_ik.py -v`
Expected: all 12 tests PASS (5 pre-existing + 4 new/changed + the 3 `solve()` tests that were already passing remain passing since `solve()` itself is untouched).

**Step 5: Commit**

```bash
git add franka_gaze/franka_gaze/ik.py franka_gaze/tests/test_ik.py
git commit -m "franka_gaze: replace Euler-composition target with real look-at aim + lean-in"
```

---

### Task 2: Manual sim verification

**Files:** none (verification only)

**Step 1: Run the full system**

Run: `.venv\Scripts\python.exe -m franka_gaze.main`

A MuJoCo viewer window opens with the Franka Panda arm. Sit in front of the
webcam and move — the end-effector should now visibly reorient to point
toward you (not just roll its wrist), with a small lean-in, as you move
left/right/up/down.

**Step 2: Tune REFERENCE_DIRECTION if the arm points the wrong way**

If the arm's aim direction doesn't correspond sensibly to your position
(e.g. it points away from you, or up into the air instead of toward you),
edit `REFERENCE_DIRECTION` in `franka_gaze/franka_gaze/ik.py` (try
`[-1, 0, 0]`, `[0, 1, 0]`, or `[0, -1, 0]` — this is the same class of
empirical sign-tuning `apps/social_app/social_app/gaze.py`'s own
`yaw_sign`/`pitch_sign` needed) and re-run.

**Step 3: Record the result**

Update `franka_gaze/plan.md`'s `## Status` section with the outcome:
whether the look-at now works, which `REFERENCE_DIRECTION` value was
correct, and any remaining issues.

**Step 4: Commit**

```bash
git add franka_gaze/plan.md
git commit -m "franka_gaze: record look-at fix verification results"
```
