# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stick handling must match the robot's, so the two feel the same.

The reference is ``pico_manager_thread_server.py:1795-1876`` -- the planner loop the real robot
runs. It is reached from Isaac Teleop as well as from XRoboToolkit (``get_controller_axes``
branches on ``IsaacTeleopReader``, :640), so it is the shared definition of "how the sticks
behave", not merely one of two options.

Deliberately *not* the gamepad path in ``g1_deploy_onnx_ref``: that snaps movement to eight
45-degree sectors and turns at 1.0 rad/s, where PICO is continuous at 1.5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gear_sonic.lab_teleop.retargeters.sonic_pico_locomotion_retargeter import (
    PICO_JOYSTICK_DEADZONE,
    PICO_LOCOMOTION_DIM,
    PICO_PLANNER_DEFAULT,
    PICO_YAW_GAIN,
    SonicPicoLocomotionRetargeter,
)


class _Stick:
    """Minimal controller stand-in exposing thumbstick axes by index."""

    is_none = False

    def __init__(self, x: float, y: float) -> None:
        self._axes = {2: float(x), 3: float(y)}

    def __getitem__(self, key):  # noqa: ANN001, ANN204
        from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex

        if int(key) == int(ControllerInputIndex.THUMBSTICK_X):
            return self._axes[2]
        if int(key) == int(ControllerInputIndex.THUMBSTICK_Y):
            return self._axes[3]
        return 0.0


def _step(node, lx=0.0, ly=0.0, rx=0.0):  # noqa: ANN001
    """Run one frame and return the emitted block."""

    class _Events:
        reset = False

    class _Ctx:
        execution_events = _Events()

    out = {node.OUTPUT_NAME: [np.zeros(PICO_LOCOMOTION_DIM, dtype=np.float32)]}
    node._compute_fn(  # noqa: SLF001
        {"controller_left": _Stick(lx, ly), "controller_right": _Stick(rx, 0.0)}, out, _Ctx()
    )
    return np.asarray(out[node.OUTPUT_NAME][0], dtype=np.float32)


def test_constants_match_upstream() -> None:
    """The three numbers that set the feel."""
    assert PICO_YAW_GAIN == pytest.approx(1.5)  # YawAccumulator(yaw_gain=1.5)
    assert PICO_JOYSTICK_DEADZONE == pytest.approx(0.15)  # JOYSTICK_DEADZONE
    assert PICO_PLANNER_DEFAULT == pytest.approx(-1.0)  # speed/height "use the default"


def test_heading_sweeps_at_the_upstream_rate_and_sign() -> None:
    """``dyaw = 1.5 * (-rx) * dt``: a second of full right deflection turns 1.5 rad clockwise."""
    node = SonicPicoLocomotionRetargeter(dt=0.02)
    for _ in range(50):
        _step(node, rx=1.0)
    assert node._yaw == pytest.approx(-PICO_YAW_GAIN, abs=1e-6)  # noqa: SLF001
    assert math.degrees(node._yaw) == pytest.approx(-85.9, abs=0.5)  # noqa: SLF001


def test_turn_respects_the_deadzone() -> None:
    """A stick resting off-centre must not creep the heading around."""
    node = SonicPicoLocomotionRetargeter(dt=0.02)
    for _ in range(100):
        _step(node, rx=0.14)
    assert node._yaw == pytest.approx(0.0)  # noqa: SLF001


def test_movement_magnitude_is_deadzoned_then_rescaled() -> None:
    """``mag = (raw - 0.15) / 0.85`` -- motion ramps from zero rather than jumping to 0.15."""
    node = SonicPicoLocomotionRetargeter(dt=0.02)

    assert np.linalg.norm(_step(node, ly=0.10)[1:4]) == pytest.approx(0.0)
    assert _step(node, ly=0.10)[9] == pytest.approx(0.0)  # not moving

    out = _step(node, ly=0.15 + 0.85 / 2)  # halfway through the usable range
    assert np.linalg.norm(out[1:4]) == pytest.approx(0.5, abs=1e-4)
    assert out[9] == pytest.approx(1.0)

    assert np.linalg.norm(_step(node, ly=1.0)[1:4]) == pytest.approx(1.0, abs=1e-4)


def test_speed_and_height_ride_as_sentinels() -> None:
    """Throttle is the movement magnitude; target_vel and height say "use the default"."""
    node = SonicPicoLocomotionRetargeter(dt=0.02)
    out = _step(node, ly=1.0)
    assert out[0] == pytest.approx(PICO_PLANNER_DEFAULT)
    assert out[7] == pytest.approx(PICO_PLANNER_DEFAULT)


def test_movement_is_relative_to_facing_and_not_binned() -> None:
    """Forward is along the heading, and a sub-sector angle stays where it is."""
    node = SonicPicoLocomotionRetargeter(dt=0.02)
    node._yaw = math.radians(90.0)  # noqa: SLF001

    out = _step(node, ly=1.0)  # stick forward
    assert math.degrees(math.atan2(out[2], out[1])) == pytest.approx(90.0, abs=0.5)

    # 20 degrees off forward stays 20 degrees off, where the gamepad would snap it to 0 or 45.
    node._yaw = 0.0  # noqa: SLF001
    out = _step(node, lx=math.sin(math.radians(20.0)), ly=math.cos(math.radians(20.0)))
    assert math.degrees(math.atan2(out[2], out[1])) == pytest.approx(-20.0, abs=1.0)


def test_reset_returns_the_heading_to_zero() -> None:
    """``YawAccumulator.reset`` puts the heading back to (1, 0, 0)."""
    node = SonicPicoLocomotionRetargeter(dt=0.02)
    for _ in range(25):
        _step(node, rx=1.0)
    assert node._yaw != pytest.approx(0.0)  # noqa: SLF001

    class _Events:
        reset = True

    class _Ctx:
        execution_events = _Events()

    out = {node.OUTPUT_NAME: [np.zeros(PICO_LOCOMOTION_DIM, dtype=np.float32)]}
    node._compute_fn(  # noqa: SLF001
        {"controller_left": _Stick(0, 0), "controller_right": _Stick(0, 0)}, out, _Ctx()
    )
    assert node._yaw == pytest.approx(0.0)  # noqa: SLF001
    assert np.asarray(out[node.OUTPUT_NAME][0])[4:7] == pytest.approx([1.0, 0.0, 0.0])


def test_a_commanded_turn_reaches_the_planner_while_standing_still() -> None:
    """Turning on the spot must replan, and `_last_command` must not alias the action buffer.

    Two separate faults made the right stick do nothing in walking mode. The first substituted the
    robot's *measured* facing whenever the stick was centred, discarding the command. The second
    is what this test pins: `_last_command` held a view into `actions`, which is the
    ActionManager's reused buffer, so it tracked the live command and the change detection could
    never fire. Storing a copy is the fix, and the aliasing form passes every shape and dtype
    check while being silently inert.
    """
    import numpy as np
    import torch

    from gear_sonic.lab_teleop.assets.g1_sonic import G1_MODEL_12_ACTION_SCALE
    from gear_sonic.lab_teleop.mdp.modal_actions import (
        SONIC_MODAL_ACTION_DIM,
        SonicModalWholeBodyAction,
        SonicModalWholeBodyActionCfg,
    )
    from gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter import (
        SONIC_REFERENCE_DIM,
        SonicReferenceSlice,
    )
    from gear_sonic.lab_teleop.tests.sonic_action_harness import FakeArticulation, FakeEnv

    asset = FakeArticulation(num_envs=1, device="cpu")
    env = FakeEnv(asset, num_envs=1, device="cpu")
    env.step_dt = 0.02
    term = SonicModalWholeBodyAction(
        SonicModalWholeBodyActionCfg(
            asset_name="robot",
            checkpoint_dir="gear_sonic_deploy/policy/low_latency",
            joint_names=[".*"],
            action_scale=G1_MODEL_12_ACTION_SCALE,
        ),
        env,
    )
    term.reset()

    planner = term._ensure_planner()  # noqa: SLF001
    real_plan, headings = planner.plan, []

    def recording(command, mode=None):  # noqa: ANN001, ANN202
        headings.append(float(np.degrees(np.arctan2(command[5], command[4]))))
        return real_plan(command, mode)

    planner.plan = recording

    reference = np.zeros(SONIC_REFERENCE_DIM, dtype=np.float32)
    reference[SonicReferenceSlice.ROOT_QUAT.start] = 1.0
    reference[SonicReferenceSlice.VR3_ORN][0::4] = 1.0
    reference[SonicReferenceSlice.VALID] = 1.0
    # One reused action buffer, exactly as the ActionManager supplies.
    action = torch.zeros(1, SONIC_MODAL_ACTION_DIM)
    action[0, :SONIC_REFERENCE_DIM] = torch.from_numpy(reference)
    base = SONIC_REFERENCE_DIM + 1

    def step(facing_deg: float) -> None:
        action[0, SONIC_REFERENCE_DIM] = 1.0  # teleop
        action[0, base + 0] = -1.0  # target_vel sentinel
        action[0, base + 1] = 0.0  # centred stick: no movement
        theta = np.radians(facing_deg)
        action[0, base + 4] = np.cos(theta)
        action[0, base + 5] = np.sin(theta)
        action[0, base + 7] = -1.0
        action[0, base + 10] = 0.0  # not moving
        term.process_actions(action)
        term.apply_actions()

    step(0.0)  # entry: the idle plan
    for degrees in (20.0, 45.0, 70.0, 90.0):
        step(degrees)

    # Each distinct heading reached the planner, in order, despite the stick being centred.
    assert headings[1:] == pytest.approx([20.0, 45.0, 70.0, 90.0], abs=0.5)

    # And a repeated heading does not replan: the comparison still works in both directions.
    before = len(headings)
    step(90.0)
    step(90.0)
    assert len(headings) == before


@pytest.mark.parametrize("tracked", [(0.0, 0.9, 0.0), (1.5, 0.9, 0.0), (-2.5, 0.9, 1.5)])
@pytest.mark.parametrize("operator_yaw_deg", [0.0, 90.0, 180.0, -135.0])
def test_entering_teleop_puts_the_operator_in_the_robot(tracked, operator_yaw_deg) -> None:  # noqa: ANN001
    """The operator must land *in* the robot, facing the way it faces.

    Both halves are checked by reproducing the runtime's own composition,
    ``R_anchor @ R_oxr_to_usd @ pose_oxr`` (``xr_anchor_manager.py:_build_matrix``), rather than
    re-using this module's conventions -- a test that does the latter passes for any transform.

    The heading half is not incidental. Ignoring the operator's own facing leaves them mirrored
    through the robot: standing across from it, looking at it, rather than standing in it. The
    180 degree case reproduces exactly that.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as sRot
    import torch

    from gear_sonic.lab_teleop.assets.g1_sonic import G1_MODEL_12_ACTION_SCALE
    from gear_sonic.lab_teleop.mdp.modal_actions import (
        OXR_TO_USD,
        SONIC_MODAL_ACTION_DIM,
        SonicModalWholeBodyAction,
        SonicModalWholeBodyActionCfg,
    )
    from gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter import (
        SONIC_REFERENCE_DIM,
        SonicReferenceSlice,
    )
    from gear_sonic.lab_teleop.tests.sonic_action_harness import FakeArticulation, FakeEnv

    class _Xr:
        anchor_pos = (0.0, 0.0, -0.19)
        anchor_rot = (0.0, 0.0, 0.0, 1.0)

    class _TeleopCfg:
        xr_cfg = _Xr()

    asset = FakeArticulation(num_envs=1, device="cpu")
    env = FakeEnv(asset, num_envs=1, device="cpu")
    env.step_dt = 0.02
    env.cfg.isaac_teleop = _TeleopCfg()
    term = SonicModalWholeBodyAction(
        SonicModalWholeBodyActionCfg(
            asset_name="robot",
            checkpoint_dir="gear_sonic_deploy/policy/low_latency",
            joint_names=[".*"],
            action_scale=G1_MODEL_12_ACTION_SCALE,
        ),
        env,
    )
    term.reset()

    robot_xy = (3.0, -1.0)
    asset.data.root_pos_w.torch[0, 0] = robot_xy[0]
    asset.data.root_pos_w.torch[0, 1] = robot_xy[1]
    # Robot facing along +X (identity), matching the harness's XYZW identity root quaternion.

    # The operator's tracked pelvis: OpenXR axes, so yaw is a rotation about that frame's Y.
    operator_rot = sRot.from_euler("y", operator_yaw_deg, degrees=True)

    reference = np.zeros(SONIC_REFERENCE_DIM, dtype=np.float32)
    reference[SonicReferenceSlice.ROOT_QUAT.start] = 1.0
    reference[SonicReferenceSlice.VR3_ORN][0::4] = 1.0
    reference[SonicReferenceSlice.VALID] = 1.0
    reference[SonicReferenceSlice.OPERATOR_ROOT_POS] = tracked
    reference[SonicReferenceSlice.OPERATOR_ROOT_QUAT] = operator_rot.as_quat()  # xyzw

    action = torch.zeros(1, SONIC_MODAL_ACTION_DIM)
    action[0, :SONIC_REFERENCE_DIM] = torch.from_numpy(reference)
    base = SONIC_REFERENCE_DIM + 1
    action[0, base + 0] = -1.0
    action[0, base + 4] = 1.0
    action[0, base + 7] = -1.0

    action[0, SONIC_REFERENCE_DIM] = 2.0  # tracking: the operator moves away from the anchor
    for _ in range(3):
        term.process_actions(action)
        term.apply_actions()

    action[0, SONIC_REFERENCE_DIM] = 1.0  # enter walking mode
    term.process_actions(action)
    term.apply_actions()

    anchor_pos = np.asarray(term._xr_cfg.anchor_pos, dtype=np.float64)  # noqa: SLF001
    r_anchor = sRot.from_quat(np.asarray(term._xr_cfg.anchor_rot)).as_matrix()  # noqa: SLF001

    # Position: the operator is standing in the robot.
    operator_world = r_anchor @ OXR_TO_USD @ np.asarray(tracked) + anchor_pos
    assert operator_world[:2] == pytest.approx(robot_xy, abs=1e-4)

    # Heading: and facing the way it faces, not across from it.
    facing = r_anchor @ OXR_TO_USD @ operator_rot.as_matrix()
    assert float(np.arctan2(facing[1, 0], facing[0, 0])) == pytest.approx(0.0, abs=1e-4)
