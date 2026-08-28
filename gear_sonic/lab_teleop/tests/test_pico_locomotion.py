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
