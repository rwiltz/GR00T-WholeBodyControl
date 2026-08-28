# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Joystick locomotion with the semantics of the shipped PICO teleop server.

A port of the planner-loop block in ``gear_sonic/scripts/pico_manager_thread_server.py:1795-1876``
-- the code that drives the real robot from a PICO headset -- rather than of the gamepad path in
``g1_deploy_onnx_ref``. The two differ in ways an operator feels immediately: the gamepad snaps
movement to eight 45 degree sectors and turns at 1.0 rad/s, while PICO is continuous and turns at
1.5.

What upstream does, line for line
---------------------------------
Facing, from the right stick (``YawAccumulator``, :519)::

    dyaw = 1.5 * (-rx) * dt
    if abs(rx) >= 0.15:  yaw += dyaw
    facing = [cos(yaw), sin(yaw), 0]

Movement, from the left stick (:1802-1830)::

    raw = clip(hypot(lx, ly), 0, 1)
    if raw < 0.15:  mag, speed = 0, IDLE
    else:           mag = (raw - 0.15) / (1 - 0.15)
    local  = [-lx, ly] * (mag / raw)
    global = [[-fy, fx], [fx, fy]] @ local

so the vector carries the **throttle in its magnitude** while ``target_vel`` and ``height`` are
sent as ``-1.0``.

Why the sentinels are safe here
-------------------------------
``-1.0`` is not a velocity; it means "use the clip's default". Measured against
``planner_sonic.onnx`` rather than assumed, because the sentinel crossing into a different planner
wrapper is exactly the kind of thing that silently misbehaves::

    target_vel  |movement|   steady speed
       0.8         1.00        0.928 m/s
       0.8         0.50        0.382         speed tracks target_vel * |movement|
       0.4         1.00        0.382
      -1.0         1.00        1.129         sentinel behaves as full speed
      -1.0         0.50        0.627
    height = -1.0  ->  root z 0.755 m, identical to height = 0.78

The deadzone is this node's job: ``LocomotionRootCmdRetargeter`` applies none, so without it the
robot creeps whenever a stick rests off-centre.
"""

from __future__ import annotations

import math

from isaacteleop.retargeting_engine.interface import BaseRetargeter, RetargeterIOType
from isaacteleop.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaacteleop.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaacteleop.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
    DLDataType,
    NDArrayType,
)
import numpy as np

__all__ = [
    "PICO_JOYSTICK_DEADZONE",
    "PICO_LOCOMOTION_DIM",
    "PICO_PLANNER_DEFAULT",
    "PICO_YAW_GAIN",
    "SonicPicoLocomotionRetargeter",
]

#: ``JOYSTICK_DEADZONE`` (``pico_manager_thread_server.py:517``). Applied to the turn axis as a
#: gate and to the movement magnitude as a gate *plus* a rescale, so motion starts from zero
#: instead of jumping to 0.15.
PICO_JOYSTICK_DEADZONE = 0.15

#: ``YawAccumulator(yaw_gain=1.5)``, radians per second (:523).
PICO_YAW_GAIN = 1.5

#: The "use the clip's default" sentinel upstream sends for both speed and height (:1876).
PICO_PLANNER_DEFAULT = -1.0

#: ``[target_vel, movement(3), facing(3), height, turn_rate, moving]``.
PICO_LOCOMOTION_DIM = 10


class SonicPicoLocomotionRetargeter(BaseRetargeter):
    """Turn the two thumbsticks into a planner command, exactly as the PICO server does.

    Replaces ``LocomotionRootCmdRetargeter`` rather than sitting after it: that node applies no
    deadzone and maps the sticks into ``[vel_x, vel_y, rot_vel_z, hip_height]``, which is not the
    shape PICO's semantics fit.

    Args:
        name: Node name within the retargeting graph.
        dt: Control period used to integrate the heading, seconds.
        yaw_gain: Heading sweep rate, rad/s.
        deadzone: Stick deadzone, on both the turn axis and the movement magnitude.
    """

    OUTPUT_NAME = "pico_locomotion"

    def __init__(
        self,
        name: str = "pico_locomotion",
        dt: float = 0.02,
        yaw_gain: float = PICO_YAW_GAIN,
        deadzone: float = PICO_JOYSTICK_DEADZONE,
    ) -> None:
        self._dt = float(dt)
        self._yaw_gain = float(yaw_gain)
        self._deadzone = float(deadzone)
        #: Accumulated heading, radians. Upstream resets this to zero and the heading to
        #: ``(1, 0, 0)`` on reset (``YawAccumulator.reset``).
        self._yaw = 0.0
        self._out = np.zeros(PICO_LOCOMOTION_DIM, dtype=np.float32)
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        """Both controllers: left drives movement, right drives heading."""
        return {
            "controller_left": OptionalType(ControllerInput()),
            "controller_right": OptionalType(ControllerInput()),
        }

    def output_spec(self) -> RetargeterIOType:
        """``[target_vel, movement(3), facing(3), height, turn_rate, moving]``."""
        return {
            self.OUTPUT_NAME: TensorGroupType(
                self.OUTPUT_NAME,
                [
                    NDArrayType(
                        "command",
                        shape=(PICO_LOCOMOTION_DIM,),
                        dtype=DLDataType.FLOAT,
                        dtype_bits=32,
                    )
                ],
            )
        }

    @staticmethod
    def _axes(controller) -> tuple[float, float]:  # noqa: ANN001
        """``(x, y)`` for one thumbstick, or zeros when the controller is not tracked."""
        if controller.is_none:
            return 0.0, 0.0
        return (
            float(controller[ControllerInputIndex.THUMBSTICK_X]),
            float(controller[ControllerInputIndex.THUMBSTICK_Y]),
        )

    def _compute_fn(
        self, inputs: RetargeterIO, outputs: RetargeterIO, context
    ) -> None:  # noqa: ANN001
        """Reproduce the upstream planner-loop block for one frame."""
        if context.execution_events.reset:
            self._yaw = 0.0

        lx, ly = self._axes(inputs["controller_left"])
        rx, _ry = self._axes(inputs["controller_right"])

        # Heading: gated by the deadzone, integrated in seconds. The sign is upstream's ``-rx``,
        # so pushing the stick right turns the robot clockwise.
        if abs(rx) >= self._deadzone:
            self._yaw += self._yaw_gain * (-rx) * self._dt
        facing_x, facing_y = math.cos(self._yaw), math.sin(self._yaw)

        # Movement: deadzone the magnitude, then rescale so it ramps from zero.
        raw = min(math.hypot(lx, ly), 1.0)
        moving = raw >= self._deadzone
        mag = (raw - self._deadzone) / (1.0 - self._deadzone) if moving else 0.0
        mag = min(mag, 1.0)
        scale = mag / raw if raw > 0.0 else 0.0

        # Upstream's local frame is (lateral, forward) = (-lx, ly), rotated by the facing basis
        # ``[[-fy, fx], [fx, fy]]``. That is a rotation of (forward, lateral) by the heading:
        # forward maps onto the facing vector, lateral onto its left perpendicular.
        lateral, forward = -lx * scale, ly * scale
        move_x = -facing_y * lateral + facing_x * forward
        move_y = facing_x * lateral + facing_y * forward

        self._out[:] = 0.0
        # Speed and height ride as sentinels; the throttle is the movement vector's magnitude.
        self._out[0] = PICO_PLANNER_DEFAULT
        self._out[1] = move_x
        self._out[2] = move_y
        self._out[4] = facing_x
        self._out[5] = facing_y
        self._out[7] = PICO_PLANNER_DEFAULT
        self._out[8] = rx
        self._out[9] = 1.0 if moving else 0.0
        outputs[self.OUTPUT_NAME][0] = self._out
