# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Operator mode selection and locomotion command, for SONIC's ``teleop`` encoder mode.

Sits in series after the stock
:class:`~isaacteleop.retargeters.LocomotionRootCmdRetargeter`, adapting its generic root command
into the form ``planner_sonic.onnx`` consumes and adding the operator's mode selection::

    ControllersSource -> LocomotionRootCmdRetargeter -> SonicTeleopCommandRetargeter -> action
                         [vel_x, vel_y, rot_vel_z,      [mode, target_vel, movement_dir(3),
                          hip_height]                    facing_dir(3), height,
                                                         ground_visible]

Controls: the **left** primary click toggles the encoder mode. A click rather than trigger or
squeeze because those are already claimed by the tri-hand retargeters, and the left hand because
the right one drives the session (start/stop and reset).

Ground-plane visibility is **derived from the mode**, not toggled separately: the floor is shown in
``teleop`` mode and hidden in ``smpl`` mode. That makes the floor the operator's mode indicator --
visible in a headset without looking away from the robot -- and removes a control that only ever
existed to work around not knowing which mode you were in.

Everything here is a pure function of **operator input**, which is why it belongs on the Isaac
Teleop side of the boundary. The planner it feeds does not: that is closed-loop on the robot's
measured pose and runs in the action term. See
:mod:`gear_sonic.lab_teleop.mdp.sonic_planner`.

Heading follows the deployed gamepad
------------------------------------
``LocomotionRootCmdRetargeter`` supplies ``vel_x``, ``vel_y``, ``rot_vel_z`` and ``hip_height``,
and all four are used, mirroring the robot's own gamepad path
(``gamepad_manager.hpp:751-763``)::

    facing_angle    -= 0.02 * right_stick_x       # integrated, per tick
    moving_direction = bin45(left_stick_angle) + facing_angle

so the right stick turns and the left stick drives *relative to where the robot faces*.

An earlier version of this node discarded ``rot_vel_z``, arguing that integrating a turn rate
would drift with nothing to correct against. That was wrong: drift is only meaningful against a
ground truth, and a commanded heading has none -- it **is** the command, and the operator closes
the loop by looking at the robot.

Two sign details, both easy to get backwards. ``rot_vel_z = -right_stick_x``
(``locomotion_retargeter.py:167``) while upstream *subtracts* ``right_stick_x``, so the negations
cancel and this node adds. And upstream's extra ``- pi/2`` on the movement angle is **not**
reproduced: it converts their raw ``atan2(ly, lx)``, whereas ``vel_x = +left_stick_y`` already
puts forward at zero here.

Idle handling crosses the boundary
----------------------------------
With the stick centred, the reference implementation derives both directions from the robot's
*measured* velocity and root yaw rather than holding the last command
(``controllers.py:209-218``). That needs robot state, so it cannot happen here. This node emits
``target_vel = 0`` with zero direction vectors, and the action term substitutes the idle
directions. Zero is unambiguous: a real command always carries a unit direction.

.. note::
   ``facing_direction`` is derived from the operator's root orientation as delivered by the
   retargeting graph, which is expressed in the XR anchor frame. With a world-fixed anchor that is
   a fixed rotation away from world, so a commanded heading is offset by the anchor yaw. Resolving
   that is deferred with the rest of the anchor work.
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
    "SONIC_COMMAND_DIM",
    "SONIC_ENCODER_MODE_SMPL",
    "SONIC_ENCODER_MODE_TELEOP",
    "SonicTeleopCommandRetargeter",
]

#: SONIC encoder modes this environment switches between. Values match the checkpoint's
#: ``observation_config.yaml`` ``encoder_modes`` block.
SONIC_ENCODER_MODE_TELEOP = 1
SONIC_ENCODER_MODE_SMPL = 2

#: ``[mode, target_vel, movement_dir(3), facing_dir(3), height, ground_visible, turn_rate]``.
SONIC_COMMAND_DIM = 11

#: Rate the right stick sweeps the commanded heading, rad/s. Upstream applies ``0.02`` rad per
#: tick on its 50 Hz loop (``gamepad_manager.hpp:753``); expressed as a duration here so the feel
#: survives a change of control rate.
TURN_RATE_RAD_S = 0.02 * 50.0

#: Movement direction is quantized to eight 45 degree sectors, as upstream does
#: (``gamepad_manager.hpp:760-763``), so the planner is driven toward cardinal and diagonal
#: headings rather than an arbitrary stick angle.
MOVEMENT_BIN_RAD = math.pi / 4.0


class SonicTeleopCommandRetargeter(BaseRetargeter):
    """Emit the operator's encoder mode and locomotion command.

    Args:
        name: Node name within the retargeting graph.
        default_mode: Mode selected at start and restored on reset.
        toggle_side: Controller whose primary button toggles the mode. Trigger and squeeze are
            already claimed by the tri-hand retargeters, so clicks are used.

    Raises:
        ValueError: If ``toggle_side`` is not ``"left"`` or ``"right"``.
    """

    OUTPUT_NAME = "sonic_command"

    def __init__(
        self,
        name: str = "sonic_command",
        default_mode: int = SONIC_ENCODER_MODE_SMPL,
        toggle_side: str = "left",
        dt: float = 0.02,
        turn_rate: float = TURN_RATE_RAD_S,
    ) -> None:
        if toggle_side not in ("left", "right"):
            raise ValueError(f"toggle_side must be 'left' or 'right', got {toggle_side!r}")
        # Set before super().__init__: BaseRetargeter calls input_spec() during construction.
        self._toggle_side = toggle_side
        self._default_mode = int(default_mode)
        self._mode = int(default_mode)
        self._toggle_was_down = False
        self._dt = float(dt)
        self._turn_rate = float(turn_rate)
        #: Commanded heading in radians, integrated from the right stick. Zeroed on reset so an
        #: episode always starts pointing the way the operator does.
        self._facing_angle = 0.0
        self._out = np.zeros(SONIC_COMMAND_DIM, dtype=np.float32)
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        """The root command and the controller that toggles the mode."""
        controllers = {f"controller_{self._toggle_side}": OptionalType(ControllerInput())}
        return {
            **controllers,
            "root_command": TensorGroupType(
                "root_command",
                [NDArrayType("command", shape=(4,), dtype=DLDataType.FLOAT, dtype_bits=32)],
            ),
        }

    def output_spec(self) -> RetargeterIOType:
        """Emit ``[mode, target_vel, movement_dir(3), facing_dir(3), height]``."""
        return {
            self.OUTPUT_NAME: TensorGroupType(
                self.OUTPUT_NAME,
                [
                    NDArrayType(
                        "command",
                        shape=(SONIC_COMMAND_DIM,),
                        dtype=DLDataType.FLOAT,
                        dtype_bits=32,
                    )
                ],
            )
        }

    def _compute_fn(
        self, inputs: RetargeterIO, outputs: RetargeterIO, context
    ) -> None:  # noqa: ANN001
        """Latch the mode toggle and build the locomotion command."""
        if context.execution_events.reset:
            self._mode = self._default_mode
            self._toggle_was_down = False
            self._facing_angle = 0.0

        controller = inputs[f"controller_{self._toggle_side}"]
        if not controller.is_none:
            down = bool(controller[ControllerInputIndex.PRIMARY_CLICK])
            # Rising edge only: holding the button must not free-run the toggle.
            if down and not self._toggle_was_down:
                self._mode = (
                    SONIC_ENCODER_MODE_TELEOP
                    if self._mode == SONIC_ENCODER_MODE_SMPL
                    else SONIC_ENCODER_MODE_SMPL
                )
            self._toggle_was_down = down

        root_command = np.asarray(
            np.from_dlpack(inputs["root_command"][0]), dtype=np.float32
        ).reshape(4)
        vel_x, vel_y, rot_vel_z, hip_height = (float(v) for v in root_command)
        speed = math.hypot(vel_x, vel_y)
        self._facing_angle += self._turn_rate * rot_vel_z * self._dt

        self._out[:] = 0.0
        self._out[0] = float(self._mode)
        self._out[1] = speed
        self._out[5] = math.cos(self._facing_angle)
        self._out[6] = math.sin(self._facing_angle)
        if speed > 1e-4:
            # Quantize the stick angle, then express it relative to where the robot faces.
            stick = math.atan2(vel_y, vel_x)
            binned = round(stick / MOVEMENT_BIN_RAD) * MOVEMENT_BIN_RAD
            heading = binned + self._facing_angle
            self._out[2] = math.cos(heading)
            self._out[3] = math.sin(heading)
        # else: leave the movement vector zero -- the action term substitutes an idle direction
        # derived from measured robot state, which is not available here. Facing is still sent: it
        # is a commanded heading and holds while the operator stands still.
        self._out[8] = hip_height
        # The floor is the mode indicator: shown while walking, hidden while tracking.
        self._out[9] = 1.0 if self._mode == SONIC_ENCODER_MODE_TELEOP else 0.0
        # Raw turn command, for the action term to swing the XR anchor in smpl mode, where there
        # is no planner to face anywhere.
        self._out[10] = rot_vel_z
        outputs[self.OUTPUT_NAME][0] = self._out
