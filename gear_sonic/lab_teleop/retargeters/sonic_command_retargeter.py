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

What is deliberately *not* reused
---------------------------------
``LocomotionRootCmdRetargeter`` supplies ``vel_x``, ``vel_y`` and ``hip_height`` directly. Its
``rot_vel_z`` is **not** used: the planner wants an absolute ``facing_direction`` vector, and
integrating a turn rate into a heading would drift with no reference to correct against. The
reference implementation instead takes facing from an absolute source -- the operator's view
direction (``motionbricks/.../demo/controllers.py:196-207``) -- which is what this node does using
the operator's own root yaw.

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

from gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter import (
    SONIC_REFERENCE_DIM,
    SonicReferenceSlice,
)

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

#: ``[mode, target_vel, movement_dir(3), facing_dir(3), height, ground_visible]``.
SONIC_COMMAND_DIM = 10


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
    ) -> None:
        if toggle_side not in ("left", "right"):
            raise ValueError(f"toggle_side must be 'left' or 'right', got {toggle_side!r}")
        # Set before super().__init__: BaseRetargeter calls input_spec() during construction.
        self._toggle_side = toggle_side
        self._default_mode = int(default_mode)
        self._mode = int(default_mode)
        self._toggle_was_down = False
        self._out = np.zeros(SONIC_COMMAND_DIM, dtype=np.float32)
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        """Root command, the toggling controller, and the operator's own reference frame."""
        controllers = {f"controller_{self._toggle_side}": OptionalType(ControllerInput())}
        return {
            **controllers,
            "root_command": TensorGroupType(
                "root_command",
                [NDArrayType("command", shape=(4,), dtype=DLDataType.FLOAT, dtype_bits=32)],
            ),
            "sonic_reference": TensorGroupType(
                "sonic_reference",
                [
                    NDArrayType(
                        "reference",
                        shape=(SONIC_REFERENCE_DIM,),
                        dtype=DLDataType.FLOAT,
                        dtype_bits=32,
                    )
                ],
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

    @staticmethod
    def _operator_yaw(reference: np.ndarray) -> float:
        """Yaw of the operator's root orientation, in the reference's own frame.

        Args:
            reference: ``(95,)`` SONIC reference frame.

        Returns:
            Yaw in radians. Zero when the frame is not marked valid, so an untracked operator
            commands a fixed heading rather than a wildly varying one.
        """
        if reference[SonicReferenceSlice.VALID][0] <= 0.5:
            return 0.0
        w, x, y, z = (float(v) for v in reference[SonicReferenceSlice.ROOT_QUAT])
        # Standard yaw extraction for a wxyz quaternion.
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _compute_fn(
        self, inputs: RetargeterIO, outputs: RetargeterIO, context
    ) -> None:  # noqa: ANN001
        """Latch the mode toggle and build the locomotion command."""
        if context.execution_events.reset:
            self._mode = self._default_mode
            self._toggle_was_down = False

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
        reference = np.asarray(
            np.from_dlpack(inputs["sonic_reference"][0]), dtype=np.float32
        ).reshape(SONIC_REFERENCE_DIM)

        vel_x, vel_y, _rot_vel_z, hip_height = (float(v) for v in root_command)
        speed = math.hypot(vel_x, vel_y)
        yaw = self._operator_yaw(reference)

        self._out[:] = 0.0
        self._out[0] = float(self._mode)
        self._out[1] = speed
        if speed > 1e-4:
            # Stick direction is relative to the operator; rotate it into the shared frame, as
            # ``abs_heading_angle = view_angle + relative_angle`` does upstream.
            heading = yaw + math.atan2(vel_y, vel_x)
            self._out[2] = math.cos(heading)
            self._out[3] = math.sin(heading)
            self._out[5] = math.cos(yaw)
            self._out[6] = math.sin(yaw)
        # else: leave both direction vectors zero -- the action term substitutes idle directions
        # derived from measured robot state, which is not available here.
        self._out[8] = hip_height
        # The floor is the mode indicator: shown while walking, hidden while tracking.
        self._out[9] = 1.0 if self._mode == SONIC_ENCODER_MODE_TELEOP else 0.0
        outputs[self.OUTPUT_NAME][0] = self._out
