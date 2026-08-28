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

Locomotion comes from the PICO port
-----------------------------------
Stick handling lives in :class:`SonicPicoLocomotionRetargeter`, a port of the planner loop the
real robot runs -- the same ``pico_manager_thread_server.py`` block drives the robot whether its
input arrives from XRoboToolkit or from Isaac Teleop (``get_controller_axes`` branches on
``IsaacTeleopReader``, :640), so the sticks feel the same in both places.

This node adds only what is specific to *this* environment -- the encoder mode and the floor that
indicates it -- and passes the locomotion block through untouched.

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

from gear_sonic.lab_teleop.retargeters.sonic_pico_locomotion_retargeter import (
    PICO_LOCOMOTION_DIM,
    SonicPicoLocomotionRetargeter,
)

PICO_OUTPUT = SonicPicoLocomotionRetargeter.OUTPUT_NAME

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

#: ``[mode, target_vel, movement(3), facing(3), height, ground_visible, turn_rate, moving]``.
#: Slots 1..8 are :class:`SonicPicoLocomotionRetargeter`'s block, passed through untouched.
SONIC_COMMAND_DIM = 12


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
        """The PICO locomotion block and the controller that toggles the mode."""
        controllers = {f"controller_{self._toggle_side}": OptionalType(ControllerInput())}
        return {
            **controllers,
            PICO_OUTPUT: TensorGroupType(
                PICO_OUTPUT,
                [
                    NDArrayType(
                        "command",
                        shape=(PICO_LOCOMOTION_DIM,),
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

        locomotion = np.asarray(
            np.from_dlpack(inputs[PICO_OUTPUT][0]), dtype=np.float32
        ).reshape(PICO_LOCOMOTION_DIM)

        self._out[:] = 0.0
        self._out[0] = float(self._mode)
        # target_vel, movement, facing and height, exactly as the PICO loop produced them.
        self._out[1:9] = locomotion[0:8]
        # The floor is the mode indicator: shown while walking, hidden while tracking.
        self._out[9] = 1.0 if self._mode == SONIC_ENCODER_MODE_TELEOP else 0.0
        # Raw turn, for the action term to swing the XR anchor in smpl mode where there is no
        # planner to face anywhere, and the moving flag it uses to pick the idle clip.
        self._out[10] = locomotion[8]
        self._out[11] = locomotion[9]
        outputs[self.OUTPUT_NAME][0] = self._out
