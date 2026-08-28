# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Teleop retargeting pipeline for SONIC full-body control.

Wires the XR full-body tracker to :class:`SonicFullBodyRetargeter` and exposes the result as the
pipeline's ``"action"`` output, which is what ``IsaacTeleopDevice.advance()`` returns and what the
teleop runner feeds to ``env.step()``.

Pass :func:`make_sonic_full_pipeline_builder` as the ``pipeline_builder`` of an
``isaaclab_teleop.IsaacTeleopCfg``::

    from isaaclab_teleop import IsaacTeleopCfg, XrCfg

    self.isaac_teleop = IsaacTeleopCfg(
        pipeline_builder=make_sonic_full_pipeline_builder(),
        sim_device=self.sim.device,
        xr_cfg=self.xr,
    )

The Isaac Lab process owns the OpenXR/CloudXR session in this arrangement. That is deliberate: it
is what allows the environment to stream a robot-POV camera back to the headset and to manage the
XR anchor. It also means you must **not** simultaneously run
``gear_sonic/scripts/pico_manager_thread_server.py``, which opens its own session to the same
headset via ``CloudXRLauncher``.

``TeleopSession`` auto-discovers trackers and their required OpenXR extensions by walking the
pipeline's source nodes, so adding :class:`FullBodySource` here is sufficient to enable body
tracking; no separate registration step is needed.
"""

from __future__ import annotations

from isaacteleop.retargeting_engine.deviceio_source_nodes import FullBodySource
from isaacteleop.retargeting_engine.interface import OutputCombiner

from gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter import (
    SONIC_REFERENCE_DIM,
    SonicFullBodyRetargeter,
    SonicFullBodyRetargeterConfig,
)

__all__ = [
    "DEFAULT_BODY_TRACKER_VENDOR",
    "make_sonic_fullbody_pipeline_builder",
    "make_sonic_full_pipeline_builder",
]

#: DeviceIO vendor string for PICO body tracking (``XR_BD_body_tracking``).
#: Upstream ``gear_sonic`` reaches the same backend via the ``FullBodyTrackerPico`` subclass.
DEFAULT_BODY_TRACKER_VENDOR = "body.pico-xr"

#: MCAP channel base name that :class:`FullBodySource` records/replays under.
BODY_CHANNEL_NAME = "full_body"

#: Slot names for the 7 joints a tri-hand retargeter emits, in its own output order.
_TRIHAND_SLOT_NAMES = (
    "thumb_rotation",
    "thumb_proximal",
    "thumb_distal",
    "index_proximal",
    "index_distal",
    "middle_proximal",
    "middle_distal",
)


def make_sonic_fullbody_pipeline_builder(vendor: str | None = DEFAULT_BODY_TRACKER_VENDOR):
    """Create a ``pipeline_builder`` callable for ``IsaacTeleopCfg``.

    Args:
        vendor: DeviceIO tracker vendor id, or ``None`` to leave the tracker's default.

            **Must be ``None`` for MCAP replay.** ``TeleopSession`` rejects vendor-carrying
            sources when ``mode`` is ``SessionMode.REPLAY`` (see
            ``teleop_session.py:313-333``): replay reads the recorded channel regardless of
            vendor, so a vendor selection would be silently ignored and it fails fast instead.

    Returns:
        A zero-argument callable returning the pipeline's ``OutputCombiner``.
    """

    def _build() -> OutputCombiner:
        if vendor is None:
            body = FullBodySource(name=BODY_CHANNEL_NAME)
        else:
            from isaacteleop.deviceio import TrackerVendor

            body = FullBodySource(name=BODY_CHANNEL_NAME, vendor=TrackerVendor(vendor))

        retargeter = SonicFullBodyRetargeter(
            SonicFullBodyRetargeterConfig(),
            name="sonic_fullbody",
        )
        connected = retargeter.connect(
            {"full_body": body.output(FullBodySource.FULL_BODY)},
        )
        return OutputCombiner({"action": connected.output("sonic_reference")})

    return _build


def make_sonic_full_pipeline_builder(vendor: str | None = DEFAULT_BODY_TRACKER_VENDOR):
    """The complete SONIC teleoperation graph: tracking, hands, and mode selection.

    One pipeline rather than a variant per capability. Hand grasping and mode switching are things
    an operator wants together, so they are not selectable at task-id level; the only axis that
    changes behaviour is the SONIC checkpoint::

        ControllersSource -+-> TriHandMotionControllerRetargeter (x2) ------------+
                           |                                                      |
                           +-> SonicPicoLocomotion -> SonicTeleopCommand ----------+
                                                                                  |
        FullBodySource -> SonicFullBodyRetargeter ----------------------------+---+-> TensorReorderer

    Controls: **trigger** pinches (index + thumb), **squeeze** grasps (middle + thumb), and the
    left **primary click** toggles between full-body tracking and stick-driven walking.

    Action layout, matching the action-term declaration order in ``SonicActionsCfg``::

        [ sonic_reference(95) | mode(1) | locomotion_command(11) | left_hand(7) | right_hand(7) ]

    == 121. Only operator-derived quantities are computed here; the velocity planner that turns
    the locomotion command into a lower-body reference is closed-loop on the robot and lives in
    the action term.

    Args:
        vendor: Body tracker vendor id, or ``None`` for MCAP replay.

    Returns:
        A zero-argument callable returning the pipeline's ``OutputCombiner``.
    """

    def _build() -> OutputCombiner:
        from isaacteleop.retargeters import (
            TensorReorderer,
            TriHandMotionControllerConfig,
            TriHandMotionControllerRetargeter,
        )
        from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource

        from gear_sonic.lab_teleop.retargeters.sonic_command_retargeter import (
            SONIC_COMMAND_DIM,
            SonicTeleopCommandRetargeter,
        )
        from gear_sonic.lab_teleop.retargeters.sonic_pico_locomotion_retargeter import (
            SonicPicoLocomotionRetargeter,
        )

        if vendor is None:
            body = FullBodySource(name=BODY_CHANNEL_NAME)
        else:
            from isaacteleop.deviceio import TrackerVendor

            body = FullBodySource(name=BODY_CHANNEL_NAME, vendor=TrackerVendor(vendor))

        reference = SonicFullBodyRetargeter(
            SonicFullBodyRetargeterConfig(), name="sonic_fullbody"
        ).connect({"full_body": body.output(FullBodySource.FULL_BODY)})

        controllers = ControllersSource(name="controllers")

        hands = {}
        for side, source_key in (
            ("left", ControllersSource.LEFT),
            ("right", ControllersSource.RIGHT),
        ):
            hands[side] = (
                TriHandMotionControllerRetargeter(
                    TriHandMotionControllerConfig(
                        hand_joint_names=list(_TRIHAND_SLOT_NAMES), controller_side=side
                    ),
                    name=f"trihand_{side}",
                )
                .connect({f"controller_{side}": controllers.output(source_key)})
                .output("hand_joints")
            )

        # dt matches the 50 Hz control rate, so the heading integrates in real seconds.
        pico = SonicPicoLocomotionRetargeter(name="pico_locomotion", dt=1.0 / 50.0).connect(
            {
                "controller_left": controllers.output(ControllersSource.LEFT),
                "controller_right": controllers.output(ControllersSource.RIGHT),
            }
        )
        command = SonicTeleopCommandRetargeter(name="sonic_command").connect(
            {
                SonicPicoLocomotionRetargeter.OUTPUT_NAME: pico.output(
                    SonicPicoLocomotionRetargeter.OUTPUT_NAME
                ),
                "controller_left": controllers.output(ControllersSource.LEFT),
            }
        )

        ref_names = [f"ref_{i}" for i in range(SONIC_REFERENCE_DIM)]
        cmd_names = [f"cmd_{i}" for i in range(SONIC_COMMAND_DIM)]
        left_names = [f"l_{n}" for n in _TRIHAND_SLOT_NAMES]
        right_names = [f"r_{n}" for n in _TRIHAND_SLOT_NAMES]
        reorderer = TensorReorderer(
            input_config={
                "sonic_reference": ref_names,
                "sonic_command": cmd_names,
                "left_hand": left_names,
                "right_hand": right_names,
            },
            output_order=ref_names + cmd_names + left_names + right_names,
            # The reference and command are single NDArrays; the tri-hand outputs are groups of
            # scalar floats, which is TensorReorderer's default.
            input_types={"sonic_reference": "array", "sonic_command": "array"},
            name="sonic_full_action",
        ).connect(
            {
                "sonic_reference": reference.output("sonic_reference"),
                "sonic_command": command.output(SonicTeleopCommandRetargeter.OUTPUT_NAME),
                "left_hand": hands["left"],
                "right_hand": hands["right"],
            }
        )
        return OutputCombiner({"action": reorderer.output("output")})

    return _build
