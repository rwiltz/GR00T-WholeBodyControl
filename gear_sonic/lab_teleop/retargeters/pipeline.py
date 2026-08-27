# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Teleop retargeting pipeline for SONIC full-body control.

Wires the XR full-body tracker to :class:`SonicFullBodyRetargeter` and exposes the result as the
pipeline's ``"action"`` output, which is what ``IsaacTeleopDevice.advance()`` returns and what the
teleop runner feeds to ``env.step()``.

Pass :func:`build_sonic_fullbody_pipeline` as the ``pipeline_builder`` of an
``isaaclab_teleop.IsaacTeleopCfg``::

    from isaaclab_teleop import IsaacTeleopCfg, XrCfg

    self.isaac_teleop = IsaacTeleopCfg(
        pipeline_builder=build_sonic_fullbody_pipeline,
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
    "build_sonic_fullbody_pipeline",
    "build_sonic_fullbody_replay_pipeline",
    "make_sonic_fullbody_pipeline_builder",
    "make_sonic_hands_pipeline_builder",
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


def build_sonic_fullbody_pipeline() -> OutputCombiner:
    """Live-session pipeline: XR full-body tracking -> SONIC reference.

    Returns:
        An ``OutputCombiner`` whose single ``"action"`` output is the 83-wide SONIC reference
        frame described in :mod:`~gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter`.
    """
    return make_sonic_fullbody_pipeline_builder()()


def build_sonic_fullbody_replay_pipeline() -> OutputCombiner:
    """MCAP-replay pipeline: identical graph, but with no vendor selection.

    Use with ``IsaacTeleopCfg`` when the teleop session runs in ``SessionMode.REPLAY``, so an
    end-to-end environment test can be driven from a recording with no headset attached.
    """
    return make_sonic_fullbody_pipeline_builder(vendor=None)()


def make_sonic_hands_pipeline_builder(vendor: str | None = DEFAULT_BODY_TRACKER_VENDOR):
    """Full-body reference **plus** controller-driven hands.

    Extends the base graph with a controller branch per side::

        ControllersSource -> TriHandMotionControllerRetargeter --+
                                                                 |
        FullBodySource -> SonicFullBodyRetargeter ---------------+-> TensorReorderer

    The generic tri-hand retargeter maps trigger to index+thumb (pinch) and squeeze to
    middle+thumb (grasp).

    No sign adaptation sits in between, which is worth recording because the joint limits invite
    the opposite conclusion: this URDF's hands are *not* mirrored (left ``index_0`` travels
    ``[-1.571, 0]`` while right travels ``[0, +1.571]``), so the generic output looks like it
    would drive the left hand into its zero bound. It does not --
    ``TriHandMotionControllerRetargeter._compute_fn`` already negates the whole vector for the
    left side, and every one of the 14 joints then lands inside its travel range. A correction
    node here would be a no-op.

    The action is ordered ``[sonic_reference(83) | left_hand(7) | right_hand(7)]`` == 97 to match
    the action-term order in ``SonicHandsActionsCfg``.

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

        if vendor is None:
            body = FullBodySource(name=BODY_CHANNEL_NAME)
        else:
            from isaacteleop.deviceio import TrackerVendor

            body = FullBodySource(name=BODY_CHANNEL_NAME, vendor=TrackerVendor(vendor))

        reference = SonicFullBodyRetargeter(
            SonicFullBodyRetargeterConfig(), name="sonic_fullbody"
        ).connect({"full_body": body.output(FullBodySource.FULL_BODY)})

        controllers = ControllersSource(name="controllers")
        hand_outputs = {}
        for side, source_key in (
            ("left", ControllersSource.LEFT),
            ("right", ControllersSource.RIGHT),
        ):
            trihand = TriHandMotionControllerRetargeter(
                TriHandMotionControllerConfig(
                    hand_joint_names=list(_TRIHAND_SLOT_NAMES), controller_side=side
                ),
                name=f"trihand_{side}",
            ).connect({f"controller_{side}": controllers.output(source_key)})
            hand_outputs[side] = trihand.output("hand_joints")

        reorderer = TensorReorderer(
            input_config={
                "sonic_reference": [f"ref_{i}" for i in range(SONIC_REFERENCE_DIM)],
                "left_hand": [f"l_{n}" for n in _TRIHAND_SLOT_NAMES],
                "right_hand": [f"r_{n}" for n in _TRIHAND_SLOT_NAMES],
            },
            output_order=(
                [f"ref_{i}" for i in range(SONIC_REFERENCE_DIM)]
                + [f"l_{n}" for n in _TRIHAND_SLOT_NAMES]
                + [f"r_{n}" for n in _TRIHAND_SLOT_NAMES]
            ),
            # The reference arrives as one 83-wide NDArray; the tri-hand outputs are groups of
            # scalar floats. TensorReorderer defaults every input to "scalar", so the reference
            # must be declared explicitly or it expects 83 separate tensors.
            input_types={"sonic_reference": "array"},
            name="sonic_hands_action",
        ).connect(
            {
                "sonic_reference": reference.output("sonic_reference"),
                "left_hand": hand_outputs["left"],
                "right_hand": hand_outputs["right"],
            }
        )
        return OutputCombiner({"action": reorderer.output("output")})

    return _build
