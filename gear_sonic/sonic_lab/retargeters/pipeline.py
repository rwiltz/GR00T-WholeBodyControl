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

from gear_sonic.sonic_lab.retargeters.sonic_fullbody_retargeter import (
    SonicFullBodyRetargeter,
    SonicFullBodyRetargeterConfig,
)

__all__ = [
    "DEFAULT_BODY_TRACKER_VENDOR",
    "build_sonic_fullbody_pipeline",
    "build_sonic_fullbody_replay_pipeline",
    "make_sonic_fullbody_pipeline_builder",
]

#: DeviceIO vendor string for PICO body tracking (``XR_BD_body_tracking``).
#: Upstream ``gear_sonic`` reaches the same backend via the ``FullBodyTrackerPico`` subclass.
DEFAULT_BODY_TRACKER_VENDOR = "body.pico-xr"

#: MCAP channel base name that :class:`FullBodySource` records/replays under.
BODY_CHANNEL_NAME = "full_body"


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
        frame described in :mod:`~gear_sonic.sonic_lab.retargeters.sonic_fullbody_retargeter`.
    """
    return make_sonic_fullbody_pipeline_builder()()


def build_sonic_fullbody_replay_pipeline() -> OutputCombiner:
    """MCAP-replay pipeline: identical graph, but with no vendor selection.

    Use with ``IsaacTeleopCfg`` when the teleop session runs in ``SessionMode.REPLAY``, so an
    end-to-end environment test can be driven from a recording with no headset attached.
    """
    return make_sonic_fullbody_pipeline_builder(vendor=None)()
