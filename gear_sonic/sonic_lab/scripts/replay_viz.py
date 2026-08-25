# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visualize an Isaac Teleop MCAP replay retargeted for SONIC, inside Isaac Lab.

Spawns SONIC's G1 next to a marker skeleton driven by
:class:`~gear_sonic.sonic_lab.retargeters.sonic_fullbody_retargeter.SonicFullBodyRetargeter`,
replaying a recorded full-body tracking session. No headset required.

.. important::
   This visualizes the **retargeting stage only**. The SONIC policy is not in the loop yet, so the
   robot's legs and torso stay at their default pose. Only the six G1 wrist joints are driven,
   because those fall out of retargeting directly as joint angles. Full-body robot motion requires
   the SONIC ``ActionTerm``, which is not implemented yet.

   Orange spheres  = retargeted SMPL joints (what SONIC's ``smpl`` encoder would receive)
   Cyan spheres    = left/right wrist and head, highlighted for orientation

Usage::

    /path/to/IsaacLab/.venv/bin/python -m gear_sonic.sonic_lab.scripts.replay_viz \\
        --mcap /path/to/full_body_*.mcap --viz kit
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mcap", type=str, required=True, help="full_body MCAP recording")
parser.add_argument("--channel", type=str, default="full_body", help="MCAP channel base name")
parser.add_argument("--fps", type=float, default=50.0, help="Replay rate (SONIC runs at 50 Hz)")
parser.add_argument("--skeleton-offset", type=float, nargs=3, default=(0.0, 1.2, 1.0),
                    help="World offset for the marker skeleton, so it sits beside the robot")
parser.add_argument("--no-loop", action="store_true", help="Stop at the end instead of looping")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ruff: noqa: E402  -- Isaac Sim requires the app to launch before these imports resolve.
import numpy as np
import torch

from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

from gear_sonic.sonic_lab.assets import make_g1_sonic_cfg
from gear_sonic.sonic_lab.retargeters.sonic_fullbody_retargeter import (
    SonicFullBodyRetargeter,
    SonicFullBodyRetargeterConfig,
    SonicReferenceSlice,
)
from gear_sonic.sonic_lab.tests.replay_mcap import read_body_frames

#: Highlighted joints: left wrist, right wrist, head (see ``BodyJointIndex``).
_HIGHLIGHT_JOINTS = (20, 21, 15)

#: G1 wrist joints, ordered to match the retargeter output
#: ``[l_roll, r_roll, l_pitch, r_pitch, l_yaw, r_yaw]``.
_WRIST_JOINT_NAMES = (
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


def _build_markers() -> VisualizationMarkers:
    """Spheres for every retargeted SMPL joint, with the wrists/head picked out in cyan."""
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/sonic_smpl_joints",
        markers={
            "joint": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.0)),
            ),
            "highlight": sim_utils.SphereCfg(
                radius=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.85, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def main() -> None:
    print(f"[replay_viz] reading {args_cli.mcap}")
    frames = read_body_frames(args_cli.mcap, args_cli.channel)
    if not frames:
        raise SystemExit("No valid body-tracking frames in recording.")
    print(f"[replay_viz] {len(frames)} frames; replaying at {args_cli.fps} Hz")

    retargeter = SonicFullBodyRetargeter(SonicFullBodyRetargeterConfig(device="cpu"), name="viz")
    references = np.stack([retargeter._retarget(f) for f in frames])  # noqa: SLF001

    dt = 1.0 / args_cli.fps
    sim = SimulationContext(SimulationCfg(dt=dt, device=args_cli.device))
    sim.set_camera_view(eye=(2.6, 2.6, 1.8), target=(0.0, 0.6, 0.9))

    cfg_ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    cfg_ground.spawn.func("/World/GroundPlane", cfg_ground.spawn)
    cfg_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.78, 0.78, 0.78)),
    )
    cfg_light.spawn.func("/World/Light", cfg_light.spawn)

    print("[replay_viz] spawning SONIC G1 (URDF->USD conversion may take a minute on first run)")
    robot = Articulation(make_g1_sonic_cfg(prim_path="/World/Robot"))
    markers = _build_markers()

    sim.reset()
    print("[replay_viz] scene ready")

    wrist_ids, wrist_names = robot.find_joints(list(_WRIST_JOINT_NAMES), preserve_order=True)
    print(f"[replay_viz] driving {len(wrist_names)} wrist joints: {wrist_names}")
    print("[replay_viz] NOTE: legs/torso hold the default pose - SONIC is not in the loop yet.")

    offset = torch.tensor(args_cli.skeleton_offset, device=sim.device, dtype=torch.float32)
    marker_indices = torch.zeros(24, dtype=torch.long, device=sim.device)
    for j in _HIGHLIGHT_JOINTS:
        marker_indices[j] = 1

    default_joint_pos = robot.data.default_joint_pos.clone()
    frame_idx = 0
    while simulation_app.is_running():
        if frame_idx >= len(references):
            if args_cli.no_loop:
                break
            frame_idx = 0

        reference = references[frame_idx]
        joints = torch.from_numpy(
            reference[SonicReferenceSlice.SMPL_JOINTS].reshape(24, 3).copy()
        ).to(sim.device)
        markers.visualize(translations=joints + offset, marker_indices=marker_indices)

        wrist_targets = torch.from_numpy(
            reference[SonicReferenceSlice.WRIST_JOINT_POS].copy()
        ).to(sim.device)
        targets = default_joint_pos.clone()
        targets[0, wrist_ids] = wrist_targets
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()

        sim.step()
        robot.update(dt)
        frame_idx += 1

    print("[replay_viz] done")


if __name__ == "__main__":
    main()
    simulation_app.close()
