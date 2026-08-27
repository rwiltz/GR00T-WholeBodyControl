# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: G1 driven by SONIC from Isaac Teleop full-body tracking, inside Isaac Lab.

Two input sources:

``--mcap PATH``
    Replay a recorded full-body session. No headset needed. Body frames are read through
    ``ReplaySession`` and pushed through the same retargeter the live path uses, so this exercises
    retargeter -> SONIC -> G1 exactly as a live session would.

``--live``
    Open a CloudXR/OpenXR session via ``IsaacTeleopCfg`` and drive from a headset.

Examples::

    # replay, with a viewport
    python -m gear_sonic.lab_teleop.scripts.run_sonic_teleop \\
        --mcap ~/recordings/full_body_*.mcap --viz kit

    # live headset
    python -m gear_sonic.lab_teleop.scripts.run_sonic_teleop --live --viz kit

.. warning::
   ONNX inference falls back to CPU unless ``onnxruntime-gpu`` is installed, and the SONIC decoder
   costs ~17 ms per step there against a 20 ms control period. Replay still produces correct
   results (it simply runs slower than wall clock), but live teleoperation needs the CUDA
   execution provider.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--mcap", type=str, help="Replay this full_body MCAP recording")
source.add_argument("--live", action="store_true", help="Drive from a live XR headset")
parser.add_argument("--channel", type=str, default="full_body", help="MCAP channel base name")
parser.add_argument("--checkpoint-dir", type=str, default=None, help="SONIC ONNX directory")
parser.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0 = unlimited)")
parser.add_argument("--no-loop", action="store_true", help="Stop at the end of the recording")
parser.add_argument(
    "--loop-mode",
    choices=["pingpong", "wrap"],
    default="pingpong",
    help=(
        "How to repeat a finite recording. 'wrap' restarts at frame 0, which teleports the "
        "reference (in the sample clip, a 136 deg yaw jump) and makes the robot visibly snap. "
        "'pingpong' plays forward then backward so the reference stays continuous."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ruff: noqa: E402  -- Isaac Sim must launch before these resolve.
import time

from isaaclab.envs import ManagerBasedRLEnv
import numpy as np
import torch

from gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter import (
    SonicFullBodyRetargeter,
    SonicFullBodyRetargeterConfig,
)
from gear_sonic.lab_teleop.sonic_teleop_env_cfg import (
    SonicTeleopG1EnvCfg,
    checkpoint_dir_or_raise,
)


def _load_replay_references(path: str, channel: str) -> np.ndarray:
    """Replay an MCAP and retarget every frame up front."""
    from gear_sonic.lab_teleop.tests.replay_mcap import read_body_frames

    frames = read_body_frames(path, channel)
    if not frames:
        raise SystemExit(f"No valid body-tracking frames in {path}")
    retargeter = SonicFullBodyRetargeter(SonicFullBodyRetargeterConfig(device="cpu"), name="replay")
    references = np.stack([retargeter._retarget(f) for f in frames])  # noqa: SLF001
    print(f"[sonic-teleop] retargeted {len(references)} frames from {path}")
    return references


def _build_playback_order(num_frames: int) -> list[int]:
    """Frame indices to emit, one per control step, before repeating.

    A finite recording has to be repeated somehow, and the naive choice teleports the reference.
    In the sample clip the operator turns ~136 deg over 5.6 s, so restarting at frame 0 is a 136 deg
    yaw discontinuity — 430x the median per-frame step. SONIC follows it faithfully, which reads as
    the robot turning and then snapping back.

    ``pingpong`` walks forward then backward, excluding the endpoints on the return leg so no frame
    is emitted twice in a row. The reference stays continuous, which is what you want for a looping
    demo capture.
    """
    forward = list(range(num_frames))
    if args_cli.loop_mode == "wrap" or num_frames < 3:
        return forward
    return forward + list(range(num_frames - 2, 0, -1))


def main() -> None:
    checkpoint_dir = checkpoint_dir_or_raise(args_cli.checkpoint_dir)

    env_cfg = SonicTeleopG1EnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.actions.sonic.checkpoint_dir = checkpoint_dir
    env_cfg.actions.sonic.policy_device = args_cli.device

    if args_cli.live:
        from isaaclab_teleop import IsaacTeleopCfg, XrCfg

        from gear_sonic.lab_teleop.retargeters import make_sonic_full_pipeline_builder

        env_cfg.xr = XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(1.0, 0.0, 0.0, 0.0))
        env_cfg.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=make_sonic_full_pipeline_builder(),
            sim_device=env_cfg.sim.device,
            xr_cfg=env_cfg.xr,
        )

    print(f"[sonic-teleop] SONIC checkpoint: {checkpoint_dir}")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"[sonic-teleop] action space: {env.action_space}")

    references = None if args_cli.live else _load_replay_references(args_cli.mcap, args_cli.channel)

    teleop_device = None
    if args_cli.live:
        from isaaclab_teleop import IsaacTeleopDevice

        teleop_device = IsaacTeleopDevice(env_cfg.isaac_teleop, env=env)
        print("[sonic-teleop] waiting for the headset to connect...")

    playback_order = None if references is None else _build_playback_order(len(references))

    obs, _ = env.reset()
    step = 0
    index = 0
    started = time.perf_counter()

    with torch.inference_mode():
        while simulation_app.is_running():
            if args_cli.max_steps and step >= args_cli.max_steps:
                break

            if teleop_device is not None:
                action = teleop_device.advance()
                if action is None:
                    env.sim.render()
                    continue
                actions = action.unsqueeze(0).to(env.device)
            else:
                if index >= len(playback_order):
                    if args_cli.no_loop:
                        break
                    index = 0
                actions = (
                    torch.from_numpy(references[playback_order[index]]).to(env.device).unsqueeze(0)
                )
                index += 1

            obs, rew, terminated, truncated, info = env.step(actions)
            step += 1

            if step % 100 == 0:
                rate = step / (time.perf_counter() - started)
                height = float(env.scene["robot"].data.root_pos_w.torch[0, 2])
                print(f"[sonic-teleop] step {step:6d}  {rate:5.1f} Hz  " f"pelvis z={height:.3f} m")

    print(f"[sonic-teleop] ran {step} steps")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
