# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal Isaac Lab environment for SONIC whole-body teleoperation of the G1.

A deliberately small manager-based environment whose only action term is
:class:`~gear_sonic.sonic_lab.mdp.actions.SonicWholeBodyAction`. The point is to exercise the
full chain — Isaac Teleop full-body tracking -> retargeter -> SONIC -> G1 joint targets — without
dragging in task-specific scene objects, rewards or curricula.

Timing matches both SONIC and the reference locomanip environment: ``sim.dt = 1/200`` with
``decimation = 4`` gives a 50 Hz control rate, which is exactly SONIC's ``control_dt_ = 0.02``.

Attach teleop by pointing ``IsaacTeleopCfg.pipeline_builder`` at
:func:`~gear_sonic.sonic_lab.retargeters.pipeline.build_sonic_fullbody_pipeline` for a live
headset, or :func:`~gear_sonic.sonic_lab.retargeters.pipeline.build_sonic_fullbody_replay_pipeline`
together with ``SessionMode.REPLAY`` to drive it from a recorded MCAP.
"""

from __future__ import annotations

import pathlib

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.configclass import configclass

from gear_sonic.sonic_lab.assets import (
    G1_MODEL_12_ACTION_SCALE,
    make_g1_sonic_cfg,
    repo_root,
)
from gear_sonic.sonic_lab.mdp.actions import SonicWholeBodyActionCfg

__all__ = ["DEFAULT_CHECKPOINT_DIR", "SonicTeleopG1EnvCfg"]

#: Where ``download_from_hf.py --sonic-v1-1`` puts the deployment ONNX graphs.
DEFAULT_CHECKPOINT_DIR = str(
    repo_root() / "gear_sonic_deploy" / "policy" / "sonic_v1_1"
)


@configclass
class SonicTeleopSceneCfg(InteractiveSceneCfg):
    """Ground, light, and SONIC's G1. Nothing task-specific."""

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.78, 0.78, 0.78)),
    )

    robot = make_g1_sonic_cfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class SonicActionsCfg:
    """SONIC is the only action term; it drives all 29 joints."""

    sonic = SonicWholeBodyActionCfg(
        asset_name="robot",
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
        joint_names=[".*"],
        action_scale=G1_MODEL_12_ACTION_SCALE,
    )


@configclass
class SonicObservationsCfg:
    """Minimal observations. SONIC builds its own inputs inside the action term."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SonicTerminationsCfg:
    """Time-out only; add fall detection when you move past bring-up."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class SonicTeleopG1EnvCfg(ManagerBasedRLEnvCfg):
    """G1 driven by SONIC from Isaac Teleop full-body tracking."""

    scene: SonicTeleopSceneCfg = SonicTeleopSceneCfg(num_envs=1, env_spacing=2.5)
    observations: SonicObservationsCfg = SonicObservationsCfg()
    actions: SonicActionsCfg = SonicActionsCfg()
    terminations: SonicTerminationsCfg = SonicTerminationsCfg()

    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self) -> None:
        # 200 Hz physics / 50 Hz control == SONIC's control_dt_ of 0.02.
        self.decimation = 4
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 120.0

        self.viewer.eye = (2.4, 2.4, 1.6)
        self.viewer.lookat = (0.0, 0.0, 0.8)


def checkpoint_dir_or_raise(path: str | pathlib.Path | None = None) -> str:
    """Return a validated SONIC checkpoint directory.

    Args:
        path: Override directory. Defaults to :data:`DEFAULT_CHECKPOINT_DIR`.

    Raises:
        FileNotFoundError: If the ONNX graphs are absent, with the README fetch command.
    """
    resolved = pathlib.Path(path or DEFAULT_CHECKPOINT_DIR)
    missing = [
        name
        for name in ("model_encoder.onnx", "model_decoder.onnx")
        if not (resolved / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {resolved}.\nFetch per the README:\n"
            "    python download_from_hf.py --sonic-v1-1"
        )
    return str(resolved)
