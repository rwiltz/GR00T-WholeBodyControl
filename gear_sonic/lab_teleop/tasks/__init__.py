# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gymnasium registration for the SONIC teleoperation environments.

Registering makes the environment usable from Isaac Lab's stock entry points, which all resolve the
task by name via ``parse_env_cfg`` + ``gym.make``::

    scripts/environments/teleoperation/teleop_se3_agent.py     --task <id>
    scripts/environments/teleoperation/teleop_replay_agent.py  --task <id> --replay_file <mcap>
    scripts/tools/record_demos.py                              --task <id>

Import this module (or :mod:`gear_sonic.lab_teleop`) before calling ``gym.make``.

Naming: the id deliberately avoids the substrings ``Lift`` and ``Reach``. ``teleop_se3_agent.py``
pattern-matches on the task name (``if "Lift" in args_cli.task``, ``if "Reach" in ...``) and injects
task-specific terms when they appear.
"""

from __future__ import annotations

import gymnasium as gym

SONIC_TELEOP_G1_TASK_ID = "IsaacContrib-Teleop-Sonic-WholeBody-G1-v0"
"""Live-headset id: G1 driven by SONIC from Isaac Teleop full-body tracking."""

SONIC_TELEOP_G1_REPLAY_TASK_ID = "IsaacContrib-Teleop-Sonic-WholeBody-G1-Replay-v0"
"""MCAP-replay id. Same environment with the vendor-less pipeline REPLAY sessions require."""

SONIC_TELEOP_G1_LOW_LATENCY_TASK_ID = "IsaacContrib-Teleop-Sonic-WholeBody-G1-LowLatency-v0"
"""Live-headset id running the low-latency checkpoint (4 reference frames, ~80 ms lag)."""

SONIC_TELEOP_G1_LOW_LATENCY_REPLAY_TASK_ID = (
    "IsaacContrib-Teleop-Sonic-WholeBody-G1-LowLatency-Replay-v0"
)
"""MCAP-replay id for the low-latency checkpoint."""

SONIC_TELEOP_G1_HANDS_TASK_ID = "IsaacContrib-Teleop-Sonic-WholeBody-G1-Hands-v0"
"""Live-headset id adding controller-driven tri-hand grasping (trigger pinch, squeeze grasp)."""

SONIC_TELEOP_G1_HANDS_REPLAY_TASK_ID = "IsaacContrib-Teleop-Sonic-WholeBody-G1-Hands-Replay-v0"
"""MCAP-replay id for the hands variant."""

SONIC_TELEOP_G1_HANDS_LOW_LATENCY_TASK_ID = (
    "IsaacContrib-Teleop-Sonic-WholeBody-G1-Hands-LowLatency-v0"
)
"""Live-headset id: controller-driven hands plus the low-latency SONIC checkpoint."""

SONIC_TELEOP_G1_HANDS_LOW_LATENCY_REPLAY_TASK_ID = (
    "IsaacContrib-Teleop-Sonic-WholeBody-G1-Hands-LowLatency-Replay-v0"
)
"""MCAP-replay id for the low-latency hands variant."""

gym.register(
    id=SONIC_TELEOP_G1_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ("gear_sonic.lab_teleop.sonic_teleop_env_cfg:SonicTeleopG1EnvCfg"),
    },
    disable_env_checker=True,
)

gym.register(
    id=SONIC_TELEOP_G1_REPLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "gear_sonic.lab_teleop.sonic_teleop_env_cfg:SonicTeleopG1ReplayEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id=SONIC_TELEOP_G1_LOW_LATENCY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "gear_sonic.lab_teleop.sonic_teleop_env_cfg:SonicTeleopG1LowLatencyEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id=SONIC_TELEOP_G1_LOW_LATENCY_REPLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "gear_sonic.lab_teleop.sonic_teleop_env_cfg:" "SonicTeleopG1LowLatencyReplayEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id=SONIC_TELEOP_G1_HANDS_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "gear_sonic.lab_teleop.sonic_teleop_env_cfg:SonicTeleopG1HandsEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id=SONIC_TELEOP_G1_HANDS_REPLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "gear_sonic.lab_teleop.sonic_teleop_env_cfg:SonicTeleopG1HandsReplayEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id=SONIC_TELEOP_G1_HANDS_LOW_LATENCY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "gear_sonic.lab_teleop.sonic_teleop_env_cfg:SonicTeleopG1HandsLowLatencyEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id=SONIC_TELEOP_G1_HANDS_LOW_LATENCY_REPLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "gear_sonic.lab_teleop.sonic_teleop_env_cfg:" "SonicTeleopG1HandsLowLatencyReplayEnvCfg"
        ),
    },
    disable_env_checker=True,
)

__all__ = [
    "SONIC_TELEOP_G1_HANDS_LOW_LATENCY_REPLAY_TASK_ID",
    "SONIC_TELEOP_G1_HANDS_LOW_LATENCY_TASK_ID",
    "SONIC_TELEOP_G1_HANDS_REPLAY_TASK_ID",
    "SONIC_TELEOP_G1_HANDS_TASK_ID",
    "SONIC_TELEOP_G1_LOW_LATENCY_REPLAY_TASK_ID",
    "SONIC_TELEOP_G1_LOW_LATENCY_TASK_ID",
    "SONIC_TELEOP_G1_REPLAY_TASK_ID",
    "SONIC_TELEOP_G1_TASK_ID",
]
