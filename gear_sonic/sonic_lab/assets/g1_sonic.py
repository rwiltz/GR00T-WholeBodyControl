# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 articulation config for running SONIC inside Isaac Lab.

SONIC was trained against :obj:`gear_sonic.envs.manager_env.robots.g1.G1_CYLINDER_MODEL_12_DEX_CFG`
(selected in training via ``robot.type: g1_model_12_dex``). We reuse that config verbatim rather
than Isaac Lab's stock ``G1_29DOF_CFG``, because the two are not interchangeable:

===================  ==========================  ==============================
Actuator group       SONIC (g1_model_12_dex)     Isaac Lab ``G1_29DOF_CFG``
===================  ==========================  ==============================
arms stiffness       ~14.3 (5020) / ~16.8 (4010) 3000.0
waist stiffness      ~28.5 / ~40.2 (yaw)         5000.0
knee stiffness       ~99.1                       200.0
actuator model       ``ImplicitActuatorCfg``     ``DCMotorCfg`` (legs/feet)
armature             per-motor (0.0036..0.0251)  0.03 legs / 0.001 arms
self-collisions      enabled                     disabled
===================  ==========================  ==============================

The stock config's arm/waist gains are ~200x stiffer because that environment drives the upper
body with an IK position servo; they are deliberately non-physical. SONIC expects compliant gains
matched to the real G1 motors, so it will not transfer onto the stock config.

Two fixes are applied here on top of the upstream config:

1. ``ASSET_DIR`` upstream is the *relative* path ``"gear_sonic/data/assets"``, which only resolves
   when the process cwd happens to be the repo root. We rewrite it to an absolute path.
2. ``effort_limit_sim`` / ``velocity_limit_sim`` are deprecated aliases in Isaac Lab 3.0 (removed
   in 4.0). They still work, but we migrate them to ``joint_effort_limit`` / ``joint_velocity_limit``
   so the config is 4.0-clean and does not emit deprecation warnings on every launch.
"""

from __future__ import annotations

import copy
import pathlib

from isaaclab.assets.articulation import ArticulationCfg

from gear_sonic.envs.manager_env.robots.g1 import (
    G1_CYLINDER_MODEL_12_DEX_CFG as _UPSTREAM_CFG,
)
from gear_sonic.envs.manager_env.robots.g1 import (
    G1_ISAACLAB_TO_MUJOCO_MAPPING,
    G1_MODEL_12_ACTION_SCALE,
)

__all__ = [
    "G1_ISAACLAB_TO_MUJOCO_MAPPING",
    "G1_MODEL_12_ACTION_SCALE",
    "G1_SONIC_CFG",
    "SONIC_NUM_JOINTS",
    "repo_root",
]

# gear_sonic/sonic_lab/assets/g1_sonic.py -> repo root is 4 levels up.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: SONIC's G1 is 29 DoF (no hand joints in ``robot_description/urdf/g1/main.urdf``).
SONIC_NUM_JOINTS = 29


def repo_root() -> pathlib.Path:
    """Absolute path to the GR00T-WholeBodyControl checkout containing this module."""
    return _REPO_ROOT


def _absolutize_asset_path(cfg: ArticulationCfg) -> None:
    """Rewrite the upstream relative ``gear_sonic/data/assets/...`` URDF path to an absolute one."""
    asset_path = getattr(cfg.spawn, "asset_path", None)
    if asset_path is None:
        raise ValueError(
            "Expected the SONIC G1 spawn config to be a UrdfFileCfg with an 'asset_path'; "
            f"got {type(cfg.spawn).__name__} with no asset_path."
        )
    if pathlib.Path(asset_path).is_absolute():
        return
    resolved = _REPO_ROOT / asset_path
    if not resolved.is_file():
        raise FileNotFoundError(
            f"SONIC G1 URDF not found at {resolved}. Expected the upstream relative path "
            f"{asset_path!r} to resolve against the repo root {_REPO_ROOT}. "
            "If this repo was cloned without Git LFS, run 'git lfs pull'."
        )
    cfg.spawn.asset_path = str(resolved)


def _migrate_deprecated_actuator_limits(cfg: ArticulationCfg) -> None:
    """Move ``*_limit_sim`` onto the Isaac Lab 3.0 ``joint_*_limit`` fields.

    Both spellings are accepted in 3.0 (``joint_*_limit`` is canonical, ``*_limit_sim`` is a
    deprecated alias slated for removal in 4.0). We only move a value when the canonical field is
    still unset, so an explicit override is never clobbered.
    """
    for actuator in cfg.actuators.values():
        for deprecated, canonical in (
            ("effort_limit_sim", "joint_effort_limit"),
            ("velocity_limit_sim", "joint_velocity_limit"),
        ):
            value = getattr(actuator, deprecated, None)
            if value is None:
                continue
            if getattr(actuator, canonical, None) is None:
                setattr(actuator, canonical, value)
            setattr(actuator, deprecated, None)


def make_g1_sonic_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    """Build the SONIC-compatible G1 articulation config.

    Args:
        prim_path: Prim path for the spawned articulation.

    Returns:
        A deep copy of the upstream SONIC G1 config with an absolute asset path and
        Isaac Lab 3.0 actuator-limit fields.
    """
    cfg = copy.deepcopy(_UPSTREAM_CFG)
    _absolutize_asset_path(cfg)
    _migrate_deprecated_actuator_limits(cfg)
    return cfg.replace(prim_path=prim_path)


#: Ready-to-use config; prefer :func:`make_g1_sonic_cfg` when you need a distinct prim path.
G1_SONIC_CFG = make_g1_sonic_cfg()
