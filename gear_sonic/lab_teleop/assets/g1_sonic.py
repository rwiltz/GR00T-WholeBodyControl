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

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from gear_sonic.envs.manager_env.robots.g1 import (
    G1_CYLINDER_MODEL_12_DEX_CFG as _UPSTREAM_CFG,
    G1_ISAACLAB_TO_MUJOCO_MAPPING,
    G1_MODEL_12_ACTION_SCALE,
)

__all__ = [
    "G1_HAND_JOINT_NAMES",
    "G1_ISAACLAB_TO_MUJOCO_MAPPING",
    "G1_MODEL_12_ACTION_SCALE",
    "G1_SONIC_CFG",
    "SONIC_NUM_JOINTS",
    "repo_root",
]

# gear_sonic/lab_teleop/assets/g1_sonic.py -> repo root is 4 levels up.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: Body joints SONIC drives. The articulation itself is 43 DoF: the URDF's 14 finger joints were
#: shipped welded (``type="fixed"``) and are now un-welded for hand teleoperation, so SONIC must
#: name its joints explicitly rather than matching ``.*``.
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


def _set_ros_package_paths(cfg: ArticulationCfg) -> None:
    """Let the URDF importer resolve ``package://robot_description/...`` mesh URIs.

    ``main.urdf`` references all 49 of its visual meshes as
    ``package://robot_description/meshes/g1/*.STL``. Resolving that scheme needs a ROS package
    search path, and ``UrdfConverterCfg.ros_package_paths`` defaults to ``[]``.

    Without this the conversion **silently succeeds** but emits zero visual geometry: the USD ends
    up with no ``Mesh`` prims, only collision shapes tagged ``purpose = "guide"``, which are not
    rendered. The robot then simulates correctly while being completely invisible in the viewport.

    The importer expects ``ros_package_paths`` as a list of ``{"name": ..., "path": ...}`` mappings
    (see ``urdf_usd_converter/_impl/convert.py:70-74``), not bare directory strings — passing bare
    strings raises ``AttributeError: 'str' object has no attribute 'get'``. ``path`` is the package
    root, so ``package://robot_description/meshes/...`` resolves to ``<path>/meshes/...``.
    """
    package_dir = _REPO_ROOT / "gear_sonic" / "data" / "assets" / "robot_description"
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"Expected the 'robot_description' package at {package_dir}. "
            "If this repo was cloned without Git LFS, run 'git lfs pull'."
        )
    cfg.spawn.ros_package_paths = [{"name": package_dir.name, "path": str(package_dir)}]


#: Hand joints, 7 per side, in the order
#: :class:`~isaacteleop.retargeters.TriHandMotionControllerRetargeter` emits them:
#: ``[thumb_rotation, thumb_proximal, thumb_distal, index_proximal, index_distal,
#: middle_proximal, middle_distal]``.
G1_HAND_JOINT_NAMES: tuple[str, ...] = tuple(
    f"{side}_hand_{finger}_{index}_joint"
    for side in ("left", "right")
    for finger, index in (
        ("thumb", 0),
        ("thumb", 1),
        ("thumb", 2),
        ("index", 0),
        ("index", 1),
        ("middle", 0),
        ("middle", 1),
    )
)


def _add_hand_actuators(cfg: ArticulationCfg) -> None:
    """Give the 14 finger joints an actuator group.

    The URDF ships these joints welded (``type="fixed"``) but fully specified -- axis, effort,
    velocity and travel limits are all present -- so un-welding them yields real DoFs with no
    other asset change. Isaac Lab requires every articulated joint to be covered by some
    actuator, so an uncovered finger would fail at startup rather than simply going limp.

    Gains are deliberately stiff relative to the body: fingers are light, carry no load in this
    scene, and a soft grip visibly lags the trigger. Effort and velocity are taken from the URDF
    rather than invented.
    """
    if any("hand" in name for name in cfg.actuators):
        return
    cfg.actuators["hands"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hand_(thumb|index|middle)_[0-2]_joint"],
        effort_limit=1.4,
        velocity_limit=12.0,
        stiffness=20.0,
        damping=1.0,
    )


def _set_usd_cache_dir(cfg: ArticulationCfg) -> None:
    """Give the URDF converter a stable, **content-addressed** output directory.

    ``UrdfConverterCfg.usd_dir`` defaults to ``None``, and the converter then writes to
    ``<tmp>/IsaacLab/usd_<timestamp>_<random>`` (``asset_converter_base.py:74``). The name differs
    every launch, so the converter's lazy-conversion check can never hit and each run re-imports
    all 68 STL meshes (~66 MB) from scratch, leaving the output behind.

    A *fixed* directory is not sufficient on its own, and getting this wrong is dangerous rather
    than merely wasteful. The URDF importer names its output after the source stem and
    **increments** instead of overwriting, so a changed asset produces ``main_1``, ``main_2``, ...
    alongside the original while the loader may still resolve the stale ``main``. That silently
    simulates the *old* robot: un-welding the hand joints produced a 43-DoF ``main_3`` while the
    articulation loaded 29 DoF from ``main``, and the only symptom was an actuator regex matching
    nothing.

    Addressing the directory by a digest of the asset and the converter settings that affect its
    output means one directory only ever holds one conversion:

    * unchanged asset -> same directory -> genuine reuse (measured 6.0 s cold, 4.1 s warm, with no
      new directories created across repeated runs);
    * changed asset -> different directory -> fresh conversion, and the previous one can no longer
      be selected.

    Superseded digests are inert and safe to delete; they are not pruned automatically because
    doing so would race a concurrently starting run. Disk is now bounded by the number of asset
    revisions (~11 MB each) rather than by the number of launches.
    """
    import hashlib

    asset_path = pathlib.Path(cfg.spawn.asset_path)
    digest = hashlib.sha256()
    digest.update(asset_path.read_bytes())
    # Converter settings that change the emitted USD. Folded in so a settings change also lands in
    # a fresh directory rather than incrementing inside an existing one.
    for field in (
        "merge_fixed_joints",
        "fix_base",
        "convert_mimic_joints_to_normal_joints",
        "replace_cylinders_with_capsules",
        "joint_drive",
    ):
        digest.update(repr(getattr(cfg.spawn, field, None)).encode())

    usd_dir = _REPO_ROOT / ".cache" / "urdf_usd" / f"g1-{digest.hexdigest()[:16]}"
    usd_dir.mkdir(parents=True, exist_ok=True)
    cfg.spawn.usd_dir = str(usd_dir)


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
    _set_ros_package_paths(cfg)
    _add_hand_actuators(cfg)
    _set_usd_cache_dir(cfg)
    _migrate_deprecated_actuator_limits(cfg)
    return cfg.replace(prim_path=prim_path)


#: Ready-to-use config; prefer :func:`make_g1_sonic_cfg` when you need a distinct prim path.
G1_SONIC_CFG = make_g1_sonic_cfg()
