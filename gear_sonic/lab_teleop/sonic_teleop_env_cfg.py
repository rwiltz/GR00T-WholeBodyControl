# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal Isaac Lab environment for SONIC whole-body teleoperation of the G1.

A deliberately small manager-based environment whose only action term is
:class:`~gear_sonic.lab_teleop.mdp.actions.SonicWholeBodyAction`. The point is to exercise the
full chain — Isaac Teleop full-body tracking -> retargeter -> SONIC -> G1 joint targets — without
dragging in task-specific scene objects, rewards or curricula.

Timing matches both SONIC and the reference locomanip environment: ``sim.dt = 1/200`` with
``decimation = 4`` gives a 50 Hz control rate, which is exactly SONIC's ``control_dt_ = 0.02``.

Attach teleop by pointing ``IsaacTeleopCfg.pipeline_builder`` at
:func:`~gear_sonic.lab_teleop.retargeters.pipeline.make_sonic_full_pipeline_builder` for a live
headset, or the same builder with ``vendor=None`` together with ``SessionMode.REPLAY`` to drive it
from a recorded MCAP.
"""

from __future__ import annotations

import pathlib

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
from gear_sonic.lab_teleop.assets import (
    G1_MODEL_12_ACTION_SCALE,
    make_g1_sonic_cfg,
    repo_root,
)
from gear_sonic.lab_teleop.assets.g1_sonic import G1_HAND_JOINT_NAMES
from gear_sonic.lab_teleop.mdp.modal_actions import SonicModalWholeBodyActionCfg

__all__ = [
    "ANCHOR_ROT_YAW_RIGHT_90",
    "DEFAULT_CHECKPOINT_DIR",
    "LOW_LATENCY_CHECKPOINT_DIR",
    "PACKING_TABLE_Z",
    "PELVIS_ANCHOR_Z_OFFSET",
    "PROP_FORWARD_DISTANCE",
    "STEERING_WHEEL_Z",
    "SONIC_G1_PELVIS_PRIM",
    "SonicTeleopG1EnvCfg",
    "SonicTeleopG1LowLatencyBareEnvCfg",
    "SonicTeleopG1LowLatencyEnvCfg",
    "SonicTeleopG1LowLatencyReplayEnvCfg",
    "SonicTeleopG1ReplayEnvCfg",
]

#: XR anchor yaw, as an XYZW quaternion: -90 deg about +Z, i.e. rotated 90 deg to the right.
#:
#: ``XrCfg.anchor_rot`` is **XYZW**, not WXYZ -- ``xr_anchor_manager.py:105`` unpacks it as
#: ``x, y, z, w``. Flip the sign of the Z term for 90 deg to the left.
_SQRT_HALF = 0.7071067811865476
ANCHOR_ROT_YAW_RIGHT_90 = (0.0, 0.0, -_SQRT_HALF, _SQRT_HALF)

#: How far in front of the robot the manipulation props sit, in metres (~5 ft).
#:
#: ``locomanip_pick_place`` puts its table at 0.55 m, close enough to reach while standing still.
#: Here the props are deliberately out of reach so the operator has to walk to them, which is the
#: point of having a locomotion mode at all.
PROP_FORWARD_DISTANCE = 1.524

#: Table origin height. The USD's surface sits ~1.0 m above its own origin, so a -0.3 origin puts
#: the top near 0.70 m -- the same relationship ``locomanip_pick_place`` relies on.
PACKING_TABLE_Z = -0.3

#: Resting height of the steering wheel, i.e. on the table surface.
STEERING_WHEEL_Z = 0.6996

#: Vertical offset from the G1 pelvis down to the operator's floor plane, in metres.
#:
#: Retained from the pelvis-anchored configuration so the static anchor lands at the same height
#: the following anchor had on frame 0. Making this less negative raises the operator relative to
#: the robot; making it more negative lowers them.
PELVIS_ANCHOR_Z_OFFSET = -0.95

#: Prim path of the G1 pelvis. No longer used as the XR anchor -- see ``__post_init__`` on why the
#: anchor is world-fixed for locomotion -- but kept because it documents the ``Geometry/`` nesting
#: gotcha below, which applies to any prim path taken against this asset.
#:
#: Note the ``Geometry/`` segment. Isaac Lab's URDF importer nests links hierarchically under a
#: ``Geometry`` scope keyed by the URDF root link, whereas the shipped G1 USD that
#: ``locomanip_pick_place`` anchors against exposes links flat under the articulation root. Copying
#: that config's ``.../Robot/pelvis`` therefore points at a prim that does not exist here, and the
#: anchor **silently falls back to the world origin** rather than erroring -- the operator's frame
#: simply stops riding with the robot.
#:
#: This prim carries ``PhysicsRigidBodyAPI`` and ``PhysicsArticulationRootAPI``, i.e. it is the
#: physics body itself; ``Robot/Physics/`` holds only the joints.
SONIC_G1_PELVIS_PRIM = "/World/envs/env_0/Robot/Geometry/pelvis"

#: Where ``download_from_hf.py --sonic-v1-1`` puts the deployment ONNX graphs.
DEFAULT_CHECKPOINT_DIR = str(repo_root() / "gear_sonic_deploy" / "policy" / "sonic_v1_1")

#: Where ``download_from_hf.py --low-latency`` puts the low-latency ONNX graphs.
#:
#: This checkpoint shortens the ``smpl`` reference window from 10 frames to 4, cutting the
#: operator-to-robot lag from ~200 ms to ~80 ms at the 50 Hz control rate. It also switches the
#: anchor orientation from heading-normalized to body-frame; both differences are described by
#: :class:`~gear_sonic.lab_teleop.mdp.sonic_policy.SonicVariant` and applied automatically, since
#: the action term reads them off the encoder graph.
LOW_LATENCY_CHECKPOINT_DIR = str(repo_root() / "gear_sonic_deploy" / "policy" / "low_latency")


@configclass
class SonicTeleopSceneCfg(InteractiveSceneCfg):
    """Ground, light, and SONIC's G1. Nothing task-specific."""

    # ``visible=False`` hides the plane from the renderer but still spawns the collider, so the
    # robot has something to stand on. Dropping the asset entirely would let it fall through.
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg(visible=False))

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.78, 0.78, 0.78)),
    )

    robot = make_g1_sonic_cfg(prim_path="{ENV_REGEX_NS}/Robot")

    # Manipulation props, borrowed from ``locomanip_pick_place``. Placed at
    # PROP_FORWARD_DISTANCE rather than that env's 0.55 m, which puts them within arm's reach of a
    # standing robot: here the operator is expected to *walk* to them in teleop mode, so they need
    # to start outside the reach of full-body tracking alone.
    #
    # The robot spawns with identity rotation, i.e. facing +X, so "in front" is +X. The locomanip
    # scene has its own convention and its coordinates are not transferable verbatim.
    packing_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[PROP_FORWARD_DISTANCE, 0.0, PACKING_TABLE_Z], rot=[0.0, 0.0, 0.0, 1.0]
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            # Kinematic: the table is scenery and must not be shoved around by a stumbling robot.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    #: Grabbable prop, resting on the table. Left dynamic so the hands can actually move it.
    steering_wheel = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SteeringWheel",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[PROP_FORWARD_DISTANCE - 0.1, 0.0, STEERING_WHEEL_Z], rot=[0.0, 0.0, 0.0, 1.0]
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/"
                "steering_wheel.usd"
            ),
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )


@configclass
class SonicActionsCfg:
    """SONIC is the only action term; it drives all 29 joints."""

    # Named explicitly rather than ".*": the articulation is 43 DoF since the URDF's 14 finger
    # joints were un-welded for hand teleoperation, and SONIC drives only its own 29.
    #
    # The modal term is the default rather than a variant. Mode switching and hand grasping are
    # capabilities an operator wants together, not alternatives to choose between at task-id
    # level, and the real robot's teleop stack does not make you pick. The velocity planner that
    # teleop mode needs is constructed lazily, so an operator who never leaves full-body tracking
    # pays none of its ~1 GB.
    sonic = SonicModalWholeBodyActionCfg(
        asset_name="robot",
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
        joint_names=list(G1_ISAACLab_ORDER),
        action_scale=G1_MODEL_12_ACTION_SCALE,
    )

    hands = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(G1_HAND_JOINT_NAMES),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
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
        # 100 Hz rendering (2 renders per 50 Hz control step), matching
        # contrib/locomanip_pick_place. Isaac Lab logs a warning that render_interval < decimation
        # causes multiple renders per step; that is intended here.
        #
        # This trades control rate for render throughput. Measured on this machine replaying a
        # capture (env steps/s vs renders/s):
        #
        #     render_interval=4, no XR : 31.8 steps/s   33.8 renders/s
        #     render_interval=2, no XR : 22.3 steps/s   47.1 renders/s
        #     render_interval=4, XR    : 20.6 steps/s   20.6 renders/s
        #     render_interval=2, XR    : 16.4 steps/s   32.7 renders/s
        #
        # Note the replay agent's reported "fps" counts *renders*, not env steps, so a higher
        # figure there is not by itself a speedup.
        #
        # The control-rate cost matters less than those numbers suggest, because Isaac Teleop does
        # not retarget synchronously with the env step. It resolves by default to
        # ``RetargetingExecutionConfig(mode="pipelined", pacing=DeadlinePacingConfig(...))``
        # (``session_lifecycle.py:942``): retargeting runs on a background worker and the app
        # consumes the most recent completed result, and XR pose prediction covers the offset. So
        # operator input is decoupled from the env-step rate, while the headset only ever sees
        # rendered frames -- which makes render throughput the thing worth spending on.
        #
        # These ratios were measured under replay, which is not paced by OpenXR, so they need not
        # hold live.
        self.sim.render_interval = 2
        self.episode_length_s = 120.0

        self.viewer.eye = (2.4, 2.4, 1.6)
        self.viewer.lookat = (0.0, 0.0, 0.8)

        # Attaching the teleop pipeline to the cfg is what lets the stock Isaac Lab entry points
        # drive this environment. `teleop_se3_agent.py:277-279` selects the Isaac Teleop path with
        #     use_isaac_teleop = not <--teleop_device given> and env_cfg.isaac_teleop is not None
        # and its main loop is device-agnostic: it passes whatever `IsaacTeleopDevice.advance()`
        # returns straight to `env.step()`. Our pipeline's "action" output is the 83-wide SONIC
        # reference, which is exactly what the action term consumes.
        #
        # Note the SE(3) devices (`--teleop_device keyboard|spacemouse|gamepad`) can NOT drive this
        # env: they emit a 6-DoF delta plus gripper, which cannot express a whole-body pose.
        from isaaclab_teleop import IsaacTeleopCfg, XrCfg

        from gear_sonic.lab_teleop.retargeters.pipeline import (
            make_sonic_full_pipeline_builder,
        )

        # Pin the operator's XR frame to the robot's *start* pose; do not let it ride the pelvis.
        #
        # Anchoring to the pelvis (as `isaaclab_tasks.contrib.locomanip_pick_place` does) is right
        # for a stationary manipulator, but it defeats locomotion teleoperation. SONIC walks
        # because the operator walks, and the operator's displacement is measured *relative to the
        # anchor*. If the anchor rides the robot, every step the robot takes carries the frame
        # forward by the same amount, so the operator can never gain ground on it and the
        # commanded displacement collapses toward zero. A world-fixed anchor makes room-scale
        # walking map directly to world displacement, which is what SONIC's gait consumes.
        #
        # The anchor is placed at exactly the pose the pelvis-relative anchor resolved to on frame
        # 0, so the operator's initial framing -- eye height and facing -- is unchanged; only the
        # following behaviour is dropped. The dynamic path composes
        # ``anchor_world = pelvis_world + anchor_pos`` (``xr_anchor_utils.py:140``), so the
        # equivalent static world position is the robot's start pose plus that same offset.
        # Derived from the configured articulation rather than hardcoded, so moving the robot's
        # spawn keeps the anchor with it.
        start_pos = self.scene.robot.init_state.pos
        self.xr = XrCfg(
            anchor_pos=(
                start_pos[0],
                start_pos[1],
                start_pos[2] + PELVIS_ANCHOR_Z_OFFSET,
            ),
            anchor_rot=ANCHOR_ROT_YAW_RIGHT_90,
        )
        # `anchor_prim_path` deliberately left None: the anchor prim is then created under
        # `/World` instead of as a child of `SONIC_G1_PELVIS_PRIM`, and is never re-synced to a
        # moving prim (`xr_anchor_manager.py:95-101`). `fixed_anchor_height` and
        # `anchor_rotation_mode` are consulted only on the prim-following branch
        # (`xr_anchor_utils.py:135` and below), so they are inert here and left unset.
        #
        # This assumes the robot spawns with identity rotation, which it does -- the static branch
        # applies `anchor_rot` as an absolute world orientation with no prim rotation to compose
        # against, so ANCHOR_ROT_YAW_RIGHT_90 reproduces the current frame-0 facing exactly. A
        # non-identity spawn rotation would need composing in here.

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=make_sonic_full_pipeline_builder(),
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )


@configclass
class SonicTeleopG1ReplayEnvCfg(SonicTeleopG1EnvCfg):
    """Same environment, wired for MCAP replay instead of a live headset.

    ``TeleopSession`` rejects source nodes that carry a tracker vendor when the session mode is
    ``SessionMode.REPLAY`` (``teleop_session.py:313-333``): replay reads whatever channel was
    recorded regardless of vendor, so a vendor selection would be silently ignored and it fails
    fast instead. This variant therefore swaps in the vendor-less pipeline builder.

    Use with ``scripts/environments/teleoperation/teleop_replay_agent.py --replay_file <mcap>``.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from gear_sonic.lab_teleop.retargeters.pipeline import (
            make_sonic_full_pipeline_builder,
        )

        self.isaac_teleop.pipeline_builder = make_sonic_full_pipeline_builder(vendor=None)


@configclass
class SonicTeleopG1LowLatencyEnvCfg(SonicTeleopG1EnvCfg):
    """Live-headset environment driven by the **low-latency** SONIC checkpoint.

    Identical scene, actuation and teleop wiring to :class:`SonicTeleopG1EnvCfg`; only the
    checkpoint changes. Everything downstream adapts automatically because
    :class:`~gear_sonic.lab_teleop.mdp.sonic_policy.SonicOnnxPolicy` identifies the checkpoint from
    its encoder input width and exposes a
    :class:`~gear_sonic.lab_teleop.mdp.sonic_policy.SonicVariant` describing it.

    Two things differ from ``sonic_v1_1`` and both are handled by that variant:

    * **4 reference frames instead of 10.** The encoder input narrows from 1751 to 1247, and the
      robot trails the operator by ~80 ms rather than ~200 ms at the 50 Hz control rate. The
      *decoder* is unchanged at 994, so the proprioception history is still 10 frames.
    * **Body-frame anchor orientation instead of heading-normalized.** The relative rotation uses
      the full base quaternion rather than the robot's yaw alone, so the robot's own pitch and
      roll now enter the reference term.

    That second difference is why this is a distinct config rather than a ``checkpoint_dir``
    override on the base class: applying v1.1's heading-normalized math to this checkpoint would
    produce a well-formed but wrong rotation, degrading control silently rather than raising.

    Fetch the checkpoint with ``python download_from_hf.py --low-latency``.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.sonic.checkpoint_dir = LOW_LATENCY_CHECKPOINT_DIR


@configclass
class SonicTeleopG1LowLatencyReplayEnvCfg(SonicTeleopG1LowLatencyEnvCfg):
    """Low-latency environment wired for MCAP replay instead of a live headset.

    Stands in the same relation to :class:`SonicTeleopG1LowLatencyEnvCfg` as
    :class:`SonicTeleopG1ReplayEnvCfg` does to :class:`SonicTeleopG1EnvCfg`: it swaps in the
    vendor-less pipeline that ``SessionMode.REPLAY`` requires (see that class for why).

    A capture recorded against either checkpoint replays here -- the MCAP holds raw tracker
    streams, not encoder inputs, so the reference window is rebuilt at replay time from whatever
    the active checkpoint asks for.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        from gear_sonic.lab_teleop.retargeters.pipeline import (
            make_sonic_full_pipeline_builder,
        )

        self.isaac_teleop.pipeline_builder = make_sonic_full_pipeline_builder(vendor=None)


@configclass
class SonicModalActionsCfg:
    """Mode-switching SONIC action term."""

    sonic = SonicModalWholeBodyActionCfg(
        asset_name="robot",
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
        joint_names=list(G1_ISAACLab_ORDER),
        action_scale=G1_MODEL_12_ACTION_SCALE,
    )


@configclass
class SonicTeleopG1LowLatencyBareEnvCfg(SonicTeleopG1LowLatencyEnvCfg):
    """Low-latency environment with the manipulation props removed.

    Same robot, controls and modes as :class:`SonicTeleopG1LowLatencyEnvCfg`; only the scene
    dressing differs. The packing table brings crates, boxes, a container and their 1k PBR texture
    sets with it, which measured ~7.7 ms per frame -- replay runs at ~30.4 ms and 33 fps with the
    props against ~22.7 ms and 44 fps without. That headroom matters for a headset, and the props
    are irrelevant when the task is locomotion or gait tuning rather than manipulation.

    Removing them also drops the dependency on the Omniverse content bucket, so this variant
    starts faster on a cold machine and works offline.

    ``InteractiveScene`` skips entries whose config is ``None`` (``interactive_scene.py:796``), so
    nulling the fields is the supported way to drop an inherited asset rather than redeclaring the
    scene.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.packing_table = None
        self.scene.steering_wheel = None


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
        from gear_sonic.lab_teleop.mdp.sonic_policy import sonic_download_hint

        raise FileNotFoundError(
            f"Missing {missing} in {resolved}.\nFetch this checkpoint with:\n"
            f"    {sonic_download_hint(resolved)}"
        )
    return str(resolved)
