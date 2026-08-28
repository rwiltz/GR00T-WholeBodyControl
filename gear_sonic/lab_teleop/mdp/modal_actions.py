# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mode-switching SONIC action term: full-body tracking or stick-driven walking.

Extends :class:`~gear_sonic.lab_teleop.mdp.actions.SonicWholeBodyAction` with the operator's
encoder-mode selection, mirroring the real robot's teleop stack. The environment action becomes::

    [ sonic_reference(95) | mode(1) | locomotion(8) | ground(1) | turn(1) | moving(1) ] -> 107

``mode`` selects which encoder observation block is populated. SONIC's encoder is mode-exclusive:
terms outside the active mode stay zero, and the mode id plus its one-hot occupy slots ``[0]`` and
``[1:4]``. The deploy stack writes those from the active reference source
(``g1_deploy_onnx_ref.cpp:1688``); this term writes them per step instead, because the operator can
change mode at any time.

============  ====================================  ==============================================
mode          encoder slots populated               source
============  ====================================  ==============================================
``smpl`` (2)  ``[911:1751]``                        operator full-body tracking
``teleop``(1) ``[644:650]``, ``[650:890]``,         velocity planner + operator upper body
              ``[890:911]``
============  ====================================  ==============================================

Boundary
--------
Everything here consumes **simulated robot state**, which is why it is on the Isaac Lab side:

* the planner is closed-loop on measured ``qpos`` (see :mod:`.sonic_planner`);
* the idle-direction fallback needs the robot's own velocity and root yaw;
* the anchor orientation term needs the robot's heading, as in ``smpl`` mode.

Isaac Teleop supplies only the operator's mode selection and 8-scalar command; see
:mod:`gear_sonic.lab_teleop.retargeters.sonic_command_retargeter`.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
import numpy as np
import torch

from gear_sonic.lab_teleop.mdp.actions import SonicWholeBodyAction, SonicWholeBodyActionCfg
from gear_sonic.lab_teleop.mdp.sonic_planner import (
    PLANNER_CLIP_IDLE,
    PLANNER_CLIP_WALK,
    PLANNER_CONTEXT_FRAMES,
    PLANNER_LOOKAHEAD_S,
    PLANNER_PERIODIC_REPLAN_S,
    PLANNER_QPOS_DIM,
    SONIC_PLANNER_COMMAND_DIM,
    SONIC_REFERENCE_DT,
    PlannerMotion,
    SonicVelocityPlanner,
)
from gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter import (
    SONIC_REFERENCE_DIM,
    SonicReferenceSlice,
)
from gear_sonic.lab_teleop.retargeters.sonic_pico_locomotion_retargeter import (
    PICO_JOYSTICK_DEADZONE,
    PICO_YAW_GAIN,
)

__all__ = [
    "SONIC_MODAL_ACTION_DIM",
    "SonicModalWholeBodyAction",
    "SonicModalWholeBodyActionCfg",
]

#: ``[reference(95) | mode(1) | planner_command(8) | ground(1) | turn(1) | moving(1)]``.
SONIC_MODAL_ACTION_DIM = SONIC_REFERENCE_DIM + 1 + SONIC_PLANNER_COMMAND_DIM + 3

#: Prim path of the scene's ground plane, toggled from the controller.
GROUND_PLANE_PRIM_PATH = "/World/GroundPlane"

#: OpenXR -> USD axis conversion, ``[x, y, z] -> [x, -z, y]``. Must match
#: ``XrAnchorManager._OXR_TO_USD_ROTATION`` (``xr_anchor_manager.py:71``), because the runtime
#: composes the anchor as ``R_anchor @ R_oxr_to_usd`` and anything predicting where the operator
#: will appear has to use the same conversion. Note this is *not* the Unity matrix the
#: ``vr_3point`` port uses: both map Y to height, so confusing them yields a correct height with
#: the horizontal offset rotated 180 degrees.
OXR_TO_USD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

#: Encoder mode ids, matching the checkpoint's ``encoder_modes``.
ENCODER_MODE_TELEOP = 1
ENCODER_MODE_SMPL = 2

#: Encoder slots ``teleop`` mode populates, recovered from the graph's ``Slice`` nodes.
TELEOP_ANCHOR_ORI = slice(644, 650)  # single frame, 6D rotation
#: Two *separate contiguous blocks*, not interleaved per frame: the encoder declares
#: ``motion_joint_positions_lowerbody_10frame_step5`` (120) followed by
#: ``motion_joint_velocities_lowerbody_10frame_step5`` (120). Each is frame-major -- 10 frames of
#: 12 joints (``g1_deploy_onnx_ref.cpp:1735``, ``GatherMotionJointPositionsMultiFrame``).
TELEOP_LOWER_POS = slice(650, 770)
TELEOP_LOWER_VEL = slice(770, 890)
TELEOP_LOWER_BODY = slice(650, 890)
TELEOP_VR3_POS = slice(890, 899)  # head + 2 hands, xyz
TELEOP_VR3_ORN = slice(899, 911)  # head + 2 hands, 4-wide each

#: The twelve leg joints the ``teleop`` block carries, **in the order the encoder wants them**:
#: whole left leg, then whole right leg. That is MuJoCo's grouping, not Isaac Lab's interleaved
#: left/right one, and it is spelled out upstream as indices into the Isaac Lab ordering --
#: ``lower_body_joint_mujoco_order_in_isaaclab_index = {0,3,6,9,13,17, 1,4,7,10,14,18}``
#: (``policy_parameters.hpp:92``).
LOWER_BODY_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

#: Isaac Lab joint indices for :data:`LOWER_BODY_JOINTS`, matching ``policy_parameters.hpp:92``.
LOWER_BODY_ISAACLAB_INDICES = (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18)

#: Reference frames the ``teleop`` lower-body block spans.
TELEOP_REFERENCE_FRAMES = 10

#: Spacing of those frames on SONIC's reference clock: ``step5`` = 5 ticks at 50 Hz = 0.1 s, so
#: the window looks 0.9 s into the future (``observation_config.md:96-99``). Expressed as a
#: duration, not a frame count, so it stays correct if the environment's control rate changes.
TELEOP_REFERENCE_STRIDE_S = 5.0 * SONIC_REFERENCE_DT


#: Time constant for smoothing the followed anchor yaw, seconds.
#:
#: Matches the shape of Isaac Lab's own ``FOLLOW_PRIM_SMOOTHED`` blend
#: (``xr_anchor_utils.py:168``). Following raw base yaw would feed per-step gait jitter straight
#: into the headset.
ANCHOR_YAW_SMOOTHING_TIME = 0.25


@configclass
class SonicModalWholeBodyActionCfg(SonicWholeBodyActionCfg):
    """Configuration for :class:`SonicModalWholeBodyAction`."""

    class_type: type["SonicModalWholeBodyAction"] | str = (
        "gear_sonic.lab_teleop.mdp.modal_actions:SonicModalWholeBodyAction"
    )

    planner_checkpoint: str = ""
    """Path to ``planner_sonic.onnx``. Empty resolves to the repo's default location."""

    planner_clip: int = PLANNER_CLIP_WALK
    """Planner gait/style clip. **Not** the SONIC encoder mode; clip 0 is idle and ignores speed."""

    #: Print a one-off trace of the SMPL -> teleop handoff: the discontinuity between the robot's
    #: measured legs and the first planned pose, the planned velocity scale, and every clock in
    #: play. Off by default -- this is a diagnostic for the transition, not routine logging.
    debug_transitions: bool = False
    """Frames consumed before re-planning. ``0`` consumes each plan fully."""

    anchor_pan_speed: float = 1.0
    """Metres per second the left stick pans the XR anchor in ``smpl`` mode.

    Lets the operator reposition themselves in the world without physically walking, which matters
    because the play space is small and the props sit 5 ft away. Set to ``0.0`` to disable.

    This is smooth VR locomotion and induces vection in some people. The default is deliberately
    walking-pace rather than fast; raise it only if the drift feels sluggish rather than
    uncomfortable.
    """


class SonicModalWholeBodyAction(SonicWholeBodyAction):
    """SONIC control with operator-selectable encoder mode."""

    cfg: SonicModalWholeBodyActionCfg

    def __init__(self, cfg: SonicModalWholeBodyActionCfg, env) -> None:  # noqa: ANN001
        super().__init__(cfg, env)
        from gear_sonic.lab_teleop.assets.g1_sonic import repo_root

        checkpoint = cfg.planner_checkpoint or str(
            repo_root()
            / "gear_sonic_deploy"
            / "planner"
            / "target_vel"
            / "V2"
            / "planner_sonic.onnx"
        )
        # Built here, not on first entry to teleop mode. Deferring it put a 1.7 s stall in the
        # control loop at the moment the operator switches -- 1.25 s to create the onnxruntime
        # session and 0.49 s for the first graph run, together ~87 control steps at 50 Hz. It
        # costs ~1 GB of GPU memory that an operator who never leaves tracking mode does not use,
        # which is the price of not stalling the one who does.
        #
        # The graph is a download rather than a repo artefact, so a missing file must not take
        # down environments that would never have entered teleop mode. In that one case the error
        # is still deferred to first use, where it can name the mode that needed it.
        self._planner_checkpoint = checkpoint
        self._planner: SonicVelocityPlanner | None = None
        try:
            self._planner = self._build_planner()
        except FileNotFoundError:
            print(
                "[SONIC] velocity planner not found; walking mode will fail if selected. "
                f"Expected at {checkpoint}",
                flush=True,
            )
        self._planner_clip = int(cfg.planner_clip)

        lower_ids, _ = self._asset.find_joints(list(LOWER_BODY_JOINTS), preserve_order=True)
        self._lower_body_ids = torch.as_tensor(lower_ids, device=self.device)
        #: Trajectory currently being played, and how far into it we are. The plan is addressed
        #: by *time*, not by frame index: it is generated at 30 Hz, resampled to SONIC's 50 Hz
        #: reference clock, and consumed at whatever the environment's control rate is.
        #: Raw right-stick turn command for this step. Drives the planner's heading in teleop
        #: mode (integrated upstream in the retargeter) and swings the XR anchor in smpl mode,
        #: where there is no planner to face anywhere.
        self._turn_command = 0.0
        self._anchor_pan_yaw = 0.0
        #: Whether the operator is asking to move at all. Comes from the PICO port's deadzoned
        #: stick magnitude, not from ``target_vel`` -- that is the ``-1.0`` "clip default"
        #: sentinel every step and says nothing about intent.
        self._moving = False
        #: The operator's tracked pelvis in the anchor frame, from the reference. Used to place the
        #: anchor so the operator -- not the anchor -- ends up on the robot.
        self._operator_root = np.zeros(3, dtype=np.float32)
        self._operator_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # xyzw
        #: The operator's pose in the anchor frame at the instant walking mode was entered, in USD
        #: axes. Held constant afterwards so the anchor rides the robot while the operator keeps
        #: their own local freedom -- the same relationship Isaac Lab's ``FOLLOW_PRIM`` gives.
        self._entry_operator_pos: np.ndarray | None = None
        self._entry_operator_yaw = 0.0
        #: Heading-alignment state for the ``teleop`` anchor-orientation observation, mirroring
        #: ``ComputeApplyDeltaHeading`` (``g1_deploy_onnx_ref.cpp:583-601``) applied to the
        #: *planner's own motion* rather than the operator's tracked pose -- that source is
        #: correct only for ``smpl`` mode. ``_teleop_heading_anchor`` is the robot's heading at
        #: teleop-mode entry (latched once, like the C++ side's ``reinitialize_heading_`` event);
        #: ``_motion_start_quat`` is the current plan's own first-frame root orientation,
        #: re-latched every time a fresh trajectory starts (``current_frame_ == 0`` there).
        self._teleop_heading_anchor = torch.zeros(self.num_envs, 4, device=self.device)
        self._teleop_heading_anchor[:, 0] = 1.0
        self._motion_start_quat = self._teleop_heading_anchor.clone()
        self._motion: PlannerMotion | None = None
        self._plan_time = 0.0
        self._since_replan = 0.0
        self._last_command: np.ndarray | None = None
        #: Sample times of the encoder's look-ahead window, in seconds from the current plan time.
        self._reference_offsets = (
            np.arange(TELEOP_REFERENCE_FRAMES, dtype=np.float32) * TELEOP_REFERENCE_STRIDE_S
        )
        self._qpos_scratch = np.zeros(PLANNER_QPOS_DIM, dtype=np.float32)
        #: Rolling measured-pose history feeding the planner's context.
        #:
        #: Owned here rather than inside the planner so it stays current in **every** mode. The
        #: planner is built lazily on first entry to teleop mode, so a history kept inside it
        #: would only advance while walking: re-entering teleop after an interval of full-body
        #: tracking would then condition the first plan on wherever the robot was when the
        #: operator last walked, and being autoregressive, that error would propagate through the
        #: whole horizon.
        self._qpos_history = np.zeros((PLANNER_CONTEXT_FRAMES, PLANNER_QPOS_DIM), dtype=np.float32)
        self._qpos_seeded = False
        self._mode = ENCODER_MODE_SMPL
        self._debug_forced_yaw = 0.0  # TEMP DEBUG - remove

        #: The environment's control period. Prefer Isaac Lab's canonical ``step_dt`` over
        #: recomputing ``sim.dt * decimation`` so there is one authority for it.
        #:
        #: Distinct from three other clocks that must not be conflated: physics (``sim.dt``),
        #: render (``sim.dt * render_interval``, 100 Hz here and irrelevant to control), the
        #: planner's native 30 Hz, and SONIC's own 50 Hz reference rate.
        self._control_dt = float(getattr(env, "step_dt", 0.0)) or float(
            getattr(getattr(env.cfg, "sim", None), "dt", 0.005)
        ) * float(getattr(env.cfg, "decimation", 4))
        if abs(self._control_dt - SONIC_REFERENCE_DT) > 1e-6:
            # SONIC's reference windows are defined in ticks of its own 50 Hz clock, and its
            # decoder was trained at that rate. Running the environment at a different control
            # rate would silently reinterpret every temporal observation, so refuse rather than
            # quietly change the model's semantics. The planner path resamples by time and would
            # cope; the policy itself is what does not.
            raise ValueError(
                f"SONIC requires a {1 / SONIC_REFERENCE_DT:.0f} Hz control rate "
                f"(step_dt = {SONIC_REFERENCE_DT}), but this environment steps at "
                f"{self._control_dt:.5f} s ({1 / self._control_dt:.1f} Hz). "
                "Set sim.dt and decimation so their product is 0.02 -- e.g. 1/200 with "
                "decimation 4, or 1/100 with decimation 2."
            )

        from gear_sonic.lab_teleop.assets.g1_sonic import G1_ISAACLAB_TO_MUJOCO_MAPPING

        # A **gather** index: ``mujoco[i] = isaaclab[isaaclab_to_mujoco[i]]``. Reading it as a
        # scatter silently permutes every joint -- it puts right_shoulder_pitch in MuJoCo slot 2,
        # where left_hip_yaw belongs -- and the planner then conditions on a pose the robot is not
        # in. Verified by resolving the mapping to names: under the gather reading MuJoCo slots
        # 0-11 come out as the left leg then the right leg, which is the canonical G1 ordering.
        self._isaaclab_to_mujoco = np.asarray(
            G1_ISAACLAB_TO_MUJOCO_MAPPING["isaaclab_to_mujoco_dof"], dtype=np.int64
        )
        #: MuJoCo slots of :data:`LOWER_BODY_JOINTS`, for selecting legs out of a planner pose.
        #: The plan speaks MuJoCo order, so the Isaac Lab indices must be translated -- indexing
        #: MuJoCo-ordered data with Isaac Lab indices is the same class of error as the one
        #: :data:`LOWER_BODY_ISAACLAB_INDICES` documents.
        _isaac_slot_of_mujoco = np.argsort(self._isaaclab_to_mujoco)
        self._lower_indices_mujoco = _isaac_slot_of_mujoco[
            np.asarray(LOWER_BODY_ISAACLAB_INDICES, dtype=np.int64)
        ]

        # Anchor handling. The live XrCfg is the one held by IsaacTeleopCfg, **not** env.cfg.xr:
        # @configclass copies the latter, so mutating it moves nothing and fails silently.
        # XrAnchorManager and XrAnchorSynchronizer both store this object by reference, and
        # sync_headset_to_anchor() reads anchor_pos/anchor_rot from it every frame, so writing
        # here repositions the operator's frame with no prim writes or carb settings.
        teleop_cfg = getattr(getattr(env, "cfg", None), "isaac_teleop", None)
        self._xr_cfg = getattr(teleop_cfg, "xr_cfg", None)
        self._anchor_z = float(self._xr_cfg.anchor_pos[2]) if self._xr_cfg is not None else 0.0
        # Captured so reset can put the operator back where the episode started. Without this the
        # anchor stays wherever teleop mode last dragged it, and because reset also returns the
        # mode to smpl it then stays there: the robot respawns at the origin while the operator is
        # left standing metres away looking at empty floor, which reads as "reset did nothing".
        self._initial_anchor_pos = tuple(self._xr_cfg.anchor_pos) if self._xr_cfg else None
        self._initial_anchor_rot = tuple(self._xr_cfg.anchor_rot) if self._xr_cfg else None
        self._anchor_yaw: float | None = None
        #: World-frame locomotion command for the current step, used to pan the anchor in smpl
        #: mode. Held here because the anchor update runs before the mode branch consumes it.
        self._pan_command: np.ndarray | None = None
        self._prev_mode = ENCODER_MODE_SMPL
        #: ``None`` until the first action arrives, so the first frame always applies and logs.
        self._ground_visible: bool | None = None

    @property
    def action_dim(self) -> int:
        """Width of the modal action: reference, mode, locomotion command."""
        return SONIC_MODAL_ACTION_DIM

    @property
    def mode(self) -> int:
        """Encoder mode currently selected by the operator."""
        return self._mode

    def reset(self, env_ids=None) -> None:  # noqa: ANN001
        """Clear rolling state, including the planner's context and pending plan.

        The planner must be reset or the first frames of a new episode continue the previous
        episode's gait, and its context conditions the next plan on a pose the robot no longer has.
        """
        super().reset(env_ids)
        if self._planner is not None:
            self._planner.reset()
        self._mode = ENCODER_MODE_SMPL
        self._prev_mode = ENCODER_MODE_SMPL
        self._anchor_yaw = None
        self._pan_command = None
        self._restore_anchor()
        self._qpos_history[:] = 0.0
        self._qpos_seeded = False
        self._motion = None
        self._plan_time = 0.0
        self._since_replan = 0.0
        self._last_command = None
        self._turn_command = 0.0
        self._anchor_pan_yaw = 0.0
        self._moving = False
        self._operator_root[:] = 0.0
        self._operator_quat[:] = (0.0, 0.0, 0.0, 1.0)
        self._entry_operator_pos = None
        self._entry_operator_yaw = 0.0
        self._debug_forced_yaw = 0.0  # TEMP DEBUG - remove
        self._teleop_heading_anchor[:, :] = 0.0
        self._teleop_heading_anchor[:, 0] = 1.0
        self._motion_start_quat = self._teleop_heading_anchor.clone()
        self._ground_visible = None

    def _build_planner(self) -> SonicVelocityPlanner:
        """Create the velocity planner, warmed up and ready to run inside the control loop."""
        return SonicVelocityPlanner(
            checkpoint_path=self._planner_checkpoint,
            device=self._policy_device,
        )

    def _ensure_planner(self) -> SonicVelocityPlanner:
        """Return the planner, building it now only if construction found no graph to load.

        Raises:
            FileNotFoundError: If the planner graph is still absent, naming the download.
        """
        if self._planner is None:
            self._planner = self._build_planner()
        return self._planner

    @staticmethod
    def _mode_name(mode: int) -> str:
        """Human-readable encoder mode, for operator-facing logs."""
        return {ENCODER_MODE_TELEOP: "teleop", ENCODER_MODE_SMPL: "smpl"}.get(mode, str(mode))

    def _apply_ground_visibility(self, visible: bool) -> None:
        """Show or hide the ground plane, and say so.

        The operator has no other feedback about which mode they are in, and mode changes are only
        visible through behaviour -- legs starting to walk, or the anchor snapping. Logging the
        transition makes a session where "nothing happened" tell the two failure modes apart: the
        toggle never firing, versus the toggle firing and the mode behaving wrongly.
        """
        if self._ground_visible is not None and visible == self._ground_visible:
            return
        self._ground_visible = visible
        try:
            from isaaclab.sim.utils import set_prim_visibility
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(GROUND_PLANE_PRIM_PATH) if stage is not None else None
            if prim is not None and prim.IsValid():
                set_prim_visibility(prim, visible)
                print(f"[SONIC] ground plane {'shown' if visible else 'hidden'}", flush=True)
                return
        except Exception as exc:  # noqa: BLE001 - outside Isaac Sim there is no stage to touch
            print(f"[SONIC] ground plane toggle unavailable: {exc}", flush=True)
            return
        print(
            f"[SONIC] ground plane prim {GROUND_PLANE_PRIM_PATH!r} not found; nothing toggled",
            flush=True,
        )

    @staticmethod
    def _yaw_of(quat_wxyz: np.ndarray) -> float:
        """Yaw about +Z of a wxyz quaternion."""
        w, x, y, z = (float(v) for v in quat_wxyz)
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _restore_anchor(self) -> None:
        """Return the XR anchor to the pose it had when the episode began.

        Called on reset. The robot respawns at its initial pose, so the operator's frame has to as
        well -- otherwise the two are separated by however far the robot travelled before the
        reset, which looks from inside the headset like the reset did not happen.
        """
        if self._xr_cfg is None or self._initial_anchor_pos is None:
            return
        moved = tuple(self._xr_cfg.anchor_pos) != self._initial_anchor_pos
        self._xr_cfg.anchor_pos = self._initial_anchor_pos
        self._xr_cfg.anchor_rot = self._initial_anchor_rot
        if moved:
            print(
                f"[SONIC] reset: XR anchor restored to {self._initial_anchor_pos}",
                flush=True,
            )

    def _update_anchor(self, mode: int) -> None:
        """Move the operator's XR anchor according to the active mode.

        Three states, mirroring how ``locomanip_pick_place`` anchors to the robot while it walks:

        * ``smpl`` -- leave the anchor untouched, so it stays wherever it was. Operator
          displacement then maps one-to-one into the world, which is what drives the gait.
        * entering ``teleop`` -- snap the anchor to the robot's current pose. Without this the
          operator would be commanding a robot metres away from their own frame.
        * ``teleop`` -- follow the robot, so the operator rides along and never walks out of the
          play space.

        Height is held at its configured value rather than followed: tracking the robot's z would
        feed the gait's vertical bob into the headset. Yaw is followed alone and smoothed, for the
        same reason Isaac Lab's own ``FOLLOW_PRIM_SMOOTHED`` does both.

        The anchor is placed so the **operator** lands on the robot, not so the anchor does. Those
        differ by however far room-scale walking has carried the operator from the anchor origin,
        which in ``smpl`` mode is the whole point and can be metres. Putting the anchor itself on
        the robot leaves that offset intact, so the operator arrives beside the robot rather than
        in it. The tracked pelvis reaches us through the reference because only the retargeter sees
        raw tracker poses.
        """
        if self._xr_cfg is None:
            return
        if mode != ENCODER_MODE_TELEOP:
            self._pan_anchor()
            return
        from gear_sonic.lab_teleop.mdp.actions import isaaclab_quat_to_wxyz

        data = self._asset.data
        root_pos = data.root_pos_w.torch[0].detach().float().cpu().numpy()
        base_quat = isaaclab_quat_to_wxyz(data.root_quat_w.torch)[0].detach().float().cpu().numpy()
        yaw = self._yaw_of(base_quat)

        if self._prev_mode != ENCODER_MODE_TELEOP or self._entry_operator_pos is None:
            # Capture where the operator stands and which way they face, once. Holding these
            # constant afterwards makes the anchor ride the robot while leaving the operator free
            # to move locally; recomputing them every step would pin them rigidly to the robot and
            # cancel their own motion.
            self._entry_operator_pos = OXR_TO_USD @ self._operator_root
            self._entry_operator_yaw = self._operator_yaw_usd()
            self._anchor_yaw = None

        # Align the operator's heading with the robot's, not the anchor's. The runtime composes
        # ``R_anchor @ R_oxr_to_usd @ R_pelvis``, so the anchor has to carry the *difference*:
        # ignoring the operator's own facing leaves them mirrored, standing across from the robot
        # looking at it rather than standing in it.
        target_yaw = yaw - self._entry_operator_yaw
        if self._anchor_yaw is None:
            self._anchor_yaw = target_yaw  # snap on entry, no blend from a stale heading
        else:
            dt = self._control_dt
            alpha = 1.0 - float(np.exp(-dt / max(ANCHOR_YAW_SMOOTHING_TIME, 1e-6)))
            delta = (target_yaw - self._anchor_yaw + np.pi) % (2.0 * np.pi) - np.pi
            self._anchor_yaw += float(np.clip(alpha, 0.05, 1.0)) * delta

        half = 0.5 * self._anchor_yaw
        cos_y, sin_y = float(np.cos(self._anchor_yaw)), float(np.sin(self._anchor_yaw))
        ox, oy = float(self._entry_operator_pos[0]), float(self._entry_operator_pos[1])
        offset_x = cos_y * ox - sin_y * oy
        offset_y = sin_y * ox + cos_y * oy
        self._xr_cfg.anchor_pos = (
            float(root_pos[0]) - offset_x,
            float(root_pos[1]) - offset_y,
            self._anchor_z,
        )
        # XrCfg.anchor_rot is XYZW, not WXYZ (xr_anchor_manager.py:110).
        self._xr_cfg.anchor_rot = (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))

    def _operator_yaw_usd(self) -> float:
        """The operator's heading in USD world axes, before the anchor rotation is applied.

        Their tracked orientation arrives in raw OpenXR axes, so it has to go through the same
        ``R_oxr_to_usd`` the runtime uses before a yaw can be read off it.
        """
        from scipy.spatial.transform import Rotation as sRot

        rot = sRot.from_quat(np.asarray(self._operator_quat, dtype=np.float64)).as_matrix()
        converted = OXR_TO_USD @ rot
        return float(np.arctan2(converted[1, 0], converted[0, 0]))

    def _pan_anchor(self) -> None:
        """Slide the anchor across the ground plane from the operator's stick, in ``smpl`` mode.

        Safe precisely because the reference carries **no root translation**: it is root-local
        joints plus a root quaternion, and SONIC infers locomotion from body pose rather than from
        a commanded displacement (``sonic_fullbody_retargeter.py:286``). Moving the anchor
        therefore changes only where the operator sees the world from; it cannot inject a spurious
        displacement into the reference.

        The left stick is free in this mode -- it only drives the planner in ``teleop`` -- so the
        same control walks the robot there and pans the operator here.

        Height and rotation are untouched: this is a translation on the ground plane, so the
        operator's sense of up and of facing are preserved.
        """
        dt = self._control_dt

        # Right stick: swing the operator's viewpoint. In teleop mode this same stick turns the
        # robot, so keeping it as "turn" here means the control does one thing in the operator's
        # head regardless of mode -- the difference is only whether there is a robot to point.
        if abs(self._turn_command) >= PICO_JOYSTICK_DEADZONE:
            # Same gain and sign as the robot: dyaw = 1.5 * (-rx) * dt.
            self._anchor_pan_yaw += PICO_YAW_GAIN * -self._turn_command * dt
            half = 0.5 * self._anchor_pan_yaw
            # XrCfg.anchor_rot is XYZW, not WXYZ (xr_anchor_manager.py:110).
            self._xr_cfg.anchor_rot = (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))

        speed = self.cfg.anchor_pan_speed
        if speed <= 0.0 or self._pan_command is None:
            return
        if not self._moving:
            return
        target_vel = float(np.hypot(self._pan_command[1], self._pan_command[2]))
        # Slide along the operator's own facing, so "forward" stays forward after turning.
        cos_y, sin_y = np.cos(self._anchor_pan_yaw), np.sin(self._anchor_pan_yaw)
        dx, dy = float(self._pan_command[1]), float(self._pan_command[2])
        direction = (cos_y * dx - sin_y * dy, sin_y * dx + cos_y * dy)
        step = target_vel * speed * dt
        pos = self._xr_cfg.anchor_pos
        self._xr_cfg.anchor_pos = (
            float(pos[0] + direction[0] * step),
            float(pos[1] + direction[1] * step),
            float(pos[2]),
        )

    def _to_world_directions(self, command: np.ndarray) -> np.ndarray:
        """Rotate the operator's commanded directions into the world frame.

        The retargeter emits directions in the operator's own frame, because the reference stream
        is anchor-local: this pipeline declares no ``world_T_anchor`` leaf, so tracker poses are
        never rebased to world. ``smpl`` mode is unaffected -- ``apply_delta_heading`` is latched
        at engage and absorbs the offset -- but the planner mixes ``facing_direction`` with a
        world-frame ``context_mujoco_qpos``, so the two must be expressed alike.

        The conversion uses the latched alignment, which is derived from robot state, and is
        therefore done here rather than upstream.

        Args:
            command: ``(8,)`` operator command with directions in the operator frame.

        Returns:
            A copy with both direction vectors rotated into world.
        """
        yaw = self._yaw_of(self._apply_delta_heading[0].detach().float().cpu().numpy())
        if abs(yaw) < 1e-9:
            return command
        cos_y, sin_y = float(np.cos(yaw)), float(np.sin(yaw))
        out = command.copy()
        for start in (1, 4):
            x, y = float(command[start]), float(command[start + 1])
            out[start] = cos_y * x - sin_y * y
            out[start + 1] = sin_y * x + cos_y * y
        return out

    def _robot_qpos(self) -> np.ndarray:
        """Measured pose as the planner's 36-wide MuJoCo-order ``qpos``."""
        from gear_sonic.lab_teleop.mdp.actions import isaaclab_quat_to_wxyz

        data = self._asset.data
        root_pos = data.root_pos_w.torch[0].detach().float().cpu().numpy()
        root_quat = isaaclab_quat_to_wxyz(data.root_quat_w.torch)[0].detach().float().cpu().numpy()
        joints = data.joint_pos.torch[0, self._joint_ids].detach().float().cpu().numpy()

        self._qpos_scratch[:3] = root_pos
        self._qpos_scratch[3:7] = root_quat
        # The planner speaks MuJoCo joint order; SONIC speaks Isaac Lab order.
        self._qpos_scratch[7:] = joints[self._isaaclab_to_mujoco]
        return self._qpos_scratch

    def _track_robot_pose(self) -> None:
        """Append the measured pose to the planner context history. Runs every step, every mode."""
        frame = self._robot_qpos()
        if not self._qpos_seeded:
            self._qpos_history[:] = frame
            self._qpos_seeded = True
        else:
            self._qpos_history[:-1] = self._qpos_history[1:]
            self._qpos_history[-1] = frame

    def _enter_teleop(self) -> None:
        """Build the first trajectory for a fresh entry into walking mode.

        Mirrors upstream ``Initialize``: canonical context, an IDLE zero-movement plan, resampled
        onto the reference clock -- and only then is the teleop encoder given anything. The
        operator must never see a partially initialized reference, so this runs to completion
        before the first teleop observation is assembled.
        """
        planner = self._ensure_planner()
        joints_mujoco = self._robot_qpos()[7:]
        self._motion = planner.initialize_from_robot(joints_mujoco)
        self._plan_time = 0.0
        self._since_replan = 0.0
        self._last_command = None
        self._latch_motion_start(latch_heading_anchor=True)
        if self.cfg.debug_transitions:
            self._log_transition()

    def _latch_motion_start(self, latch_heading_anchor: bool) -> None:
        """Re-anchor the ``teleop`` orientation observation to a fresh trajectory.

        Args:
            latch_heading_anchor: Also re-latch the robot-heading side of the alignment. Only
                true on teleop-mode entry: mirrors ``reinitialize_heading_``
                (``g1_deploy_onnx_ref.cpp:559-570``), which fires on an operator-triggered reset,
                not on every replan.
        """
        from gear_sonic.lab_teleop.mdp.actions import isaaclab_quat_to_wxyz

        self._motion_start_quat[:] = torch.as_tensor(
            self._motion.qpos[0, 3:7], device=self.device
        ).unsqueeze(0)
        if latch_heading_anchor:
            base_quat = isaaclab_quat_to_wxyz(self._asset.data.root_quat_w.torch)
            self._teleop_heading_anchor[:] = base_quat

    def _log_transition(self) -> None:
        """Trace the SMPL -> teleop handoff, so a bad transition is measurable rather than felt."""
        robot_mujoco = self._robot_qpos()
        robot_legs = robot_mujoco[7:][self._lower_indices_mujoco]
        times = self._plan_time + self._reference_offsets
        pos, vel = self._motion.sample_joints(times, self._lower_indices_mujoco)
        delta = pos[0] - robot_legs
        print(
            "[SONIC] smpl -> teleop handoff\n"
            f"    clocks: control {self._control_dt * 1e3:.1f} ms | "
            f"SONIC reference {SONIC_REFERENCE_DT * 1e3:.1f} ms | "
            f"planner native {1e3 / 30.0:.1f} ms\n"
            f"    look-ahead window: {times[0] - self._plan_time:.2f}..."
            f"{times[-1] - self._plan_time:.2f} s in {len(times)} samples\n"
            f"    plan: {len(self._motion.qpos)} frames, {self._motion.duration:.2f} s\n"
            f"    first planned legs vs measured: L2 {np.linalg.norm(delta):.4f} rad, "
            f"max |delta| {np.abs(delta).max():.4f} rad\n"
            f"    planned leg velocity: max |qvel| {np.abs(vel).max():.3f} rad/s",
            flush=True,
        )

    def _canonical_command(self, command: np.ndarray) -> np.ndarray:
        """The command as the planner receives it.

        A pass-through. The retargeter already produces exactly what upstream sends: a zero
        movement vector when the stick is inside the deadzone, and the accumulated heading
        regardless (``pico_manager_thread_server.py:1800-1830``).

        Nothing is substituted from measured robot state here. Doing so cost the operator the
        right stick entirely: with the left stick centred, facing was overwritten with the robot's
        own measured heading every step, so a commanded turn could never survive to reach the
        planner. Facing is a *command*, and it holds while the operator is standing still --
        that is how you turn on the spot.

        Args:
            command: ``(8,)`` operator command in world frame.

        Returns:
            The same command. Kept as a named step so the planner call, the change detection and
            ``_last_command`` provably share one representation.
        """
        return command

    def _replan_needed(self, command: np.ndarray) -> bool:
        """Whether this control step should generate a new trajectory.

        Time-based, mirroring the upstream planner thread rather than counting environment frames:
        always replan when the operator's command changes, replan periodically only while actually
        moving, and replan when the current plan is about to run out
        (``planner_onnx.md:362-385``).

        Args:
            command: Canonical command from :meth:`_canonical_command`.
        """
        if self._motion is None or self._last_command is None:
            return True
        if not np.allclose(command, self._last_command, atol=1e-4):
            return True
        # Keep enough trajectory ahead to fill the encoder's look-ahead window. Past the end
        # PlannerMotion repeats its final pose, so this is what stops the reference going static
        # rather than what stops an out-of-range read.
        if self._plan_time + float(self._reference_offsets[-1]) >= self._motion.duration:
            return True
        return self._moving and self._since_replan >= PLANNER_PERIODIC_REPLAN_S

    def _advance_planner(self, command: np.ndarray, dt: float) -> None:
        """Advance plan time by one control step, replanning when required.

        Args:
            command: ``(8,)`` operator command in world frame.
            dt: The environment's control period, in seconds.
        """
        planner = self._ensure_planner()
        canonical = self._canonical_command(command)

        # TEMP DEBUG - remove: bypass the joystick's facing entirely and force a full rotation
        # every 2 seconds, to isolate whether ANY commanded facing_direction can turn the robot.
        self._debug_forced_yaw = (self._debug_forced_yaw + dt * (2.0 * np.pi / 2.0)) % (2.0 * np.pi)
        canonical = canonical.copy()
        canonical[4] = float(np.cos(self._debug_forced_yaw))
        canonical[5] = float(np.sin(self._debug_forced_yaw))

        if self._motion is None:
            # Fresh entry into walking mode. Build the idle trajectory and stop there: storing the
            # canonical command means the next step compares like with like and lets that
            # trajectory actually play, instead of replanning it away before it delivers a frame.
            self._enter_teleop()
            # Copy, never alias. ``actions`` is the ActionManager's reused buffer, so storing a
            # view means `_last_command` silently tracks the live command and the change
            # detection below can never fire -- the operator's turn reaches nothing.
            self._last_command = np.array(canonical, dtype=np.float32)
            self._since_replan = 0.0
            return

        if self._replan_needed(canonical):
            # Context comes from the plan being played, not from measured robot state, so
            # successive trajectories are continuous with one another. Sampled a short look-ahead
            # in, because the new plan takes effect a step or two from now.
            planner.context_from_motion(self._motion, self._plan_time + PLANNER_LOOKAHEAD_S)
            # Upstream drops to the idle clip whenever the stick is centred
            # (``pico_manager_thread_server.py:1806``); the walk clip at zero throttle is a
            # different motion from standing still.
            clip = self._planner_clip if self._moving else PLANNER_CLIP_IDLE
            from gear_sonic.lab_teleop.mdp.actions import isaaclab_quat_to_wxyz  # TEMP DEBUG - remove

            robot_yaw = self._yaw_of(  # TEMP DEBUG - remove
                isaaclab_quat_to_wxyz(self._asset.data.root_quat_w.torch)[0]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            print(  # TEMP DEBUG - remove
                f"[DEBUG-TURN] replan: clip={clip} facing=({canonical[4]:.3f},{canonical[5]:.3f}) "
                f"move=({canonical[1]:.3f},{canonical[2]:.3f}) robot_yaw={robot_yaw:.3f} "
                f"anchor_yaw={self._anchor_yaw}",
                flush=True,
            )
            self._motion = planner.plan(canonical, mode=clip)
            self._plan_time = 0.0
            self._since_replan = 0.0
            self._last_command = np.array(canonical, dtype=np.float32)
            self._latch_motion_start(latch_heading_anchor=False)
        else:
            self._plan_time += dt
            self._since_replan += dt

    def _clear_teleop_slots(self) -> None:
        """Zero the ``teleop`` blocks when the encoder returns to ``smpl`` mode.

The checkpoint's contract is that terms outside the active mode are zero -- the deploy stack
        sends them as zeros rather than omitting them. The encoder observation is a single
        persistent buffer and ``fill_smpl_encoder_obs`` rewrites only the ``smpl`` blocks, so
        without this every ``smpl`` frame after a walking excursion would carry the last planner
        window and anchor orientation into the encoder.

        Called on the transition rather than every step: nothing writes these slots while
        ``smpl`` mode is active, so one pass is enough.
        """
        obs = self._policy.encoder_obs
        for block in (
            TELEOP_ANCHOR_ORI,
            TELEOP_LOWER_POS,
            TELEOP_LOWER_VEL,
            TELEOP_VR3_POS,
            TELEOP_VR3_ORN,
        ):
            obs[:, block] = 0.0

    def _fill_lower_body(self) -> None:
        """Write the encoder's 0.9 s look-ahead window of planned leg motion.

        Two contiguous frame-major blocks -- all ten frames of positions, then all ten of
        velocities -- sampled at ``current plan time + i * 0.1 s``. ``10frame_step5`` names a
        window into the plan's *future*: ten samples 0.1 s apart on the checkpoint's own 50 Hz
        clock, spanning 0.9 s ahead (``observation_config.md:96-99``).
        """
        obs = self._policy.encoder_obs
        times = self._plan_time + self._reference_offsets
        pos, vel = self._motion.sample_joints(times, self._lower_indices_mujoco)
        obs[:, TELEOP_LOWER_POS] = torch.as_tensor(
            pos.reshape(-1), device=self.device
        ).unsqueeze(0)
        obs[:, TELEOP_LOWER_VEL] = torch.as_tensor(
            vel.reshape(-1), device=self.device
        ).unsqueeze(0)

    def _write_mode_slots(self, mode: int) -> None:
        """Write the encoder's mode id and one-hot, which are per-step under mode switching."""
        from gear_sonic.lab_teleop.mdp.sonic_policy import SmplEncoderSlots

        obs = self._policy.encoder_obs
        obs[:, SmplEncoderSlots.MODE_ID] = float(mode)
        obs[:, SmplEncoderSlots.ENCODER_INDEX] = 0.0
        obs[:, SmplEncoderSlots.ENCODER_INDEX.start + mode] = 1.0

    def _fill_teleop_obs(self, reference: torch.Tensor, base_quat: torch.Tensor) -> None:
        """Populate the ``teleop`` observation block and zero the ``smpl`` one."""
        from gear_sonic.lab_teleop.mdp.sonic_policy import smpl_anchor_orientation

        obs = self._policy.encoder_obs
        variant = self._policy.variant
        obs[:, variant.smpl_joints] = 0.0
        obs[:, variant.smpl_anchor_ori] = 0.0
        obs[:, variant.wrist_joint_pos] = 0.0

        self._fill_lower_body()

        # Single-frame anchor orientation. Unlike smpl mode, the reference here is the SONIC
        # planner's own predicted trajectory (``self._motion``), not the operator's tracked pose:
        # per ``GatherMotionAnchorOrientationMutiFrame`` (``g1_deploy_onnx_ref.cpp:612-687``), the
        # "current_motion_" this observation reads from is the planner's plan in teleop mode and
        # the operator's kinematic retarget in smpl mode -- they are not interchangeable. Feeding
        # the operator's pose here left the commanded turn (``facing_direction``) unable to reach
        # this observation at all: the anchor tracked the operator's own body orientation instead.
        from gear_sonic.isaac_utils.rotations import calc_heading_quat, calc_heading_quat_inv, quat_mul

        motion_root_quat = torch.as_tensor(
            self._motion.sample_qpos(self._plan_time)[3:7], device=self.device
        ).unsqueeze(0)
        teleop_apply_delta_heading = quat_mul(
            calc_heading_quat(self._teleop_heading_anchor, w_last=False),
            calc_heading_quat_inv(self._motion_start_quat, w_last=False),
            w_last=False,
        )
        anchor = smpl_anchor_orientation(
            reference_root_quat=motion_root_quat[:, None, :],
            robot_base_quat=base_quat,
            apply_delta_heading=teleop_apply_delta_heading,
            orientation_mode=variant.orientation_mode,
        )
        obs[:, TELEOP_ANCHOR_ORI] = anchor.reshape(1, -1)

        # vr_3point: measured hand and neck poses relative to the operator's root, built by the
        # retargeter because it is the only node that sees the raw tracker poses. See
        # ``SonicFullBodyRetargeter._vr_three_point`` -- a port of what the real robot runs.
        obs[:, TELEOP_VR3_POS] = reference[:, SonicReferenceSlice.VR3_POS]
        obs[:, TELEOP_VR3_ORN] = reference[:, SonicReferenceSlice.VR3_ORN]

    def process_actions(self, actions: torch.Tensor) -> None:
        """Run one control step in whichever mode the operator selected.

        Args:
            actions: ``(num_envs, 105)`` ``[reference | mode | command | ground]``.

        Raises:
            ValueError: If the action width does not match the contract.
        """
        if actions.shape[-1] != SONIC_MODAL_ACTION_DIM:
            raise ValueError(
                f"expected a {SONIC_MODAL_ACTION_DIM}-wide action "
                f"({SONIC_REFERENCE_DIM} reference + 1 mode + {SONIC_PLANNER_COMMAND_DIM} "
                f"command + 1 ground_visible), got {actions.shape[-1]}"
            )

        actions = actions.to(self.device)
        reference = actions[:, :SONIC_REFERENCE_DIM]
        mode = int(actions[0, SONIC_REFERENCE_DIM].item())
        tail = actions[0, SONIC_REFERENCE_DIM + 1 :].detach().float().cpu().numpy()
        command = tail[:SONIC_PLANNER_COMMAND_DIM]
        ground_visible = bool(tail[SONIC_PLANNER_COMMAND_DIM] > 0.5)
        self._turn_command = float(tail[SONIC_PLANNER_COMMAND_DIM + 1])
        self._moving = bool(tail[SONIC_PLANNER_COMMAND_DIM + 2] > 0.5)
        self._operator_root[:] = (
            reference[0, SonicReferenceSlice.OPERATOR_ROOT_POS].detach().float().cpu().numpy()
        )
        self._operator_quat[:] = (
            reference[0, SonicReferenceSlice.OPERATOR_ROOT_QUAT].detach().float().cpu().numpy()
        )
        self._mode = mode if mode in (ENCODER_MODE_TELEOP, ENCODER_MODE_SMPL) else ENCODER_MODE_SMPL
        if self._mode != self._prev_mode:
            print(
                f"[SONIC] mode {self._mode_name(self._prev_mode)} -> "
                f"{self._mode_name(self._mode)}",
                flush=True,
            )
        self._apply_ground_visibility(ground_visible)

        # Tracked in every mode so a return to teleop plans from where the robot actually is.
        self._track_robot_pose()
        # Rotated once, up front: teleop feeds it to the planner and smpl pans the anchor with it,
        # and both want world frame.
        command = self._to_world_directions(command)
        self._pan_command = command
        self._update_anchor(self._mode)
        if self._mode == ENCODER_MODE_TELEOP:
            # Planner runs outside inference_mode: it is numpy/onnxruntime and reads articulation
            # state, which the surrounding context does not need to guard.
            self._advance_planner(command, self._control_dt)

        from gear_sonic.lab_teleop.mdp.actions import isaaclab_quat_to_wxyz

        with torch.inference_mode(), self._policy.compute_stream():
            with self._profiler.stage("total_process_actions"):
                with self._profiler.stage("reference_history_update"):
                    self._push_reference(reference)
                with self._profiler.stage("proprio_history_update"):
                    self._append_proprioception()

                base_quat = isaaclab_quat_to_wxyz(self._asset.data.root_quat_w.torch)
                with self._profiler.stage("encoder_obs_construction"):
                    if self._mode == ENCODER_MODE_SMPL:
                        if self._prev_mode != ENCODER_MODE_SMPL:
                            self._clear_teleop_slots()
                            # Drop the trajectory so re-entry re-initializes from the robot's
                            # pose at that moment rather than resuming a stale excursion.
                            self._motion = None
                        encoder_obs = self._build_encoder_obs()
                    else:
                        self._fill_teleop_obs(reference, base_quat)
                        encoder_obs = self._policy.encoder_obs
                    self._write_mode_slots(self._mode)
                with self._profiler.stage("encoder_inference"):
                    token = self._policy.encode(encoder_obs)

                with self._profiler.stage("decoder_obs_construction"):
                    proprio = self._history.flat(out=self._policy.decoder_proprio_view)
                with self._profiler.stage("decoder_inference"):
                    raw = self._policy.decode(token, proprio)

                with self._profiler.stage("output_postprocessing"):
                    self._raw_actions.copy_(raw)
                    torch.mul(self._raw_actions, self._scale, out=self._processed_actions)
                    self._processed_actions.add_(self._offset)
                    self._hold_default_if_untracked(reference)
        self._prev_mode = self._mode
