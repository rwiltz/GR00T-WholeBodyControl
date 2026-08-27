# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mode-switching SONIC action term: full-body tracking or stick-driven walking.

Extends :class:`~gear_sonic.lab_teleop.mdp.actions.SonicWholeBodyAction` with the operator's
encoder-mode selection, mirroring the real robot's teleop stack. The environment action becomes::

    [ sonic_reference(83) | mode(1) | locomotion_command(8) | ground_visible(1) ]   -> 93

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
    PLANNER_CLIP_WALK,
    PLANNER_CONTEXT_FRAMES,
    PLANNER_QPOS_DIM,
    SONIC_PLANNER_COMMAND_DIM,
    SonicVelocityPlanner,
)
from gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter import (
    SONIC_REFERENCE_DIM,
    SonicReferenceSlice,
)

__all__ = [
    "SONIC_MODAL_ACTION_DIM",
    "SonicModalWholeBodyAction",
    "SonicModalWholeBodyActionCfg",
]

#: ``[reference(83) | mode(1) | planner_command(8) | ground_visible(1)]``.
SONIC_MODAL_ACTION_DIM = SONIC_REFERENCE_DIM + 1 + SONIC_PLANNER_COMMAND_DIM + 1

#: Prim path of the scene's ground plane, toggled from the controller.
GROUND_PLANE_PRIM_PATH = "/World/GroundPlane"

#: Encoder mode ids, matching the checkpoint's ``encoder_modes``.
ENCODER_MODE_TELEOP = 1
ENCODER_MODE_SMPL = 2

#: Encoder slots ``teleop`` mode populates, recovered from the graph's ``Slice`` nodes.
TELEOP_ANCHOR_ORI = slice(644, 650)  # single frame, 6D rotation
TELEOP_LOWER_BODY = slice(650, 890)  # 10 frames x (12 pos + 12 vel)
TELEOP_VR3_POS = slice(890, 899)  # head + 2 hands, xyz
TELEOP_VR3_ORN = slice(899, 911)  # head + 2 hands, 4-wide each

#: Lower-body joints the ``teleop`` block carries, in Isaac Lab order.
LOWER_BODY_JOINTS = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)

#: Reference frames the ``teleop`` lower-body block spans.
TELEOP_REFERENCE_FRAMES = 10

#: SMPL joint indices, into the reference's 24-joint **root-local** block, for the three points
#: ``vr_3point`` carries. Order matches ``GatherVR3PointPosition``: left wrist, right wrist, head.
#:
#: The retargeter's ``_SMPL_L_WRIST_IDX = 19`` and friends index the root-*excluded* 21-joint
#: ``body_pose`` array, so the full-array indices are one higher.
VR3_SMPL_INDICES = (20, 21, 15)

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

    planner_replan_interval: int = 0
    """Frames consumed before re-planning. ``0`` consumes each plan fully."""


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
        # Constructed lazily on first entry to teleop mode. The planner's onnxruntime session
        # costs ~1 GB of GPU memory, and an operator who never leaves full-body tracking should
        # not pay for it -- which is what lets mode switching live in every environment rather
        # than in a separate variant.
        self._planner_checkpoint = checkpoint
        self._planner: SonicVelocityPlanner | None = None
        self._planner_clip = int(cfg.planner_clip)

        lower_ids, _ = self._asset.find_joints(list(LOWER_BODY_JOINTS), preserve_order=True)
        self._lower_body_ids = torch.as_tensor(lower_ids, device=self.device)
        #: Rolling window of planned lower-body pos+vel, oldest first, mirroring the encoder's
        #: expectation of a 10-frame history.
        self._lower_history = np.zeros(
            (TELEOP_REFERENCE_FRAMES, 2 * len(LOWER_BODY_JOINTS)), dtype=np.float32
        )
        self._prev_lower_pos: np.ndarray | None = None
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

        from gear_sonic.lab_teleop.assets.g1_sonic import G1_ISAACLAB_TO_MUJOCO_MAPPING

        self._isaaclab_to_mujoco = np.asarray(
            G1_ISAACLAB_TO_MUJOCO_MAPPING["isaaclab_to_mujoco_dof"], dtype=np.int64
        )

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
        self._lower_history[:] = 0.0
        self._prev_lower_pos = None
        self._mode = ENCODER_MODE_SMPL
        self._prev_mode = ENCODER_MODE_SMPL
        self._anchor_yaw = None
        self._restore_anchor()
        self._qpos_history[:] = 0.0
        self._qpos_seeded = False
        self._ground_visible = None

    def _ensure_planner(self) -> SonicVelocityPlanner:
        """Construct the velocity planner on first use.

        Raises:
            FileNotFoundError: If the planner graph is absent. Raised here rather than at env
                construction so environments that never enter teleop mode do not require it.
        """
        if self._planner is None:
            print("[SONIC] building the velocity planner (first entry to teleop mode)", flush=True)
            self._planner = SonicVelocityPlanner(
                checkpoint_path=self._planner_checkpoint,
                device=self._policy_device,
                replan_interval=self.cfg.planner_replan_interval,
            )
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
        """
        if self._xr_cfg is None or mode != ENCODER_MODE_TELEOP:
            return
        from gear_sonic.lab_teleop.mdp.actions import isaaclab_quat_to_wxyz

        data = self._asset.data
        root_pos = data.root_pos_w.torch[0].detach().float().cpu().numpy()
        base_quat = isaaclab_quat_to_wxyz(data.root_quat_w.torch)[0].detach().float().cpu().numpy()
        yaw = self._yaw_of(base_quat)

        if self._prev_mode != ENCODER_MODE_TELEOP or self._anchor_yaw is None:
            self._anchor_yaw = yaw  # snap on entry, no blend from a stale heading
        else:
            dt = float(getattr(self._env, "step_dt", 0.02))
            alpha = 1.0 - float(np.exp(-dt / max(ANCHOR_YAW_SMOOTHING_TIME, 1e-6)))
            delta = (yaw - self._anchor_yaw + np.pi) % (2.0 * np.pi) - np.pi
            self._anchor_yaw += float(np.clip(alpha, 0.05, 1.0)) * delta

        half = 0.5 * self._anchor_yaw
        self._xr_cfg.anchor_pos = (float(root_pos[0]), float(root_pos[1]), self._anchor_z)
        # XrCfg.anchor_rot is XYZW, not WXYZ (xr_anchor_manager.py:110).
        self._xr_cfg.anchor_rot = (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))

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
        self._qpos_scratch[7:][self._isaaclab_to_mujoco] = joints
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

    def _advance_planner(self, command: np.ndarray) -> None:
        """Run the planner and push one frame of lower-body pos+vel into the history."""
        planner = self._ensure_planner()
        planner.set_context(self._qpos_history)

        if float(command[0]) <= 1e-4:
            # No operator input: continue from measured motion rather than a stale command, as the
            # reference implementation does. Needs robot state, hence here rather than upstream.
            movement, facing = SonicVelocityPlanner.idle_directions(self._qpos_history)
            command = command.copy()
            command[1:4] = movement
            command[4:7] = facing

        frame = planner.next_frame(command, mode=self._planner_clip)
        # Planned joints are MuJoCo-ordered; take the lower body back in Isaac Lab order.
        planned_joints = frame[7:][self._isaaclab_to_mujoco]
        lower_pos = planned_joints[: len(LOWER_BODY_JOINTS)]

        dt = self._env.step_dt if hasattr(self._env, "step_dt") else 0.02
        if self._prev_mode != ENCODER_MODE_TELEOP:
            # Differencing against the last frame of a previous excursion would manufacture a huge
            # spurious velocity across the gap.
            self._prev_lower_pos = None
        if self._prev_lower_pos is None:
            lower_vel = np.zeros_like(lower_pos)
        else:
            lower_vel = (lower_pos - self._prev_lower_pos) / max(dt, 1e-6)
        self._prev_lower_pos = lower_pos.copy()

        if self._prev_mode != ENCODER_MODE_TELEOP:
            # Entering teleop: backfill the whole window with this frame rather than continuing a
            # window whose older frames come from a previous walk, possibly minutes ago. Mixing
            # them would hand the encoder a reference trajectory that jumps discontinuously
            # between two unrelated excursions. Same priming semantics as the proprioception
            # history uses after a reset.
            self._lower_history[:, : len(LOWER_BODY_JOINTS)] = lower_pos
            self._lower_history[:, len(LOWER_BODY_JOINTS) :] = lower_vel
        else:
            self._lower_history[:-1] = self._lower_history[1:]
            self._lower_history[-1, : len(LOWER_BODY_JOINTS)] = lower_pos
            self._lower_history[-1, len(LOWER_BODY_JOINTS) :] = lower_vel

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

        lower = torch.as_tensor(self._lower_history, device=self.device).reshape(1, -1)
        obs[:, TELEOP_LOWER_BODY] = lower

        # Single-frame anchor orientation, same maths as smpl mode over one frame.
        root_quat = reference[:, None, SonicReferenceSlice.ROOT_QUAT]
        anchor = smpl_anchor_orientation(
            reference_root_quat=root_quat,
            robot_base_quat=base_quat,
            apply_delta_heading=self._apply_delta_heading,
            orientation_mode=variant.orientation_mode,
        )
        obs[:, TELEOP_ANCHOR_ORI] = anchor.reshape(1, -1)

        # vr_3point positions: the operator's own wrists and head, relative to their own root.
        #
        # "local" here means root-normalized, not robot-relative: ``GatherVR3PointPosition``
        # subtracts the reference's root position and rotates by the inverse root quaternion
        # (``g1_deploy_onnx_ref.cpp:1157-1170``). The reference already carries its SMPL joints in
        # exactly that frame, so the points are a slice rather than a transform.
        #
        # No positional offsets are applied. The C++ adds +0.18 along each wrist's X and +0.35 Z
        # on the torso only when *synthesizing* VR points from a robot skeleton; with real tracked
        # head and hands it takes the buffered values directly, which is the case here.
        smpl = reference[:, SonicReferenceSlice.SMPL_JOINTS].reshape(1, 24, 3)
        obs[:, TELEOP_VR3_POS] = smpl[:, VR3_SMPL_INDICES, :].reshape(1, -1)

        # Orientations are left zero: the 83-wide reference carries only the root quaternion, not
        # per-joint orientations, so head and hand rotations are not recoverable from it. Filling
        # these needs the reference format widened, which would break compatibility with existing
        # MCAP captures -- deliberately out of scope here.
        obs[:, TELEOP_VR3_ORN] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        """Run one control step in whichever mode the operator selected.

        Args:
            actions: ``(num_envs, 92)`` ``[reference | mode | command]``.

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
        self._update_anchor(self._mode)
        if self._mode == ENCODER_MODE_TELEOP:
            command = self._to_world_directions(command)
            # Planner runs outside inference_mode: it is numpy/onnxruntime and reads articulation
            # state, which the surrounding context does not need to guard.
            self._advance_planner(command)

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
        self._prev_mode = self._mode
