# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mode-switching SONIC action term: full-body tracking or stick-driven walking.

Extends :class:`~gear_sonic.lab_teleop.mdp.actions.SonicWholeBodyAction` with the operator's
encoder-mode selection, mirroring the real robot's teleop stack. The environment action becomes::

    [ sonic_reference(83) | mode(1) | locomotion_command(8) ]   -> 92

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

import numpy as np
import torch
from isaaclab.utils.configclass import configclass

from gear_sonic.lab_teleop.mdp.actions import SonicWholeBodyAction, SonicWholeBodyActionCfg
from gear_sonic.lab_teleop.mdp.sonic_planner import (
    PLANNER_CLIP_WALK,
    PLANNER_QPOS_DIM,
    SONIC_PLANNER_COMMAND_DIM,
    SonicVelocityPlanner,
)
from gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter import SONIC_REFERENCE_DIM

__all__ = [
    "SONIC_MODAL_ACTION_DIM",
    "SonicModalWholeBodyAction",
    "SonicModalWholeBodyActionCfg",
]

#: ``[reference(83) | mode(1) | command(8)]``.
SONIC_MODAL_ACTION_DIM = SONIC_REFERENCE_DIM + 1 + SONIC_PLANNER_COMMAND_DIM

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
        self._planner = SonicVelocityPlanner(
            checkpoint_path=checkpoint,
            device=self._policy_device,
            replan_interval=cfg.planner_replan_interval,
        )
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
        self._mode = ENCODER_MODE_SMPL

        from gear_sonic.lab_teleop.assets.g1_sonic import G1_ISAACLAB_TO_MUJOCO_MAPPING

        self._isaaclab_to_mujoco = np.asarray(
            G1_ISAACLAB_TO_MUJOCO_MAPPING["isaaclab_to_mujoco_dof"], dtype=np.int64
        )

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
        self._planner.reset()
        self._lower_history[:] = 0.0
        self._prev_lower_pos = None
        self._mode = ENCODER_MODE_SMPL

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

    def _advance_planner(self, command: np.ndarray) -> None:
        """Run the planner and push one frame of lower-body pos+vel into the history."""
        qpos = self._robot_qpos()
        self._planner.push_state(qpos)

        if float(command[0]) <= 1e-4:
            # No operator input: continue from measured motion rather than a stale command, as the
            # reference implementation does. Needs robot state, hence here rather than upstream.
            movement, facing = SonicVelocityPlanner.idle_directions(
                self._planner._context[0]  # noqa: SLF001 - deliberate: the measured history
            )
            command = command.copy()
            command[1:4] = movement
            command[4:7] = facing

        frame = self._planner.next_frame(command, mode=self._planner_clip)
        # Planned joints are MuJoCo-ordered; take the lower body back in Isaac Lab order.
        planned_joints = frame[7:][self._isaaclab_to_mujoco]
        lower_pos = planned_joints[: len(LOWER_BODY_JOINTS)]

        dt = self._env.step_dt if hasattr(self._env, "step_dt") else 0.02
        if self._prev_lower_pos is None:
            lower_vel = np.zeros_like(lower_pos)
        else:
            lower_vel = (lower_pos - self._prev_lower_pos) / max(dt, 1e-6)
        self._prev_lower_pos = lower_pos.copy()

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
        root_quat = reference[:, None, 73:77]
        anchor = smpl_anchor_orientation(
            reference_root_quat=root_quat,
            robot_base_quat=base_quat,
            apply_delta_heading=self._apply_delta_heading,
            orientation_mode=variant.orientation_mode,
        )
        obs[:, TELEOP_ANCHOR_ORI] = anchor.reshape(1, -1)

        # vr_3point: operator head and hands. Left zero for now -- deriving robot-local targets
        # needs the anchor handling that is deliberately deferred, and a wrong frame here is a
        # silent tracking error rather than a loud failure.
        obs[:, TELEOP_VR3_POS] = 0.0
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
                f"command), got {actions.shape[-1]}"
            )

        actions = actions.to(self.device)
        reference = actions[:, :SONIC_REFERENCE_DIM]
        mode = int(actions[0, SONIC_REFERENCE_DIM].item())
        command = actions[0, SONIC_REFERENCE_DIM + 1 :].detach().float().cpu().numpy()
        self._mode = mode if mode in (ENCODER_MODE_TELEOP, ENCODER_MODE_SMPL) else ENCODER_MODE_SMPL

        if self._mode == ENCODER_MODE_TELEOP:
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
