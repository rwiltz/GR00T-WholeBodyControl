# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""``ActionTerm`` that runs the SONIC whole-body controller inside Isaac Lab.

Follows the pattern established by ``isaaclab_tasks.contrib.locomanip_pick_place``'s
``AgileBasedLowerBodyAction``: the trained policy lives *inside* the action term, so ``env.step()``
keeps the standard gym contract and the incoming action is a low-dimensional reference rather than
raw joint targets.

Action contract
---------------
The action passed to ``env.step()`` is the 83-wide SONIC reference frame emitted by
:class:`~gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter.SonicFullBodyRetargeter`.
Per control step this term:

1. pushes the reference into a 10-frame rolling window,
2. builds the ``smpl``-mode encoder observation and runs the encoder to a 64-dim token,
3. concatenates the token with the 930-wide proprioception history and runs the decoder,
4. converts the 29 raw outputs to joint position targets via ``raw * scale + default_joint_pos``.

Induced latency
---------------
SONIC's ``smpl`` encoder consumes 10 reference frames at 20 ms spacing, which the training data
supplied as *future* motion. A live operator has no future, so — exactly as the C++ deploy stack
does by pinning its playback cursor at ``timesteps - 11``
(``g1_deploy_onnx_ref.cpp:3394-3398``) — we present the 10 most recent frames and let the policy
treat the oldest as "now". The robot therefore trails the operator by ~200 ms. This is inherent to
the checkpoint, not a bug: the ``low_latency`` checkpoint reduces the window and hence the lag.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from gear_sonic.lab_teleop.mdp.proprio_history import (
    SONIC_HISTORY_LENGTH,
    SonicProprioHistory,
)
from gear_sonic.lab_teleop.mdp.sonic_policy import (
    SONIC_NUM_ACTIONS,
    SonicOnnxPolicy,
    smpl_anchor_orientation_heading,
)
from gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter import (
    SONIC_REFERENCE_DIM,
    SonicReferenceSlice,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["SonicWholeBodyAction", "SonicWholeBodyActionCfg"]


def isaaclab_quat_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert an Isaac Lab quaternion to the WXYZ order ``gear_sonic`` expects.

    Isaac Lab 3.0 changed its quaternion convention from WXYZ to XYZW
    (``docs/source/migration/migrating_to_isaaclab_3-0.rst:1317``), but ``gear_sonic`` was written
    against 2.3.2 and its rotation helpers are all ``w_last=False``. Mixing the two silently
    produces a valid-looking unit quaternion with the wrong orientation — in this integration it
    corrupted the heading term and made the robot spin continuously.

    Args:
        quat_xyzw: ``(..., 4)`` quaternion in Isaac Lab 3.0 XYZW order.

    Returns:
        ``(..., 4)`` quaternion in WXYZ order.
    """
    return quat_xyzw[..., [3, 0, 1, 2]]


@configclass
class SonicWholeBodyActionCfg(ActionTermCfg):
    """Configuration for :class:`SonicWholeBodyAction`."""

    # Resolved lazily from this entry point so the cfg can be declared before the class, matching
    # the pattern used by isaaclab_tasks.contrib.locomanip_pick_place.
    class_type: type["SonicWholeBodyAction"] | str = (
        "gear_sonic.lab_teleop.mdp.actions:SonicWholeBodyAction"
    )

    checkpoint_dir: str = MISSING
    """Directory containing ``model_encoder.onnx`` and ``model_decoder.onnx``."""

    joint_names: list[str] = MISSING
    """Joints SONIC drives, in SONIC's own (IsaacLab) order. Normally all 29."""

    action_scale: dict[str, float] = MISSING
    """Per-joint action scale, keyed by joint-name regex (``G1_MODEL_12_ACTION_SCALE``)."""

    policy_device: str = "cuda:0"
    """Device for ONNX inference. See the module note on CPU cost at 50 Hz."""


class SonicWholeBodyAction(ActionTerm):
    """Runs SONIC end to end: teleop reference in, 29 joint position targets out."""

    cfg: SonicWholeBodyActionCfg
    _asset: Articulation

    def __init__(self, cfg: SonicWholeBodyActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._env = env

        self._policy = SonicOnnxPolicy(cfg.checkpoint_dir, device=self.device)

        joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=True
        )
        self._joint_ids = torch.as_tensor(joint_ids, device=self.device)
        if len(self._joint_names) != SONIC_NUM_ACTIONS:
            raise ValueError(
                f"SONIC drives {SONIC_NUM_ACTIONS} joints, matched {len(self._joint_names)}: "
                f"{self._joint_names}"
            )

        self._scale = self._resolve_action_scale(cfg.action_scale)
        self._offset = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()

        self._history = SonicProprioHistory(
            num_envs=self.num_envs,
            num_joints=SONIC_NUM_ACTIONS,
            device=self.device,
        )
        # Rolling window of reference frames; the encoder consumes all of it each step.
        self._reference_window = torch.zeros(
            self.num_envs, SONIC_HISTORY_LENGTH, SONIC_REFERENCE_DIM, device=self.device
        )
        self._window_primed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Operator->robot heading alignment, latched on first valid reference after a reset.
        self._apply_delta_heading = torch.zeros(self.num_envs, 4, device=self.device)
        self._apply_delta_heading[:, 0] = 1.0
        self._heading_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._raw_actions = torch.zeros(
            self.num_envs, SONIC_NUM_ACTIONS, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._warn_if_xr_anchor_missing(env)

    @staticmethod
    def _warn_if_xr_anchor_missing(env: ManagerBasedEnv) -> None:
        """Warn when the configured XR anchor prim does not exist.

        A bad ``anchor_prim_path`` does not raise: the XR session falls back to the world origin, so
        the operator's frame silently stops riding with the robot. That is hard to attribute from
        inside a headset, so check it once at setup instead.
        """
        xr_cfg = getattr(env.cfg, "xr", None)
        prim_path = getattr(xr_cfg, "anchor_prim_path", None)
        if not prim_path:
            return
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
        except Exception:  # noqa: BLE001 - no stage outside Isaac Sim; nothing to check
            return
        if stage is None or stage.GetPrimAtPath(prim_path).IsValid():
            return

        import warnings

        warnings.warn(
            f"XR anchor prim {prim_path!r} does not exist; the anchor will fall back to the world "
            "origin and will not follow the robot. Note Isaac Lab's URDF importer nests links "
            "under a 'Geometry' scope, so the path is e.g. "
            "'/World/envs/env_0/Robot/Geometry/pelvis', not '/World/envs/env_0/Robot/pelvis'.",
            RuntimeWarning,
            stacklevel=2,
        )

    @property
    def action_dim(self) -> int:
        """Width of the SONIC reference frame produced by the retargeter."""
        return SONIC_REFERENCE_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw, unscaled decoder output. This is what feeds the ``last_action`` history."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Joint position targets actually written to the articulation."""
        return self._processed_actions

    def _resolve_action_scale(self, scale_cfg: dict[str, float]) -> torch.Tensor:
        """Expand a ``{joint-name-regex: scale}`` mapping into a per-joint tensor."""
        scale = torch.ones(SONIC_NUM_ACTIONS, device=self.device)
        matched = torch.zeros(SONIC_NUM_ACTIONS, dtype=torch.bool, device=self.device)
        for pattern, value in scale_cfg.items():
            ids, _ = self._asset.find_joints(pattern, preserve_order=False)
            for joint_id in ids:
                local = (self._joint_ids == joint_id).nonzero(as_tuple=False)
                if local.numel():
                    scale[local[0, 0]] = value
                    matched[local[0, 0]] = True
        if not bool(matched.all()):
            missing = [n for n, m in zip(self._joint_names, matched.tolist(), strict=True) if not m]
            raise ValueError(f"No action scale matched these joints: {missing}")
        return scale.unsqueeze(0)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear all rolling state.

        ``AgileBasedLowerBodyAction`` omits this, which leaks the previous episode's final action
        into the next episode's first observation. SONIC carries far more state, so resetting is
        mandatory here.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._history.reset(env_ids)
        self._reference_window[env_ids] = 0.0
        self._window_primed[env_ids] = False
        self._heading_latched[env_ids] = False
        self._apply_delta_heading[env_ids] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        """Run one SONIC control step.

        Args:
            actions: ``(num_envs, 83)`` SONIC reference frames from the teleop retargeter.
        """
        with torch.inference_mode():
            self._push_reference(actions.to(self.device))
            self._append_proprioception()

            encoder_obs = self._build_encoder_obs()
            token = self._policy.encode(encoder_obs)
            raw = self._policy.decode(token, self._history.flat())

            self._raw_actions[:] = raw
            self._processed_actions[:] = raw * self._scale + self._offset

    def apply_actions(self) -> None:
        """Write the joint position targets to the articulation."""
        self._asset.set_joint_position_target_index(
            target=self._processed_actions, joint_ids=self._joint_ids
        )

    def _push_reference(self, reference: torch.Tensor) -> None:
        """Roll the reference window, backfilling it on the first frame after a reset."""
        unprimed = ~self._window_primed
        self._reference_window[:] = torch.roll(self._reference_window, shifts=-1, dims=1)
        self._reference_window[:, -1] = reference
        if bool(unprimed.any()):
            self._reference_window[unprimed] = (
                reference[unprimed].unsqueeze(1).expand(-1, SONIC_HISTORY_LENGTH, -1)
            )
        self._window_primed[:] = True

        # Latch the operator/robot heading alignment on the first valid reference.
        valid = reference[:, SonicReferenceSlice.VALID].squeeze(-1) > 0.5
        to_latch = valid & (~self._heading_latched)
        if bool(to_latch.any()):
            self._latch_heading(to_latch, reference)

    def _latch_heading(self, mask: torch.Tensor, reference: torch.Tensor) -> None:
        """Align the operator's initial facing to the robot's, so yaw does not accumulate.

        Mirrors ``ComputeApplyDeltaHeading`` (``g1_deploy_onnx_ref.cpp:589-602``)::

            apply_delta_heading = heading(robot_base_quat) * heading_inv(reference_root_quat)
        """
        from gear_sonic.isaac_utils.rotations import (
            calc_heading_quat,
            calc_heading_quat_inv,
            quat_mul,
        )

        base_quat = isaaclab_quat_to_wxyz(self._asset.data.root_quat_w.torch)
        init_heading = calc_heading_quat(base_quat, w_last=False)
        ref_quat = reference[:, SonicReferenceSlice.ROOT_QUAT]
        ref_heading_inv = calc_heading_quat_inv(ref_quat, w_last=False)
        delta = quat_mul(init_heading, ref_heading_inv, w_last=False)

        self._apply_delta_heading[mask] = delta[mask]
        self._heading_latched[mask] = True

    def _append_proprioception(self) -> None:
        """Push the current robot state into the decoder's history buffers."""
        data = self._asset.data
        joint_pos_rel = (
            data.joint_pos.torch[:, self._joint_ids]
            - data.default_joint_pos.torch[:, self._joint_ids]
        )
        joint_vel_rel = (
            data.joint_vel.torch[:, self._joint_ids]
            - data.default_joint_vel.torch[:, self._joint_ids]
        )
        self._history.append(
            base_ang_vel=data.root_ang_vel_b.torch,
            joint_pos_rel=joint_pos_rel,
            joint_vel_rel=joint_vel_rel,
            last_action=self._raw_actions,
            gravity_dir=data.projected_gravity_b.torch,
        )

    def _build_encoder_obs(self) -> torch.Tensor:
        """Assemble the ``smpl``-mode encoder observation from the reference window."""
        window = self._reference_window
        num_envs = window.shape[0]

        smpl_joints = window[:, :, SonicReferenceSlice.SMPL_JOINTS].reshape(
            num_envs, SONIC_HISTORY_LENGTH, 24, 3
        )
        root_quat = window[:, :, SonicReferenceSlice.ROOT_QUAT]
        wrist = window[:, :, SonicReferenceSlice.WRIST_JOINT_POS]

        anchor_ori = smpl_anchor_orientation_heading(
            reference_root_quat=root_quat,
            robot_base_quat=isaaclab_quat_to_wxyz(self._asset.data.root_quat_w.torch),
            apply_delta_heading=self._apply_delta_heading,
        )
        return SonicOnnxPolicy.assemble_smpl_encoder_obs(smpl_joints, anchor_ori, wrist)


