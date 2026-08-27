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
The action passed to ``env.step()`` is the 95-wide SONIC reference frame emitted by
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

Hot-path discipline
-------------------
``process_actions`` runs at the control rate and is written to stay GPU-resident:

* Every fixed-shape tensor is preallocated. The step writes into slices rather than building
  fresh tensors, so steady state performs no allocation of policy inputs or outputs.
* Steady state never reads a CUDA tensor's *value* from Python. Guards of the form
  ``if bool(mask.any())`` forced a device synchronize on every control step just to learn that
  the branch was not taken. Two different replacements were needed, because the two guards asked
  different questions. History/window priming depends only on whether a reset has occurred, which
  the host already knows, so it is gated on a plain Python mirror flag and the per-environment
  detail is resolved with a mask select. The heading latch instead depends on a validity flag
  carried inside the reference data, so it is read back -- but only while some environment is
  still unlatched, never once latching has completed.
* The reference window is a mirrored ring (see :mod:`.proprio_history`) rather than a
  ``torch.roll``, so advancing costs two frame writes instead of rewriting the window.
* All of it, including onnxruntime, issues into a single CUDA stream owned by
  :class:`~gear_sonic.lab_teleop.mdp.sonic_policy.SonicOnnxPolicy`, joined to the caller's stream
  with events at entry and exit.

Set ``enable_profiling`` on the config to get a per-stage CUDA-event breakdown; it is off by
default and costs nothing when off. See :meth:`SonicWholeBodyAction.profiling_report`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass
import torch

from gear_sonic.lab_teleop.mdp.profiling import StageProfiler, StageStats
from gear_sonic.lab_teleop.mdp.proprio_history import (
    SonicProprioHistory,
)
from gear_sonic.lab_teleop.mdp.sonic_policy import (
    SONIC_NUM_ACTIONS,
    SonicOnnxPolicy,
    smpl_anchor_orientation,
)
from gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter import (
    SONIC_REFERENCE_DIM,
    SonicReferenceSlice,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["SONIC_PROFILE_STAGES", "SonicWholeBodyAction", "SonicWholeBodyActionCfg"]

#: Stages reported by :meth:`SonicWholeBodyAction.profiling_report`, in control-flow order.
SONIC_PROFILE_STAGES = [
    "reference_history_update",
    "proprio_history_update",
    "encoder_obs_construction",
    "encoder_inference",
    "decoder_obs_construction",
    "decoder_inference",
    "output_postprocessing",
    "total_process_actions",
]


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

    policy_device: str = "auto"
    """Device for ONNX inference, independent of the physics device.

    ``"auto"`` (default) puts inference on a GPU whenever one is available, **even when physics
    runs on CPU**. That split is almost always what you want: the SONIC decoder is ~37M params and
    costs ~17 ms per step on CPU against a 20 ms control period, so CPU inference alone puts the
    environment below real time, while CPU *physics* is merely slower. Resolution order:

    * physics already on CUDA -> the same GPU, so nothing crosses the bus;
    * physics on CPU and CUDA available -> the current CUDA device;
    * no CUDA -> CPU, with the runner warning about the cost.

    Pass an explicit device (``"cuda:1"``, ``"cpu"``) to override. Note that when this differs
    from the physics device the encoder input and the resulting action must cross the bus each
    step; that transfer is far cheaper than CPU inference, but it is not free.
    """

    enable_profiling: bool = False
    """Record per-stage CUDA-event timings. Off by default; see :meth:`profiling_report`."""

    profile_capacity: int = 512
    """Samples retained per stage when profiling is enabled."""


class SonicWholeBodyAction(ActionTerm):
    """Runs SONIC end to end: teleop reference in, 29 joint position targets out."""

    cfg: SonicWholeBodyActionCfg
    _asset: Articulation

    def __init__(self, cfg: SonicWholeBodyActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._env = env

        self._policy_device = self._resolve_policy_device(
            getattr(cfg, "policy_device", "auto"), self.device
        )
        self._policy = SonicOnnxPolicy(cfg.checkpoint_dir, device=self._policy_device)
        if self.num_envs != self._policy.BATCH:
            # Fail here rather than at the first control step. The shipped graphs have a static
            # batch axis, so a vectorized scene can never work -- and letting it through produces
            # an opaque broadcast error from inside observation assembly instead.
            raise ValueError(
                f"SonicWholeBodyAction requires num_envs == {self._policy.BATCH}, got "
                f"{self.num_envs}. The SONIC encoder/decoder ONNX graphs are exported with a "
                "static batch of 1; re-export them with a dynamic batch axis to run a "
                "vectorized scene."
            )

        joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._joint_ids = torch.as_tensor(joint_ids, device=self.device)
        if len(self._joint_names) != SONIC_NUM_ACTIONS:
            raise ValueError(
                f"SONIC drives {SONIC_NUM_ACTIONS} joints, matched {len(self._joint_names)}: "
                f"{self._joint_names}"
            )

        self._scale = self._resolve_action_scale(cfg.action_scale)
        self._offset = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        # Constant for the episode, so the per-step relative velocity is a subtraction rather
        # than a second gather.
        self._default_joint_vel = self._asset.data.default_joint_vel.torch[
            :, self._joint_ids
        ].clone()

        self._history = SonicProprioHistory(
            num_envs=self.num_envs,
            num_joints=SONIC_NUM_ACTIONS,
            device=self.device,
        )
        # Mirrored ring of reference frames; the encoder consumes the whole window each step.
        # Doubled length keeps the chronological window a contiguous slice -- see .proprio_history.
        #
        # The window length comes from the *checkpoint*, not from SONIC_HISTORY_LENGTH: the
        # low-latency export consumes 4 reference frames while its decoder still takes the same
        # 10-frame proprioception history. Conflating the two would silently feed the encoder a
        # wrongly-shaped reference block.
        self._reference_frames = self._policy.variant.reference_frames
        self._reference_window = torch.zeros(
            self.num_envs, 2 * self._reference_frames, SONIC_REFERENCE_DIM, device=self.device
        )
        self._reference_write = self._reference_frames - 1
        self._window_primed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        #: Host mirror of ``_window_primed.all()``; keeps the backfill off the steady-state path.
        self._window_all_primed = False
        # Operator->robot heading alignment, latched on first valid reference after a reset.
        self._apply_delta_heading = torch.zeros(self.num_envs, 4, device=self.device)
        self._apply_delta_heading[:, 0] = 1.0
        self._identity_quat = self._apply_delta_heading.clone()
        self._heading_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        #: Host mirror of ``_heading_latched.all()``. False only during the brief pre-latch phase,
        #: which is the sole window in which this term reads a CUDA value from Python.
        self._heading_all_latched = False
        self._warned_nonfinite = False
        #: Set once a reference has ever been marked valid. Until then the term holds the default
        #: pose rather than driving SONIC from a frame that carries no tracking.
        self._seen_valid_reference = False

        self._raw_actions = torch.zeros(self.num_envs, SONIC_NUM_ACTIONS, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # Scratch for the proprioception frame, refilled in place each step.
        self._joint_pos_rel = torch.zeros_like(self._raw_actions)
        self._joint_vel_rel = torch.zeros_like(self._raw_actions)

        self._profiler = StageProfiler(
            SONIC_PROFILE_STAGES,
            enabled=bool(getattr(cfg, "enable_profiling", False)),
            device=self.device,
            capacity=int(getattr(cfg, "profile_capacity", 512)),
        )

        self._warn_if_xr_anchor_missing(env)

    @staticmethod
    def _resolve_policy_device(
        configured: str | None, env_device: torch.device | str
    ) -> torch.device:
        """Choose the ONNX inference device.

        See :attr:`SonicWholeBodyActionCfg.policy_device` for why the default prefers a GPU even
        when physics is on CPU.

        Args:
            configured: The config value; ``"auto"``/``None`` selects automatically.
            env_device: Device the environment's tensors live on.

        Returns:
            A concrete device. CUDA devices are always pinned to an explicit ordinal so
            onnxruntime and torch cannot end up on different GPUs.
        """
        if configured and configured != "auto":
            device = torch.device(configured)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError(
                    f"policy_device={configured!r} requests CUDA, but torch reports no CUDA "
                    "devices. Use 'cpu', or 'auto' to pick whatever is present."
                )
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            return device

        env_device = torch.device(env_device)
        if env_device.type == "cuda":
            # Physics is already on a GPU -- run inference on that same one, so the hot path
            # stays entirely device-resident and multi-GPU hosts do not split across cards.
            if env_device.index is None:
                return torch.device("cuda", torch.cuda.current_device())
            return env_device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

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

    def profiling_report(self) -> StageStats:
        """Per-stage timing summary (mean / p50 / p95, milliseconds).

        Empty unless ``cfg.enable_profiling`` is set. Synchronizes the device once per call, so
        call it on a reporting cadence, never inside the control loop.
        """
        return self._profiler.report()

    def reset_profiling(self) -> None:
        """Drop accumulated profiling samples."""
        self._profiler.reset()

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

        The reference ring's write index is shared across environments and deliberately left
        alone: a reset environment is marked unprimed, and its next append backfills every slot,
        so the window is correct for it regardless of where the cursor happens to sit.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._history.reset(env_ids)
        self._reference_window[env_ids] = 0.0
        self._window_primed[env_ids] = False
        self._window_all_primed = False
        self._heading_latched[env_ids] = False
        self._heading_all_latched = False
        self._seen_valid_reference = False
        self._apply_delta_heading[env_ids] = self._identity_quat[env_ids]
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        """Run one SONIC control step.

        Args:
            actions: ``(num_envs, 95)`` SONIC reference frames from the teleop retargeter.
        """
        with torch.inference_mode(), self._policy.compute_stream():
            with self._profiler.stage("total_process_actions"):
                with self._profiler.stage("reference_history_update"):
                    self._push_reference(actions.to(self.device))
                with self._profiler.stage("proprio_history_update"):
                    self._append_proprioception()

                with self._profiler.stage("encoder_obs_construction"):
                    encoder_obs = self._build_encoder_obs()
                with self._profiler.stage("encoder_inference"):
                    token = self._policy.encode(encoder_obs)

                with self._profiler.stage("decoder_obs_construction"):
                    # Write the history straight into the decoder's bound input slice, so the
                    # 930-wide vector is never materialized separately.
                    proprio = self._history.flat(out=self._policy.decoder_proprio_view)
                with self._profiler.stage("decoder_inference"):
                    raw = self._policy.decode(token, proprio)

                with self._profiler.stage("output_postprocessing"):
                    self._raw_actions.copy_(raw)
                    # raw * scale + offset, via out-parameters so no temporary is built and the
                    # arithmetic stays bit-identical to the original expression (no fused FMA).
                    torch.mul(self._raw_actions, self._scale, out=self._processed_actions)
                    self._processed_actions.add_(self._offset)
                    self._hold_default_if_untracked(actions)

    def _hold_default_if_untracked(self, reference: torch.Tensor) -> None:
        """Fall back to the default pose when there is nothing trustworthy to track.

        Two cases, both of which otherwise reach the physics solver as garbage:

        * **No tracking yet.** With the body tracker absent the retargeter emits a neutral frame
          with the valid flag clear. Driving SONIC from a collapsed skeleton makes the robot thrash;
          standing in its default pose until tracking arrives is the honest behaviour.
        * **Non-finite output.** A NaN reaching ``set_joint_position_target`` is unrecoverable and
          silent -- the robot simply goes limp or explodes, with nothing in the log. Guarding here
          converts that into a single visible warning.

        Args:
            reference: ``(num_envs, 95)`` reference frame for this step.
        """
        if bool((reference[:, SonicReferenceSlice.VALID] > 0.5).any()):
            self._seen_valid_reference = True
        finite = bool(torch.isfinite(self._processed_actions).all())
        if self._seen_valid_reference and finite:
            return
        if not finite and not self._warned_nonfinite:
            print(
                "[SONIC] non-finite policy output; holding the default pose. "
                "This usually means the reference carried no valid tracking.",
                flush=True,
            )
            self._warned_nonfinite = True
        self._processed_actions.copy_(self._offset)
        self._raw_actions.zero_()

    def apply_actions(self) -> None:
        """Write the joint position targets to the articulation."""
        self._asset.set_joint_position_target_index(
            target=self._processed_actions, joint_ids=self._joint_ids
        )

    def _reference_view(self) -> torch.Tensor:
        """``(num_envs, reference_frames, 95)`` reference window, oldest frame first."""
        start = self._reference_write + 1
        return self._reference_window[:, start : start + self._reference_frames]

    def _push_reference(self, reference: torch.Tensor) -> None:
        """Advance the reference ring, backfilling it on the first frame after a reset."""
        write = (self._reference_write + 1) % self._reference_frames
        self._reference_window[:, write] = reference
        self._reference_window[:, write + self._reference_frames] = reference
        # Pending-backfill is a reset-driven fact the host already knows, so steady state skips
        # this without touching the device; *which* rows need it stays a GPU-side mask select.
        if not self._window_all_primed:
            backfill = (~self._window_primed).view(-1, 1, 1)
            self._reference_window.copy_(
                torch.where(backfill, reference.unsqueeze(1), self._reference_window)
            )
            self._window_primed[:] = True
            self._window_all_primed = True
        self._reference_write = write

        # Unlike priming, latching depends on the validity flag carried *inside* the reference, so
        # the host cannot know when it completes without a read. We therefore read -- but only
        # while at least one environment is still unlatched. Once every environment has latched,
        # this whole block is skipped by a Python bool and steady state never synchronizes.
        if not self._heading_all_latched:
            self._latch_heading(reference)
            self._heading_all_latched = bool(self._heading_latched.all())

    def _latch_heading(self, reference: torch.Tensor) -> None:
        """Align the operator's initial facing to the robot's, so yaw does not accumulate.

        Mirrors ``ComputeApplyDeltaHeading`` (``g1_deploy_onnx_ref.cpp:589-602``)::

            apply_delta_heading = heading(robot_base_quat) * heading_inv(reference_root_quat)

        The delta is computed every step and written only where an environment has a valid
        reference and has not latched yet. Computing unconditionally trades a few small kernels
        for the device synchronize that the previous ``if bool(to_latch.any())`` guard cost on
        every control step. Values from environments that should not latch are discarded by the
        select, so a degenerate quaternion on an invalid frame cannot leak in.
        """
        from gear_sonic.isaac_utils.rotations import (
            calc_heading_quat,
            calc_heading_quat_inv,
            quat_mul,
        )

        valid = reference[:, SonicReferenceSlice.VALID].squeeze(-1) > 0.5
        to_latch = (valid & (~self._heading_latched)).unsqueeze(-1)

        base_quat = isaaclab_quat_to_wxyz(self._asset.data.root_quat_w.torch)
        init_heading = calc_heading_quat(base_quat, w_last=False)
        ref_quat = reference[:, SonicReferenceSlice.ROOT_QUAT]
        ref_heading_inv = calc_heading_quat_inv(ref_quat, w_last=False)
        delta = quat_mul(init_heading, ref_heading_inv, w_last=False)

        self._apply_delta_heading.copy_(torch.where(to_latch, delta, self._apply_delta_heading))
        self._heading_latched |= valid

    def _append_proprioception(self) -> None:
        """Push the current robot state into the decoder's history buffers."""
        data = self._asset.data
        # Gather into preallocated scratch, then subtract the (constant) defaults in place.
        torch.index_select(data.joint_pos.torch, 1, self._joint_ids, out=self._joint_pos_rel)
        self._joint_pos_rel.sub_(self._offset)
        torch.index_select(data.joint_vel.torch, 1, self._joint_ids, out=self._joint_vel_rel)
        self._joint_vel_rel.sub_(self._default_joint_vel)
        self._history.append(
            base_ang_vel=data.root_ang_vel_b.torch,
            joint_pos_rel=self._joint_pos_rel,
            joint_vel_rel=self._joint_vel_rel,
            last_action=self._raw_actions,
            gravity_dir=data.projected_gravity_b.torch,
        )

    def _build_encoder_obs(self) -> torch.Tensor:
        """Assemble the ``smpl``-mode encoder observation from the reference window.

        Writes into the policy's bound encoder buffer; the returned tensor is that buffer.
        """
        window = self._reference_view()
        num_envs = window.shape[0]

        smpl_joints = window[:, :, SonicReferenceSlice.SMPL_JOINTS].reshape(
            num_envs, self._reference_frames, 24, 3
        )
        root_quat = window[:, :, SonicReferenceSlice.ROOT_QUAT]
        wrist = window[:, :, SonicReferenceSlice.WRIST_JOINT_POS]

        anchor_ori = smpl_anchor_orientation(
            reference_root_quat=root_quat,
            robot_base_quat=isaaclab_quat_to_wxyz(self._asset.data.root_quat_w.torch),
            apply_delta_heading=self._apply_delta_heading,
            orientation_mode=self._policy.variant.orientation_mode,
        )
        return self._policy.fill_smpl_encoder_obs(smpl_joints, anchor_ori, wrist)
