# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rolling proprioception history feeding SONIC's ``g1_dyn`` decoder.

The decoder input is a fixed 994-wide vector::

    token(64) | base_ang_vel(30) | joint_pos(290) | joint_vel(290) | last_actions(290) | gravity_dir(30)

i.e. a 64-dim motion token followed by five 10-frame histories. **The order is load-bearing** and
matches the deployment spec (``gear_sonic_deploy/policy/*/observation_config.yaml``) and the
attribute declaration order of the training ``PolicyCfg``
(``gear_sonic/envs/manager_env/mdp/observations.py``). Note ``gravity_dir`` is *last*, not first.

Within each history, frames run **oldest-first, newest-last**, then flatten to ``(num_envs, H * D)``.
That matches both Isaac Lab's ``CircularBuffer`` and the C++ deploy path, which gathers with
``newest_first = false``.

Reset semantics
---------------
Isaac Lab's :class:`~isaaclab.utils.buffers.CircularBuffer` does *not* leave zeros after a reset:
on the first ``append`` following a reset it **backfills all H slots with that first observation**.
We reproduce that here, because a policy whose history says "the robot has been still at the
current pose for 200 ms" behaves very differently from one whose history says "the robot was at
the origin with zero everything".

This deliberately differs from the C++ deploy stack, whose ``StateLogger`` zero-pads for the first
~10 ticks at startup. We follow the Isaac Lab / training-time convention, since that is the
distribution SONIC was trained under.

Storage: mirrored ring
----------------------
Each term is stored as ``(num_envs, 2H, dim)`` and every frame is written twice -- at ``w`` and at
``w + H``. That makes the chronological window a **contiguous slice** ``[w+1 : w+1+H]`` for any
write position, so reading needs neither a gather nor an index-dependent two-piece copy::

    H = 4, newest just written at w = 1

    index:   0   1   2   3 | 4   5   6   7
    content: d   a   b   c | d   a   b   c
                 ^-------------^
                 window [2:6] = b c d a   -> oldest .. newest

Advancing therefore writes 2 frames and increments an integer, where ``torch.roll`` would
allocate a full copy of each term and rewrite all H frames every tick. The doubled buffer costs
930 extra floats per environment, negligible against five per-tick copies avoided.

Backfill without synchronizing
------------------------------
Guarding the backfill with ``if bool(unprimed.any())`` would read a CUDA tensor from Python and
force a device synchronize on *every* control step, purely to discover that -- in steady state --
there is nothing to backfill.

The question is therefore split in two. *Whether* any backfill is pending depends only on
whether :meth:`SonicProprioHistory.reset` has been called since the last append, which the host
already witnessed; that is tracked in a plain Python bool, so steady state skips the work entirely
without consulting the device. *Which* environments need it is a per-env mask, applied with
:func:`torch.where` so the answer never leaves the GPU. ``where`` selects between operands rather
than blending them, so untouched environments keep bit-identical values.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = ["SONIC_DECODER_PROPRIO_DIM", "SONIC_HISTORY_LENGTH", "SonicProprioHistory"]

#: Number of past frames retained per term (10 frames at the 50 Hz control rate = 200 ms).
SONIC_HISTORY_LENGTH = 10

#: Flattened width of the five histories: (3 + 29 + 29 + 29 + 3) * 10.
SONIC_DECODER_PROPRIO_DIM = 930


class SonicProprioHistory:
    """Batched, per-environment rolling history of SONIC's five proprioceptive terms.

    Args:
        num_envs: Number of parallel environments.
        num_joints: Actuated DoF count. SONIC's G1 is 29.
        history_length: Frames retained per term.
        device: Torch device for the buffers.

    Example:
        >>> hist = SonicProprioHistory(num_envs=1, device="cpu")
        >>> hist.reset()
        >>> hist.append(
        ...     base_ang_vel=torch.zeros(1, 3),
        ...     joint_pos_rel=torch.zeros(1, 29),
        ...     joint_vel_rel=torch.zeros(1, 29),
        ...     last_action=torch.zeros(1, 29),
        ...     gravity_dir=torch.zeros(1, 3),
        ... )
        >>> hist.flat().shape
        torch.Size([1, 930])
    """

    #: Term name -> per-frame width. Iteration order defines the decoder concatenation order.
    _TERM_DIMS: tuple[tuple[str, str], ...] = (
        ("base_ang_vel", "ang_vel_dim"),
        ("joint_pos_rel", "joint_dim"),
        ("joint_vel_rel", "joint_dim"),
        ("last_action", "joint_dim"),
        ("gravity_dir", "gravity_dim"),
    )

    def __init__(
        self,
        num_envs: int,
        num_joints: int = 29,
        history_length: int = SONIC_HISTORY_LENGTH,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if history_length <= 0:
            raise ValueError(f"history_length must be positive, got {history_length}")

        self.num_envs = num_envs
        self.num_joints = num_joints
        self.history_length = history_length
        self.device = torch.device(device)

        self._dims = {
            "ang_vel_dim": 3,
            "joint_dim": num_joints,
            "gravity_dim": 3,
        }
        # Mirrored ring: 2H frames so the ordered window is always contiguous.
        self._buffers: dict[str, torch.Tensor] = {
            term: torch.zeros(num_envs, 2 * history_length, self._dims[dim_key], device=self.device)
            for term, dim_key in self._TERM_DIMS
        }
        #: Index of the newest frame within ``[0, history_length)``.
        self._write = history_length - 1
        # False => the next append for this env backfills the whole window.
        self._primed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        #: Host mirror of "``_primed`` is all True". Lets the hot path skip the backfill without
        #: reading ``_primed`` back from the device.
        self._all_primed = False

        # Persistent flat output plus per-term (N, H, D) views into it, so ``flat`` is a set of
        # contiguous copies into an existing allocation rather than a fresh ``torch.cat``.
        self._flat = torch.zeros(num_envs, self.flat_dim, device=self.device)
        self._flat_views: dict[str, torch.Tensor] = {}
        offset = 0
        for term, dim_key in self._TERM_DIMS:
            width = self._dims[dim_key] * history_length
            self._flat_views[term] = self._flat[:, offset : offset + width].view(
                num_envs, history_length, self._dims[dim_key]
            )
            offset += width

    @property
    def flat_dim(self) -> int:
        """Width of :meth:`flat`'s output."""
        return sum(self._dims[dim_key] * self.history_length for _, dim_key in self._TERM_DIMS)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear history for the given environments.

        Buffers are zeroed and marked unprimed, so the next :meth:`append` backfills every slot
        with that first observation.

        Args:
            env_ids: Environments to reset. ``None`` or ``slice(None)`` resets all. Accepts the
                ``slice(None)`` that Isaac Lab's ``ActionManager.reset`` passes on a full reset.
        """
        if env_ids is None:
            env_ids = slice(None)
        for buf in self._buffers.values():
            buf[env_ids] = 0.0
        self._primed[env_ids] = False
        self._all_primed = False

    def append(
        self,
        *,
        base_ang_vel: torch.Tensor,
        joint_pos_rel: torch.Tensor,
        joint_vel_rel: torch.Tensor,
        last_action: torch.Tensor,
        gravity_dir: torch.Tensor,
    ) -> None:
        """Push one frame for every environment.

        Unprimed environments have the whole window filled with this frame; primed ones advance by
        one. Neither path reads a tensor value from Python, so no device synchronization occurs.

        Args:
            base_ang_vel: ``(num_envs, 3)`` base angular velocity in the body frame.
            joint_pos_rel: ``(num_envs, num_joints)`` joint position **relative to default**.
            joint_vel_rel: ``(num_envs, num_joints)`` joint velocity relative to default.
            last_action: ``(num_envs, num_joints)`` previous **raw** (pre-scale) policy output.
            gravity_dir: ``(num_envs, 3)`` gravity direction in the pelvis frame.
        """
        frame = {
            "base_ang_vel": base_ang_vel,
            "joint_pos_rel": joint_pos_rel,
            "joint_vel_rel": joint_vel_rel,
            "last_action": last_action,
            "gravity_dir": gravity_dir,
        }
        for term, dim_key in self._TERM_DIMS:
            value = frame[term]
            expected = (self.num_envs, self._dims[dim_key])
            if tuple(value.shape) != expected:
                raise ValueError(f"{term}: expected shape {expected}, got {tuple(value.shape)}")

        history = self.history_length
        write = (self._write + 1) % history
        # Whether a backfill is pending depends only on whether ``reset`` has been called since
        # the last append -- an event the host already witnessed. Tracking it in a Python bool
        # means steady state neither reads a CUDA tensor nor runs the select at all.
        needs_backfill = not self._all_primed
        backfill = (~self._primed).view(-1, 1, 1) if needs_backfill else None

        for term, _ in self._TERM_DIMS:
            buf = self._buffers[term]
            value = frame[term].to(device=buf.device, dtype=buf.dtype)
            # Write the newest frame into both mirrors of the ring...
            buf[:, write] = value
            buf[:, write + history] = value
            # ...then, only when a reset is outstanding, fill every slot for the environments it
            # touched. Mask-selected rather than index-selected, so no host sync is needed to
            # discover *which* environments those are.
            if needs_backfill:
                buf.copy_(torch.where(backfill, value.unsqueeze(1), buf))

        self._write = write
        if needs_backfill:
            self._primed[:] = True
        self._all_primed = True

    def window(self, term: str) -> torch.Tensor:
        """Return ``(num_envs, H, dim)`` for one term, oldest frame first."""
        start = self._write + 1
        return self._buffers[term][:, start : start + self.history_length]

    def flat(self, out: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``(num_envs, 930)`` in the decoder's expected concatenation order.

        Args:
            out: Optional destination. Passing the decoder's own bound input slice lets the
                history land straight in the buffer onnxruntime reads, avoiding another copy.

        Returns:
            ``out`` when given, else an internal persistent buffer that is overwritten on the next
            call.
        """
        if out is None:
            for term, _ in self._TERM_DIMS:
                self._flat_views[term].copy_(self.window(term))
            return self._flat

        if out.shape != (self.num_envs, self.flat_dim):
            raise ValueError(
                f"out must be {(self.num_envs, self.flat_dim)}, got {tuple(out.shape)}"
            )
        offset = 0
        for term, dim_key in self._TERM_DIMS:
            width = self._dims[dim_key] * self.history_length
            view = out[:, offset : offset + width].view(
                self.num_envs, self.history_length, self._dims[dim_key]
            )
            view.copy_(self.window(term))
            offset += width
        return out
