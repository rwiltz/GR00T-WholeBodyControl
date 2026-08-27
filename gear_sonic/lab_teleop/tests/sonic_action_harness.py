# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Headless harness that drives the real :class:`SonicWholeBodyAction` without Isaac Sim.

The ActionTerm only touches a small, well-defined slice of the articulation API, so a fake
articulation is enough to exercise the *production* class -- not a reimplementation of it. That
matters for parity testing: the numbers this harness produces come from the same code path the
simulator runs.

Articulation surface the term depends on (verified against ``actions.py``)::

    find_joints(patterns, preserve_order)      -> (ids, names)
    set_joint_position_target_index(target=, joint_ids=)
    data.default_joint_pos.torch   (N, 29)     data.default_joint_vel.torch  (N, 29)
    data.joint_pos.torch           (N, 29)     data.joint_vel.torch          (N, 29)
    data.root_quat_w.torch         (N, 4)  XYZW (Isaac Lab 3.0 order)
    data.root_ang_vel_b.torch      (N, 3)     data.projected_gravity_b.torch (N, 3)

Everything is driven from a seeded generator so a run is reproducible across processes, which is
what lets us diff a baseline capture against an optimized one.
"""

from __future__ import annotations

import re

import torch

from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
from gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter import (
    SONIC_REFERENCE_DIM,
    SonicReferenceSlice,
)

__all__ = [
    "FakeArticulation",
    "FakeEnv",
    "build_action_term",
    "make_reference_sequence",
    "make_robot_state_sequence",
]


class _Proxy:
    """Stands in for Isaac Lab's ``ProxyArray``; only ``.torch`` is used by the term."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.torch = tensor


class _FakeArticulationData:
    def __init__(self, num_envs: int, num_joints: int, device: torch.device) -> None:
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.default_joint_pos = _Proxy(z(num_envs, num_joints))
        self.default_joint_vel = _Proxy(z(num_envs, num_joints))
        self.joint_pos = _Proxy(z(num_envs, num_joints))
        self.joint_vel = _Proxy(z(num_envs, num_joints))
        # XYZW identity, matching Isaac Lab 3.0's quaternion order.
        root_quat = z(num_envs, 4)
        root_quat[:, 3] = 1.0
        self.root_quat_w = _Proxy(root_quat)
        self.root_ang_vel_b = _Proxy(z(num_envs, 3))
        root_pos = z(num_envs, 3)
        root_pos[:, 2] = 0.76
        self.root_pos_w = _Proxy(root_pos)
        gravity = z(num_envs, 3)
        gravity[:, 2] = -1.0
        self.projected_gravity_b = _Proxy(gravity)


class FakeArticulation:
    """Minimal articulation exposing exactly what the ActionTerm reads and writes."""

    def __init__(
        self,
        num_envs: int,
        joint_names: list[str] | None = None,
        device: torch.device | str = "cuda:0",
    ) -> None:
        self.device = torch.device(device)
        self.joint_names = list(joint_names or G1_ISAACLab_ORDER)
        self.data = _FakeArticulationData(num_envs, len(self.joint_names), self.device)
        #: Last value passed to :meth:`set_joint_position_target_index`.
        self.applied_target: torch.Tensor | None = None
        self.applied_joint_ids: torch.Tensor | None = None

    def find_joints(
        self, name_keys: str | list[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Regex joint lookup mirroring Isaac Lab's ``find_joints`` semantics."""
        keys = [name_keys] if isinstance(name_keys, str) else list(name_keys)
        ids: list[int] = []
        if preserve_order:
            for key in keys:
                for idx, name in enumerate(self.joint_names):
                    if re.fullmatch(key, name) and idx not in ids:
                        ids.append(idx)
        else:
            for idx, name in enumerate(self.joint_names):
                if any(re.fullmatch(key, name) for key in keys):
                    ids.append(idx)
        return ids, [self.joint_names[i] for i in ids]

    def set_joint_position_target_index(
        self, target: torch.Tensor, joint_ids: torch.Tensor
    ) -> None:
        self.applied_target = target.clone()
        self.applied_joint_ids = joint_ids


class _FakeScene:
    def __init__(self, asset: FakeArticulation) -> None:
        self._asset = asset

    def __getitem__(self, _name: str) -> FakeArticulation:
        return self._asset


class _FakeEnvCfg:
    """Env cfg without an ``xr`` attribute, so the anchor-prim warning short-circuits."""


class FakeEnv:
    """Just enough of ``ManagerBasedEnv`` for ``ActionTerm.__init__``."""

    def __init__(self, asset: FakeArticulation, num_envs: int, device: torch.device | str) -> None:
        self.scene = _FakeScene(asset)
        self.num_envs = num_envs
        self.device = str(device)
        self.cfg = _FakeEnvCfg()
        self.sim = None


def build_action_term(
    num_envs: int = 1,
    device: torch.device | str = "cuda:0",
    checkpoint_dir: str = "gear_sonic_deploy/policy/sonic_v1_1",
    **cfg_overrides,
):
    """Construct the production ActionTerm against a fake articulation.

    Returns:
        ``(term, env, asset)``.
    """
    from gear_sonic.lab_teleop.assets.g1_sonic import G1_MODEL_12_ACTION_SCALE
    from gear_sonic.lab_teleop.mdp.actions import (
        SonicWholeBodyAction,
        SonicWholeBodyActionCfg,
    )

    asset = FakeArticulation(num_envs=num_envs, device=device)
    env = FakeEnv(asset, num_envs=num_envs, device=device)
    cfg = SonicWholeBodyActionCfg(
        asset_name="robot",
        checkpoint_dir=checkpoint_dir,
        joint_names=[".*"],
        action_scale=G1_MODEL_12_ACTION_SCALE,
        **cfg_overrides,
    )
    term = SonicWholeBodyAction(cfg, env)
    return term, env, asset


def make_reference_sequence(
    num_steps: int,
    num_envs: int = 1,
    device: torch.device | str = "cuda:0",
    seed: int = 0,
    invalid_steps: set[int] | None = None,
) -> list[torch.Tensor]:
    """Deterministic 95-wide reference frames.

    Root quaternions are normalized so the heading math is well-conditioned, and the validity
    flag is driven explicitly so invalid-reference handling can be exercised.

    Args:
        num_steps: Frames to generate.
        num_envs: Batch width.
        device: Device to allocate on.
        seed: Generator seed.
        invalid_steps: Step indices whose validity flag is forced to 0.

    Returns:
        List of ``(num_envs, 95)`` tensors.
    """
    device = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    invalid_steps = invalid_steps or set()
    out: list[torch.Tensor] = []
    for step in range(num_steps):
        ref = torch.randn(num_envs, SONIC_REFERENCE_DIM, generator=gen) * 0.3
        quat = torch.randn(num_envs, 4, generator=gen)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        ref[:, SonicReferenceSlice.ROOT_QUAT] = quat
        ref[:, SonicReferenceSlice.VALID] = 0.0 if step in invalid_steps else 1.0
        out.append(ref.to(device))
    return out


def make_robot_state_sequence(
    num_steps: int,
    num_envs: int = 1,
    num_joints: int = 29,
    device: torch.device | str = "cuda:0",
    seed: int = 1,
) -> list[dict[str, torch.Tensor]]:
    """Deterministic articulation state per step, to be written into the fake asset."""
    device = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out: list[dict[str, torch.Tensor]] = []
    for _ in range(num_steps):
        quat = torch.randn(num_envs, 4, generator=gen)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        gravity = torch.randn(num_envs, 3, generator=gen)
        gravity = gravity / gravity.norm(dim=-1, keepdim=True)
        out.append(
            {
                "joint_pos": (torch.randn(num_envs, num_joints, generator=gen) * 0.2).to(device),
                "joint_vel": (torch.randn(num_envs, num_joints, generator=gen) * 0.5).to(device),
                "root_quat_w": quat.to(device),  # XYZW
                "root_ang_vel_b": (torch.randn(num_envs, 3, generator=gen) * 0.4).to(device),
                "projected_gravity_b": gravity.to(device),
            }
        )
    return out


def apply_robot_state(asset: FakeArticulation, state: dict[str, torch.Tensor]) -> None:
    """Write one step of articulation state into the fake asset."""
    asset.data.joint_pos.torch.copy_(state["joint_pos"])
    asset.data.joint_vel.torch.copy_(state["joint_vel"])
    asset.data.root_quat_w.torch.copy_(state["root_quat_w"])
    asset.data.root_ang_vel_b.torch.copy_(state["root_ang_vel_b"])
    asset.data.projected_gravity_b.torch.copy_(state["projected_gravity_b"])
