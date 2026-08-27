# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture every SONIC hot-path intermediate for a deterministic input sequence.

Run once on the baseline implementation and again on the optimized one, then diff the two
``.npz`` files with :mod:`compare_sonic_golden`. Intermediates are captured by wrapping the
policy's ``encode``/``decode`` at runtime, so the production modules stay untouched by the
instrumentation itself.

Usage::

    python -m gear_sonic.lab_teleop.tests.capture_sonic_golden --out baseline.npz
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

from gear_sonic.lab_teleop.tests.sonic_action_harness import (
    apply_robot_state,
    build_action_term,
    make_reference_sequence,
    make_robot_state_sequence,
)

__all__ = ["capture_all", "main"]


def _instrument(term, sink: dict[str, list[np.ndarray]]):
    """Wrap encode/decode so each call records its inputs and outputs."""
    policy = term._policy
    real_encode, real_decode = policy.encode, policy.decode

    def encode(encoder_obs):
        sink["encoder_obs"].append(encoder_obs.detach().float().cpu().numpy().copy())
        token = real_encode(encoder_obs)
        sink["latent"].append(token.detach().float().cpu().numpy().copy())
        return token

    def decode(token, proprio):
        sink["proprio"].append(proprio.detach().float().cpu().numpy().copy())
        sink["decoder_obs"].append(
            torch.cat([token, proprio], dim=-1).detach().float().cpu().numpy().copy()
        )
        raw = real_decode(token, proprio)
        sink["raw_action"].append(raw.detach().float().cpu().numpy().copy())
        return raw

    policy.encode, policy.decode = encode, decode
    return lambda: (setattr(policy, "encode", real_encode), setattr(policy, "decode", real_decode))


def _run_scenario(
    name: str,
    num_steps: int,
    device: str,
    checkpoint_dir: str,
    invalid_steps: set[int] | None = None,
    reset_at: int | None = None,
) -> dict[str, np.ndarray]:
    """Drive one scenario end to end and return its stacked intermediates."""
    term, _env, asset = build_action_term(num_envs=1, device=device, checkpoint_dir=checkpoint_dir)
    refs = make_reference_sequence(
        num_steps, num_envs=1, device=device, seed=0, invalid_steps=invalid_steps
    )
    states = make_robot_state_sequence(num_steps, num_envs=1, device=device, seed=1)

    sink: dict[str, list[np.ndarray]] = {
        k: [] for k in ("encoder_obs", "latent", "proprio", "decoder_obs", "raw_action")
    }
    restore = _instrument(term, sink)
    targets: list[np.ndarray] = []
    try:
        term.reset()
        for step in range(num_steps):
            if reset_at is not None and step == reset_at:
                term.reset()
            apply_robot_state(asset, states[step])
            term.process_actions(refs[step])
            term.apply_actions()
            targets.append(asset.applied_target.detach().float().cpu().numpy().copy())
    finally:
        restore()

    out = {f"{name}/{k}": np.stack(v) for k, v in sink.items()}
    out[f"{name}/joint_target"] = np.stack(targets)
    out[f"{name}/delta_heading"] = term._apply_delta_heading.detach().float().cpu().numpy().copy()
    return out


def _run_buffer_scenarios(device: str) -> dict[str, np.ndarray]:
    """Exercise history/window semantics that the batch-1 ONNX graphs cannot reach.

    The shipped encoder/decoder are exported with a static batch of 1, so vectorized-env parity
    can only be checked on the pure-torch buffers. Those are the pieces the optimization actually
    rewrites, so this is where partial-reset correctness is pinned down.
    """
    from gear_sonic.lab_teleop.mdp.proprio_history import SonicProprioHistory

    num_envs, num_joints, steps = 4, 29, 12
    hist = SonicProprioHistory(num_envs=num_envs, num_joints=num_joints, device=device)
    gen = torch.Generator(device="cpu").manual_seed(7)
    hist.reset()

    flats: list[np.ndarray] = []
    for step in range(steps):
        if step == 5:
            # Partial reset of a non-contiguous subset, the case index arithmetic gets wrong.
            hist.reset(torch.tensor([0, 2], device=device))
        if step == 8:
            hist.reset(slice(None))
        frame = {
            "base_ang_vel": (torch.randn(num_envs, 3, generator=gen) * 0.3).to(device),
            "joint_pos_rel": (torch.randn(num_envs, num_joints, generator=gen) * 0.2).to(device),
            "joint_vel_rel": (torch.randn(num_envs, num_joints, generator=gen) * 0.4).to(device),
            "last_action": (torch.randn(num_envs, num_joints, generator=gen) * 0.1).to(device),
            "gravity_dir": (torch.randn(num_envs, 3, generator=gen) * 0.5).to(device),
        }
        hist.append(**frame)
        flats.append(hist.flat().detach().float().cpu().numpy().copy())
    return {"buffers/proprio_flat": np.stack(flats)}


def capture_all(device: str, checkpoint_dir: str) -> dict[str, np.ndarray]:
    """Run every scenario and return one flat dict suitable for ``np.savez``."""
    out: dict[str, np.ndarray] = {}
    out.update(_run_scenario("basic", 20, device, checkpoint_dir))
    out.update(_run_scenario("invalid_ref", 20, device, checkpoint_dir, invalid_steps={0, 1, 2, 9}))
    out.update(_run_scenario("reset_mid", 20, device, checkpoint_dir, reset_at=10))
    out.update(_run_buffer_scenarios(device))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Destination .npz path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint_dir", default="gear_sonic_deploy/policy/sonic_v1_1")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    data = capture_all(args.device, args.checkpoint_dir)
    np.savez(args.out, **data)
    print(f"wrote {args.out} with {len(data)} arrays")
    for key in sorted(data):
        print(f"  {key:34s} {data[key].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
