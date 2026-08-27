# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Differential tests: the Isaac Lab retargeter vs. the shipped PICO teleop server.

:class:`SonicFullBodyRetargeter` is a restructuring of the retargeting inside
``gear_sonic/scripts/pico_manager_thread_server.py``. These tests assert it is *numerically
identical* to that reference, so the Isaac Lab path and the real-robot path feed SONIC the same
numbers.

Run from the repo root with Isaac Lab's interpreter::

    /path/to/IsaacLab/.venv/bin/python -m pytest gear_sonic/lab_teleop/tests -q
"""

from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRot
import torch

# The upstream server imports pyzmq at module scope for transport only; none of the retargeting
# math touches it. Stub it so this test runs in an Isaac Lab venv without pulling in pyzmq.
sys.modules.setdefault("zmq", mock.MagicMock())

from gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter import (  # noqa: E402
    _SMPL_BASE_ROT_WXYZ,
    _SMPL_PARENT_INDICES,
    SONIC_REFERENCE_DIM,
    VR3_POINT_SMPL_INDICES,
    SonicFullBodyRetargeter,
    SonicFullBodyRetargeterConfig,
    SonicReferenceSlice,
)
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa  # noqa: E402

import gear_sonic.scripts.pico_manager_thread_server as upstream  # noqa: E402  isort: skip

_TOL = 1e-5


def _make_body_frame(seed: int) -> np.ndarray:
    """Synthesize one ``(24, 7)`` XR body frame: ``[x, y, z, qx, qy, qz, qw]`` per joint."""
    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(24, 3)).astype(np.float32)
    quats = rng.normal(size=(24, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return np.concatenate([positions, quats], axis=1).astype(np.float32)


def _upstream_wrists(smpl_pose: torch.Tensor) -> np.ndarray:
    """Re-derive the six G1 wrist angles exactly as ``pico_manager_thread_server.py:1418-1476``."""
    pose = smpl_pose.detach().cpu().numpy()[:, :63].reshape(-1, 21, 3)
    axis = np.array([0, 1, 0])
    _, l_swing = decompose_rotation_aa(pose[:, 17], axis)
    _, r_swing = decompose_rotation_aa(pose[:, 18], axis)
    l_swing_e = sRot.from_quat(l_swing[:, [1, 2, 3, 0]]).as_euler("XYZ")
    r_swing_e = sRot.from_quat(r_swing[:, [1, 2, 3, 0]]).as_euler("XYZ")
    l_wrist_e = sRot.from_rotvec(pose[:, 19]).as_euler("XYZ")
    r_wrist_e = sRot.from_rotvec(pose[:, 20]).as_euler("XYZ")

    joint_pos = np.zeros(29)
    joint_pos[23] = l_swing_e[:, 0][0] + l_wrist_e[:, 0][0]
    joint_pos[25] = -(-l_wrist_e[:, 1][0])
    joint_pos[27] = l_swing_e[:, 2][0] + l_wrist_e[:, 2][0]
    joint_pos[24] = -(r_swing_e[:, 0][0] + r_wrist_e[:, 0][0])
    joint_pos[26] = -r_wrist_e[:, 1][0]
    joint_pos[28] = r_swing_e[:, 2][0] + r_wrist_e[:, 2][0]
    return joint_pos[[23, 24, 25, 26, 27, 28]]


@pytest.fixture
def retargeter() -> SonicFullBodyRetargeter:
    return SonicFullBodyRetargeter(
        SonicFullBodyRetargeterConfig(device="cpu"), name="test"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_matches_upstream(retargeter: SonicFullBodyRetargeter, seed: int) -> None:
    """SMPL joints, root quaternion and wrist angles all match the shipped implementation."""
    body = _make_body_frame(seed)
    expected = upstream.compute_from_body_poses(
        _SMPL_PARENT_INDICES, torch.device("cpu"), body
    )
    actual = retargeter._retarget(body)  # noqa: SLF001

    np.testing.assert_allclose(
        actual[SonicReferenceSlice.SMPL_JOINTS],
        expected["smpl_joints_local"].reshape(-1).numpy(),
        atol=_TOL,
    )
    np.testing.assert_allclose(
        actual[SonicReferenceSlice.ROOT_QUAT],
        expected["global_orient_quat"].reshape(-1).numpy(),
        atol=_TOL,
    )
    np.testing.assert_allclose(
        actual[SonicReferenceSlice.WRIST_JOINT_POS],
        _upstream_wrists(expected["smpl_pose"]),
        atol=_TOL,
    )


def test_output_layout(retargeter: SonicFullBodyRetargeter) -> None:
    """The flat frame is the advertised width and flags itself valid."""
    frame = retargeter._retarget(_make_body_frame(1))  # noqa: SLF001
    assert frame.shape == (SONIC_REFERENCE_DIM,) == (95,)
    assert frame.dtype == np.float32
    assert frame[SonicReferenceSlice.VALID] == pytest.approx(1.0)


def test_holds_last_good_frame_on_tracking_loss(
    retargeter: SonicFullBodyRetargeter,
) -> None:
    """After a good frame, a dropout re-emits it with the valid flag cleared."""
    good = retargeter._retarget(_make_body_frame(1))  # noqa: SLF001
    retargeter._last_good = good  # noqa: SLF001
    retargeter._have_good_frame = True  # noqa: SLF001

    held = retargeter._fallback_frame()  # noqa: SLF001
    assert held[SonicReferenceSlice.VALID] == pytest.approx(0.0)
    np.testing.assert_array_equal(
        held[SonicReferenceSlice.SMPL_JOINTS], good[SonicReferenceSlice.SMPL_JOINTS]
    )


def test_neutral_frame_before_first_good_frame() -> None:
    """With no good frame yet the fallback is neutral: zeros, but *unit* quaternions.

    A zero quaternion has zero norm, so the heading maths downstream divides by it and NaN
    reaches the physics solver. Every quaternion slot must therefore be identity, not zero.
    """
    fresh = SonicFullBodyRetargeter(
        SonicFullBodyRetargeterConfig(device="cpu"), name="fresh"
    )
    frame = fresh._fallback_frame()  # noqa: SLF001

    expected = np.zeros(SONIC_REFERENCE_DIM, dtype=np.float32)
    expected[SonicReferenceSlice.ROOT_QUAT.start] = 1.0
    expected[SonicReferenceSlice.VR3_ORN][0::4] = 1.0
    np.testing.assert_array_equal(frame, expected)


def test_vr_3point_orientations_match_the_training_definition(
    retargeter: SonicFullBodyRetargeter,
) -> None:
    """The shipped closed form equals the long way round through the FK chain.

    ``vr_3point_local_orn_target`` is ``quat_inv(anchor_quat_w) * point_quat_w``
    (``gear_sonic/envs/manager_env/mdp/observations.py:1430``). Here the anchor is the retargeted
    root after ``smpl_root_ytoz_up`` and ``remove_smpl_base_rot``, and the point rotation comes
    from chaining the SMPL local rotations. The retargeter skips both because the chain
    telescopes; this test rebuilds them the long way and demands the same answer.
    """
    body_poses = _make_body_frame(7)
    actual = retargeter._retarget(body_poses)[SonicReferenceSlice.VR3_ORN].reshape(3, 4)  # noqa: SLF001

    # Global XR rotations, exactly as ``_xr_to_smpl_local`` forms them.
    globals_ = sRot.from_quat(body_poses[:, [6, 3, 4, 5]], scalar_first=True)
    globals_ = globals_ * sRot.from_euler("y", 180, degrees=True)
    local = [
        globals_[i] if _SMPL_PARENT_INDICES[i] == -1
        else globals_[_SMPL_PARENT_INDICES[i]].inv() * globals_[i]
        for i in range(24)
    ]

    # Anchor: Y-up -> Z-up left-multiplies the root; remove_smpl_base_rot right-multiplies it.
    root_z_up = sRot.from_rotvec([np.pi / 2, 0.0, 0.0]) * local[0]
    base = sRot.from_quat(_SMPL_BASE_ROT_WXYZ, scalar_first=True)
    anchor = root_z_up * base.inv()

    # Point rotations by walking the FK chain from that same root.
    chain = [None] * 24
    chain[0] = root_z_up
    for i in range(1, 24):
        chain[i] = chain[_SMPL_PARENT_INDICES[i]] * local[i]

    for slot, joint in enumerate(VR3_POINT_SMPL_INDICES):
        expected = anchor.inv() * chain[joint]
        residual = (sRot.from_quat(actual[slot], scalar_first=True).inv() * expected).magnitude()
        assert residual == pytest.approx(0.0, abs=_TOL)


def test_pipeline_builds() -> None:
    """The retargeting graph wires up and exposes a single ``action`` output."""
    from gear_sonic.lab_teleop.retargeters.pipeline import make_sonic_full_pipeline_builder

    combiner = make_sonic_full_pipeline_builder()()
    mapping = getattr(combiner, "_output_mapping", None) or getattr(
        combiner, "output_mapping", {}
    )
    assert list(mapping.keys()) == ["action"]


def test_isaaclab_quat_conversion_is_wxyz() -> None:
    """Isaac Lab 3.0 hands out XYZW; gear_sonic's helpers are all ``w_last=False``.

    Regression guard: feeding an Isaac Lab quaternion straight into ``calc_heading_quat`` yields a
    valid-looking unit quaternion with the wrong orientation, which corrupted the heading term and
    made the robot spin continuously. See ``migrating_to_isaaclab_3-0.rst:1317``.
    """
    from gear_sonic.lab_teleop.mdp.actions import isaaclab_quat_to_wxyz

    # XYZW ordering, distinct components so a wrong permutation cannot pass.
    quat_xyzw = torch.tensor([[0.1, 0.2, 0.3, 0.9]])
    expected_wxyz = torch.tensor([[0.9, 0.1, 0.2, 0.3]])
    torch.testing.assert_close(isaaclab_quat_to_wxyz(quat_xyzw), expected_wxyz)

    # Batched / multi-frame inputs keep their leading dims.
    batched = torch.randn(4, 10, 4)
    assert isaaclab_quat_to_wxyz(batched).shape == batched.shape
    torch.testing.assert_close(isaaclab_quat_to_wxyz(batched)[..., 0], batched[..., 3])
