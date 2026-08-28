# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Timing semantics of the planner and of the SMPL -> teleop handoff.

Four clocks meet in the walking path and conflating any two of them destabilizes the robot:

===================  ========  =====================================================
clock                rate      source of truth
===================  ========  =====================================================
physics              200 Hz    ``sim.dt`` -- environment
control / ActionTerm  50 Hz    ``env.step_dt`` = ``sim.dt * decimation`` -- environment
render               100 Hz    ``sim.dt * render_interval`` -- irrelevant to control
planner native        30 Hz    the planner graph -- model property
SONIC reference       50 Hz    the checkpoint -- model property
===================  ========  =====================================================

These tests pin the model-derived rates and the observation semantics that depend on them, so a
future change to ``sim.dt`` or a re-exported graph cannot silently reinterpret them.

Run from the repo root with the Isaac Lab interpreter::

    /path/to/venv/bin/python -m pytest gear_sonic/lab_teleop/tests/test_planner_timing.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.lab_teleop.mdp.sonic_planner import (
    PLANNER_NATIVE_DT,
    PLANNER_QPOS_DIM,
    SONIC_REFERENCE_DT,
    PlannerMotion,
)


def _ramp_plan(frames: int = 10) -> np.ndarray:
    """A native-rate plan whose every channel is a known linear ramp in frame index."""
    plan = np.zeros((frames, PLANNER_QPOS_DIM), dtype=np.float32)
    plan[:, 0] = np.arange(frames)  # root x
    plan[:, 3] = 1.0  # identity quaternion
    plan[:, 7:] = np.arange(frames)[:, None]  # every joint
    return plan


def test_resampling_stretches_30hz_onto_the_50hz_clock() -> None:
    """A 30 Hz plan must occupy the same wall-clock duration after resampling, not 3/5 of it.

    This is the bug that played every plan at ``50/30 = 1.67x``: one native frame was consumed per
    control step, so a 1.43 s trajectory finished in 0.86 s and the robot was asked to walk 67%
    faster than commanded.
    """
    native = _ramp_plan(31)  # 30 intervals at 1/30 s = exactly 1.0 s
    motion = PlannerMotion(native)

    assert motion.dt == pytest.approx(SONIC_REFERENCE_DT)
    assert motion.duration == pytest.approx(1.0, abs=SONIC_REFERENCE_DT)
    # Upstream keeps num_pred_frames * 50/30 frames, rounded down.
    assert len(motion.qpos) == pytest.approx(51, abs=1)


def test_sampling_is_by_time_not_by_frame() -> None:
    """Sampling at t seconds must land on the ramp value the native plan holds at that time."""
    motion = PlannerMotion(_ramp_plan(31))
    for t in (0.0, 0.1, 0.5, 0.97):
        expected = t / PLANNER_NATIVE_DT  # the ramp equals the native frame index
        assert motion.sample_qpos(t)[7] == pytest.approx(expected, abs=0.02)


def test_velocities_use_the_resampled_timestep() -> None:
    """Velocity must be d(pos)/dt on the *resampled* series.

    Differencing native frames but dividing by the control period inflates every velocity by
    ``50/30``. Here the ramp advances one unit per native frame, i.e. 30 units/s.
    """
    motion = PlannerMotion(_ramp_plan(31))
    assert np.abs(motion.qvel[:-1] - 30.0).max() < 0.5


def test_out_of_range_samples_hold_the_final_pose() -> None:
    """Past the end, repeat the last frame -- the encoder's own documented rule."""
    motion = PlannerMotion(_ramp_plan(11))
    last = motion.sample_qpos(motion.duration)
    assert motion.sample_qpos(motion.duration + 5.0) == pytest.approx(last)


def test_quaternions_are_slerped_not_lerped() -> None:
    """A resampled quaternion stays unit-norm; component-wise lerp would not."""
    native = _ramp_plan(11)
    native[:, 3:7] = 0.0
    angles = np.linspace(0.0, np.pi / 2, 11)
    native[:, 3] = np.cos(angles / 2)  # w
    native[:, 6] = np.sin(angles / 2)  # z: yaw sweep
    motion = PlannerMotion(native)
    norms = np.linalg.norm(motion.qpos[:, 3:7], axis=1)
    assert np.abs(norms - 1.0).max() < 1e-5


def test_lower_body_indices_match_the_deployment_stack() -> None:
    """The encoder wants whole-left-leg-then-whole-right-leg, MuJoCo's grouping.

    Isaac Lab's own joint order interleaves left/right and puts waist joints among the legs, so
    taking "the first twelve" yields three waist joints, a shoulder, and no ankles at all.
    """
    from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
    from gear_sonic.lab_teleop.mdp.modal_actions import (
        LOWER_BODY_ISAACLAB_INDICES,
        LOWER_BODY_JOINTS,
    )

    # policy_parameters.hpp:92
    assert LOWER_BODY_ISAACLAB_INDICES == (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18)
    resolved = [G1_ISAACLab_ORDER[i] for i in LOWER_BODY_ISAACLAB_INDICES]
    assert resolved == list(LOWER_BODY_JOINTS)
    assert all("ankle" not in n for n in G1_ISAACLab_ORDER[:12])  # why the naive slice was wrong
    assert sum("ankle" in n for n in resolved) == 4


def test_reference_window_is_a_future_lookahead() -> None:
    """``10frame_step5`` is 10 samples 0.1 s apart, spanning 0.9 s ahead of the current plan time."""
    from gear_sonic.lab_teleop.mdp.modal_actions import (
        TELEOP_REFERENCE_FRAMES,
        TELEOP_REFERENCE_STRIDE_S,
    )

    assert TELEOP_REFERENCE_STRIDE_S == pytest.approx(0.1)
    offsets = np.arange(TELEOP_REFERENCE_FRAMES) * TELEOP_REFERENCE_STRIDE_S
    assert offsets[0] == pytest.approx(0.0)
    assert offsets[-1] == pytest.approx(0.9)


def test_position_and_velocity_blocks_do_not_overlap() -> None:
    """Two contiguous 120-wide blocks, not per-frame interleaving.

    Interleaving wrote velocities into the positions the encoder reads, which is a much larger
    insult to the model than any timing error.
    """
    from gear_sonic.lab_teleop.mdp.modal_actions import (
        TELEOP_LOWER_BODY,
        TELEOP_LOWER_POS,
        TELEOP_LOWER_VEL,
    )

    assert TELEOP_LOWER_POS.stop - TELEOP_LOWER_POS.start == 120
    assert TELEOP_LOWER_VEL.stop - TELEOP_LOWER_VEL.start == 120
    assert TELEOP_LOWER_POS.stop == TELEOP_LOWER_VEL.start
    assert (TELEOP_LOWER_POS.start, TELEOP_LOWER_VEL.stop) == (
        TELEOP_LOWER_BODY.start,
        TELEOP_LOWER_BODY.stop,
    )
