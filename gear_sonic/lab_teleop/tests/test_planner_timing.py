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


def test_joint_mapping_is_a_gather_resolved_by_name() -> None:
    """``isaaclab_to_mujoco_dof`` is a gather: ``mujoco[i] = isaaclab[m[i]]``.

    Resolved by **name**, not by index arithmetic. The previous test in this file compared an
    index list against the one in ``policy_parameters.hpp`` and stopped there, which cannot
    detect the array being applied in the wrong direction -- both readings are permutations of
    0..28 and both "look right" against a static fake robot whose joint angles are all zero.

    Under the correct reading MuJoCo slots 0-11 are the left leg then the right leg, the canonical
    G1 ordering. Under the scatter reading slot 2 holds ``right_shoulder_pitch_joint``.
    """
    import numpy as np

    from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
    from gear_sonic.lab_teleop.assets.g1_sonic import G1_ISAACLAB_TO_MUJOCO_MAPPING
    from gear_sonic.lab_teleop.mdp.modal_actions import LOWER_BODY_JOINTS

    mapping = np.asarray(G1_ISAACLAB_TO_MUJOCO_MAPPING["isaaclab_to_mujoco_dof"], dtype=np.int64)
    isaac_names = np.asarray(G1_ISAACLab_ORDER, dtype=object)

    mujoco_names = isaac_names[mapping]  # the gather reading
    assert list(mujoco_names[:12]) == list(LOWER_BODY_JOINTS)

    scattered = np.empty(len(isaac_names), dtype=object)
    scattered[mapping] = isaac_names  # the scatter reading, for contrast
    assert scattered[2] == "right_shoulder_pitch_joint"
    assert list(scattered[:12]) != list(LOWER_BODY_JOINTS)


def test_robot_qpos_places_each_joint_in_its_mujoco_slot() -> None:
    """End-to-end through the real term: a distinct angle per joint must land where it belongs.

    Uses a different value per joint precisely so a permutation cannot pass. The planner
    conditions on this pose, so getting it wrong asks the graph to plan from a body the robot is
    not in.
    """
    import numpy as np
    import torch

    from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
    from gear_sonic.lab_teleop.assets.g1_sonic import (
        G1_ISAACLAB_TO_MUJOCO_MAPPING,
        G1_MODEL_12_ACTION_SCALE,
    )
    from gear_sonic.lab_teleop.mdp.modal_actions import (
        SonicModalWholeBodyAction,
        SonicModalWholeBodyActionCfg,
    )
    from gear_sonic.lab_teleop.tests.sonic_action_harness import FakeArticulation, FakeEnv

    asset = FakeArticulation(num_envs=1, device="cpu")
    env = FakeEnv(asset, num_envs=1, device="cpu")
    env.step_dt = 0.02
    term = SonicModalWholeBodyAction(
        SonicModalWholeBodyActionCfg(
            asset_name="robot",
            checkpoint_dir="gear_sonic_deploy/policy/low_latency",
            joint_names=[".*"],
            action_scale=G1_MODEL_12_ACTION_SCALE,
        ),
        env,
    )
    angles = torch.arange(len(G1_ISAACLab_ORDER), dtype=torch.float32) * 0.01
    asset.data.joint_pos.torch[0] = angles

    qpos = term._robot_qpos()  # noqa: SLF001
    mapping = np.asarray(G1_ISAACLAB_TO_MUJOCO_MAPPING["isaaclab_to_mujoco_dof"], dtype=np.int64)
    for mujoco_slot, isaac_index in enumerate(mapping):
        assert qpos[7 + mujoco_slot] == pytest.approx(float(angles[isaac_index]))

    # And the leg selector must pull the legs back out of that MuJoCo-ordered pose.
    legs = qpos[7:][term._lower_indices_mujoco]  # noqa: SLF001
    from gear_sonic.lab_teleop.mdp.modal_actions import LOWER_BODY_JOINTS

    expected = [float(angles[G1_ISAACLab_ORDER.index(n)]) for n in LOWER_BODY_JOINTS]
    assert list(legs) == pytest.approx(expected)


def _build_term(checkpoint: str = "gear_sonic_deploy/policy/low_latency"):
    """A modal action term on a fake articulation, plus a neutral-but-valid action vector."""
    import numpy as np
    import torch

    from gear_sonic.lab_teleop.assets.g1_sonic import G1_MODEL_12_ACTION_SCALE
    from gear_sonic.lab_teleop.mdp.modal_actions import (
        SONIC_MODAL_ACTION_DIM,
        SonicModalWholeBodyAction,
        SonicModalWholeBodyActionCfg,
    )
    from gear_sonic.lab_teleop.retargeters.sonic_fullbody_retargeter import (
        SONIC_REFERENCE_DIM,
        SonicReferenceSlice,
    )
    from gear_sonic.lab_teleop.tests.sonic_action_harness import FakeArticulation, FakeEnv

    asset = FakeArticulation(num_envs=1, device="cpu")
    env = FakeEnv(asset, num_envs=1, device="cpu")
    env.step_dt = 0.02
    term = SonicModalWholeBodyAction(
        SonicModalWholeBodyActionCfg(
            asset_name="robot",
            checkpoint_dir=checkpoint,
            joint_names=[".*"],
            action_scale=G1_MODEL_12_ACTION_SCALE,
        ),
        env,
    )
    term.reset()

    reference = np.zeros(SONIC_REFERENCE_DIM, dtype=np.float32)
    reference[SonicReferenceSlice.ROOT_QUAT.start] = 1.0
    reference[SonicReferenceSlice.VR3_ORN][0::4] = 1.0
    reference[SonicReferenceSlice.VALID] = 1.0
    action = torch.zeros(1, SONIC_MODAL_ACTION_DIM)
    action[0, :SONIC_REFERENCE_DIM] = torch.from_numpy(reference)
    return term, asset, action, SONIC_REFERENCE_DIM


def test_entering_teleop_plans_once_and_lets_the_idle_trajectory_play() -> None:
    """Entry must build the idle plan and stop, not replan it away on the same step.

    ``_advance_planner`` substitutes idle directions for a centred stick before planning. If the
    stored ``_last_command`` is not in that same canonical form, the comparison never matches and
    the idle trajectory is replaced before it delivers a single frame -- the initialization is
    real but invisible, and the robot runs a walk clip at zero speed instead.
    """
    from gear_sonic.lab_teleop.mdp.sonic_planner import PLANNER_CLIP_IDLE

    term, _asset, action, ref_dim = _build_term()
    planner = term._ensure_planner()  # noqa: SLF001
    real_plan, calls = planner.plan, []

    def counting(command, mode=None):
        calls.append(mode)
        return real_plan(command, mode) if mode is not None else real_plan(command)

    planner.plan = counting

    action[0, ref_dim] = 2.0  # smpl
    for _ in range(5):
        term.process_actions(action)
        term.apply_actions()
    assert calls == [], "the planner must not run while in smpl mode"

    action[0, ref_dim] = 1.0  # enter teleop, operator commanding nothing
    term.process_actions(action)
    term.apply_actions()
    assert calls == [PLANNER_CLIP_IDLE], f"entry should plan exactly once, as IDLE; got {calls}"

    for _ in range(10):
        term.process_actions(action)
        term.apply_actions()
    assert calls == [PLANNER_CLIP_IDLE], "a stationary operator must not trigger replans"


def test_returning_to_smpl_clears_the_teleop_encoder_slots() -> None:
    """No teleop values may survive into an ``smpl`` observation.

    The checkpoint's contract is that terms outside the active mode are zero. The encoder
    observation is one persistent buffer and ``fill_smpl_encoder_obs`` rewrites only the ``smpl``
    blocks, so without an explicit clear every ``smpl`` frame after a walking excursion carries
    the last planner window and anchor orientation into the encoder.
    """
    import torch

    from gear_sonic.lab_teleop.mdp.modal_actions import (
        TELEOP_ANCHOR_ORI,
        TELEOP_LOWER_POS,
        TELEOP_LOWER_VEL,
        TELEOP_VR3_ORN,
        TELEOP_VR3_POS,
    )

    term, _asset, action, ref_dim = _build_term()
    blocks = (
        TELEOP_ANCHOR_ORI,
        TELEOP_LOWER_POS,
        TELEOP_LOWER_VEL,
        TELEOP_VR3_POS,
        TELEOP_VR3_ORN,
    )

    for cycle in range(3):
        action[0, ref_dim] = 1.0
        for _ in range(20):
            term.process_actions(action)
            term.apply_actions()
        obs = term._policy.encoder_obs[0]  # noqa: SLF001
        assert sum(float(obs[b].abs().sum()) for b in blocks) > 0.0, "teleop mode should fill them"

        action[0, ref_dim] = 2.0
        for _ in range(10):
            term.process_actions(action)
            term.apply_actions()
        obs = term._policy.encoder_obs[0]  # noqa: SLF001
        leaked = sum(float(obs[b].abs().sum()) for b in blocks)
        assert leaked == pytest.approx(0.0), f"cycle {cycle}: {leaked} leaked into smpl mode"
        assert bool(torch.isfinite(term._policy.encoder_obs).all())  # noqa: SLF001


def test_right_stick_heading_matches_the_deployed_gamepad() -> None:
    """Turn rate, movement binning and both sign conventions.

    Mirrors ``gamepad_manager.hpp:751-763``. The signs are the easy part to get backwards:
    ``rot_vel_z = -right_stick_x`` while upstream *subtracts* ``right_stick_x``, so the negations
    cancel; and upstream's ``- pi/2`` is not reproduced because ``vel_x = +left_stick_y`` already
    puts forward at zero here.
    """
    import math

    from gear_sonic.lab_teleop.retargeters.sonic_command_retargeter import (
        MOVEMENT_BIN_RAD,
        TURN_RATE_RAD_S,
    )

    assert TURN_RATE_RAD_S == pytest.approx(1.0)  # 0.02 rad/tick at 50 Hz
    assert MOVEMENT_BIN_RAD == pytest.approx(math.pi / 4)

    dt, facing = 0.02, 0.0
    for _ in range(50):  # one second of full deflection
        facing += TURN_RATE_RAD_S * 1.0 * dt
    assert math.degrees(facing) == pytest.approx(57.3, abs=0.5)

    def movement(facing_rad: float, vel_x: float, vel_y: float) -> float:
        binned = round(math.atan2(vel_y, vel_x) / MOVEMENT_BIN_RAD) * MOVEMENT_BIN_RAD
        return math.degrees(binned + facing_rad)

    # Forward on the stick means forward along the robot's heading, not along world +X.
    assert movement(math.radians(90), 1.0, 0.0) == pytest.approx(90.0)
    assert movement(math.radians(90), 1.0, 1.0) == pytest.approx(135.0)
    # Sub-sector stick angles snap to the nearest of the eight directions.
    assert movement(0.0, 1.0, 0.30) == pytest.approx(0.0)
    assert movement(0.0, 1.0, 0.60) == pytest.approx(45.0)

    # Pushing the right stick right (right_stick_x = +1, so rot_vel_z = -1) turns clockwise.
    clockwise = TURN_RATE_RAD_S * -1.0 * dt
    assert clockwise < 0.0
