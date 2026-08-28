# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Target-velocity motion planner that supplies SONIC's ``teleop``-mode lower-body reference.

SONIC's encoder has three modes, declared by the checkpoint itself
(``observation_config.yaml``: ``g1`` = 0, ``teleop`` = 1, ``smpl`` = 2). Our environments have so
far run ``smpl`` exclusively -- full-body tracking. ``teleop`` mode is the "walk it around with the
sticks" mode, and it does **not** consume joystick values directly: it consumes a *lower-body
reference*, which this planner generates.

Graph I/O, read off ``planner_sonic.onnx`` rather than documentation::

    context_mujoco_qpos  (1, 4, 36)    4 frames of history: 7 root + 29 joints, MuJoCo order
    target_vel           (1,)          speed
    movement_direction   (1, 3)        world-frame XY unit vector, z = 0
    facing_direction     (1, 3)        world-frame XY unit vector, z = 0
    height               (1,)          hip height
    mode, random_seed, has_specific_target, specific_target_*, allowed_pred_num_tokens
        ->
    mujoco_qpos          (1, 64, 36)   64 planned future frames
    num_pred_frames      (1,)

Why this lives in Isaac Lab, not Isaac Teleop
---------------------------------------------
``context_mujoco_qpos`` is filled from the robot's **measured** state -- the reference
implementation appends ``mj_data.qpos`` each tick
(``motionbricks/.../demo/controllers.py:160``). The planner is therefore closed-loop on the robot,
and running it upstream of ``env.step()`` would feed it a stale pose whose error compounds through
the autoregressive rollout. Only the operator's *command* crosses the boundary from Isaac Teleop
(8 scalars); the 36-wide state stays local.

Frame conventions
-----------------
``movement_direction`` carries the throttle in its magnitude and ``facing_direction`` is a
world-frame XY unit vector. Both come from the operator and are passed through untouched: a
centred stick sends zero movement and the heading still holds, which is how the operator turns on
the spot (``pico_manager_thread_server.py:1800-1830``).
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

__all__ = [
    "PLANNER_BLEND_S",
    "PLANNER_CLIP_IDLE",
    "PLANNER_CLIP_SLOW_WALK",
    "PLANNER_CLIP_WALK",
    "PLANNER_CONTEXT_FRAMES",
    "PLANNER_QPOS_DIM",
    "PLANNER_DEFAULT_HEIGHT",
    "PLANNER_IDLE_COMMAND",
    "PLANNER_LOOKAHEAD_S",
    "PLANNER_NATIVE_DT",
    "PLANNER_NATIVE_HZ",
    "PLANNER_PERIODIC_REPLAN_S",
    "PLANNER_TICK_S",
    "PLANNER_WALK_TOKENS",
    "SONIC_PLANNER_COMMAND_DIM",
    "SONIC_REFERENCE_DT",
    "SONIC_REFERENCE_HZ",
    "PlannerMotion",
    "SonicVelocityPlanner",
]

#: Frames of robot history the planner conditions on.
PLANNER_CONTEXT_FRAMES = 4

#: ``qpos`` width: 3 root position + 4 root quaternion + 29 joints.
PLANNER_QPOS_DIM = 36

#: Operator command width: ``[target_vel, movement_dir(3), facing_dir(3), height]``.
SONIC_PLANNER_COMMAND_DIM = 8

#: **Planner model property.** The graph emits frames at 30 Hz regardless of what the simulation
#: runs at, so this must never be derived from ``sim.dt``. Confirmed empirically as well as from
#: ``docs/source/references/planner_onnx.md``: commanding a steady 0.5/0.8/1.0 m/s walk and
#: measuring the per-frame root displacement over the settled tail of the plan implies 29.4/30.9/
#: 32.5 Hz.
PLANNER_NATIVE_HZ = 30.0
PLANNER_NATIVE_DT = 1.0 / PLANNER_NATIVE_HZ

#: **SONIC checkpoint property.** The reference timeline the encoder's ``Nframe_stepS`` windows are
#: expressed on. ``step5`` means five ticks of *this* clock -- 0.1 s -- not five environment steps
#: (``docs/source/references/observation_config.md:96``). Also never derived from ``sim.dt``.
SONIC_REFERENCE_HZ = 50.0
SONIC_REFERENCE_DT = 1.0 / SONIC_REFERENCE_HZ

#: Planner look-ahead when building context from the current plan: upstream's
#: ``motion_look_ahead_steps = 2`` at 50 Hz (``localmotion_kplanner.hpp:215``).
PLANNER_LOOKAHEAD_S = 2.0 * SONIC_REFERENCE_DT

#: Cross-fade duration between an old and a newly arrived plan: upstream's 8-frame blend at 50 Hz.
#:
#: **Currently unused, deliberately.** Upstream blends successive trajectories over this window;
#: we do not, because the instability this integration suffered came from the timing and layout
#: semantics rather than from plan-to-plan discontinuity, and fixing those resolved it without
#: blending. Kept because it is the known next lever if changing walking direction or speed shows
#: a visible step: implement the blend in physical time against this duration, not as a frame
#: count (``planner_onnx.md:398-410``).
PLANNER_BLEND_S = 8.0 * SONIC_REFERENCE_DT

#: Planner thread cadence upstream (10 Hz) and the periodic replan interval for walking.
PLANNER_TICK_S = 0.1
PLANNER_PERIODIC_REPLAN_S = 1.0

#: Standing height the upstream planner canonicalizes its initial context to
#: (``InitializeContext``: x = y = 0, z = default height, identity quaternion).
PLANNER_DEFAULT_HEIGHT = 0.78

#: The zero-movement command used to seed the first plan on entry to walking mode. Held as a
#: constant so the caller can recognise "still idle" and avoid replanning the idle plan away
#: before it has played a single frame.
PLANNER_IDLE_COMMAND = np.array([0.0, 1, 0, 0, 1, 0, 0, PLANNER_DEFAULT_HEIGHT], dtype=np.float32)


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-arc spherical interpolation between two wxyz quaternions.

    Args:
        q0: ``(4,)`` start quaternion, wxyz.
        q1: ``(4,)`` end quaternion, wxyz.
        alpha: Interpolation weight in ``[0, 1]``.

    Returns:
        ``(4,)`` interpolated unit quaternion, wxyz.
    """
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # take the shorter arc; q and -q are the same rotation
        q1 = -q1
        dot = -dot
    if dot > 0.9995:  # nearly parallel: lerp is numerically better behaved than slerp
        out = q0 + alpha * (q1 - q0)
        return out / (np.linalg.norm(out) + 1e-12)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / sin_theta


class PlannerMotion:
    """A planned trajectory resampled onto SONIC's reference clock, sampled by *time*.

    The planner graph speaks 30 Hz and SONIC speaks 50 Hz. Keeping the two apart is the whole
    point of this class: a bare array of planner frames invites the reader to advance it one entry
    per control step, which plays the motion at ``50/30 = 1.67x``. Everything here is addressed in
    seconds.

    Mirrors ``ResampleGeneratedSequence50Hz`` in the deployment stack: linear interpolation for
    joint and root positions, slerp for the root quaternion, and joint velocities finite-differenced
    from the *resampled* series so they carry the right timestep
    (``docs/source/references/planner_onnx.md:387-396``).

    Attributes:
        dt: Sample spacing of the resampled trajectory, ``SONIC_REFERENCE_DT``.
        qpos: ``(frames, 36)`` resampled poses: 3 root position, 4 root quaternion (wxyz), 29 joints.
        qvel: ``(frames, 29)`` joint velocities derived from :attr:`qpos`.
    """

    def __init__(self, native_qpos: np.ndarray, native_dt: float = PLANNER_NATIVE_DT) -> None:
        """Resample a native-rate plan onto the reference clock.

        Args:
            native_qpos: ``(n, 36)`` planner output at ``native_dt`` spacing.
            native_dt: Sample spacing of ``native_qpos``. Defaults to the planner's 30 Hz.
        """
        native = np.asarray(native_qpos, dtype=np.float32).reshape(-1, PLANNER_QPOS_DIM)
        self.dt = SONIC_REFERENCE_DT
        n_native = len(native)
        # Upstream keeps num_pred_frames * 50/30 frames, rounded down.
        n_out = max(1, int(np.floor((n_native - 1) * native_dt / self.dt)) + 1)

        self.qpos = np.empty((n_out, PLANNER_QPOS_DIM), dtype=np.float32)
        for k in range(n_out):
            f = (k * self.dt) / native_dt
            f0 = min(int(np.floor(f)), n_native - 1)
            f1 = min(f0 + 1, n_native - 1)
            alpha = float(f - f0)
            a, b = native[f0], native[f1]
            self.qpos[k, :3] = a[:3] + alpha * (b[:3] - a[:3])
            self.qpos[k, 3:7] = _slerp(a[3:7], b[3:7], alpha)
            self.qpos[k, 7:] = a[7:] + alpha * (b[7:] - a[7:])

        # Finite difference on the resampled series, holding the last frame's velocity.
        self.qvel = np.zeros((n_out, PLANNER_QPOS_DIM - 7), dtype=np.float32)
        if n_out > 1:
            self.qvel[:-1] = (self.qpos[1:, 7:] - self.qpos[:-1, 7:]) / self.dt
            self.qvel[-1] = self.qvel[-2]

    @property
    def duration(self) -> float:
        """Play time of the trajectory, in seconds."""
        return (len(self.qpos) - 1) * self.dt

    def _frame_at(self, t: float) -> tuple[int, int, float]:
        """Bracketing frame indices and blend weight for time ``t``, clamped to the ends.

        Clamping rather than wrapping matches the encoder's own out-of-range rule: "if future
        frames exceed the motion length, the last frame is repeated"
        (``docs/source/references/observation_config.md:99``).
        """
        f = max(0.0, t) / self.dt
        f0 = min(int(np.floor(f)), len(self.qpos) - 1)
        f1 = min(f0 + 1, len(self.qpos) - 1)
        return f0, f1, float(min(max(f - f0, 0.0), 1.0))

    def sample_qpos(self, t: float) -> np.ndarray:
        """Pose at time ``t`` seconds into the trajectory."""
        f0, f1, alpha = self._frame_at(t)
        a, b = self.qpos[f0], self.qpos[f1]
        out = np.empty(PLANNER_QPOS_DIM, dtype=np.float32)
        out[:3] = a[:3] + alpha * (b[:3] - a[:3])
        out[3:7] = _slerp(a[3:7], b[3:7], alpha)
        out[7:] = a[7:] + alpha * (b[7:] - a[7:])
        return out

    def sample_joints(self, times: np.ndarray, joint_indices: np.ndarray) -> np.ndarray:
        """Joint positions and velocities for a set of sample times.

        Args:
            times: ``(k,)`` times in seconds from the start of the trajectory.
            joint_indices: Indices selecting and ordering the joints to return.

        Returns:
            ``(positions, velocities)``, each ``(k, len(joint_indices))``.
        """
        pos = np.empty((len(times), len(joint_indices)), dtype=np.float32)
        vel = np.empty_like(pos)
        for i, t in enumerate(times):
            f0, f1, alpha = self._frame_at(float(t))
            jp = self.qpos[:, 7:]
            pos[i] = (jp[f0] + alpha * (jp[f1] - jp[f0]))[joint_indices]
            vel[i] = (self.qvel[f0] + alpha * (self.qvel[f1] - self.qvel[f0]))[joint_indices]
        return pos, vel


#: Planner clip ids, indexing ``clip_holder_G1.CLIPS`` (``motionbricks/.../demo/clips.py:131``).
#:
#: This is a **gait/style** selector and is unrelated to SONIC's encoder mode, despite both being
#: called "mode". Feeding ``idle`` is why a commanded ``target_vel`` can produce no motion at all:
#: measured over a 64-frame horizon at ``target_vel=1.0``, idle advances 0.008 m while walk
#: advances 1.10 m.
PLANNER_CLIP_IDLE = 0
PLANNER_CLIP_SLOW_WALK = 1
PLANNER_CLIP_WALK = 2

#: Token mask the walking clips declare. Not all-ones: that belongs to the crawling and boxing
#: styles, and it changes how many frames the graph actually predicts (44 vs 64).
PLANNER_WALK_TOKENS = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0)


class SonicVelocityPlanner:
    """Run ``planner_sonic.onnx`` and serve its planned frames one at a time.

    The graph emits 64 future frames per invocation, so it is *not* run every control step. It is
    re-planned when the buffer is exhausted or when :meth:`reset` invalidates it, and the caller
    pops one frame per step.

    Args:
        checkpoint_path: Path to ``planner_sonic.onnx``.
        device: Torch device for the returned tensors. Inference itself runs through onnxruntime;
            the graph is small next to the SONIC decoder.
        allowed_tokens: Clip token mask; see :data:`PLANNER_WALK_TOKENS`.
        warmup: Run one throwaway inference during construction. The first call into a fresh
            onnxruntime session pays lazy kernel and CUDA-graph setup -- measured at ~490 ms
            against a 20 ms control period -- so paying it here keeps that cost out of the
            control loop.

    Raises:
        FileNotFoundError: If the planner graph is absent.
    """

    def __init__(
        self,
        checkpoint_path: str | pathlib.Path,
        device: torch.device | str = "cuda:0",
        allowed_tokens: tuple[int, ...] = PLANNER_WALK_TOKENS,
        warmup: bool = True,
    ) -> None:
        import onnxruntime as ort

        path = pathlib.Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"SONIC velocity planner not found: {path}\n"
                "Fetch it with:\n"
                "    python download_from_hf.py --low-latency"
            )
        self.device = torch.device(device)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if self.device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": self.device.index or 0}))
        self._session = ort.InferenceSession(str(path), options, providers=providers)
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._horizon = self._session.get_outputs()[0].shape[1]
        self._allowed_tokens = tuple(allowed_tokens)

        self._context = np.zeros((1, PLANNER_CONTEXT_FRAMES, PLANNER_QPOS_DIM), dtype=np.float32)
        self._seeded = False

        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Run the graph once on a neutral standing pose, then discard everything it produced.

        Leaves no state behind: :meth:`reset` clears the plan, the cursor and the seeded flag, so
        the first real call still plans from the caller's own context rather than from this.
        """
        self._context[0, :, 2] = 0.76  # hip height
        self._context[0, :, 3] = 1.0  # wxyz identity root quaternion
        self.plan(np.array([0.0, 1, 0, 0, 1, 0, 0, PLANNER_DEFAULT_HEIGHT], dtype=np.float32),
                  PLANNER_CLIP_WALK)
        self.reset()

    @property
    def horizon(self) -> int:
        """Frames the graph plans per invocation."""
        return self._horizon

    def reset(self) -> None:
        """Drop the context history and any pending plan.

        Mandatory on episode reset: a retained context conditions the next plan on a pose the
        robot no longer has. The trajectory itself is owned by the caller.
        """
        self._context[:] = 0.0
        self._seeded = False

    def set_context(self, history: np.ndarray) -> None:
        """Replace the whole context window with a caller-owned history.

        Preferred over :meth:`push_state` when the caller already tracks the robot's recent poses,
        because it decouples *keeping the history current* from *owning the planner*. The action
        term uses this so it can maintain the history on every control step while still building
        the planner lazily on first use.

        Args:
            history: ``(PLANNER_CONTEXT_FRAMES, 36)`` recent measured poses, oldest first.
        """
        self._context[0, :] = np.asarray(history, dtype=np.float32).reshape(
            PLANNER_CONTEXT_FRAMES, PLANNER_QPOS_DIM
        )
        self._seeded = True

    def push_state(self, qpos: np.ndarray) -> None:
        """Append one measured robot pose to the planner's context.

        Args:
            qpos: ``(36,)`` MuJoCo-order pose: 3 position, 4 quaternion (wxyz), 29 joints.
        """
        frame = np.asarray(qpos, dtype=np.float32).reshape(PLANNER_QPOS_DIM)
        if not self._seeded:
            # Backfill so the first plan is conditioned on a plausible history rather than zeros,
            # matching the reference implementation's initialization.
            self._context[0, :] = frame
            self._seeded = True
        else:
            self._context[0, :-1] = self._context[0, 1:]
            self._context[0, -1] = frame

    def plan(self, command: np.ndarray, mode: int = PLANNER_CLIP_WALK) -> PlannerMotion:
        """Run the graph once and return the whole planned trajectory.

        Returns the whole trajectory rather than a frame, so a caller cannot advance a native
        30 Hz plan once per 50 Hz control step and play it at 1.67x.

        Args:
            command: ``(8,)`` ``[target_vel, movement_dir(3), facing_dir(3), height]``.
            mode: Planner clip id; see :data:`PLANNER_CLIP_WALK`.

        Returns:
            The planned motion, resampled onto SONIC's reference clock.
        """
        target_vel = np.asarray([command[0]], dtype=np.float32)
        movement = np.asarray(command[1:4], dtype=np.float32).reshape(1, 3)
        facing = np.asarray(command[4:7], dtype=np.float32).reshape(1, 3)
        height = np.asarray([command[7]], dtype=np.float32)

        feeds: dict[str, np.ndarray] = {
            "context_mujoco_qpos": self._context,
            "target_vel": target_vel,
            "movement_direction": movement,
            "facing_direction": facing,
            "height": height,
        }
        # Optional inputs the graph declares but which have sensible neutral values. Fed only when
        # present so a re-exported planner with a reduced signature still runs. Dtypes are taken
        # from the graph (mode / random_seed / has_specific_target / allowed_pred_num_tokens are
        # int64); onnxruntime rejects a float where it wants int64 rather than coercing.
        optional = {
            "mode": np.asarray([mode], dtype=np.int64),
            "random_seed": np.asarray([0], dtype=np.int64),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": np.asarray([self._allowed_tokens], dtype=np.int64),
        }
        for name, value in optional.items():
            if name in self._input_names:
                feeds[name] = value

        outputs = self._session.run(None, feeds)
        native = np.asarray(outputs[0], dtype=np.float32).reshape(-1, PLANNER_QPOS_DIM)
        # The output tensor is always 64 frames wide, but only ``num_pred_frames`` of them are
        # predictions -- 44 under the walking token mask. Consuming the tail would replay
        # uninitialized frames as if they were motion.
        num_valid = int(np.asarray(outputs[1]).reshape(-1)[0])
        return PlannerMotion(native[: max(2, min(num_valid, len(native)))])

    def initialize_from_robot(self, joint_positions: np.ndarray) -> PlannerMotion:
        """Build the first trajectory when the operator enters walking mode.

        Reproduces upstream ``Initialize`` / ``InitializeContext``
        (``localmotion_kplanner.hpp:332,591``): the context is canonicalized to the origin at the
        default standing height with an identity (zero-yaw) quaternion, carrying only the robot's
        *current joint positions*, and the first inference is an IDLE plan with no movement.

        The canonicalization is deliberate and is not a loss of fidelity: the planner works in its
        own canonical frame and the caller re-anchors the result. Feeding the robot's true world
        pose here would ask the graph to extrapolate from a position it never trains on.

        Args:
            joint_positions: ``(29,)`` measured joint positions in MuJoCo order.

        Returns:
            A valid idle trajectory, ready to sample before the first teleop observation is built.
        """
        self._context[:] = 0.0
        self._context[0, :, 2] = PLANNER_DEFAULT_HEIGHT
        self._context[0, :, 3] = 1.0  # wxyz identity
        self._context[0, :, 7:] = np.asarray(joint_positions, dtype=np.float32).reshape(1, -1)
        self._seeded = True
        return self.plan(PLANNER_IDLE_COMMAND, PLANNER_CLIP_IDLE)

    def context_from_motion(self, motion: PlannerMotion, gen_time: float) -> None:
        """Refill the context by sampling the *current plan*, as upstream does.

        Upstream's ``UpdateContextFromMotion`` samples four frames at 30 Hz spacing starting at
        ``gen_frame = current_frame + motion_look_ahead_steps``, taken from the planner motion
        rather than from measured robot state. Replanning from the plan keeps successive
        trajectories continuous with each other; replanning from the robot would fold whatever
        tracking error the controller has into the next plan's starting pose.

        Args:
            motion: Trajectory currently being played.
            gen_time: Time in the trajectory at which the next plan should begin.
        """
        for n in range(PLANNER_CONTEXT_FRAMES):
            self._context[0, n] = motion.sample_qpos(gen_time + n * PLANNER_NATIVE_DT)
        self._seeded = True
