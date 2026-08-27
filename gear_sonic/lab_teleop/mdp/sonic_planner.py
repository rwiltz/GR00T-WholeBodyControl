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
``movement_direction`` and ``facing_direction`` are **absolute world-frame** XY unit vectors with
``z = 0``, not robot-relative and not integrated turn rates (``controllers.py:196-207``). When the
operator gives no input the reference implementation falls back to the robot's own measured
velocity and root yaw rather than holding the last command, which is what
:meth:`SonicVelocityPlanner.idle_directions` reproduces.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

__all__ = [
    "PLANNER_CLIP_IDLE",
    "PLANNER_CLIP_SLOW_WALK",
    "PLANNER_CLIP_WALK",
    "PLANNER_CONTEXT_FRAMES",
    "PLANNER_QPOS_DIM",
    "PLANNER_WALK_TOKENS",
    "SONIC_PLANNER_COMMAND_DIM",
    "SonicVelocityPlanner",
]

#: Frames of robot history the planner conditions on.
PLANNER_CONTEXT_FRAMES = 4

#: ``qpos`` width: 3 root position + 4 root quaternion + 29 joints.
PLANNER_QPOS_DIM = 36

#: Operator command width: ``[target_vel, movement_dir(3), facing_dir(3), height]``.
SONIC_PLANNER_COMMAND_DIM = 8

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
        replan_interval: Frames to consume before re-planning. ``0`` consumes the whole valid
            plan, which is ``num_pred_frames`` long -- not necessarily the full 64.
        allowed_tokens: Clip token mask; see :data:`PLANNER_WALK_TOKENS`.

    Raises:
        FileNotFoundError: If the planner graph is absent.
    """

    def __init__(
        self,
        checkpoint_path: str | pathlib.Path,
        device: torch.device | str = "cuda:0",
        replan_interval: int = 0,
        allowed_tokens: tuple[int, ...] = PLANNER_WALK_TOKENS,
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
        self._replan_interval = replan_interval
        self._allowed_tokens = tuple(allowed_tokens)

        self._context = np.zeros((1, PLANNER_CONTEXT_FRAMES, PLANNER_QPOS_DIM), dtype=np.float32)
        self._plan: np.ndarray | None = None
        self._cursor = 0
        self._seeded = False

    @property
    def horizon(self) -> int:
        """Frames the graph plans per invocation."""
        return self._horizon

    def reset(self) -> None:
        """Drop the context history and any pending plan.

        Mandatory on episode reset. A retained plan describes the *previous* episode's motion, and
        a retained context conditions the next plan on a pose the robot no longer has.
        """
        self._context[:] = 0.0
        self._plan = None
        self._cursor = 0
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

    @staticmethod
    def idle_directions(qpos_history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Directions to use when the operator is giving no input.

        Reproduces ``controllers.py:209-218``: continue along the robot's own measured velocity and
        facing rather than holding the last command, so an idle operator does not command a stale
        heading.

        Args:
            qpos_history: ``(frames, 36)`` recent measured poses.

        Returns:
            ``(movement_direction, facing_direction)``, each a world-frame XY unit vector.
        """
        from scipy.spatial.transform import Rotation

        deltas = np.diff(qpos_history[:, :3], axis=0)
        velocity = deltas.mean(axis=0) * np.array([1.0, 1.0, 0.0]) if len(deltas) else np.zeros(3)
        movement = velocity / (np.linalg.norm(velocity) + 1e-5)
        facing = Rotation.from_quat(qpos_history[-1, 3:7], scalar_first=True).apply(
            np.array([1.0, 0.0, 0.0])
        ) * np.array([1.0, 1.0, 0.0])
        facing = facing / (np.linalg.norm(facing) + 1e-5)
        return movement.astype(np.float32), facing.astype(np.float32)

    def _run(self, command: np.ndarray, mode: int) -> None:
        """Invoke the graph and refill the plan buffer."""
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
        plan = np.asarray(outputs[0], dtype=np.float32).reshape(-1, PLANNER_QPOS_DIM)
        # The output tensor is always 64 frames wide, but only ``num_pred_frames`` of them are
        # predictions -- 44 under the walking token mask. Consuming the tail would replay
        # uninitialized frames as if they were motion.
        num_valid = int(np.asarray(outputs[1]).reshape(-1)[0])
        self._plan = plan[: max(1, min(num_valid, len(plan)))]
        self._cursor = 0

    def next_frame(self, command: np.ndarray, mode: int = PLANNER_CLIP_WALK) -> np.ndarray:
        """Return the next planned pose, re-planning when the buffer runs dry.

        Args:
            command: ``(8,)`` ``[target_vel, movement_dir(3), facing_dir(3), height]``.
            mode: Planner clip id; see :data:`PLANNER_CLIP_WALK`. Defaults to walking, because the
                idle clip ignores ``target_vel`` entirely.

        Returns:
            ``(36,)`` planned MuJoCo-order pose.
        """
        limit = len(self._plan) if self._plan is not None else 0
        if self._replan_interval:
            limit = min(self._replan_interval, limit)
        if self._plan is None or self._cursor >= limit:
            self._run(np.asarray(command, dtype=np.float32).reshape(-1), mode)
        frame = self._plan[self._cursor]
        self._cursor += 1
        return frame
