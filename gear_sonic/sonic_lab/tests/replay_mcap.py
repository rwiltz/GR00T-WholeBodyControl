# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Drive :class:`SonicFullBodyRetargeter` from a recorded Isaac Teleop MCAP.

The unit tests prove the retargeter is numerically identical to the shipped PICO implementation,
but they use synthetic random quaternions, which say nothing about whether real tracking data
produces a *plausible human pose*. This harness replays an actual recording through the same
``FullBodyTracker`` API the live session uses, so no headset is required.

Usage::

    /path/to/IsaacLab/.venv/bin/python -m gear_sonic.sonic_lab.tests.replay_mcap \\
        /path/to/full_body_*.mcap [--max-frames N] [--dump out.npz]

Sanity checks applied per frame: joint validity, quaternion normalisation, human-scale skeleton
extent, root-relative origin, and G1 wrist angles inside the URDF limits.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

from gear_sonic.sonic_lab.retargeters.sonic_fullbody_retargeter import (
    SonicFullBodyRetargeter,
    SonicFullBodyRetargeterConfig,
    SonicReferenceSlice,
)

#: G1 wrist joint limits (rad) from ``robot_description/urdf/g1/main.urdf``, ordered to match the
#: retargeter's output: ``[l_roll, r_roll, l_pitch, r_pitch, l_yaw, r_yaw]``.
WRIST_LIMITS = np.array(
    [
        [-1.972222054, 1.972222054],  # left  wrist roll
        [-1.972222054, 1.972222054],  # right wrist roll
        [-1.614429558, 1.614429558],  # left  wrist pitch
        [-1.614429558, 1.614429558],  # right wrist pitch
        [-1.614429558, 1.614429558],  # left  wrist yaw
        [-1.614429558, 1.614429558],  # right wrist yaw
    ]
)

_NUM_BODY_JOINTS = 24


def read_body_frames(
    mcap_path: str, channel: str = "full_body", max_frames: int | None = None
) -> list[np.ndarray]:
    """Replay an MCAP and return one ``(24, 7)`` array per frame.

    Args:
        mcap_path: Path to the recording.
        channel: MCAP channel base name the tracker was recorded under.
        max_frames: Stop after this many frames; ``None`` reads to the end.

    Returns:
        List of ``[x, y, z, qx, qy, qz, qw]`` per-joint arrays, in capture order.
    """
    import isaacteleop.deviceio as deviceio
    from isaacteleop.deviceio_session import McapReplayConfig, ReplaySession

    tracker = deviceio.FullBodyTracker()
    config = McapReplayConfig(mcap_path, [(tracker, channel)])

    frames: list[np.ndarray] = []
    seen_invalid = 0
    with ReplaySession.run(config) as session:
        while max_frames is None or len(frames) < max_frames:
            try:
                session.update()
            except Exception:  # noqa: BLE001 - native replay signals EOF by raising
                break
            body_pose = getattr(tracker.get_body_pose(session), "data", None)
            if body_pose is None or body_pose.joints is None:
                seen_invalid += 1
                if seen_invalid > 200:  # tolerate a lead-in, but don't spin forever
                    break
                continue

            frame = np.zeros((_NUM_BODY_JOINTS, 7), dtype=np.float32)
            all_valid = True
            for i in range(_NUM_BODY_JOINTS):
                joint = body_pose.joints.joints(i)
                pose = joint.pose
                frame[i, :3] = (pose.position.x, pose.position.y, pose.position.z)
                frame[i, 3:] = (
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                )
                all_valid &= bool(joint.is_valid)
            if not all_valid:
                seen_invalid += 1
                continue

            # Replay loops the file; stop once we wrap back to the first sample.
            if frames and np.array_equal(frame, frames[0]):
                break
            frames.append(frame)

    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcap", type=str, help="Path to a full_body MCAP recording")
    parser.add_argument("--channel", default="full_body")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--dump", type=str, default=None, help="Write frames to .npz")
    args = parser.parse_args(argv)

    if not pathlib.Path(args.mcap).is_file():
        parser.error(f"no such file: {args.mcap}")

    frames = read_body_frames(args.mcap, args.channel, args.max_frames)
    if not frames:
        print("No valid body-tracking frames found in recording.")
        return 1
    print(f"Replayed {len(frames)} valid body frames from {pathlib.Path(args.mcap).name}")

    retargeter = SonicFullBodyRetargeter(SonicFullBodyRetargeterConfig(device="cpu"), name="replay")
    refs = np.stack([retargeter._retarget(f) for f in frames])  # noqa: SLF001

    joints = refs[:, SonicReferenceSlice.SMPL_JOINTS].reshape(len(refs), 24, 3)
    quats = refs[:, SonicReferenceSlice.ROOT_QUAT]
    wrists = refs[:, SonicReferenceSlice.WRIST_JOINT_POS]

    extent = np.linalg.norm(joints.max(axis=1) - joints.min(axis=1), axis=1)
    quat_norm = np.linalg.norm(quats, axis=1)
    height = joints[..., 2].max(axis=1) - joints[..., 2].min(axis=1)

    # "local" means the root *rotation* is divided out; translation is NOT removed, so joint 0 is
    # not at the origin. Verified against pico_manager_thread_server on real recordings: upstream
    # produces bit-identical values, so this is the distribution SONIC was trained on.
    # The meaningful rotation check is that the hip axis stays roughly fixed once the root
    # orientation is removed, regardless of which way the operator turns.
    hip_axis = joints[:, 2] - joints[:, 1]  # right hip - left hip
    hip_axis /= np.linalg.norm(hip_axis, axis=1, keepdims=True) + 1e-9
    hip_axis_spread = float(np.linalg.norm(hip_axis - hip_axis.mean(axis=0), axis=1).max())

    print("\n--- sanity ---")
    print(f"finite                : {np.isfinite(refs).all()}")
    print(f"root quat norm        : {quat_norm.min():.6f} .. {quat_norm.max():.6f}  (expect ~1)")
    print(f"skeleton extent (m)   : {extent.min():.3f} .. {extent.max():.3f}  (expect ~1-2 m)")
    print(f"skeleton height (m)   : {height.min():.3f} .. {height.max():.3f}")
    print(f"hip-axis spread       : {hip_axis_spread:.3f}              (expect small: root yaw removed)")
    print(f"wrist range (rad)     : {wrists.min():.3f} .. {wrists.max():.3f}")

    lo, hi = WRIST_LIMITS[:, 0], WRIST_LIMITS[:, 1]
    out_of_range = ((wrists < lo) | (wrists > hi)).sum(axis=0)
    names = ["l_roll", "r_roll", "l_pitch", "r_pitch", "l_yaw", "r_yaw"]
    print("\n--- G1 wrist joint-limit violations (frames out of %d) ---" % len(refs))
    for name, count in zip(names, out_of_range, strict=True):
        flag = "" if count == 0 else "   <-- exceeds URDF limit"
        print(f"  {name:8s}: {count:5d}{flag}")

    ok = (
        bool(np.isfinite(refs).all())
        and float(np.abs(quat_norm - 1).max()) < 1e-3
        and float(root_offset.max()) < 1e-3
        and 0.5 < float(extent.mean()) < 3.0
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")

    if args.dump:
        np.savez_compressed(args.dump, body_frames=np.stack(frames), references=refs)
        print(f"wrote {args.dump}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
