# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Teleop retargeter: XR full-body tracking -> SONIC ``smpl`` encoder reference.

SONIC's ``smpl`` encoder (``mode_id 2``) is the only mode in which the operator's *legs* drive the
robot's legs; the ``teleop`` mode (``mode_id 1``) is upper-body-only and delegates the lower body to
a kinematic planner. This retargeter therefore targets ``smpl`` mode.

It is a port of the retargeting performed by ``gear_sonic/scripts/pico_manager_thread_server.py``
(the shipped PICO/CloudXR teleop server), restructured as an ``isaacteleop`` ``BaseRetargeter`` so
the Isaac Lab environment can own the OpenXR session. All non-trivial math is *imported* from
``gear_sonic`` rather than re-derived — only the ~40 lines of glue are restated here.

Upstream references (``gear_sonic/scripts/pico_manager_thread_server.py``):
    * ``compute_from_body_poses``  :555-586   XR global quats -> SMPL local axis-angle
    * ``process_smpl_joints``      :450-489   Y-up->Z-up, canonical FK, root-local joints
    * wrist retarget block         :1418-1476 SMPL elbow swing/twist -> G1 wrist angles
    * ``PoseStreamer.parent_indices`` :1273-1299 SMPL parent tree

Output layout
-------------
A single flat ``(107,)`` float32 vector, so it composes with the standard
``OutputCombiner({"action": ...})`` convention that ``IsaacTeleopDevice.advance()`` expects::

    [0]      valid flag (1.0 = body tracking live this frame, 0.0 = holding last good frame)
    [1:73]   smpl_joints_local  (24, 3) flattened - root-local joint positions
    [73:77]  smpl_root_quat     (4,)    wxyz, Z-up, SMPL base rotation removed
    [77:83]  wrist_joint_pos    (6,)    G1 wrist angles, IsaacLab joint indices [23..28]
    [83:92]  vr_3point_pos      (3, 3)  root-relative; left hand, right hand, neck
    [92:104] vr_3point_orn      (3, 4)  wxyz, root-relative; left hand, right hand, neck
    [104:107] operator_root_pos (3,)    tracked pelvis in the XR anchor frame, robot axes

Consumers must **not** treat this as SONIC's encoder input directly. It is one *reference frame*.
The downstream ``ActionTerm`` is responsible for (a) stacking a rolling window of these frames,
(b) running the policy ~N frames behind the newest sample so the encoder's "future" frames are
real measured data, and (c) computing the robot-heading-relative 6D root orientation, which needs
robot state this retargeter cannot see.

Scale note: there is deliberately **no** per-operator height or limb-length calibration. Upstream
runs SMPL forward kinematics on a fixed canonical skeleton
(``gear_sonic/data/human/human_joints_info.pkl``) using only the operator's tracked *rotations*, so
operator proportions are replaced wholesale by the canonical body. We preserve that behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
import pathlib

from isaacteleop.retargeting_engine.interface import BaseRetargeter, RetargeterIOType
from isaacteleop.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaacteleop.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaacteleop.retargeting_engine.tensor_types import (
    DLDataType,
    FullBodyInput,
    FullBodyInputIndex,
    NDArrayType,
)
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import torch

from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
)

__all__ = [
    "SONIC_REFERENCE_DIM",
    "VR3_KEYPOINT_SMPL_IDS",
    "SonicFullBodyRetargeter",
    "SonicFullBodyRetargeterConfig",
    "SonicReferenceSlice",
]

#: Number of XR body joints (``XR_BD_body_tracking``); also SMPL's body joint count.
_NUM_BODY_JOINTS = 24

#: SMPL parent tree. Copied verbatim from ``PoseStreamer.parent_indices``
#: (``pico_manager_thread_server.py:1273-1299``), including its 25-entry-then-``[:24]`` form.
#: Note index 23 resolves to parent 22 rather than SMPL's canonical 21. This is harmless because
#: only ``pose_aa[1:22]`` (the first 63 values of ``body_pose``) is consumed downstream.
_SMPL_PARENT_INDICES: list[int] = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
][:_NUM_BODY_JOINTS]

# SMPL body-pose indices (into the 21-joint ``body_pose`` array, i.e. already root-excluded).
_SMPL_L_ELBOW_IDX = 17
_SMPL_L_WRIST_IDX = 19
_SMPL_R_ELBOW_IDX = 18
_SMPL_R_WRIST_IDX = 20

# G1 wrist joint indices in IsaacLab joint order.
_G1_WRIST_INDICES = (23, 24, 25, 26, 27, 28)

#: G1 elbow hinge axis; used for the swing/twist split.
_G1_ELBOW_AXIS = np.array([0.0, 1.0, 0.0])

#: Canonical SMPL rest skeleton driving the FK. ``compute_human_joints`` defaults this to the
#: *relative* path ``gear_sonic/data/human/human_joints_info.pkl``, which only resolves when the
#: process cwd happens to be the repo root — running from an Isaac Lab checkout fails with
#: ``FileNotFoundError``, and the teleop session reports it as an XR teardown. Resolve it here.
_HUMAN_JOINTS_INFO_PATH = str(
    pathlib.Path(__file__).resolve().parents[3]
    / "gear_sonic"
    / "data"
    / "human"
    / "human_joints_info.pkl"
)

#: Keypoints the ``vr_3point`` block carries, as SMPL joint ids: root, left hand, right hand,
#: neck. Taken from ``_process_3pt_pose`` (``pico_manager_thread_server.py:254-275``), which is
#: what the real robot runs. Note these are the **hands** (22, 23) and the **neck** (12), not the
#: wrists and head -- upstream picks the neck deliberately, "more stable than Head (joint 15) for
#: body tracking".
VR3_KEYPOINT_SMPL_IDS = (0, 22, 23, 12)

#: Per-keypoint rotation corrections aligning SMPL joint frames with the robot convention,
#: post-multiplied onto each keypoint (``pico_manager_thread_server.py:160-167``). The mirrored
#: +/-90 degree roll on the two hands is what makes an unported implementation look "rotated the
#: wrong way, and opposite on each side".
_VR3_OFFSETS_EULER_XYZ_DEG = (
    (0.0, 0.0, -90.0),  # root
    (90.0, 0.0, 0.0),  # left hand
    (-90.0, 0.0, 180.0),  # right hand
    (0.0, 0.0, -90.0),  # neck
)

#: Unity (X-right, Y-up, left-handed) -> robot (X-forward, Y-left, Z-up): ``[x, y, z] -> [-x, z, y]``
#: (``_compute_rel_transform``, ``pico_manager_thread_server.py:290``). Orthogonal with
#: determinant +1, so it is a rotation and conjugating by it maps orientations across too.
_UNITY_TO_ROBOT = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

#: Flat output width: 1 valid flag + 24*3 joints + 4 root quat + 6 wrist angles + 3*3 vr_3point
#: positions + 3*4 vr_3point orientations.
SONIC_REFERENCE_DIM = 1 + _NUM_BODY_JOINTS * 3 + 4 + 6 + 3 * 3 + 3 * 4 + 3


class SonicReferenceSlice:
    """Slice offsets into the retargeter's flat output vector."""

    VALID = slice(0, 1)
    SMPL_JOINTS = slice(1, 73)
    ROOT_QUAT = slice(73, 77)
    WRIST_JOINT_POS = slice(77, 83)
    VR3_POS = slice(83, 92)
    VR3_ORN = slice(92, 104)
    OPERATOR_ROOT_POS = slice(104, 107)


@dataclass
class SonicFullBodyRetargeterConfig:
    """Configuration for :class:`SonicFullBodyRetargeter`.

    Attributes:
        device: Torch device for the SMPL forward-kinematics pass. ``"cpu"`` is normally right:
            the FK is tiny (24 joints, batch 1) and keeping it off the GPU avoids contending with
            rendering and avoids a device sync on the XR thread.
        hold_last_on_tracking_loss: When body tracking drops out, re-emit the last good frame with
            the valid flag cleared. When ``False``, emit zeros instead.
    """

    device: str = "cpu"
    hold_last_on_tracking_loss: bool = True


class SonicFullBodyRetargeter(BaseRetargeter):
    """Convert XR 24-joint full-body tracking into a SONIC ``smpl``-mode reference frame.

    Args:
        config: Retargeter configuration.
        name: Unique node name within the retargeting pipeline.

    Example:
        >>> body = FullBodySource(name="body", vendor=TrackerVendor("body.pico-xr"))
        >>> retgt = SonicFullBodyRetargeter(SonicFullBodyRetargeterConfig(), name="sonic_ref")
        >>> connected = retgt.connect({"full_body": body.output(FullBodySource.FULL_BODY)})
        >>> OutputCombiner({"action": connected.output("sonic_reference")})
    """

    def __init__(self, config: SonicFullBodyRetargeterConfig, name: str) -> None:
        super().__init__(name=name)
        self._config = config
        self._device = torch.device(config.device)
        self._last_good = np.zeros(SONIC_REFERENCE_DIM, dtype=np.float32)
        self._have_good_frame = False

    def input_spec(self) -> RetargeterIOType:
        """Consume the standard full-body stream (Optional: absent when tracking is inactive)."""
        return {"full_body": OptionalType(FullBodyInput())}

    def output_spec(self) -> RetargeterIOType:
        """Emit one flat SONIC reference frame. See module docstring for the layout."""
        return {
            "sonic_reference": TensorGroupType(
                "sonic_reference",
                [
                    NDArrayType(
                        "reference",
                        shape=(SONIC_REFERENCE_DIM,),
                        dtype=DLDataType.FLOAT,
                        dtype_bits=32,
                    )
                ],
            )
        }

    def _compute_fn(
        self, inputs: RetargeterIO, outputs: RetargeterIO, context
    ) -> None:  # noqa: ANN001
        """Retarget one frame, or hold the previous one when tracking is unavailable."""
        body = inputs["full_body"]
        if body.is_none:
            outputs["sonic_reference"][0] = self._fallback_frame()
            return

        # Tensor groups are indexed by their generated index enum and exported via DLPack, not by
        # field name. Indexing with a string raises "only integers, slices ... are valid indices",
        # which the teleop session swallows and reports as an XR teardown.
        positions = np.from_dlpack(body[FullBodyInputIndex.JOINT_POSITIONS])
        orientations = np.from_dlpack(body[FullBodyInputIndex.JOINT_ORIENTATIONS])
        valid = np.from_dlpack(body[FullBodyInputIndex.JOINT_VALID])

        # Upstream requires every body joint; a partially-tracked skeleton would silently produce a
        # plausible-but-wrong pose, which is worse than holding the last good frame.
        if not bool(valid.all()):
            outputs["sonic_reference"][0] = self._fallback_frame()
            return

        # (24, 7) == [x, y, z, qx, qy, qz, qw], matching upstream's body_poses_np layout.
        body_poses = np.concatenate(
            [np.asarray(positions, dtype=np.float32), np.asarray(orientations, dtype=np.float32)],
            axis=1,
        )
        frame = self._retarget(body_poses)

        self._last_good = frame
        self._have_good_frame = True
        outputs["sonic_reference"][0] = frame

    def _fallback_frame(self) -> np.ndarray:
        """Last good frame with the valid flag cleared, or a neutral frame if we never had one.

        The neutral frame is zeros **except** for an identity root quaternion. An all-zero frame
        looks harmless but is not: a zero quaternion has zero norm, so the heading maths downstream
        divides by it and produces NaN, which propagates through the encoder into NaN joint targets
        and hands the physics solver garbage. Identity keeps the frame degenerate-but-finite.
        """
        if not (self._config.hold_last_on_tracking_loss and self._have_good_frame):
            neutral = np.zeros(SONIC_REFERENCE_DIM, dtype=np.float32)
            neutral[SonicReferenceSlice.ROOT_QUAT.start] = 1.0  # wxyz identity
            # Same reasoning for the three vr_3point quaternions: zeros are not a rotation.
            neutral[SonicReferenceSlice.VR3_ORN][0::4] = 1.0  # positions stay zero
            return neutral
        held = self._last_good.copy()
        held[SonicReferenceSlice.VALID] = 0.0
        return held

    def _retarget(self, body_poses: np.ndarray) -> np.ndarray:
        """Run the full XR -> SONIC-reference conversion for one frame.

        Args:
            body_poses: ``(24, 7)`` array of ``[x, y, z, qx, qy, qz, qw]`` per XR body joint.

        Returns:
            ``(SONIC_REFERENCE_DIM,)`` float32 reference frame.
        """
        body_pose, global_orient = self._xr_to_smpl_local(body_poses)
        smpl_joints_local, root_quat = self._smpl_to_reference(body_pose, global_orient)
        wrist_joint_pos = self._smpl_to_g1_wrists(body_pose)

        frame = np.empty(SONIC_REFERENCE_DIM, dtype=np.float32)
        frame[SonicReferenceSlice.VALID] = 1.0
        frame[SonicReferenceSlice.SMPL_JOINTS] = smpl_joints_local
        frame[SonicReferenceSlice.ROOT_QUAT] = root_quat
        frame[SonicReferenceSlice.WRIST_JOINT_POS] = wrist_joint_pos
        vr3_pos, vr3_orn, operator_root = self._vr_three_point(body_poses)
        frame[SonicReferenceSlice.VR3_POS] = vr3_pos.reshape(-1)
        frame[SonicReferenceSlice.VR3_ORN] = vr3_orn.reshape(-1)
        frame[SonicReferenceSlice.OPERATOR_ROOT_POS] = operator_root
        return frame

    @staticmethod
    def _vr_three_point(body_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The ``vr_3point`` block, ported from what the real robot runs.

        Port of ``_process_3pt_pose`` (``pico_manager_thread_server.py:200-310``), reached from
        Isaac Teleop as well as from XRoboToolkit, so it is the shared definition of these targets.

        The steps, in upstream's order:

        1. Unity -> robot frame for every joint: ``p' = Q p`` and ``R' = Q R Q^T``.
        2. Keep keypoints ``(0, 22, 23, 12)`` -- root, left hand, right hand, neck.
        3. Post-multiply each by its rotation offset.
        4. Express the three non-root points relative to the root, in position and orientation.

        Positions come from the **tracked** joint positions, not from forward kinematics on the
        canonical skeleton. That distinction is the whole point: the ``smpl`` reference deliberately
        replaces the operator's proportions with a canonical body driven by tracked rotations, but
        ``vr_3point`` is a measured quantity, and feeding it canonical-FK positions puts the targets
        roughly half a metre out.

        Operator calibration (``ThreePointPose._apply_calibration``) is **not** applied. Upstream
        returns the pose unchanged until a calibration has been captured, so this matches an
        uncalibrated session; adding it would need that capture step.

        Args:
            body_poses: ``(24, 7)`` ``[x, y, z, qx, qy, qz, qw]`` per XR body joint, Unity frame.

        Returns:
            ``(positions (3, 3), orientations (3, 4) wxyz, root_pos (3,))``. The three points are
            relative to the root; ``root_pos`` is the operator's tracked pelvis in the anchor
            frame, which the caller needs to place the anchor so the *operator* lands on the robot
            rather than the anchor doing so.
        """
        q = _UNITY_TO_ROBOT
        ids = VR3_KEYPOINT_SMPL_IDS
        offsets = [
            sRot.from_euler("xyz", angles, degrees=True) for angles in _VR3_OFFSETS_EULER_XYZ_DEG
        ]

        positions, rotations = [], []
        for slot, joint in enumerate(ids):
            pose = body_poses[joint]
            positions.append(q @ np.asarray(pose[:3], dtype=np.float64))
            rot = sRot.from_quat(np.asarray(pose[3:7], dtype=np.float64)).as_matrix()
            rotations.append(sRot.from_matrix(q @ rot @ q.T) * offsets[slot])

        root_pos, root_rot = positions[0], rotations[0]
        root_inv = root_rot.inv()
        out_pos = np.empty((3, 3), dtype=np.float32)
        out_rot = np.empty((3, 4), dtype=np.float32)
        for i in range(1, 4):
            out_pos[i - 1] = root_inv.apply(positions[i] - root_pos)
            out_rot[i - 1] = (root_inv * rotations[i]).as_quat(scalar_first=True)
        return out_pos, out_rot, np.asarray(root_pos, dtype=np.float32)

    def _xr_to_smpl_local(self, body_poses: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """XR global quaternions -> SMPL local axis-angle.

        Port of ``compute_from_body_poses`` (``pico_manager_thread_server.py:561-586``). The
        ``+180 deg about Y`` term reconciles the XR joint frame with SMPL's.

        Returns:
            ``(body_pose (1, 69), global_orient (1, 3))`` torch tensors on the configured device.
        """
        global_quats = body_poses[:, [6, 3, 4, 5]]  # xyzw -> wxyz
        global_rots = sRot.from_quat(global_quats, scalar_first=True)
        global_rots = global_rots * sRot.from_euler("y", 180, degrees=True)

        local_rots = []
        for i in range(_NUM_BODY_JOINTS):
            parent = _SMPL_PARENT_INDICES[i]
            if parent == -1:
                local_rots.append(global_rots[i])
            else:
                local_rots.append(global_rots[parent].inv() * global_rots[i])

        pose_aa = np.array([rot.as_rotvec() for rot in local_rots])
        body_pose = torch.from_numpy(pose_aa[1:].flatten()).float().to(self._device).unsqueeze(0)
        global_orient = torch.from_numpy(pose_aa[0]).float().to(self._device).unsqueeze(0)
        return body_pose, global_orient

    def _smpl_to_reference(
        self, body_pose: torch.Tensor, global_orient: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """SMPL params -> root-local joint positions + Z-up root quaternion.

        Port of ``process_smpl_joints`` (``pico_manager_thread_server.py:450-489``). Upstream also
        returns translation, but it is never transmitted: SONIC infers locomotion from root-local
        leg joints plus root orientation, so we drop it too.

        Returns:
            ``(smpl_joints_local (72,), root_quat (4,) wxyz)`` as float32 numpy arrays.
        """
        # Y-up (XR/SMPL) -> Z-up (robot), matching the training flag ``smpl_y_up: true``.
        global_orient_quat = angle_axis_to_quaternion(global_orient)
        global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
        global_orient_new = quaternion_to_angle_axis(global_orient_quat)

        # FK on the fixed canonical skeleton -> (1, 24, 3) world-ish joint positions.
        joints = compute_human_joints(
            body_pose=body_pose[..., :63],
            global_orient=global_orient_new,
            human_joints_info_path=_HUMAN_JOINTS_INFO_PATH,
        )

        global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)
        quat_inv_exp = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
        smpl_joints_local = quat_apply(quat_inv_exp, joints)

        return (
            smpl_joints_local.reshape(-1).detach().cpu().numpy().astype(np.float32),
            global_orient_quat.reshape(-1).detach().cpu().numpy().astype(np.float32),
        )

    def _smpl_to_g1_wrists(self, body_pose: torch.Tensor) -> np.ndarray:
        """SMPL elbow+wrist rotations -> six G1 wrist joint angles.

        Port of the wrist block at ``pico_manager_thread_server.py:1418-1476``.

        SMPL's elbow is 3-DoF but the G1's is a 1-DoF hinge about ``[0, 1, 0]``. The rotation is
        split into swing and twist about that axis; the twist becomes elbow motion and the **swing
        (roll + yaw) is pushed downstream into the wrist**, summed with SMPL's own wrist rotation.

        The left/right sign handling is asymmetric upstream and is reproduced verbatim: the right
        roll is negated as a whole, and the left pitch is negated twice (once into
        ``g1_l_wrist_pitch``, once on assignment) so it ends up as ``+l_wrist_euler[:, 1]`` while
        the right pitch ends up as ``-r_wrist_euler[:, 1]``.

        Args:
            body_pose: ``(1, 69)`` SMPL body pose (23 joints x 3, root excluded). Only the first
                63 values (21 joints) are used, matching upstream's ``[:, :63]`` truncation at
                ``pico_manager_thread_server.py:1378``.

        Returns:
            ``(6,)`` float32 array ordered by IsaacLab joint index ``[23, 24, 25, 26, 27, 28]``
            = ``[l_roll, r_roll, l_pitch, r_pitch, l_yaw, r_yaw]``.
        """
        pose = body_pose[..., :63].reshape(-1, 21, 3).detach().cpu().numpy()

        l_elbow_aa = pose[:, _SMPL_L_ELBOW_IDX]
        l_wrist_aa = pose[:, _SMPL_L_WRIST_IDX]
        r_elbow_aa = pose[:, _SMPL_R_ELBOW_IDX]
        r_wrist_aa = pose[:, _SMPL_R_WRIST_IDX]

        _, l_swing = decompose_rotation_aa(l_elbow_aa, _G1_ELBOW_AXIS)
        _, r_swing = decompose_rotation_aa(r_elbow_aa, _G1_ELBOW_AXIS)

        # decompose_rotation_aa returns wxyz; scipy wants xyzw.
        l_swing_euler = sRot.from_quat(l_swing[:, [1, 2, 3, 0]]).as_euler("XYZ")
        r_swing_euler = sRot.from_quat(r_swing[:, [1, 2, 3, 0]]).as_euler("XYZ")
        l_wrist_euler = sRot.from_rotvec(l_wrist_aa).as_euler("XYZ")
        r_wrist_euler = sRot.from_rotvec(r_wrist_aa).as_euler("XYZ")

        l_roll = l_swing_euler[:, 0] + l_wrist_euler[:, 0]
        l_pitch = -l_wrist_euler[:, 1]
        l_yaw = l_swing_euler[:, 2] + l_wrist_euler[:, 2]
        r_roll = -(r_swing_euler[:, 0] + r_wrist_euler[:, 0])
        r_pitch = -r_wrist_euler[:, 1]
        r_yaw = r_swing_euler[:, 2] + r_wrist_euler[:, 2]

        joint_pos = np.zeros(29, dtype=np.float32)
        joint_pos[23] = l_roll[0]
        joint_pos[25] = -l_pitch[0]  # deliberate double negation, see docstring
        joint_pos[27] = l_yaw[0]
        joint_pos[24] = r_roll[0]
        joint_pos[26] = r_pitch[0]
        joint_pos[28] = r_yaw[0]

        return joint_pos[list(_G1_WRIST_INDICES)]
