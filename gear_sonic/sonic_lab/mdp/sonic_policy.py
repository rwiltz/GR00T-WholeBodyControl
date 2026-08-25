# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SONIC encoder/decoder ONNX runner and ``smpl``-mode observation assembly.

The shipped SONIC deployment artefacts are two ONNX graphs::

    model_encoder.onnx : obs_dict (1, 1751) -> encoded_tokens (1, 64)
    model_decoder.onnx : obs_dict (1,  994) -> action         (1,  29)

We run those directly rather than rebuilding the model from the training checkpoint, so simulation
matches what the real robot executes bit-for-bit.

Encoder input layout
--------------------
Recovered directly from the encoder graph's ``Slice`` nodes rather than from documentation, so it
is ground truth for this checkpoint::

    [0]           encoder mode id            (Gather; smpl = 2)
    [1:4]         encoder_index one-hot      (3)
    [4:584]       motion joint pos+vel       10f step5   -- g1 mode
    [584:644]     motion anchor ori heading  10f step5   -- g1 mode
    [644:650]     motion anchor ori heading  single      -- teleop mode
    [650:890]     lower-body pos+vel         10f step5   -- teleop mode
    [890:899]     vr_3point_local_target                 -- teleop mode
    [899:911]     vr_3point_local_orn_target             -- teleop mode
    [911:1631]    smpl_joints                10f step1   -- SMPL MODE
    [1631:1691]   smpl anchor ori heading    10f step1   -- SMPL MODE
    [1691:1751]   wrist joint positions      10f step1   -- SMPL MODE

Terms outside the active mode stay zero; the deploy stack does the same rather than omitting them.

.. note::
   Both graphs are exported with a **fixed batch size of 1**, so this runner drives a single
   environment. Multi-env rollouts would need a re-export with a dynamic batch axis.

GPU execution
-------------
The decoder is ~37M parameters. On CPU it costs ~17 ms per step against a 20 ms control period at
50 Hz, which alone puts the environment below real time; on CUDA it is ~0.7 ms. Install a runtime
whose CUDA major version matches torch's::

    uv pip install --python <isaaclab-venv>/bin/python onnxruntime-gpu==1.22.0

Version matters: ``onnxruntime-gpu`` 1.28 links CUDA 13 and fails to load against a
``torch==2.11.0+cu128`` environment (``libcublasLt.so.13: cannot open shared object file``). The
1.22 line targets CUDA 12 / cuDNN 9 and matches. When the provider cannot load, onnxruntime falls
back to CPU *silently*, so :class:`SonicOnnxPolicy` warns explicitly instead.

This module imports ``torch`` at module scope on purpose: doing so loads torch's bundled CUDA
libraries into the process, which is what lets onnxruntime's CUDA provider resolve
``libcublasLt.so.12`` and friends without a hand-set ``LD_LIBRARY_PATH``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

__all__ = [
    "SONIC_DECODER_INPUT_DIM",
    "SONIC_ENCODER_INPUT_DIM",
    "SONIC_NUM_ACTIONS",
    "SONIC_TOKEN_DIM",
    "SmplEncoderSlots",
    "SonicOnnxPolicy",
    "smpl_anchor_orientation_heading",
]

SONIC_ENCODER_INPUT_DIM = 1751
SONIC_DECODER_INPUT_DIM = 994
SONIC_TOKEN_DIM = 64
SONIC_NUM_ACTIONS = 29

#: Encoder mode ids, matching ``observation_config.yaml`` ``encoder_modes``.
ENCODER_MODE_G1 = 0
ENCODER_MODE_TELEOP = 1
ENCODER_MODE_SMPL = 2


class SmplEncoderSlots:
    """Slices of the encoder input that ``smpl`` mode populates."""

    MODE_ID = slice(0, 1)
    ENCODER_INDEX = slice(1, 4)
    SMPL_JOINTS = slice(911, 1631)  # 10 frames x 24 joints x 3
    SMPL_ANCHOR_ORI = slice(1631, 1691)  # 10 frames x 6D rotation
    WRIST_JOINT_POS = slice(1691, 1751)  # 10 frames x 6


def smpl_anchor_orientation_heading(
    reference_root_quat: torch.Tensor,
    robot_base_quat: torch.Tensor,
    apply_delta_heading: torch.Tensor,
) -> torch.Tensor:
    """Reference root orientation expressed relative to the robot's own heading, as 6D rotation.

    Reproduces ``GatherMotionAnchorOrientationMutiFrame`` with ``orientation_mode == 1``
    (``g1_deploy_onnx_ref.cpp:612-691``), which is what SONIC v1.1 uses::

        base_to_ref = conj(heading(robot_base_quat)) * (apply_delta_heading * reference_root_quat)

    Taking only the robot's *yaw* makes the reference invariant to which way the operator has
    turned, so operator yaw drift does not accumulate. The operator's pitch and roll stay absolute.

    Args:
        reference_root_quat: ``(N, F, 4)`` wxyz reference root orientations.
        robot_base_quat: ``(N, 4)`` wxyz robot base orientation.
        apply_delta_heading: ``(N, 4)`` wxyz operator/robot heading alignment latched at engage.

    Returns:
        ``(N, F, 6)`` first two columns of each rotation matrix, flattened per frame.
    """
    from gear_sonic.isaac_utils.rotations import (
        calc_heading_quat_inv,
        quat_mul,
        quaternion_to_matrix,
    )

    num_frames = reference_root_quat.shape[1]
    heading_inv = calc_heading_quat_inv(robot_base_quat, w_last=False)
    heading_inv = heading_inv.unsqueeze(1).expand(-1, num_frames, -1)
    delta = apply_delta_heading.unsqueeze(1).expand(-1, num_frames, -1)

    aligned = quat_mul(delta, reference_root_quat, w_last=False)
    base_to_ref = quat_mul(heading_inv, aligned, w_last=False)

    matrices = quaternion_to_matrix(base_to_ref.reshape(-1, 4))
    six_d = matrices[..., :2].reshape(base_to_ref.shape[0], num_frames, 6)
    return six_d


class SonicOnnxPolicy:
    """Runs the SONIC encoder and decoder ONNX graphs.

    Args:
        checkpoint_dir: Directory holding ``model_encoder.onnx`` and ``model_decoder.onnx``
            (e.g. ``gear_sonic_deploy/policy/sonic_v1_1``).
        device: Torch device the caller's tensors live on. Used to move results back.
        providers: Explicit onnxruntime execution providers. Defaults to CUDA then CPU.

    Example:
        >>> policy = SonicOnnxPolicy("gear_sonic_deploy/policy/sonic_v1_1", device="cuda:0")
        >>> token = policy.encode(encoder_obs)      # (1, 1751) -> (1, 64)
        >>> action = policy.decode(token, proprio)  # (1, 64) + (1, 930) -> (1, 29)
    """

    def __init__(
        self,
        checkpoint_dir: str | pathlib.Path,
        device: torch.device | str = "cpu",
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self._dir = pathlib.Path(checkpoint_dir)
        encoder_path = self._dir / "model_encoder.onnx"
        decoder_path = self._dir / "model_decoder.onnx"
        for path in (encoder_path, decoder_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"SONIC ONNX not found: {path}\n"
                    "Fetch it per the repo README, e.g.:\n"
                    "    python download_from_hf.py --sonic-v1-1"
                )

        self.device = torch.device(device)
        if providers is None:
            providers = self._default_providers(ort, self.device)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._encoder = ort.InferenceSession(str(encoder_path), options, providers=providers)
        self._decoder = ort.InferenceSession(str(decoder_path), options, providers=providers)
        self.providers = self._encoder.get_providers()
        self._encoder_input = self._encoder.get_inputs()[0].name
        self._decoder_input = self._decoder.get_inputs()[0].name

        if self.device.type == "cuda" and "CUDAExecutionProvider" not in self.providers:
            import warnings

            warnings.warn(
                "SONIC ONNX is running on CPU while the simulation is on CUDA. The decoder is "
                "~37M params and costs ~17 ms per step there, against a 20 ms control period at "
                "50 Hz. Install the GPU runtime for real-time teleoperation:\n"
                "    uv pip install --python <isaaclab-venv>/bin/python onnxruntime-gpu",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _default_providers(ort, device: torch.device) -> list:
        """Prefer the CUDA execution provider on the simulation's own GPU.

        We bind ``device_id`` to the torch device so ONNX runs on the same GPU as the simulation,
        rather than defaulting to device 0 on a multi-GPU machine.

        TensorRT is deliberately *not* selected by default: it is faster in steady state but pays a
        multi-minute engine build on first run, which is a poor default for interactive bring-up.
        Pass ``providers=`` explicitly to opt in.
        """
        available = set(ort.get_available_providers())
        providers: list = []
        if device.type == "cuda" and "CUDAExecutionProvider" in available:
            providers.append(
                ("CUDAExecutionProvider", {"device_id": device.index or 0})
            )
        providers.append("CPUExecutionProvider")
        return providers

    def encode(self, encoder_obs: torch.Tensor) -> torch.Tensor:
        """Run the encoder.

        Args:
            encoder_obs: ``(1, 1751)`` assembled encoder observation.

        Returns:
            ``(1, 64)`` motion token.
        """
        if encoder_obs.shape[-1] != SONIC_ENCODER_INPUT_DIM:
            raise ValueError(
                f"encoder input must be {SONIC_ENCODER_INPUT_DIM}-wide, "
                f"got {encoder_obs.shape[-1]}"
            )
        arr = encoder_obs.detach().cpu().numpy().astype(np.float32)
        token = self._encoder.run(None, {self._encoder_input: arr})[0]
        return torch.from_numpy(token).to(self.device)

    def decode(self, token: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Run the decoder.

        Args:
            token: ``(1, 64)`` motion token from :meth:`encode`.
            proprio: ``(1, 930)`` flattened proprioception history.

        Returns:
            ``(1, 29)`` raw (pre-scale) joint actions.
        """
        obs = torch.cat([token, proprio], dim=-1)
        if obs.shape[-1] != SONIC_DECODER_INPUT_DIM:
            raise ValueError(
                f"decoder input must be {SONIC_DECODER_INPUT_DIM}-wide, got {obs.shape[-1]}"
            )
        arr = obs.detach().cpu().numpy().astype(np.float32)
        action = self._decoder.run(None, {self._decoder_input: arr})[0]
        return torch.from_numpy(action).to(self.device)

    @staticmethod
    def assemble_smpl_encoder_obs(
        smpl_joints: torch.Tensor,
        anchor_ori_6d: torch.Tensor,
        wrist_joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Build the 1751-wide encoder input for ``smpl`` mode.

        Args:
            smpl_joints: ``(N, 10, 24, 3)`` root-local reference joint positions.
            anchor_ori_6d: ``(N, 10, 6)`` heading-relative root orientation.
            wrist_joint_pos: ``(N, 10, 6)`` G1 wrist joint angles.

        Returns:
            ``(N, 1751)`` encoder observation, zero outside the ``smpl`` slots.
        """
        num_envs = smpl_joints.shape[0]
        obs = torch.zeros(
            num_envs,
            SONIC_ENCODER_INPUT_DIM,
            device=smpl_joints.device,
            dtype=torch.float32,
        )
        obs[:, SmplEncoderSlots.MODE_ID] = float(ENCODER_MODE_SMPL)
        obs[:, SmplEncoderSlots.ENCODER_INDEX] = torch.tensor(
            [0.0, 0.0, 1.0], device=smpl_joints.device
        )
        obs[:, SmplEncoderSlots.SMPL_JOINTS] = smpl_joints.reshape(num_envs, -1)
        obs[:, SmplEncoderSlots.SMPL_ANCHOR_ORI] = anchor_ori_6d.reshape(num_envs, -1)
        obs[:, SmplEncoderSlots.WRIST_JOINT_POS] = wrist_joint_pos.reshape(num_envs, -1)
        return obs
