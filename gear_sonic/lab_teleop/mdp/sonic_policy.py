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
   environment. Multi-env rollouts would need a re-export with a dynamic batch axis. The batch
   width is validated explicitly here rather than left to fail inside onnxruntime.

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

Zero-copy GPU I/O
-----------------
On CUDA the runner keeps every tensor resident on the device. Inputs and outputs are preallocated
torch CUDA tensors, and onnxruntime reads and writes those exact allocations through a persistent
``IOBinding`` created once at construction and reused every control step::

    torch CUDA encoder obs -> ORT encoder -> 64-D latent (torch CUDA)
                           -> ORT decoder -> 29-D action (torch CUDA)

Binding is by raw ``data_ptr()``, so no DLPack capsule is built per frame and no host memory is
touched. The alternative -- ``.cpu().numpy()`` in, ``torch.from_numpy().to(cuda)`` out -- forced a
full device synchronize twice per control step, because copying to host must wait for all prior
GPU work to retire.

Stream ownership
~~~~~~~~~~~~~~~~
onnxruntime runs on its own CUDA stream by default, which would race against torch writes into the
bound input buffers. We therefore create one dedicated torch stream and hand its handle to the CUDA
provider via ``user_compute_stream``, so torch and ORT issue into the *same* stream and ordering is
implicit -- no events, no host synchronization.

Torch's default stream cannot be used for this: its handle is the null pointer, which the provider
treats as "unset" (it silently reports ``has_user_compute_stream: 0``). Callers therefore run their
own tensor work inside :meth:`SonicOnnxPolicy.compute_stream`, which joins the caller's stream to
the SONIC stream on entry and back on exit using CUDA events -- asynchronous in both directions.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib

import numpy as np
import torch

__all__ = [
    "ORIENTATION_MODE_BODY",
    "ORIENTATION_MODE_HEADING",
    "SONIC_DECODER_INPUT_DIM",
    "SONIC_ENCODER_INPUT_DIM",
    "SONIC_NUM_ACTIONS",
    "SONIC_TOKEN_DIM",
    "SONIC_VARIANTS_BY_ENCODER_DIM",
    "SONIC_VARIANT_LOW_LATENCY",
    "SONIC_VARIANT_V1_1",
    "SmplEncoderSlots",
    "SonicOnnxPolicy",
    "SonicVariant",
    "smpl_anchor_orientation",
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
    """Slices of the SONIC v1.1 encoder input that ``smpl`` mode populates.

    Retained as the v1.1 layout for external callers. The runtime uses
    :attr:`SonicVariant.smpl_joints` and friends, which vary per checkpoint.
    """

    MODE_ID = slice(0, 1)
    ENCODER_INDEX = slice(1, 4)
    SMPL_JOINTS = slice(911, 1631)  # 10 frames x 24 joints x 3
    SMPL_ANCHOR_ORI = slice(1631, 1691)  # 10 frames x 6D rotation
    WRIST_JOINT_POS = slice(1691, 1751)  # 10 frames x 6


#: Orientation modes from ``GatherMotionAnchorOrientationMutiFrame``
#: (``g1_deploy_onnx_ref.cpp:607-610``). These select the *left* quaternion of the relative
#: rotation ``conj(left) * (apply_delta_heading * reference_root_quat)``.
ORIENTATION_MODE_BODY = 0
"""Full base quaternion (``motion_anchor_ori_b``). The robot's pitch and roll enter the term."""
ORIENTATION_MODE_HEADING = 1
"""Robot heading (yaw) only (``motion_anchor_ori_heading``). Invariant to operator turning."""


@dataclasses.dataclass(frozen=True)
class SonicVariant:
    """Per-checkpoint encoder geometry and reference semantics.

    Two SONIC checkpoints ship with this repo and they differ by more than a window length, so
    the differences are captured in one place rather than scattered through the action term.

    ============================  ===============  ==================
    field                         ``sonic_v1_1``   ``low_latency``
    ============================  ===============  ==================
    encoder input                 1751             1247
    reference frames              10               4
    anchor orientation            heading (yaw)    body (full quat)
    decoder input / proprio       994 / 10 frames  994 / 10 frames
    ============================  ===============  ==================

    The layouts were recovered from each encoder graph's ``Slice`` nodes, which slice the input
    after the mode id is gathered off the front -- hence the uniform ``+1`` against the raw graph
    offsets. Note the decoder is identical across both, so the **proprioception history stays 10
    frames either way**; only the *reference* window shortens.

    Attributes:
        name: Checkpoint directory name, used in messages.
        encoder_input_dim: Width of the encoder graph's single input.
        reference_frames: Reference frames the ``smpl`` encoder consumes. This is the induced
            operator latency in control steps: 10 frames at 50 Hz trails by ~200 ms, 4 by ~80 ms.
        smpl_joints: Slice holding ``reference_frames x 24 x 3`` root-local joint positions.
        smpl_anchor_ori: Slice holding ``reference_frames x 6`` rotations.
        wrist_joint_pos: Slice holding ``reference_frames x 6`` wrist angles.
        orientation_mode: One of :data:`ORIENTATION_MODE_BODY` / :data:`ORIENTATION_MODE_HEADING`.
    """

    name: str
    encoder_input_dim: int
    reference_frames: int
    smpl_joints: slice
    smpl_anchor_ori: slice
    wrist_joint_pos: slice
    orientation_mode: int


SONIC_VARIANT_V1_1 = SonicVariant(
    name="sonic_v1_1",
    encoder_input_dim=1751,
    reference_frames=10,
    smpl_joints=slice(911, 1631),
    smpl_anchor_ori=slice(1631, 1691),
    wrist_joint_pos=slice(1691, 1751),
    orientation_mode=ORIENTATION_MODE_HEADING,
)

SONIC_VARIANT_LOW_LATENCY = SonicVariant(
    name="low_latency",
    encoder_input_dim=1247,
    reference_frames=4,
    smpl_joints=slice(911, 1199),
    smpl_anchor_ori=slice(1199, 1223),
    wrist_joint_pos=slice(1223, 1247),
    orientation_mode=ORIENTATION_MODE_BODY,
)

#: Keyed by encoder input width so the variant is read off the graph itself. Selecting on the
#: directory name instead would silently mismatch if a checkpoint were moved or renamed.
SONIC_VARIANTS_BY_ENCODER_DIM = {
    variant.encoder_input_dim: variant
    for variant in (SONIC_VARIANT_V1_1, SONIC_VARIANT_LOW_LATENCY)
}


def smpl_anchor_orientation(
    reference_root_quat: torch.Tensor,
    robot_base_quat: torch.Tensor,
    apply_delta_heading: torch.Tensor,
    orientation_mode: int = ORIENTATION_MODE_HEADING,
) -> torch.Tensor:
    """Reference root orientation relative to the robot, as a 6D rotation.

    Reproduces ``GatherMotionAnchorOrientationMutiFrame``
    (``g1_deploy_onnx_ref.cpp:612-691``)::

        base_to_ref = conj(left) * (apply_delta_heading * reference_root_quat)

    where ``left`` is selected by ``orientation_mode``:

    * :data:`ORIENTATION_MODE_HEADING` -- ``heading(robot_base_quat)``, the robot's *yaw* only.
      Used by SONIC v1.1. Makes the reference invariant to which way the operator has turned, so
      operator yaw drift does not accumulate; the operator's pitch and roll stay absolute.
    * :data:`ORIENTATION_MODE_BODY` -- the full ``robot_base_quat``. Used by the low-latency
      checkpoint (``motion_anchor_ori_b``). The robot's own pitch and roll now enter the term, so
      the reference is expressed in the tilted body frame rather than a level heading frame.

    Picking the wrong mode is a silent failure: both produce a well-formed rotation, and the
    resulting 6D term stays in range, so it degrades control quality rather than raising.

    Args:
        reference_root_quat: ``(N, F, 4)`` wxyz reference root orientations.
        robot_base_quat: ``(N, 4)`` wxyz robot base orientation.
        apply_delta_heading: ``(N, 4)`` wxyz operator/robot heading alignment latched at engage.
        orientation_mode: See above.

    Returns:
        ``(N, F, 6)`` first two columns of each rotation matrix, flattened per frame.

    Raises:
        ValueError: If ``orientation_mode`` is not a supported mode.
    """
    from gear_sonic.isaac_utils.rotations import (
        calc_heading_quat_inv,
        quat_conjugate,
        quat_mul,
        quaternion_to_matrix,
    )

    num_frames = reference_root_quat.shape[1]
    if orientation_mode == ORIENTATION_MODE_HEADING:
        left_inv = calc_heading_quat_inv(robot_base_quat, w_last=False)
    elif orientation_mode == ORIENTATION_MODE_BODY:
        left_inv = quat_conjugate(robot_base_quat, w_last=False)
    else:
        raise ValueError(
            f"Unsupported orientation_mode {orientation_mode}; expected "
            f"{ORIENTATION_MODE_BODY} (body) or {ORIENTATION_MODE_HEADING} (heading). "
            "Mode 2 (reference first-frame heading) is not used by any shipped checkpoint."
        )
    left_inv = left_inv.unsqueeze(1).expand(-1, num_frames, -1)
    delta = apply_delta_heading.unsqueeze(1).expand(-1, num_frames, -1)

    aligned = quat_mul(delta, reference_root_quat, w_last=False)
    base_to_ref = quat_mul(left_inv, aligned, w_last=False)

    matrices = quaternion_to_matrix(base_to_ref.reshape(-1, 4))
    six_d = matrices[..., :2].reshape(base_to_ref.shape[0], num_frames, 6)
    return six_d


def smpl_anchor_orientation_heading(
    reference_root_quat: torch.Tensor,
    robot_base_quat: torch.Tensor,
    apply_delta_heading: torch.Tensor,
) -> torch.Tensor:
    """Heading-normalized anchor orientation, i.e. SONIC v1.1 semantics.

    Thin wrapper over :func:`smpl_anchor_orientation` with
    :data:`ORIENTATION_MODE_HEADING`, kept as the original entry point.
    """
    return smpl_anchor_orientation(
        reference_root_quat,
        robot_base_quat,
        apply_delta_heading,
        orientation_mode=ORIENTATION_MODE_HEADING,
    )


class SonicOnnxPolicy:
    """Runs the SONIC encoder and decoder ONNX graphs, GPU-resident where possible.

    On CUDA, inference is zero-copy: the caller writes into :attr:`encoder_obs`, and results land
    in :attr:`latent` and :attr:`action`, all of which are persistent torch CUDA tensors bound to
    onnxruntime once at construction. On CPU the runner falls back to the numpy path, which is
    correct but slow enough that teleoperation will not hold 50 Hz.

    Args:
        checkpoint_dir: Directory holding ``model_encoder.onnx`` and ``model_decoder.onnx``
            (e.g. ``gear_sonic_deploy/policy/sonic_v1_1``).
        device: Torch device the caller's tensors live on.
        providers: Explicit onnxruntime execution providers. Defaults to CUDA then CPU.

    Example:
        >>> policy = SonicOnnxPolicy("gear_sonic_deploy/policy/sonic_v1_1", device="cuda:0")
        >>> with policy.compute_stream():
        ...     token = policy.encode(encoder_obs)      # (1, 1751) -> (1, 64)
        ...     action = policy.decode(token, proprio)  # (1, 64) + (1, 930) -> (1, 29)
    """

    #: The shipped graphs are exported with a static batch axis.
    BATCH = 1

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
        self._is_cuda = self.device.type == "cuda"
        if self._is_cuda and self.device.index is None:
            # A bare "cuda" must be pinned to a concrete ordinal before anything derives an ORT
            # ``device_id`` from it. Torch would allocate on the *current* device while ORT would
            # default to 0, silently splitting the two across GPUs on a multi-GPU host.
            self.device = torch.device("cuda", torch.cuda.current_device())

        # A dedicated stream shared with ORT. Created before the sessions because its handle is a
        # provider option. See the module docstring on why torch's default stream cannot be used.
        self._stream = torch.cuda.Stream(device=self.device) if self._is_cuda else None

        if providers is None:
            providers = self._default_providers(ort, self.device, self._stream)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._encoder = ort.InferenceSession(str(encoder_path), options, providers=providers)
        self._decoder = ort.InferenceSession(str(decoder_path), options, providers=providers)
        self.providers = self._encoder.get_providers()
        self.variant = self._resolve_variant(self._encoder)
        self._encoder_input = self._encoder.get_inputs()[0].name
        self._decoder_input = self._decoder.get_inputs()[0].name
        self._encoder_output = self._encoder.get_outputs()[0].name
        self._decoder_output = self._decoder.get_outputs()[0].name

        self._gpu_resident = self._is_cuda and "CUDAExecutionProvider" in self.providers
        if self._is_cuda and not self._gpu_resident:
            import warnings

            warnings.warn(
                "SONIC ONNX is running on CPU while the simulation is on CUDA. The decoder is "
                "~37M params and costs ~17 ms per step there, against a 20 ms control period at "
                "50 Hz. Falling back to the host round-trip path. Install the GPU runtime for "
                "real-time teleoperation:\n"
                "    uv pip install --python <isaaclab-venv>/bin/python onnxruntime-gpu",
                RuntimeWarning,
                stacklevel=2,
            )

        self._init_buffers()
        if self._gpu_resident:
            self._init_bindings()

    @staticmethod
    def _resolve_variant(encoder_session) -> SonicVariant:
        """Identify the checkpoint from its encoder input width.

        Reading the graph is authoritative: the same directory name could hold a re-export, and a
        mismatched layout would corrupt the ``smpl`` slots without raising.

        Raises:
            ValueError: If the width matches no known checkpoint.
        """
        width = encoder_session.get_inputs()[0].shape[-1]
        variant = SONIC_VARIANTS_BY_ENCODER_DIM.get(width)
        if variant is None:
            known = ", ".join(
                f"{v.encoder_input_dim} ({v.name})" for v in SONIC_VARIANTS_BY_ENCODER_DIM.values()
            )
            raise ValueError(
                f"Unrecognised SONIC encoder input width {width}. Known checkpoints: {known}. "
                "A new export needs a SonicVariant entry describing its smpl slot layout and "
                "anchor orientation mode."
            )
        return variant

    def _init_buffers(self) -> None:
        """Preallocate every fixed-shape tensor the control loop touches.

        The encoder observation is zeroed once here. ``smpl`` mode only ever writes its own slots,
        and the mode id / one-hot are constants, so the hot path never re-zeros the full 1751-wide
        vector nor ships a constant from host to device.
        """
        alloc = dict(device=self.device, dtype=torch.float32)
        self.encoder_obs = torch.zeros(self.BATCH, self.variant.encoder_input_dim, **alloc)
        self.latent = torch.zeros(self.BATCH, SONIC_TOKEN_DIM, **alloc)
        self.decoder_obs = torch.zeros(self.BATCH, SONIC_DECODER_INPUT_DIM, **alloc)
        self.action = torch.zeros(self.BATCH, SONIC_NUM_ACTIONS, **alloc)

        # Constant encoder terms, written once rather than per frame.
        self.encoder_obs[:, SmplEncoderSlots.MODE_ID] = float(ENCODER_MODE_SMPL)
        self.encoder_obs[:, SmplEncoderSlots.ENCODER_INDEX] = torch.tensor([0.0, 0.0, 1.0], **alloc)

        #: Views into the decoder input, so callers can fill it without a concatenation.
        self.decoder_token_view = self.decoder_obs[:, :SONIC_TOKEN_DIM]
        self.decoder_proprio_view = self.decoder_obs[:, SONIC_TOKEN_DIM:]

    def _init_bindings(self) -> None:
        """Bind the preallocated CUDA buffers to both sessions, once."""
        device_id = self.device.index
        self._enc_binding = self._encoder.io_binding()
        self._enc_binding.bind_input(
            self._encoder_input,
            "cuda",
            device_id,
            np.float32,
            tuple(self.encoder_obs.shape),
            self.encoder_obs.data_ptr(),
        )
        self._enc_binding.bind_output(
            self._encoder_output,
            "cuda",
            device_id,
            np.float32,
            tuple(self.latent.shape),
            self.latent.data_ptr(),
        )
        self._dec_binding = self._decoder.io_binding()
        self._dec_binding.bind_input(
            self._decoder_input,
            "cuda",
            device_id,
            np.float32,
            tuple(self.decoder_obs.shape),
            self.decoder_obs.data_ptr(),
        )
        self._dec_binding.bind_output(
            self._decoder_output,
            "cuda",
            device_id,
            np.float32,
            tuple(self.action.shape),
            self.action.data_ptr(),
        )

    @staticmethod
    def _default_providers(ort, device: torch.device, stream=None) -> list:
        """Prefer the CUDA execution provider on the simulation's own GPU.

        We bind ``device_id`` to the torch device so ONNX runs on the same GPU as the simulation,
        rather than defaulting to device 0 on a multi-GPU machine, and hand over ``stream`` as the
        provider's compute stream so ORT and torch serialize against each other for free.

        TensorRT is deliberately *not* selected by default: it is faster in steady state but pays a
        multi-minute engine build on first run, which is a poor default for interactive bring-up.
        Pass ``providers=`` explicitly to opt in.
        """
        available = set(ort.get_available_providers())
        providers: list = []
        if device.type == "cuda" and "CUDAExecutionProvider" in available:
            opts: dict = {"device_id": device.index}
            if stream is not None:
                opts["has_user_compute_stream"] = "1"
                opts["user_compute_stream"] = str(stream.cuda_stream)
            providers.append(("CUDAExecutionProvider", opts))
        providers.append("CPUExecutionProvider")
        return providers

    @property
    def gpu_resident(self) -> bool:
        """Whether inference runs zero-copy on the GPU."""
        return self._gpu_resident

    @contextlib.contextmanager
    def compute_stream(self):
        """Run caller tensor work on the same stream onnxruntime uses.

        Joins are CUDA events in both directions, so neither entry nor exit blocks the host. A
        no-op when not GPU-resident.

        The device is set alongside the stream: ``torch.cuda.stream`` swaps the active stream but
        leaves the current device alone, so on a multi-GPU host a caller running with a different
        current device would otherwise allocate temporaries on the wrong GPU.
        """
        if self._stream is None:
            yield
            return
        with torch.cuda.device(self.device):
            caller = torch.cuda.current_stream(self.device)
            self._stream.wait_stream(caller)
            try:
                with torch.cuda.stream(self._stream):
                    yield
            finally:
                caller.wait_stream(self._stream)

    def _check_batch(self, tensor: torch.Tensor, what: str) -> None:
        if tensor.shape[0] != self.BATCH:
            raise ValueError(
                f"{what} has batch {tensor.shape[0]}, but the SONIC ONNX graphs are exported "
                f"with a static batch of {self.BATCH}. Run a single environment, or re-export "
                "the graphs with a dynamic batch axis."
            )

    def encode(self, encoder_obs: torch.Tensor) -> torch.Tensor:
        """Run the encoder.

        Args:
            encoder_obs: ``(1, 1751)`` assembled encoder observation. When this is
                :attr:`encoder_obs` itself the copy is skipped and the call is fully zero-copy.

        Returns:
            ``(1, 64)`` motion token. On CUDA this is :attr:`latent`, a persistent buffer that is
            overwritten by the next call -- clone it if you need to retain it.
        """
        if encoder_obs.shape[-1] != self.variant.encoder_input_dim:
            raise ValueError(
                f"encoder input must be {self.variant.encoder_input_dim}-wide for the "
                f"{self.variant.name} checkpoint, got {encoder_obs.shape[-1]}"
            )
        self._check_batch(encoder_obs, "encoder_obs")
        if not self._gpu_resident:
            arr = encoder_obs.detach().cpu().numpy().astype(np.float32)
            token = self._encoder.run(None, {self._encoder_input: arr})[0]
            return torch.from_numpy(token).to(self.device)

        if encoder_obs.data_ptr() != self.encoder_obs.data_ptr():
            self.encoder_obs.copy_(encoder_obs)
        self._encoder.run_with_iobinding(self._enc_binding)
        return self.latent

    def decode(self, token: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Run the decoder.

        The 994-wide input is assembled by writing into the two halves of the preallocated
        :attr:`decoder_obs` rather than concatenating a fresh tensor each step.

        Args:
            token: ``(1, 64)`` motion token from :meth:`encode`.
            proprio: ``(1, 930)`` flattened proprioception history.

        Returns:
            ``(1, 29)`` raw (pre-scale) joint actions. On CUDA this is :attr:`action`, a persistent
            buffer overwritten by the next call.
        """
        width = token.shape[-1] + proprio.shape[-1]
        if width != SONIC_DECODER_INPUT_DIM:
            raise ValueError(f"decoder input must be {SONIC_DECODER_INPUT_DIM}-wide, got {width}")
        self._check_batch(token, "token")
        if not self._gpu_resident:
            obs = torch.cat([token, proprio], dim=-1)
            arr = obs.detach().cpu().numpy().astype(np.float32)
            action = self._decoder.run(None, {self._decoder_input: arr})[0]
            return torch.from_numpy(action).to(self.device)

        if token.data_ptr() != self.decoder_token_view.data_ptr():
            self.decoder_token_view.copy_(token)
        if proprio.data_ptr() != self.decoder_proprio_view.data_ptr():
            self.decoder_proprio_view.copy_(proprio)
        self._decoder.run_with_iobinding(self._dec_binding)
        return self.action

    def fill_smpl_encoder_obs(
        self,
        smpl_joints: torch.Tensor,
        anchor_ori_6d: torch.Tensor,
        wrist_joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Write the ``smpl`` slots of the persistent encoder observation in place.

        The mode id and one-hot were written once at construction and the non-``smpl`` slots stay
        zero for the process lifetime, so only the three varying blocks are touched here.

        Args:
            smpl_joints: ``(1, 10, 24, 3)`` root-local reference joint positions.
            anchor_ori_6d: ``(1, 10, 6)`` heading-relative root orientation.
            wrist_joint_pos: ``(1, 10, 6)`` G1 wrist joint angles.

        Returns:
            :attr:`encoder_obs`, ready to hand to :meth:`encode`.
        """
        self._check_batch(smpl_joints, "smpl_joints")
        obs = self.encoder_obs
        batch = obs.shape[0]
        variant = self.variant
        # ``.to`` is a no-op returning self when the device already matches, so the matched case
        # stays zero-copy. When inference runs on a different device than physics (CPU physics
        # with GPU inference, the default), this is where the reference crosses the bus.
        device = obs.device
        obs[:, variant.smpl_joints] = smpl_joints.reshape(batch, -1).to(device)
        obs[:, variant.smpl_anchor_ori] = anchor_ori_6d.reshape(batch, -1).to(device)
        obs[:, variant.wrist_joint_pos] = wrist_joint_pos.reshape(batch, -1).to(device)
        return obs

    @staticmethod
    def assemble_smpl_encoder_obs(
        smpl_joints: torch.Tensor,
        anchor_ori_6d: torch.Tensor,
        wrist_joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Build a fresh 1751-wide encoder input for ``smpl`` mode.

        Allocating variant retained for callers outside the control loop (tests, tooling). The
        ActionTerm uses :meth:`fill_smpl_encoder_obs` instead, which writes into the bound buffer.

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
