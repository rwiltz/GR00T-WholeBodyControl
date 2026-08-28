# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The GPU runtime is mandatory unless CPU inference is asked for explicitly.

Installing the CPU-only ``onnxruntime`` wheel over ``onnxruntime-gpu`` is silent -- both provide
the same module, so whichever lands last wins -- and the only symptom is that everything runs
about 24x slower than it should. That reads as "this implementation is slow" rather than "my
install is wrong", which is why the mismatch is fatal instead of a warning.
"""

from __future__ import annotations

import pytest
import torch

CHECKPOINT = "gear_sonic_deploy/policy/low_latency"


def _requires_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to exercise the GPU/CPU mismatch")


def test_cpu_only_runtime_on_a_cuda_policy_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The broken-install case names the wheel conflict and how to undo it."""
    _requires_cuda()
    import onnxruntime as ort

    from gear_sonic.lab_teleop.mdp.sonic_policy import SonicOnnxPolicy

    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["AzureExecutionProvider", "CPUExecutionProvider"]
    )
    with pytest.raises(RuntimeError) as excinfo:
        SonicOnnxPolicy(CHECKPOINT, device="cuda:0")

    message = str(excinfo.value)
    assert "onnxruntime-gpu" in message
    assert "shadows" in message  # the wheel-conflict explanation, not a generic failure
    assert "policy_device='cpu'" in message  # the supported way out


def test_provider_restriction_does_not_blame_the_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller-restricted provider list is a different cause and must not misdirect.

    Telling someone to reinstall a package that is already correct wastes more time than saying
    nothing, so the two causes are reported separately.
    """
    _requires_cuda()
    import onnxruntime as ort

    from gear_sonic.lab_teleop.mdp.sonic_policy import SonicOnnxPolicy

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("needs a working onnxruntime-gpu to distinguish the two causes")

    with pytest.raises(RuntimeError) as excinfo:
        SonicOnnxPolicy(CHECKPOINT, device="cuda:0", providers=["CPUExecutionProvider"])

    message = str(excinfo.value)
    assert "provider list excludes it" in message
    assert "shadows" not in message


def test_explicit_cpu_inference_is_supported() -> None:
    """``policy_device='cpu'`` is a deliberate choice and must keep working.

    Hard-failing on every CPU execution would lock contributors without a GPU out of the whole
    suite, which buys no safety: a machine with no CUDA at all is not the misconfiguration this
    guards against.
    """
    from gear_sonic.lab_teleop.mdp.sonic_policy import SonicOnnxPolicy

    policy = SonicOnnxPolicy(CHECKPOINT, device="cpu")
    assert policy._gpu_resident is False  # noqa: SLF001


def test_healthy_install_binds_the_cuda_provider() -> None:
    """The normal path still resolves to the GPU."""
    _requires_cuda()
    import onnxruntime as ort

    from gear_sonic.lab_teleop.mdp.sonic_policy import SonicOnnxPolicy

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("needs a working onnxruntime-gpu")

    policy = SonicOnnxPolicy(CHECKPOINT, device="cuda:0")
    assert policy._gpu_resident is True  # noqa: SLF001
