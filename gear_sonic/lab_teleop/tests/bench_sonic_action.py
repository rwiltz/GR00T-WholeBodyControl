# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the real :class:`SonicWholeBodyAction`, not just the standalone ONNX graphs.

Timing uses CUDA events around each ``process_actions`` call so asynchronous GPU work is
attributed correctly; a single ``synchronize`` per step is acceptable *here* because this is the
measurement harness, not the production loop.

Also counts host round trips by patching ``torch.Tensor.cpu``/``.numpy`` and
``torch.from_numpy`` for the duration of a step, which is how the "no CPU copies in steady state"
requirement is verified mechanically rather than by reading the code.

Usage::

    python -m gear_sonic.lab_teleop.tests.bench_sonic_action --steps 300
"""

from __future__ import annotations

import argparse
import statistics
import warnings

import torch

from gear_sonic.lab_teleop.tests.sonic_action_harness import (
    apply_robot_state,
    build_action_term,
    make_reference_sequence,
    make_robot_state_sequence,
)

__all__ = ["bench", "count_host_roundtrips", "main"]


def _summarize(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    n = len(ordered)
    pick = lambda q: ordered[min(n - 1, int(q * n))]  # noqa: E731
    return {
        "mean": statistics.fmean(ordered),
        "p50": pick(0.50),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "min": ordered[0],
        "max": ordered[-1],
        "n": float(n),
    }


def count_host_roundtrips(term, reference, state, asset) -> dict[str, int]:
    """Count ``.cpu()`` / ``.numpy()`` / ``from_numpy`` calls during one steady-state step.

    Patches at the ``torch`` level so it catches calls made anywhere beneath
    ``process_actions``, including inside the policy wrapper.
    """
    counts = {"Tensor.cpu": 0, "Tensor.numpy": 0, "from_numpy": 0}
    real_cpu = torch.Tensor.cpu
    real_numpy = torch.Tensor.numpy
    real_from_numpy = torch.from_numpy

    def cpu(self, *a, **k):
        counts["Tensor.cpu"] += 1
        return real_cpu(self, *a, **k)

    def numpy(self, *a, **k):
        counts["Tensor.numpy"] += 1
        return real_numpy(self, *a, **k)

    def from_numpy(*a, **k):
        counts["from_numpy"] += 1
        return real_from_numpy(*a, **k)

    torch.Tensor.cpu = cpu
    torch.Tensor.numpy = numpy
    torch.from_numpy = from_numpy
    try:
        apply_robot_state(asset, state)
        term.process_actions(reference)
        term.apply_actions()
    finally:
        torch.Tensor.cpu = real_cpu
        torch.Tensor.numpy = real_numpy
        torch.from_numpy = real_from_numpy
    return counts


def bench(steps: int, warmup: int, device: str, checkpoint_dir: str, profile: bool = False) -> dict:
    extra = {"enable_profiling": True, "profile_capacity": steps} if profile else {}
    term, _env, asset = build_action_term(
        num_envs=1, device=device, checkpoint_dir=checkpoint_dir, **extra
    )
    total = steps + warmup
    refs = make_reference_sequence(total, num_envs=1, device=device, seed=0)
    states = make_robot_state_sequence(total, num_envs=1, device=device, seed=1)

    # Per-stage timing via wrappers, so the baseline (which has no built-in profiler) is
    # measured the same way as the optimized build.
    stage_ms: dict[str, list[float]] = {"encode": [], "decode": []}
    policy = term._policy
    real_encode, real_decode = policy.encode, policy.decode

    def timed(name, fn):
        def wrapper(*a, **k):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*a, **k)
            end.record()
            end.synchronize()
            stage_ms[name].append(start.elapsed_time(end))
            return out

        return wrapper

    policy.encode = timed("encode", real_encode)
    policy.decode = timed("decode", real_decode)

    term.reset()
    for i in range(warmup):
        apply_robot_state(asset, states[i])
        term.process_actions(refs[i])
        term.apply_actions()
    stage_ms["encode"].clear()
    stage_ms["decode"].clear()
    torch.cuda.synchronize()

    step_ms: list[float] = []
    for i in range(warmup, total):
        apply_robot_state(asset, states[i])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        term.process_actions(refs[i])
        end.record()
        end.synchronize()
        step_ms.append(start.elapsed_time(end))
        term.apply_actions()

    # Steady-state CUDA allocation count, from the caching allocator's own counters.
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats(device).get("allocation.all.allocated", 0)
    probe_steps = 20
    for i in range(probe_steps):
        apply_robot_state(asset, states[warmup + i])
        term.process_actions(refs[warmup + i])
        term.apply_actions()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats(device).get("allocation.all.allocated", 0)
    allocs_per_step = (after - before) / probe_steps

    policy.encode, policy.decode = real_encode, real_decode
    stage_report = term.profiling_report() if profile else None
    roundtrips = count_host_roundtrips(term, refs[-1], states[-1], asset)

    enc = _summarize(stage_ms["encode"])
    dec = _summarize(stage_ms["decode"])
    tot = _summarize(step_ms)
    other = tot["mean"] - enc["mean"] - dec["mean"]
    return {
        "total_process_actions": tot,
        "encode": enc,
        "decode": dec,
        "obs_history_construction_mean_ms": other,
        "host_roundtrips_per_step": roundtrips,
        "providers": list(getattr(policy, "providers", [])),
        "stages": stage_report,
        "cuda_allocs_per_step": allocs_per_step,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint_dir", default="gear_sonic_deploy/policy/sonic_v1_1")
    parser.add_argument("--label", default="run")
    parser.add_argument(
        "--profile", action="store_true", help="Enable the ActionTerm stage profiler."
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    result = bench(args.steps, args.warmup, args.device, args.checkpoint_dir, args.profile)

    print(f"=== {args.label} ===")
    print(f"providers: {result['providers']}")
    hdr = f"{'stage':<24}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}"
    print(hdr)
    print("-" * len(hdr))
    for key in ("total_process_actions", "encode", "decode"):
        s = result[key]
        print(
            f"{key:<24}{s['mean']:9.3f}{s['p50']:9.3f}{s['p95']:9.3f}"
            f"{s['p99']:9.3f}{s['max']:9.3f}"
        )
    print(f"{'obs+history (derived)':<24}{result['obs_history_construction_mean_ms']:9.3f}")
    print(f"\nhost round trips per step: {result['host_roundtrips_per_step']}")
    print(f"cuda allocations per step: {result['cuda_allocs_per_step']:.1f}")
    if result.get("stages"):
        print()
        print(result["stages"].format_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
