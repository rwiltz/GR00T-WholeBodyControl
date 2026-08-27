# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in CUDA-event profiler for the SONIC control loop.

CUDA work is asynchronous, so ``time.perf_counter()`` around a launch measures queueing, not
execution. This records :class:`torch.cuda.Event` pairs per stage and only synchronizes when a
report is requested -- never per frame -- so enabling it does not serialize the pipeline.

Zero cost when disabled
-----------------------
:meth:`StageProfiler.stage` returns a **preallocated** context manager rather than being a
``@contextmanager`` generator function. A generator-based implementation would allocate one
generator, one frame and one ``try/finally`` per stage per control step -- eight allocations per
tick here -- which is precisely the per-frame overhead this module exists to measure away. When
disabled, ``stage`` returns a single shared no-op object, so the cost is one dict lookup.

Events are likewise preallocated into a fixed-size ring at construction, so steady-state profiling
allocates nothing and simply overwrites the oldest slot.

Example:
    >>> prof = StageProfiler(["encode"], enabled=True, device="cuda:0")
    >>> with prof.stage("encode"):
    ...     pass
    >>> stats = prof.report()   # synchronizes once, here
"""

from __future__ import annotations

import statistics
import time

import torch

__all__ = ["StageProfiler", "StageStats"]


class _NullCtx:
    """Shared no-op context. Stateless, so one instance serves arbitrarily nested use."""

    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NULL = _NullCtx()


class _CudaStageCtx:
    """Records a preallocated CUDA event pair around a block."""

    __slots__ = ("_profiler", "_name")

    def __init__(self, profiler: StageProfiler, name: str) -> None:
        self._profiler = profiler
        self._name = name

    def __enter__(self):
        prof = self._profiler
        slot = prof._count[self._name] % prof.capacity
        prof._events[self._name][slot][0].record()
        return None

    def __exit__(self, *exc):
        prof = self._profiler
        slot = prof._count[self._name] % prof.capacity
        prof._events[self._name][slot][1].record()
        prof._count[self._name] += 1
        return False


class _HostStageCtx:
    """``perf_counter`` fallback for CPU devices, where CUDA events do not exist."""

    __slots__ = ("_profiler", "_name", "_t0")

    def __init__(self, profiler: StageProfiler, name: str) -> None:
        self._profiler = profiler
        self._name = name
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return None

    def __exit__(self, *exc):
        prof = self._profiler
        prof._cpu_samples[self._name].append((time.perf_counter() - self._t0) * 1e3)
        prof._count[self._name] += 1
        return False


class StageStats(dict):
    """``{stage: {mean,p50,p95,n}}`` with a readable table rendering."""

    def format_table(self, unit: str = "ms") -> str:
        if not self:
            return "(no samples)"
        width = max(len(k) for k in self)
        head = f"{'stage'.ljust(width)}  {'mean':>8} {'p50':>8} {'p95':>8} {'n':>6}"
        lines = [head, "-" * len(head)]
        for name, s in self.items():
            lines.append(
                f"{name.ljust(width)}  {s['mean']:8.3f} {s['p50']:8.3f} "
                f"{s['p95']:8.3f} {int(s['n']):6d}"
            )
        lines.append(f"(times in {unit})")
        return "\n".join(lines)


class StageProfiler:
    """Fixed-capacity, per-stage CUDA-event timer.

    Args:
        stages: Stage names to track. Declared up front so events and context objects can be
            preallocated; unknown names passed to :meth:`stage` are ignored rather than raising
            inside a control loop.
        enabled: When ``False`` every method is a no-op beyond a dict lookup.
        device: Device whose stream the events are recorded on. CPU falls back to
            ``perf_counter``.
        capacity: Samples retained per stage before the ring wraps.
    """

    def __init__(
        self,
        stages: list[str],
        enabled: bool = False,
        device: torch.device | str = "cuda:0",
        capacity: int = 512,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self.capacity = max(1, int(capacity))
        self._stages = list(stages)
        self._use_cuda = self.enabled and self.device.type == "cuda"
        self._count: dict[str, int] = {s: 0 for s in self._stages}
        self._cpu_samples: dict[str, list[float]] = {s: [] for s in self._stages}
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._ctxs: dict[str, object] = {}

        if not self.enabled:
            return

        if self._use_cuda:
            # Events bind to the device current at creation, so pin it explicitly rather than
            # inheriting whichever GPU happened to be active on a multi-GPU host.
            with torch.cuda.device(self.device):
                self._events = {
                    s: [
                        (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        for _ in range(self.capacity)
                    ]
                    for s in self._stages
                }
            self._ctxs = {s: _CudaStageCtx(self, s) for s in self._stages}
        else:
            self._ctxs = {s: _HostStageCtx(self, s) for s in self._stages}

    def stage(self, name: str):
        """Return a context manager timing ``name``. Preallocated; never allocates per call."""
        if not self.enabled:
            return _NULL
        return self._ctxs.get(name, _NULL)

    def reset(self) -> None:
        """Drop all recorded samples."""
        for name in self._count:
            self._count[name] = 0
            self._cpu_samples[name] = []

    def report(self) -> StageStats:
        """Synchronize once and summarize. Safe at any cadence, never per frame."""
        stats = StageStats()
        if not self.enabled:
            return stats
        if self._use_cuda:
            # One sync for the whole report, not one per stage or per sample.
            torch.cuda.synchronize(self.device)
        for name in self._stages:
            count = self._count[name]
            if count == 0:
                continue
            if self._use_cuda:
                samples = []
                for i in range(min(count, self.capacity)):
                    start, end = self._events[name][i]
                    try:
                        samples.append(start.elapsed_time(end))
                    except RuntimeError:
                        # Slot holds a start without a matching end; skip it.
                        continue
            else:
                samples = list(self._cpu_samples[name])
            if not samples:
                continue
            ordered = sorted(samples)
            n = len(ordered)
            stats[name] = {
                "mean": statistics.fmean(ordered),
                "p50": ordered[n // 2],
                "p95": ordered[min(n - 1, int(0.95 * n))],
                "n": float(n),
            }
        return stats
