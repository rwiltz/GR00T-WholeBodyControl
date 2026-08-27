# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Diff two :mod:`capture_sonic_golden` captures and report per-array deviation.

Usage::

    python -m gear_sonic.lab_teleop.tests.compare_sonic_golden baseline.npz optimized.npz
"""

from __future__ import annotations

import argparse

import numpy as np

__all__ = ["compare", "main"]

#: FP32 tolerance. The zero-copy path is expected to be bit-exact; this leaves headroom for
#: reassociation in the history/observation rewrites without hiding a real regression.
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-5


def compare(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> tuple[bool, list[tuple[str, float, float, bool]]]:
    """Compare two captures key by key.

    Returns:
        ``(all_ok, rows)`` where each row is ``(key, max_abs, max_rel, ok)``.
    """
    rows: list[tuple[str, float, float, bool]] = []
    all_ok = True

    missing = sorted(set(baseline) ^ set(candidate))
    for key in missing:
        rows.append((f"{key} (MISSING)", float("nan"), float("nan"), False))
        all_ok = False

    for key in sorted(set(baseline) & set(candidate)):
        a, b = baseline[key].astype(np.float64), candidate[key].astype(np.float64)
        if a.shape != b.shape:
            rows.append((f"{key} (SHAPE {a.shape}!={b.shape})", float("nan"), float("nan"), False))
            all_ok = False
            continue
        diff = np.abs(a - b)
        max_abs = float(diff.max()) if diff.size else 0.0
        denom = np.maximum(np.abs(a), np.abs(b))
        rel = np.where(denom > 0, diff / np.maximum(denom, 1e-12), 0.0)
        max_rel = float(rel.max()) if rel.size else 0.0
        ok = bool(np.allclose(a, b, atol=atol, rtol=rtol))
        all_ok &= ok
        rows.append((key, max_abs, max_rel, ok))
    return all_ok, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = parser.parse_args()

    base = dict(np.load(args.baseline))
    cand = dict(np.load(args.candidate))
    all_ok, rows = compare(base, cand, args.atol, args.rtol)

    width = max(len(r[0]) for r in rows)
    print(f"{'array'.ljust(width)}   {'max_abs':>12}  {'max_rel':>12}   status")
    print("-" * (width + 42))
    for key, max_abs, max_rel, ok in rows:
        flag = "exact" if ok and max_abs == 0.0 else ("ok" if ok else "MISMATCH")
        print(f"{key.ljust(width)}   {max_abs:12.3e}  {max_rel:12.3e}   {flag}")

    exact = sum(1 for _, a, _, ok in rows if ok and a == 0.0)
    print(f"\n{exact}/{len(rows)} arrays bit-exact; overall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
