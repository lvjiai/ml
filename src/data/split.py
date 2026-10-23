"""Unit-level splitting: engines never straddle train/validation/test."""

from __future__ import annotations

import numpy as np


def split_units(units: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(left, right)`` unit ids with ``len(left) == round(frac * n)``."""
    uniq = np.unique(units)
    if not 0.0 < frac < 1.0:
        raise ValueError("frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_left = max(1, int(round(frac * len(uniq))))
    left = np.sort(uniq[perm[:n_left]])
    right = np.sort(uniq[perm[n_left:]])
    return left, right
