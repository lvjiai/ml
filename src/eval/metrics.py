"""Evaluation metrics. Kept dependency-free so every module can import them."""

from __future__ import annotations

import numpy as np

# Test-set RUL labels are not clipped upstream (FD001 reaches 145), so the top
# bin has to cover the full range instead of stopping at the training clip.
DEFAULT_BINS: tuple[int, ...] = (0, 30, 80, 200)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def nasa_score(y: np.ndarray, p: np.ndarray) -> float:
    """Asymmetric score used by the C-MAPSS literature.

    Late predictions (p > y) are penalised harder than early ones (p < y),
    which encodes the cost asymmetry of real maintenance decisions.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    d = p - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def mae_by_bins(
    y: np.ndarray, p: np.ndarray, bins: tuple[int, ...] = DEFAULT_BINS
) -> dict[str, float]:
    """MAE restricted to RUL ranges, e.g. ``mae_0_30``."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    out: dict[str, float] = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y >= lo) & (y < hi) if hi < bins[-1] else (y >= lo) & (y <= hi)
        out[f"mae_{lo}_{hi}"] = float(np.mean(np.abs(p[mask] - y[mask]))) if mask.any() else float("nan")
    return out


def summarize(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"rmse": rmse(y, p), "nasa_score": nasa_score(y, p), **mae_by_bins(y, p)}
