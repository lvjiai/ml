"""Conformal prediction intervals for RUL.

Two methods, both distribution-free under exchangeability:

* :class:`SplitConformal` — calibrate the absolute residual of a point model
  (the standard baseline).
* :class:`CQR` — Conformalized Quantile Regression: train lower/upper quantile
  regressors, then calibrate the *conformity score* ``max(lo - y, y - hi)``.
  The interval width adapts to the local difficulty, which matters for RUL:
  predictions near failure are usually noisier than healthy-phase ones.

Calibration must use points that look like test-time inputs. Test units are
truncated trajectories, so calibrating on "last cycle of a full run" (where the
label is always 0) would be useless; the caller samples truncation cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of ``scores`` at level ``1 - alpha``.

    Uses the ``ceil((n+1)(1-alpha))/n`` order statistic, so coverage holds for
    finite n instead of only asymptotically.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    n = scores.size
    if n == 0:
        raise ValueError("cannot calibrate on zero scores")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    # k-th smallest score, k = ceil((n+1)(1-alpha)); np.quantile's linear
    # positions are 1-based-off for this purpose, so index the order statistic
    # directly.
    k = min(n, int(np.ceil((n + 1) * (1.0 - alpha))))
    return float(np.sort(scores)[k - 1])


@dataclass
class SplitConformal:
    """Symmetric interval ``[pred - q, pred + q]`` around a point prediction."""

    alpha: float = 0.1
    lo_clip: float = 0.0
    hi_clip: float = 125.0
    q_: float = field(default=float("nan"), init=False)

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "SplitConformal":
        self.q_ = conformal_quantile(np.abs(np.asarray(y_true) - np.asarray(y_pred)), self.alpha)
        return self

    def predict(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_pred = np.asarray(y_pred, dtype=float)
        lo = np.clip(y_pred - self.q_, self.lo_clip, self.hi_clip)
        hi = np.clip(y_pred + self.q_, self.lo_clip, self.hi_clip)
        return lo, hi


@dataclass
class CQR:
    """Conformalized quantile regression: ``[lo - Q, hi + Q]``."""

    alpha: float = 0.1
    lo_clip: float = 0.0
    hi_clip: float = 125.0
    q_: float = field(default=float("nan"), init=False)
    scores_: np.ndarray = field(default_factory=lambda: np.empty(0), init=False)

    def fit(self, y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> "CQR":
        y = np.asarray(y_true, dtype=float)
        self.scores_ = np.maximum(np.asarray(lo) - y, y - np.asarray(hi))
        self.q_ = conformal_quantile(self.scores_, self.alpha)
        return self

    def predict(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo_out = np.clip(np.asarray(lo, dtype=float) - self.q_, self.lo_clip, self.hi_clip)
        hi_out = np.clip(np.asarray(hi, dtype=float) + self.q_, self.lo_clip, self.hi_clip)
        # Keep lo <= hi after the corner clip at 0/125.
        return np.minimum(lo_out, hi_out), np.maximum(lo_out, hi_out)


def interval_metrics(
    y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, bins: tuple[float, ...]
) -> dict:
    """Marginal coverage/width plus coverage per true-RUL band."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    covered = (y >= lo) & (y <= hi)
    out: dict = {
        "coverage": float(covered.mean()),
        "mean_width": float(np.mean(hi - lo)),
        "median_width": float(np.median(hi - lo)),
    }
    edges = list(bins)
    for left, right in zip(edges[:-1], edges[1:]):
        m = (y >= left) & (y < right)
        out[f"cov_{left:g}_{right:g}"] = float(covered[m].mean()) if m.any() else float("nan")
        out[f"n_{left:g}_{right:g}"] = int(m.sum())
        out[f"width_{left:g}_{right:g}"] = (
            float(np.mean((hi - lo)[m])) if m.any() else float("nan")
        )
    return out
