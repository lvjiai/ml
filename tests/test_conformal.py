"""Tests for conformal intervals: finite-sample coverage and adaptivity."""

from __future__ import annotations

import numpy as np
import pytest

from src.intervals.conformal import (
    CQR,
    SplitConformal,
    conformal_quantile,
    interval_metrics,
)


def test_conformal_quantile_uses_finite_sample_level():
    scores = np.arange(1, 101, dtype=float)  # 1..100
    # level = ceil(101 * 0.9) / 100 = 0.91 -> 91st order statistic
    assert conformal_quantile(scores, 0.1) == pytest.approx(91.0)


def test_split_conformal_marginal_coverage():
    # RUL is non-negative, and intervals are clipped at 0; build the response
    # so the clip does not silently halve the target coverage.
    rng = np.random.default_rng(0)
    y_cal = np.abs(rng.normal(0.0, 1.0, 2000))
    y = np.abs(rng.normal(0.0, 1.0, 5000))
    cp = SplitConformal(alpha=0.1).fit(y_cal, np.zeros_like(y_cal))
    lo, hi = cp.predict(np.zeros_like(y))
    cov = float(((y >= lo) & (y <= hi)).mean())
    assert abs(cov - 0.9) < 0.02
    assert np.all(lo <= hi)


def test_cqr_width_adapts_to_noise():
    """Interval should be wider where the base quantile models are uncertain."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 4000)
    scale = 0.1 + 2.0 * x  # heteroscedastic noise
    y = scale * rng.normal(0.0, 1.0) + 5.0
    lo_true = 5.0 - 1.28 * scale
    hi_true = 5.0 + 1.28 * scale

    calib = slice(0, 2000)
    cqr = CQR(alpha=0.1, lo_clip=-1e9, hi_clip=1e9).fit(
        y[calib], lo_true[calib], hi_true[calib]
    )
    lo, hi = cqr.predict(lo_true[2000:], hi_true[2000:])
    width = hi - lo
    assert width[x[2000:] > 0.8].mean() > width[x[2000:] < 0.2].mean()


def test_intervals_are_clipped_to_valid_rul_range():
    cp = SplitConformal(alpha=0.1).fit(np.array([0.0, 10.0]), np.array([0.0, 10.0]))
    lo, hi = cp.predict(np.array([2.0, 120.0]))
    assert lo[0] >= 0.0 and hi[1] <= 125.0


def test_cqr_handles_crossed_quantiles():
    y = np.array([10.0, 20.0])
    lo = np.array([15.0, 25.0])  # above y
    hi = np.array([5.0, 15.0])  # below y -> score is large, interval widens
    cqr = CQR(alpha=0.1, lo_clip=-1e9, hi_clip=1e9).fit(y, lo, hi)
    assert cqr.q_ > 0
    lo_out, hi_out = cqr.predict(lo, hi)
    assert np.all(lo_out <= hi_out)


def test_interval_metrics_reports_bands():
    y = np.array([5.0, 20.0, 100.0])
    lo = np.array([0.0, 10.0, 90.0])
    hi = np.array([10.0, 30.0, 110.0])
    m = interval_metrics(y, lo, hi, (0, 30, 80, 200))
    assert m["coverage"] == pytest.approx(1.0)
    assert m["n_80_200"] == 1
    assert m["cov_0_30"] == pytest.approx(1.0)
