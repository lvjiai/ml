"""Vectorised rolling-window features (no per-group Python loops).

All features at cycle ``t`` use rows ``<= t`` of the same unit only, so they are
safe for both training and truncated test trajectories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20, 30)


def add_rolling_features(
    df: pd.DataFrame,
    sensors: list[str],
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Append mean/std/deviation/trend/cummax features for every sensor/window.

    ``trend`` is the average per-cycle change over the window, i.e. the slope of
    the straight line through its endpoints.
    """
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    unit = df["unit"]
    count = df.groupby("unit", sort=False).cumcount() + 1

    out: dict[str, np.ndarray] = {}
    for col in sensors:
        x = df[col].astype("float64")
        cumsum = x.groupby(unit).cumsum()
        cumsum_sq = (x * x).groupby(unit).cumsum()
        out[f"{col}_cummax"] = x.groupby(unit).cummax().astype("float32")

        for w in windows:
            n = count.clip(upper=w).astype("float64")
            s1 = cumsum - cumsum.groupby(unit).shift(w).fillna(0.0)
            s2 = cumsum_sq - cumsum_sq.groupby(unit).shift(w).fillna(0.0)
            mean = s1 / n
            # Population variance from the two raw moments; clipped because
            # floating-point cancellation can produce tiny negatives.
            var = (s2 / n - mean * mean).clip(lower=0.0)
            std = np.sqrt(var)
            prev = x.groupby(unit).shift(w - 1)
            trend = (x - prev) / (n - 1).clip(lower=1.0)

            out[f"{col}_mean_w{w}"] = mean.astype("float32")
            out[f"{col}_std_w{w}"] = std.astype("float32")
            out[f"{col}_diff_w{w}"] = (x - mean).astype("float32")
            out[f"{col}_trend_w{w}"] = trend.fillna(0.0).astype("float32")

    features = pd.DataFrame(out, index=df.index)
    return pd.concat([df, features], axis=1)
