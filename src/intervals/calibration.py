"""Shared calibration-sample construction for conformal intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_truncations(
    df: pd.DataFrame,
    units: np.ndarray,
    window: int,
    stride: int = 10,
    horizon: float = 150.0,
    rul_col: str = "rul_true",
) -> pd.DataFrame:
    """Cut each unit at cycle positions and label it with its remaining life.

    Test units are *truncated* trajectories, so a calibration sample that ends
    at cycle ``c`` has label ``max_cycle(c) - c``. Only points whose remaining
    life is at most ``horizon`` are kept: beyond the horizon the point model
    saturates and the label stops carrying information.
    """
    rows: list[dict] = []
    for unit in units:
        g = df[df["unit"] == unit].sort_values("cycle").reset_index(drop=True)
        for i in range(window - 1, len(g), stride):
            if g[rul_col].iloc[i] > horizon:
                continue
            rows.append(
                {
                    "unit": int(unit),
                    "cycle": int(g["cycle"].iloc[i]),
                    "i": i,
                    "rul": float(g[rul_col].iloc[i]),
                }
            )
    return pd.DataFrame(rows)
