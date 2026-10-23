"""Remaining-useful-life label construction (training set only)."""

from __future__ import annotations

import pandas as pd


def add_rul(df: pd.DataFrame, clip: int = 125) -> pd.DataFrame:
    """Append a ``rul`` column: cycles until the unit's last observed cycle.

    The value is clipped at ``clip`` because early-life RUL is unidentifiable
    from the sensors and a large unclipped target just adds variance.
    """
    if clip <= 0:
        raise ValueError("clip must be positive")
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    rul = (max_cycle - df["cycle"]).clip(upper=clip)
    return df.assign(rul=rul.astype("int32"))
