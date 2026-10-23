"""Pure helpers behind the Streamlit demo.

Keeping the data-shaping logic here (instead of inside the app file) means the
demo's behaviour is unit-testable without launching Streamlit: given a dataset
and a (unit, cutoff) pair, :func:`observable_window` returns exactly the rows a
"we have only seen up to cycle T" query would receive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.schema import SETTINGS


@dataclass(frozen=True)
class UnitView:
    """One truncated trajectory plus the metadata the UI wants to show."""

    unit: int
    cycle: int
    rows: pd.DataFrame
    rul_true: float | None
    full_life: bool
    n_cycles_total: int


def unit_cycles(df: pd.DataFrame, unit: int) -> np.ndarray:
    """Cycles of ``unit`` in ascending order."""
    sub = df.loc[df["unit"] == unit, "cycle"]
    if sub.empty:
        raise KeyError(f"unit {unit} is not in this dataset")
    return np.sort(sub.to_numpy())


def observable_window(df: pd.DataFrame, unit: int, cutoff: int) -> UnitView:
    """Rows of ``unit`` with ``cycle <= cutoff`` — what the model may see.

    ``rul_true`` is only filled for a *full-life* trajectory, because the RUL of
    a truncated C-MAPSS test unit is not knowable from the data alone (the label
    sits in RUL_FD00x for its *final* cycle only).
    """
    cycles = unit_cycles(df, unit)
    cutoff = int(np.clip(cutoff, cycles.min(), cycles.max()))
    rows = df[(df["unit"] == unit) & (df["cycle"] <= cutoff)].sort_values("cycle")
    has_truth = "rul_true" in rows.columns and not rows["rul_true"].isna().all()
    rul_true = float(rows["rul_true"].iloc[-1]) if has_truth else None
    return UnitView(
        unit=int(unit),
        cycle=int(rows["cycle"].iloc[-1]),
        rows=rows,
        rul_true=rul_true,
        full_life=has_truth,
        n_cycles_total=int(cycles.max()),
    )


def settings_defaults(df: pd.DataFrame) -> dict[str, float]:
    """Median settings of a dataset — used to patch uploaded CSVs that omit them."""
    return {c: float(df[c].median()) for c in SETTINGS}


def coerce_uploaded(
    raw: pd.DataFrame, defaults: dict[str, float]
) -> tuple[pd.DataFrame, list[str]]:
    """Validate an uploaded trajectory; return the frame plus the patched columns.

    Required: the 21 sensor columns.  ``cycle`` is optional (1..n if missing).
    Missing settings are filled with the dataset medians and reported, because
    silently inventing them would change the condition normalisation.
    """
    from src.data.schema import SENSORS

    missing = [c for c in SENSORS if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV 缺少传感器列：{missing[:5]}（共 {len(missing)} 个）")

    out = raw.copy()
    patched: list[str] = []
    if "cycle" not in out.columns:
        out["cycle"] = np.arange(1, len(out) + 1)
    out["unit"] = 0
    for col in SETTINGS:
        if col not in out.columns:
            out[col] = defaults[col]
            patched.append(col)
    keep = ["unit", "cycle", *SETTINGS, *SENSORS]
    return out[keep], patched
