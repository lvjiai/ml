"""Tests for the demo's data-shaping helpers (no Streamlit needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import SENSORS, SETTINGS
from src.demo import (
    coerce_uploaded,
    observable_window,
    settings_defaults,
    unit_cycles,
)


def _frame(unit: int, n: int, *, with_truth: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"unit": unit, "cycle": np.arange(1, n + 1)})
    for col in SETTINGS:
        df[col] = 0.0
    for col in SENSORS:
        df[col] = rng.normal(size=n)
    if with_truth:
        df["rul_true"] = (n - df["cycle"]).astype(float)
    return df


def test_unit_cycles_sorted_and_complete() -> None:
    df = pd.concat([_frame(1, 5), _frame(2, 3)], ignore_index=True)
    assert unit_cycles(df, 1).tolist() == [1, 2, 3, 4, 5]
    assert unit_cycles(df, 2).tolist() == [1, 2, 3]


def test_unit_cycles_unknown_unit_raises() -> None:
    with pytest.raises(KeyError):
        unit_cycles(_frame(1, 5), 99)


def test_observable_window_truncates_and_keeps_truth() -> None:
    view = observable_window(_frame(1, 10), 1, cutoff=6)
    assert view.cycle == 6
    assert view.rows["cycle"].max() == 6
    assert len(view.rows) == 6
    assert view.rul_true == 4.0
    assert view.full_life is True
    assert view.n_cycles_total == 10


def test_observable_window_clips_out_of_range_cutoff() -> None:
    view = observable_window(_frame(1, 10), 1, cutoff=999)
    assert view.cycle == 10
    view = observable_window(_frame(1, 10), 1, cutoff=-5)
    assert view.cycle == 1


def test_observable_window_without_truth() -> None:
    view = observable_window(_frame(1, 10, with_truth=False), 1, cutoff=4)
    assert view.rul_true is None
    assert view.full_life is False


def test_settings_defaults_are_medians() -> None:
    df = _frame(1, 4)
    df["setting_1"] = [1.0, 2.0, 3.0, 100.0]
    out = settings_defaults(df)
    assert out["setting_1"] == 2.5
    assert set(out) == set(SETTINGS)


def test_coerce_uploaded_fills_settings_and_reports_them() -> None:
    raw = pd.DataFrame({c: [0.1, 0.2] for c in SENSORS})
    defaults = {c: 7.0 for c in SETTINGS}
    out, patched = coerce_uploaded(raw, defaults)
    assert set(patched) == set(SETTINGS)
    assert out["setting_1"].tolist() == [7.0, 7.0]
    assert out["cycle"].tolist() == [1, 2]
    assert (out["unit"] == 0).all()


def test_coerce_uploaded_rejects_missing_sensors() -> None:
    raw = pd.DataFrame({"s_1": [1.0]})
    with pytest.raises(ValueError, match="缺少传感器列"):
        coerce_uploaded(raw, {c: 0.0 for c in SETTINGS})


def test_coerce_uploaded_preserves_given_settings() -> None:
    raw = pd.DataFrame({c: [0.0] for c in [*SENSORS, *SETTINGS]})
    out, patched = coerce_uploaded(raw, {c: 9.9 for c in SETTINGS})
    assert patched == []
    assert out.loc[0, "setting_1"] == 0.0
