"""Tests for the selection protocol's truncation helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import truncate_at_fraction


def _frame(lives: list[int]) -> pd.DataFrame:
    rows = []
    for unit, life in enumerate(lives, start=1):
        for cycle in range(1, life + 1):
            rows.append({"unit": unit, "cycle": cycle, "s_2": float(cycle)})
    return pd.DataFrame(rows)


def test_truncation_keeps_prefix_and_returns_labels() -> None:
    df = _frame([100, 40])
    kept, labels = truncate_at_fraction(df, 0.75)
    assert len(labels) == 2
    assert list(labels["unit"]) == [1, 2]
    assert list(labels["y"]) == [25.0, 10.0]

    first = kept[kept["unit"] == 1]
    assert first["cycle"].max() == 75
    assert len(first) == 75
    second = kept[kept["unit"] == 2]
    assert second["cycle"].max() == 30


def test_truncation_caps_remaining_life_at_125() -> None:
    df = _frame([400])
    _, labels = truncate_at_fraction(df, 0.5)
    assert labels["y"].iloc[0] == 125.0


def test_truncation_rejects_out_of_range_fraction() -> None:
    df = _frame([10])
    with pytest.raises(ValueError):
        truncate_at_fraction(df, 0.0)
    with pytest.raises(ValueError):
        truncate_at_fraction(df, 1.0)


def test_truncation_never_cuts_below_first_cycle() -> None:
    df = _frame([2])
    kept, labels = truncate_at_fraction(df, 0.1)
    assert kept["cycle"].max() == 1
    assert labels["y"].iloc[0] == 1.0
    assert np.isfinite(labels["y"]).all()
