"""Tests for the W11 deep model: windowing, padding, forward pass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.deep import (
    SENSORS,
    ConvTransformer,
    last_windows,
    make_windows,
)


def _frame(units: int = 2, cycles: int = 8) -> pd.DataFrame:
    rows = []
    for unit in range(1, units + 1):
        for cycle in range(1, cycles + 1):
            row = {"unit": unit, "cycle": cycle, "condition": 0}
            for i, name in enumerate(SENSORS):
                row[name] = float(unit * 10 + cycle + i * 0.1)
            rows.append(row)
    return pd.DataFrame(rows)


def test_make_windows_shapes_and_truncated_target() -> None:
    df = _frame(units=2, cycles=8)
    X, y, unit, cycle, cond = make_windows(df, SENSORS, window=3, cap=5.0)
    assert X.shape == (2 * 6, 3, len(SENSORS))
    # first window of unit 1 ends at cycle 3 -> 5 cycles left, capped at 5
    assert y[0] == pytest.approx(5.0)
    # last window of unit 1 ends at cycle 8 -> 0 left
    assert y[5] == pytest.approx(0.0)
    assert set(unit) == {1, 2}
    assert set(cond) == {0}
    assert cycle[0] == 3


def test_last_windows_pads_short_units() -> None:
    df = _frame(units=2, cycles=8)
    short = df[df["unit"] == 1].head(2)  # only 2 cycles observed
    X, unit, cond = last_windows(short, SENSORS, window=4)
    assert X.shape == (1, 4, len(SENSORS))
    # left padding repeats the first row, so row 0 == row 2
    assert np.allclose(X[0, 0], X[0, 2])
    assert unit.tolist() == [1]


def test_forward_shape_and_gradients() -> None:
    model = ConvTransformer(len(SENSORS), n_cond=6, d_model=16, heads=2, layers=1)
    x = torch.randn(4, 30, len(SENSORS))
    cond = torch.tensor([0, 1, 5, 3])
    out = model(x, cond)
    assert out.shape == (4,)
    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
