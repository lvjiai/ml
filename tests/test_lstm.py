"""Tests for the LSTM sequence builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data.loader import load_subset
from src.data.schema import SENSORS
from src.models.lstm import RULNet, build_sequences, last_window
import torch

CFG = Config()
WINDOW = 10


@pytest.fixture(scope="module")
def fd001_train():
    df = load_subset("FD001", CFG.data_dir).train.copy()
    df["rul"] = (df.groupby("unit")["cycle"].transform("max") - df["cycle"]).clip(upper=125)
    return df


def test_build_sequences_shapes(fd001_train) -> None:
    X, y, meta = build_sequences(fd001_train, SENSORS, WINDOW)
    n_units = fd001_train["unit"].nunique()
    assert X.shape[1:] == (WINDOW, len(SENSORS))
    assert len(X) == len(y) == len(meta) == len(fd001_train) - n_units * (WINDOW - 1)
    assert (meta["cycle"] >= WINDOW).all()


def test_build_sequences_window_is_causal(fd001_train) -> None:
    """The first sample of each unit must equal its first ``window`` raw rows."""
    X, y, meta = build_sequences(fd001_train, SENSORS, WINDOW)
    first = meta.index[0]
    unit = meta.loc[first, "unit"]
    rows = fd001_train[fd001_train["unit"] == unit].head(WINDOW)
    np.testing.assert_allclose(X[first], rows[SENSORS].to_numpy(dtype=np.float32), rtol=0)
    assert y[first] == pytest.approx(rows["rul"].iloc[-1])


def test_last_window_uses_tail_rows(fd001_train) -> None:
    X, meta = last_window(fd001_train, SENSORS, WINDOW)
    assert len(X) == fd001_train["unit"].nunique()
    unit = int(meta["unit"].iloc[0])
    tail = fd001_train[fd001_train["unit"] == unit].tail(WINDOW)
    np.testing.assert_allclose(X[0], tail[SENSORS].to_numpy(dtype=np.float32), rtol=0)
    assert meta["cycle"].iloc[0] == tail["cycle"].iloc[-1]


def test_last_window_pads_short_units() -> None:
    """Units recorded for fewer than ``window`` cycles are left-padded."""
    df = pd.DataFrame(
        {
            "unit": [1] * 5 + [2] * 12,
            "cycle": list(range(1, 6)) + list(range(1, 13)),
            **{s: np.full(17, float(i)) for i, s in enumerate(SENSORS)},
        }
    )
    X, meta = last_window(df, SENSORS, WINDOW)
    assert X.shape == (2, WINDOW, len(SENSORS))
    assert list(meta["unit"]) == [1, 2]


def test_rulnet_output_shape() -> None:
    model = RULNet(n_features=len(SENSORS), hidden=16)
    out = model(torch.randn(4, WINDOW, len(SENSORS)))
    assert out.shape == (4,)
