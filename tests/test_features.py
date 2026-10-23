"""Tests for feature engineering: causality of rolling features, normalisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data.loader import load_subset
from src.data.schema import SENSORS
from src.features.conditions import N_CONDITIONS, ConditionNormalizer
from src.features.rolling import add_rolling_features

CFG = Config()
WINDOWS = (5, 10)


@pytest.fixture(scope="module")
def fd001_train() -> pd.DataFrame:
    return load_subset("FD001", CFG.data_dir).train


@pytest.fixture(scope="module")
def fd002_train() -> pd.DataFrame:
    return load_subset("FD002", CFG.data_dir).train


def test_rolling_features_are_causal(fd001_train: pd.DataFrame) -> None:
    """A row's features must not change when later cycles are removed.

    This is the leakage test: features are recomputed on a prefix of the unit's
    trajectory and compared against the full-history computation.
    """
    unit, cutoff = 1, 80
    full = add_rolling_features(fd001_train.copy(), SENSORS, WINDOWS)
    prefix = fd001_train[fd001_train["unit"] == unit].head(cutoff).copy()
    part = add_rolling_features(prefix, SENSORS, WINDOWS)

    row_full = full[(full["unit"] == unit) & (full["cycle"] == cutoff)].iloc[0]
    row_part = part[part["cycle"] == cutoff].iloc[0]
    for col in full.columns:
        if col in {"unit", "cycle"}:
            continue
        assert row_part[col] == pytest.approx(row_full[col], rel=1e-9, abs=1e-9), col


def test_rolling_window_truncates_at_start(fd001_train: pd.DataFrame) -> None:
    """Before a full window is available the statistics use the rows so far."""
    feats = add_rolling_features(fd001_train.head(3).copy(), SENSORS, (10,))
    vals = fd001_train["s_2"].head(3).to_numpy()
    assert feats["s_2_mean_w10"].iloc[2] == pytest.approx(vals.mean(), rel=1e-6)
    assert feats["s_2_std_w10"].iloc[2] == pytest.approx(vals.std(ddof=0), rel=1e-6)


def test_condition_normalizer_centres_each_condition(fd002_train: pd.DataFrame) -> None:
    norm = ConditionNormalizer(n_clusters=N_CONDITIONS["FD002"], seed=42)
    norm.fit(fd002_train)
    out = norm.transform(fd002_train)
    assert set(out["condition"].unique()) == set(range(N_CONDITIONS["FD002"]))
    for cond, group in out.groupby("condition"):
        means = group[SENSORS].mean().abs().max()
        # Every normalised sensor should average ~0 inside its own condition.
        assert means < 1e-6, f"condition {cond}: {means}"


def test_condition_normalizer_counts():
    assert N_CONDITIONS == {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def test_condition_normalizer_single_cluster_standardises(fd001_train: pd.DataFrame):
    """With one condition the normaliser z-scores every sensor globally."""
    norm = ConditionNormalizer(n_clusters=1, seed=42)
    norm.fit(fd001_train)
    out = norm.transform(fd001_train)
    std = fd001_train[SENSORS].std(ddof=0)
    varying = std[std > 1e-9].index
    np.testing.assert_allclose(out[varying].mean().to_numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(out[varying].std(ddof=0).to_numpy(), 1.0, atol=1e-6)
