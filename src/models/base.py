"""Naive baselines. They answer "what does modelling actually buy us?".

All baselines are evaluated at the last observed cycle of each test unit, which
is the only point where the official RUL labels are defined.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def constant_value(train_df: pd.DataFrame) -> float:
    """Predict the median training RUL regardless of the input."""
    return float(train_df["rul"].median())


def fit_cycle_linear(train_df: pd.DataFrame) -> LinearRegression:
    """RUL as a linear function of the cycle index (no sensors at all)."""
    model = LinearRegression()
    model.fit(train_df[["cycle"]].to_numpy(), train_df["rul"].to_numpy())
    return model


def fit_ridge(
    train_df: pd.DataFrame, feature_cols: list[str], alpha: float = 1.0
) -> object:
    """Ridge on raw (un-engineered) features, standardised inside the pipeline."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_df[feature_cols].to_numpy(), train_df["rul"].to_numpy())
    return model


def last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unit: the last observed cycle (where RUL labels live)."""
    return df.groupby("unit", sort=True).tail(1).reset_index(drop=True)
