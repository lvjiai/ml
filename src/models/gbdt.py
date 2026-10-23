"""LightGBM point-prediction models (regression and quantile variants)."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DEFAULT_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_threads": 8,
    "deterministic": True,
    "verbose": -1,
}


@dataclass
class TrainedGBDT:
    booster: lgb.Booster
    features: list[str]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(df[self.features].to_numpy())


def train_gbdt(
    train_df: pd.DataFrame,
    features: list[str],
    params: dict | None = None,
    seed: int = 42,
    num_boost_round: int = 3000,
    val_df: pd.DataFrame | None = None,
    early_stopping: int = 100,
) -> TrainedGBDT:
    """Train on ``train_df``; early-stop on ``val_df`` when given."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    p["seed"] = seed
    p["bagging_seed"] = seed + 1
    p["feature_fraction_seed"] = seed + 2
    p["data_random_seed"] = seed + 3

    dtrain = lgb.Dataset(train_df[features].to_numpy(), label=train_df["rul"].to_numpy())
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = []
    if val_df is not None:
        valid_sets.append(
            lgb.Dataset(
                val_df[features].to_numpy(),
                label=val_df["rul"].to_numpy(),
                reference=dtrain,
            )
        )
        callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
    booster = lgb.train(
        p,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    return TrainedGBDT(booster=booster, features=features)


GRID: list[dict] = [
    {"num_leaves": 31, "min_data_in_leaf": 50},
    {"num_leaves": 63, "min_data_in_leaf": 20},
]


def select_model(
    df: pd.DataFrame,
    features: list[str],
    grid: list[dict] | None = None,
    seed: int = 42,
    n_splits: int = 3,
    max_rounds: int = 1500,
) -> dict:
    """Pick params + boosting rounds by group K-fold CV on the training units.

    Selection uses only training data: each fold validates on units the fold
    never trained on. The test set stays untouched.
    """
    grid = grid or GRID
    units = np.sort(df["unit"].unique())
    kf = GroupKFold(n_splits=n_splits)
    folds = list(kf.split(units, groups=units))

    best: dict | None = None
    for params in grid:
        fold_rmse: list[float] = []
        fold_rounds: list[int] = []
        for tr_idx, va_idx in folds:
            tr_df = df[df["unit"].isin(units[tr_idx])]
            va_df = df[df["unit"].isin(units[va_idx])]
            m = train_gbdt(
                tr_df,
                features,
                params=params,
                seed=seed,
                num_boost_round=max_rounds,
                val_df=va_df,
            )
            pred = np.clip(m.predict(va_df), 0.0, 125.0)
            err = va_df["rul"].to_numpy() - pred
            fold_rmse.append(float(np.sqrt(np.mean(err**2))))
            fold_rounds.append(m.booster.best_iteration or max_rounds)
        candidate = {
            "params": params,
            "rounds": int(np.median(fold_rounds)),
            "fold_rmse": fold_rmse,
            "cv_rmse": float(np.mean(fold_rmse)),
        }
        if best is None or candidate["cv_rmse"] < best["cv_rmse"]:
            best = candidate
    assert best is not None
    return best


def importance(booster: lgb.Booster, features: list[str]) -> pd.Series:
    gains = booster.feature_importance(importance_type="gain")
    return pd.Series(gains, index=features).sort_values(ascending=False)
