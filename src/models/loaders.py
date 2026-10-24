"""Loading saved models for evaluation and selection.

Every model that has an artifact on disk can be rebuilt here, so scripts do
not need to retrain to score a held-out split. The normalisation each model
expects is loaded from the same artifacts directory:

* GBDT  — ``norm_FD00x.joblib`` (fitted on the whole training split)
* LSTM  — ``norm_FD00x.joblib`` as well (W3/W14 trained that way)
* Deep  — fitted on the 80% fit units only (W15), so callers must pass a
    normaliser fitted on the same units, see :func:`deep_normalizer`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.features.conditions import N_CONDITIONS, ConditionNormalizer
from src.data.schema import SENSORS as ALL_SENSORS
from src.models.deep import SENSORS as DEEP_SENSORS
from src.models.deep import ConvTransformer, last_windows
from src.models.lstm import RULNet
from src.models.lstm import last_window as lstm_last_window
from src.models.lstm import predict as lstm_predict

SHIFTS = (-2, -1, 0, 1, 2)


def load_norm(artifacts_dir: Path, subset: str) -> ConditionNormalizer:
    return joblib.load(Path(artifacts_dir) / f"norm_{subset}.joblib")


def deep_normalizer(train: pd.DataFrame, subset: str, seed: int = 42) -> ConditionNormalizer:
    """Normaliser the deep model saw during training (fit units only)."""
    from src.data.split import split_units

    fit_units, _ = split_units(train["unit"].to_numpy(), 0.8, seed)
    return ConditionNormalizer(n_clusters=N_CONDITIONS[subset], seed=seed).fit(
        train[train["unit"].isin(fit_units)]
    )


def _tta_lstm(model: RULNet, x: np.ndarray) -> np.ndarray:
    preds = []
    for k in SHIFTS:
        if k == 0:
            xs = x
        else:
            xs = np.empty_like(x)
            if k > 0:
                xs[:, :-k] = x[:, k:]
                xs[:, -k:] = x[:, -1:]
            else:
                xs[:, -k:] = x[:, :k]
                xs[:, :-k] = x[:, :1]
        preds.append(lstm_predict(model, xs))
    return np.mean(preds, axis=0)


def _tta_deep(model: ConvTransformer, x: np.ndarray, cond: np.ndarray) -> np.ndarray:
    model.eval()
    preds = []
    for k in SHIFTS:
        if k == 0:
            xs = x
        else:
            xs = np.empty_like(x)
            if k > 0:
                xs[:, :-k] = x[:, k:]
                xs[:, -k:] = x[:, -1:]
            else:
                xs[:, -k:] = x[:, :k]
                xs[:, :-k] = x[:, :1]
        with torch.no_grad():
            preds.append(model(torch.from_numpy(xs), torch.from_numpy(cond)).numpy())
    return np.mean(preds, axis=0)


@dataclass
class LstmBundle:
    model: RULNet
    window: int
    sensors: list[str]


def load_lstm(artifacts_dir: Path, subset: str, window: int) -> LstmBundle:
    path = Path(artifacts_dir) / f"lstm_{subset}.pt"
    if window == 50:
        alt = Path(artifacts_dir) / f"lstm_w50_{subset}.pt"
        if alt.exists():
            path = alt
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = RULNet(n_features=len(ALL_SENSORS), hidden=int(state["lstm.weight_ih_l0"].shape[0]) // 4, layers=1)
    model.load_state_dict(state)
    model.eval()
    return LstmBundle(model=model, window=window, sensors=ALL_SENSORS)


def predict_lstm(bundle: LstmBundle, df: pd.DataFrame, norm: ConditionNormalizer, cap: float) -> pd.DataFrame:
    dfn = norm.transform(df)
    x, meta = lstm_last_window(dfn, bundle.sensors, bundle.window)
    pred = np.clip(_tta_lstm(bundle.model, x), 0.0, cap)
    return pd.DataFrame({"unit": meta["unit"].to_numpy(), "pred": pred})


def load_deep(artifacts_dir: Path, subset: str) -> tuple[ConvTransformer, dict]:
    meta = json.loads((Path(artifacts_dir) / f"deep_{subset}.json").read_text(encoding="utf-8"))
    model = ConvTransformer(
        len(meta["sensors"]),
        N_CONDITIONS[subset],
        d_model=meta["d_model"],
        layers=meta["layers"],
        dropout=meta["dropout"],
    )
    model.load_state_dict(torch.load(Path(artifacts_dir) / f"deep_{subset}.pt", map_location="cpu"))
    model.eval()
    return model, meta


def predict_deep(model: ConvTransformer, meta: dict, df: pd.DataFrame, norm: ConditionNormalizer, cap: float) -> pd.DataFrame:
    dfn = norm.transform(df)
    x, units, cond = last_windows(dfn, meta["sensors"], meta["window"])
    pred = np.clip(_tta_deep(model, x, cond), 0.0, cap)
    return pd.DataFrame({"unit": units, "pred": pred})
