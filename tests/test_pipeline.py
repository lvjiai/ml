"""End-to-end: synthetic trajectories -> artefacts -> one interval prediction.

Unit tests check modules; this one checks that the *contract* between them
holds: whatever the demo saves, the demo can load, and the loaded bundle answers
a query without ever seeing a training frame again.
"""

from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data.schema import SENSORS, SETTINGS
from src.features.conditions import ConditionNormalizer
from src.features.rolling import add_rolling_features
from src.models.gbdt import DEFAULT_PARAMS, train_gbdt
from src.pipeline import load_bundle, predict_last, predict_one, save_conformal

WINDOWS = (5, 10)
CAP = 125


def _synthetic(n_units: int = 6, cycles: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for unit in range(n_units):
        level = 1.0 + 0.2 * unit
        for cycle in range(1, cycles + 1):
            row: dict = {"unit": unit + 1, "cycle": cycle}
            for i, sensor in enumerate(SENSORS):
                row[sensor] = level * (1.0 + 0.01 * cycle * (i + 1)) + rng.normal(0, 0.05)
            for k, setting in enumerate(SETTINGS):
                row[setting] = 0.1 * k + rng.normal(0, 0.01)
            rows.append(row)
    return pd.DataFrame(rows)


def _train_and_save(dir_path, monkeypatch: pytest.MonkeyPatch) -> None:
    train = _synthetic()
    last = train.groupby("unit")["cycle"].transform("max")
    train["rul"] = (last - train["cycle"]).clip(upper=CAP).astype("int32")

    norm = ConditionNormalizer(n_clusters=1).fit(train)
    feat = add_rolling_features(norm.transform(train), SENSORS, WINDOWS)
    features = [c for c in feat.columns if c not in {"unit", "cycle", "rul", "condition", *SETTINGS}]

    model = train_gbdt(
        feat, features, params={"min_data_in_leaf": 10, "num_leaves": 15}, num_boost_round=40
    )
    quantiles = []
    for a in (0.05, 0.95):
        params = dict(DEFAULT_PARAMS)
        params.update({"objective": "quantile", "alpha": a, "min_data_in_leaf": 10})
        quantiles.append(
            lgb.train(
                params,
                lgb.Dataset(feat[features].to_numpy(), label=feat["rul"].to_numpy()),
                num_boost_round=40,
            )
        )

    (dir_path / f"gbdt_FD001.txt").write_text(model.booster.model_to_string(), encoding="utf-8")
    joblib.dump(norm, dir_path / "norm_FD001.joblib")
    (dir_path / "features_FD001.json").write_text(
        json.dumps(
            {"subset": "FD001", "features": features, "windows": list(WINDOWS), "rul_clip": CAP}
        ),
        encoding="utf-8",
    )
    save_conformal(
        "FD001", quantiles[0], quantiles[1], cqr_q=9.0, split_q=11.0,
        artifacts_dir=dir_path, alpha=0.1,
    )


def test_bundle_round_trip_and_interval(tmp_path, monkeypatch):
    _train_and_save(tmp_path, monkeypatch)
    monkeypatch.setattr(Config, "artifacts_dir", tmp_path, raising=False)

    bundle = load_bundle("FD001", artifacts_dir=tmp_path)
    assert bundle.subset == "FD001"
    assert bundle.has_interval

    # A truncated query: 25 cycles of a unit the bundle never trained on.
    query = _synthetic(n_units=1, cycles=25, seed=7)
    result = predict_one(bundle, query)

    assert 0.0 <= result["rul_lo"] <= result["rul_point"] <= result["rul_hi"] <= CAP
    assert result["cycle"] == 25


def test_predict_last_handles_several_units(tmp_path, monkeypatch):
    _train_and_save(tmp_path, monkeypatch)
    bundle = load_bundle("FD001", artifacts_dir=tmp_path)

    query = _synthetic(n_units=2, cycles=30, seed=3)
    out = predict_last(bundle, query)

    assert len(out) == 2
    assert set(out.columns) >= {"unit", "cycle", "rul_point", "rul_lo", "rul_hi"}
    assert (out["rul_lo"] <= out["rul_hi"]).all()
