"""W18: close the selection loop.

Every model so far was compared on the test split, which is the one thing a
selection protocol must not do. This script picks the model on the 20% calib
units (one cutoff per engine, at 75% of its life) and only then looks at the
test error of the single selected model.

The deep model needs its own normaliser (fitted on the 80% fit units, as in
W15), and LSTM/GBDT use the one shipped in the artifacts, so each model sees
the distribution it was trained on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data.loader import load_subset, truncate_at_fraction  # noqa: E402
from src.data.split import split_units  # noqa: E402
from src.eval.metrics import rmse  # noqa: E402
from src.models.loaders import (  # noqa: E402
    deep_normalizer,
    load_deep,
    load_lstm,
    load_norm,
    predict_deep,
    predict_lstm,
)
from src.pipeline import build_features, load_bundle, predict_last  # noqa: E402

CUT_FRAC = 0.75
CANDIDATES = {
    "gbdt": ["gbdt"],
    "lstm_w30": ["lstm_w30"],
    "lstm_w50": ["lstm_w50"],
    "deep": ["deep"],
    "gbdt+lstm_w50": ["gbdt", "lstm_w50"],
    "gbdt+lstm_w30+lstm_w50": ["gbdt", "lstm_w30", "lstm_w50"],
    "lstm_w50+deep": ["lstm_w50", "deep"],
    "all4": ["gbdt", "lstm_w30", "lstm_w50", "deep"],
}


def log(*args, **kwargs) -> None:
    print(*args, **kwargs, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="W18: calib-based model selection")
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    args = parser.parse_args()

    cfg = Config()
    art = Path(cfg.artifacts_dir)
    rows, chosen = [], []

    for name in args.subsets:
        log(f"[{name}] loading")
        ds = load_subset(name, cfg.data_dir)
        fit_units, calib_units = split_units(ds.train["unit"].to_numpy(), 1.0 - cfg.calib_frac, cfg.seed)
        calib_raw = ds.train[ds.train["unit"].isin(calib_units)]
        calib_df, calib_y = truncate_at_fraction(calib_raw, CUT_FRAC)

        norm = load_norm(art, name)
        deep_norm = deep_normalizer(ds.train, name, cfg.seed)
        bundle = load_bundle(name, art, cfg)
        lstm30 = load_lstm(art, name, 30)
        lstm50 = load_lstm(art, name, 50)
        deep_model, deep_meta = load_deep(art, name)

        y_test = np.minimum(ds.rul.astype(float), cfg.rul_clip)

        def predict_all(df: pd.DataFrame) -> pd.DataFrame:
            gb = predict_last(bundle, df)[["unit", "rul_point"]].rename(columns={"rul_point": "gbdt"})
            l30 = predict_lstm(lstm30, df, norm, cfg.rul_clip).rename(columns={"pred": "lstm_w30"})
            l50 = predict_lstm(lstm50, df, norm, cfg.rul_clip).rename(columns={"pred": "lstm_w50"})
            dp = predict_deep(deep_model, deep_meta, df, deep_norm, cfg.rul_clip).rename(columns={"pred": "deep"})
            out = gb.merge(l30, on="unit").merge(l50, on="unit").merge(dp, on="unit")
            return out.set_index("unit")

        calib_pred = predict_all(calib_df).reindex(calib_y["unit"].to_numpy())
        test_pred = predict_all(ds.test)
        test_units = ds.test.groupby("unit", sort=True)["unit"].first().to_numpy()
        test_pred = test_pred.reindex(test_units)
        assert len(test_pred) == len(y_test), (name, len(test_pred), len(y_test))

        calib_yv = calib_y["y"].to_numpy(dtype=float)
        for cand, members in CANDIDATES.items():
            p_cal = np.clip(calib_pred[members].mean(axis=1).to_numpy(), 0.0, cfg.rul_clip)
            p_te = np.clip(test_pred[members].mean(axis=1).to_numpy(), 0.0, cfg.rul_clip)
            rows.append(
                {
                    "subset": name,
                    "candidate": cand,
                    "members": "+".join(members),
                    "calib_rmse": round(rmse(calib_yv, p_cal), 4),
                    "test_rmse": round(rmse(y_test, p_te), 4),
                }
            )

        tab = pd.DataFrame([r for r in rows if r["subset"] == name])
        best = tab.loc[tab["calib_rmse"].idxmin()]
        oracle = tab.loc[tab["test_rmse"].idxmin()]
        chosen.append(
            {
                "subset": name,
                "selected": best["candidate"],
                "calib_rmse": best["calib_rmse"],
                "test_rmse": best["test_rmse"],
                "oracle": oracle["candidate"],
                "oracle_test_rmse": oracle["test_rmse"],
                "regret": round(best["test_rmse"] - oracle["test_rmse"], 4),
                "n_calib_units": len(calib_y),
                "spearman_calib_test": round(
                    tab["calib_rmse"].corr(tab["test_rmse"], method="spearman"), 3
                ),
            }
        )
        log(f"[{name}] selected {best['candidate']}  test={best['test_rmse']:.3f}")
        log(tab.sort_values("calib_rmse").to_string(index=False))

    out = pd.DataFrame(rows)
    dest = Path(cfg.reports_dir) / "tables"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "w18_select.csv", index=False)
    sel = pd.DataFrame(chosen)
    sel.to_csv(dest / "w18_selected.csv", index=False)
    log("\n=== selection summary ===")
    log(sel.to_string(index=False))


if __name__ == "__main__":
    main()
