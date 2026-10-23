"""W2 deliverable: engineered features + LightGBM point predictions.

Model selection (params, boosting rounds) is done by group K-fold CV on the
training units only; the test set is used once, for reporting.

Run from the project root:

    python scripts/w2_features_gbdt.py
    python scripts/w2_features_gbdt.py --subsets FD001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.data.loader import load_subset  # noqa: E402
from src.data.schema import SENSORS  # noqa: E402
from src.data.split import split_units  # noqa: E402
from src.eval.importance import plot_importance  # noqa: E402
from src.eval.metrics import summarize  # noqa: E402
from src.features.conditions import N_CONDITIONS, ConditionNormalizer  # noqa: E402
from src.features.rolling import add_rolling_features  # noqa: E402
from src.models.base import last_cycle_rows  # noqa: E402
from src.models.gbdt import importance, select_model, train_gbdt  # noqa: E402

NON_FEATURE = {"unit", "cycle", "setting_1", "setting_2", "setting_3", "rul"}
BLEND_W = 0.5


def build_features(
    train: pd.DataFrame, test: pd.DataFrame, subset: str, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, ConditionNormalizer]:
    norm = ConditionNormalizer(n_clusters=N_CONDITIONS[subset], seed=cfg.seed)
    norm.fit(train)
    tr = add_rolling_features(norm.transform(train), SENSORS, cfg.windows)
    te = add_rolling_features(norm.transform(test), SENSORS, cfg.windows)
    return tr, te, norm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="W2: features + LightGBM")
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = Config()
    if args.data_dir:
        cfg = cfg.replace(data_dir=args.data_dir)
    if args.subsets:
        cfg = cfg.replace(subsets=tuple(args.subsets))

    tab_dir = cfg.reports_dir / "tables"
    fig_dir = cfg.reports_dir / "figures"
    art_dir = cfg.artifacts_dir
    pred_dir = cfg.artifacts_dir / "preds"
    for d in (tab_dir, fig_dir, art_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for subset in cfg.subsets:
        t0 = time.time()
        ds = load_subset(subset, cfg.data_dir)
        train = ds.train.copy()
        train["rul"] = (
            train.groupby("unit")["cycle"].transform("max") - train["cycle"]
        ).clip(upper=cfg.rul_clip).astype("int32")

        tr, te, norm = build_features(train, ds.test, subset, cfg)
        features = [c for c in tr.columns if c not in NON_FEATURE]
        print(f"[{subset}] {len(features)} features, {len(tr)} train rows, "
              f"{norm.n_clusters} condition(s)")

        # fit/calibration split is made once and reused by W4 (conformal).
        fit_units, _calib_units = split_units(
            train["unit"].to_numpy(), 1.0 - cfg.calib_frac, cfg.seed
        )
        fit = tr[tr["unit"].isin(fit_units)]

        sel = select_model(fit, features, seed=cfg.seed, max_rounds=1500)
        print(f"[{subset}] CV chose {sel['params']} rounds={sel['rounds']} "
              f"(cv_rmse={sel['cv_rmse']:.2f}, folds={np.round(sel['fold_rmse'], 2)})")

        model = train_gbdt(
            fit,
            features,
            params=sel["params"],
            seed=cfg.seed,
            num_boost_round=sel["rounds"],
            val_df=None,
        )
        te_last = last_cycle_rows(te)
        y = ds.rul.astype(np.float64)
        pred_g = np.clip(model.predict(te_last), 0.0, float(cfg.rul_clip))

        # Ablation: same engineered features through a linear model, so the
        # feature gain is separated from the tree gain.
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ridge.fit(fit[features].to_numpy(), fit["rul"].to_numpy())
        pred_r = np.clip(
            ridge.predict(te_last[features].to_numpy()), 0.0, float(cfg.rul_clip)
        )
        pred_b = np.clip(BLEND_W * pred_g + (1 - BLEND_W) * pred_r, 0.0, float(cfg.rul_clip))

        for name, pred in (
            ("gbdt_engineered", pred_g),
            ("ridge_engineered", pred_r),
            ("blend_50", pred_b),
        ):
            rows.append({"subset": subset, "model": name, **summarize(y, pred)})

        preds = pd.DataFrame(
            {
                "unit": te_last["unit"].to_numpy(),
                "cycle": te_last["cycle"].to_numpy(),
                "y_true": y,
                "pred_gbdt": pred_g,
                "pred_ridge": pred_r,
                "pred_blend": pred_b,
            }
        )
        preds.to_csv(pred_dir / f"w2_preds_{subset}.csv", index=False)

        imp = importance(model.booster, features)
        imp.to_csv(art_dir / f"importance_{subset}.csv", header=["gain"])
        plot_importance(imp, subset, fig_dir)

        meta = {
            "subset": subset,
            "features": features,
            "windows": list(cfg.windows),
            "n_conditions": norm.n_clusters,
            "params": sel["params"],
            "rounds": sel["rounds"],
            "cv_rmse": sel["cv_rmse"],
            "fold_rmse": sel["fold_rmse"],
            "seed": cfg.seed,
        }
        (art_dir / f"features_{subset}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # LightGBM's C++ layer cannot open paths containing non-ASCII characters,
        # so the model text is written through Python's UTF-8 file API instead.
        (art_dir / f"gbdt_{subset}.txt").write_text(
            model.booster.model_to_string(), encoding="utf-8"
        )
        joblib.dump(norm, art_dir / f"norm_{subset}.joblib")
        print(f"[{subset}] done in {time.time() - t0:.1f}s")

    tab = pd.DataFrame(rows)
    tab.to_csv(tab_dir / "w2_results.csv", index=False)
    print("\n=== W2 results (test last cycle) ===")
    print(tab.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
