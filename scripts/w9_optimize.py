"""W9: does a structured model beat the flat GBDT on multi-condition subsets?

Two candidates, all sharing the W2 protocol (fit 80% / calib 20%, engineered
features, group-KFold round selection inside the fit units):

- ``linear_tree``: LightGBM piecewise-linear leaves, a natural fit for a
  degradation trend that is roughly linear near failure.
- ``moe``: one expert per KMeans operating condition, with the global model as
  fallback for conditions with too few units. Each expert gets the same
  boosting-round budget as the global model (no extra tuning), so the only
  difference is specialisation.

    python scripts/w9_optimize.py            # FD002 + FD004
    python scripts/w9_optimize.py --subsets FD001
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.data.loader import load_subset  # noqa: E402
from src.data.schema import SENSORS  # noqa: E402
from src.data.split import split_units  # noqa: E402
from src.eval.metrics import summarize  # noqa: E402
from src.features.conditions import N_CONDITIONS, ConditionNormalizer  # noqa: E402
from src.features.rolling import add_rolling_features  # noqa: E402
from src.models.base import last_cycle_rows  # noqa: E402
from src.models.gbdt import select_model, train_gbdt  # noqa: E402

NON_FEATURE = {"unit", "cycle", "setting_1", "setting_2", "setting_3", "rul"}
MIN_EXPERT_UNITS = 8


def build_features(train: pd.DataFrame, test: pd.DataFrame, subset: str):
    norm = ConditionNormalizer(n_clusters=N_CONDITIONS[subset]).fit(train)
    tr = add_rolling_features(norm.transform(train), SENSORS, (5, 10, 20, 30))
    te = add_rolling_features(norm.transform(test), SENSORS, (5, 10, 20, 30))
    feats = [c for c in tr.columns if c not in NON_FEATURE]
    return tr, te, feats


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    life = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = np.minimum(life - out["cycle"], 125).astype(float)
    return out


def params_variant(base: dict, linear_tree: bool) -> dict:
    p = dict(base)
    if linear_tree:
        p.update({"linear_tree": True, "lambda_l2": 1.0, "min_data_in_leaf": 100})
    return p


def fit_moe(
    df: pd.DataFrame, feats: list[str], params: dict, rounds: int, seed: int
) -> dict:
    """Global model + one expert per condition (fallback to global)."""
    global_model = train_gbdt(df, feats, params=params, seed=seed, num_boost_round=rounds)
    experts: dict[int, object] = {}
    for cond, part in df.groupby("condition"):
        if part["unit"].nunique() < MIN_EXPERT_UNITS or len(part) < 500:
            continue
        experts[int(cond)] = train_gbdt(
            part, feats, params=params, seed=seed, num_boost_round=rounds
        )
    return {"global": global_model, "experts": experts}


def predict_structured(models: dict, df: pd.DataFrame) -> np.ndarray:
    out = np.clip(models["global"].predict(df), 0.0, 125.0)
    cond = df["condition"].to_numpy()
    for c, m in models["experts"].items():
        mask = cond == c
        if mask.any():
            out[mask] = np.clip(m.predict(df[mask]), 0.0, 125.0)
    return out


def test_labels(subset: str, test: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    last = last_cycle_rows(test).sort_values("unit").reset_index(drop=True)
    labels = np.loadtxt(Path(cfg.data_dir) / f"RUL_{subset}.txt", dtype=float)
    assert len(labels) == len(last), f"{subset}: {len(labels)} labels vs {len(last)} units"
    return last, labels


def run_subset(subset: str, cfg: Config) -> list[dict]:
    ds = load_subset(subset, cfg.data_dir)
    train = add_rul(ds.train)
    tr, te, feats = build_features(train, ds.test, subset)

    fit_units, _ = split_units(np.sort(tr["unit"].unique()), 1.0 - cfg.calib_frac, cfg.seed)
    fit = tr[tr["unit"].isin(fit_units)]
    test_last, y = test_labels(subset, te, cfg)

    # One selection protocol for the baseline and the linear-tree variant.
    base_sel = select_model(fit, feats, grid=[{"num_leaves": 31, "min_data_in_leaf": 50}], seed=cfg.seed)
    lin_sel = select_model(
        fit, feats,
        grid=[{"num_leaves": 31, "min_data_in_leaf": 100, "linear_tree": True, "lambda_l2": 1.0}],
        seed=cfg.seed,
    )
    base_params, base_rounds = base_sel["params"], base_sel["rounds"]
    lin_params, lin_rounds = lin_sel["params"], lin_sel["rounds"]

    variants: list[tuple[str, dict]] = []
    base_global = train_gbdt(fit, feats, params=base_params, seed=cfg.seed, num_boost_round=base_rounds)
    variants.append(("gbdt", {"global": base_global, "experts": {}}))

    lin_global = train_gbdt(fit, feats, params=lin_params, seed=cfg.seed, num_boost_round=lin_rounds)
    variants.append(("gbdt_linear_tree", {"global": lin_global, "experts": {}}))

    variants.append(("moe", fit_moe(fit, feats, base_params, base_rounds, cfg.seed)))
    variants.append(("moe_linear_tree", fit_moe(fit, feats, lin_params, lin_rounds, cfg.seed)))

    rows = []
    preds = test_last[["unit", "cycle"]].copy()
    preds["y_true"] = y
    for name, models in variants:
        t0 = time.time()
        pred = predict_structured(models, test_last)
        preds[name] = pred
        s = summarize(y, pred)
        rows.append(
            {
                "subset": subset,
                "model": name,
                "rmse": s["rmse"],
                "nasa_score": s["nasa_score"],
                "mae_0_30": s["mae_0_30"],
                "rounds": base_rounds if "linear" not in name else lin_rounds,
                "n_experts": len(models["experts"]),
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(
            f"[{subset}] {name:16s} rmse={s['rmse']:.2f} nasa={s['nasa_score']:.0f} "
            f"experts={rows[-1]['n_experts']}",
            flush=True,
        )
    preds.to_csv(cfg.artifacts_dir / "preds" / f"w9_preds_{subset}.csv", index=False)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+", default=["FD002", "FD004"])
    args = parser.parse_args()

    cfg = Config()
    rows: list[dict] = []
    for subset in args.subsets:
        t0 = time.time()
        rows.extend(run_subset(subset, cfg))
        print(f"[{subset}] done in {time.time() - t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    path = cfg.reports_dir / "tables" / "w9_optimize.csv"
    out.to_csv(path, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
