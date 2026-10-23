"""W4 deliverable: conformal intervals (split conformal baseline + CQR main).

Calibration uses the 20% ``calib`` units, sampled at *truncation points* rather
than at the end of their (complete) run: test units are truncated trajectories,
so a calibration sample ending at cycle ``c`` carries label ``max_cycle - c``,
which mimics the test-time input distribution.

Run from the project root:

    python scripts/w4_intervals.py
    python scripts/w4_intervals.py --subsets FD001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.data.loader import load_subset  # noqa: E402
from src.data.schema import SENSORS  # noqa: E402
from src.data.split import split_units  # noqa: E402
from src.eval.metrics import DEFAULT_BINS  # noqa: E402
from src.features.conditions import N_CONDITIONS, ConditionNormalizer  # noqa: E402
from src.features.rolling import add_rolling_features  # noqa: E402
from src.intervals.calibration import sample_truncations as _sample_truncations  # noqa: E402
from src.intervals.conformal import (  # noqa: E402
    CQR,
    SplitConformal,
    conformal_quantile,
    interval_metrics,
)
from src.models.base import last_cycle_rows  # noqa: E402
from src.models.lstm import RULNet, predict as lstm_predict  # noqa: E402
from src.pipeline import save_conformal  # noqa: E402

NON_FEATURE = {"unit", "cycle", "setting_1", "setting_2", "setting_3", "rul"}
BLEND_W = 0.5
WINDOW = 30
ALPHAS = (0.05, 0.1, 0.2)
CAP = 125.0  # RUL horizon: at or beyond CAP we only claim "at least 125"
CALIB_HORIZON = 150.0  # calibration truncations are drawn from the last 150 cycles
BAND_EDGES = DEFAULT_BINS


def load_gbdt(path: Path) -> lgb.Booster:
    # LightGBM's C++ layer cannot read non-ASCII paths; go through Python.
    return lgb.Booster(model_str=path.read_text(encoding="utf-8"))


def sample_truncations(
    df: pd.DataFrame, units: np.ndarray, window: int, stride: int
) -> pd.DataFrame:
    return _sample_truncations(df, units, window, stride=stride, horizon=CALIB_HORIZON)


def lstm_windows(
    df: pd.DataFrame, window: int, keys: pd.DataFrame | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    """Causal windows ending at each requested ``(unit, cycle)`` key.

    Arrays are built from the full ``df`` so a window can reach back before the
    key; only the keys' windows are materialised.
    """
    arrays: dict[int, np.ndarray] = {}
    pos: dict[int, dict[int, int]] = {}
    for unit, g in df.groupby("unit", sort=False):
        arrays[int(unit)] = g[SENSORS].to_numpy(dtype=np.float32)
        pos[int(unit)] = {int(c): k for k, c in enumerate(g["cycle"])}

    wanted = keys if keys is not None else df[["unit", "cycle"]]
    xs: list[np.ndarray] = []
    meta: list[tuple[int, int]] = []
    for unit, cycle in zip(wanted["unit"], wanted["cycle"]):
        u, c = int(unit), int(cycle)
        arr, i = arrays[u], pos[u][c]
        lo = max(0, i - window + 1)
        win = arr[lo : i + 1]
        if len(win) < window:
            pad = np.repeat(win[:1], window - len(win), axis=0)
            win = np.concatenate([pad, win], axis=0)
        xs.append(win)
        meta.append((u, c))
    return np.stack(xs), pd.DataFrame(meta, columns=["unit", "cycle"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="W4: conformal intervals for RUL")
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args(argv)

    cfg = Config()
    if args.data_dir:
        cfg = cfg.replace(data_dir=args.data_dir)
    if args.subsets:
        cfg = cfg.replace(subsets=tuple(args.subsets))

    tab_dir = cfg.reports_dir / "tables"
    fig_dir = cfg.reports_dir / "figures"
    pred_dir = cfg.artifacts_dir / "preds"
    for d in (tab_dir, fig_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    sweep_rows: list[dict] = []

    for subset in cfg.subsets:
        t0 = time.time()
        ds = load_subset(subset, cfg.data_dir)
        train = ds.train.copy()
        train["rul"] = (
            train.groupby("unit")["cycle"].transform("max") - train["cycle"]
        ).clip(upper=cfg.rul_clip).astype("int32")

        norm = joblib.load(cfg.artifacts_dir / f"norm_{subset}.joblib")
        meta = json.loads(
            (cfg.artifacts_dir / f"features_{subset}.json").read_text(encoding="utf-8")
        )
        features = meta["features"]

        tr = add_rolling_features(norm.transform(train), SENSORS, cfg.windows)
        te = add_rolling_features(norm.transform(ds.test), SENSORS, cfg.windows)
        # Capped remaining life is the target: the point model saturates at 125,
        # so "cover 90% of the time" is only meaningful against min(RUL, 125);
        # beyond that horizon the honest statement is "at least 125 cycles".
        tr["rul_true"] = (
            tr.groupby("unit")["cycle"].transform("max") - tr["cycle"]
        ).astype(float)

        fit_units, calib_units = split_units(
            train["unit"].to_numpy(), 1.0 - cfg.calib_frac, cfg.seed
        )
        fit = tr[tr["unit"].isin(fit_units)]

        # --- point models -------------------------------------------------
        booster = load_gbdt(cfg.artifacts_dir / f"gbdt_{subset}.txt")
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ridge.fit(fit[features].to_numpy(), fit["rul"].to_numpy())
        net = RULNet(n_features=len(SENSORS))
        net.load_state_dict(torch.load(cfg.artifacts_dir / f"lstm_{subset}.pt", weights_only=True))

        calib = sample_truncations(tr, calib_units, WINDOW, args.stride)
        calib = calib.merge(
            tr[["unit", "cycle", *features]], on=["unit", "cycle"], how="left", validate="one_to_one"
        )
        assert calib[features].notna().all().all(), "calibration rows lost features"

        Xc = calib[features].to_numpy()
        p_g = np.clip(booster.predict(Xc), 0.0, float(cfg.rul_clip))
        p_r = np.clip(ridge.predict(Xc), 0.0, float(cfg.rul_clip))
        p_b = np.clip(BLEND_W * p_g + (1 - BLEND_W) * p_r, 0.0, float(cfg.rul_clip))
        Xl, _ = lstm_windows(tr, WINDOW, calib[["unit", "cycle"]])
        p_l = np.clip(lstm_predict(net, Xl), 0.0, float(cfg.rul_clip))

        y_c = calib["rul"].to_numpy()

        # --- test last cycle ---------------------------------------------
        te_last = last_cycle_rows(te)
        y = np.minimum(ds.rul.astype(np.float64), CAP)
        tg = np.clip(booster.predict(te_last[features].to_numpy()), 0.0, float(cfg.rul_clip))
        tr_ = np.clip(ridge.predict(te_last[features].to_numpy()), 0.0, float(cfg.rul_clip))
        tb = np.clip(BLEND_W * tg + (1 - BLEND_W) * tr_, 0.0, float(cfg.rul_clip))
        Xt, _ = lstm_windows(te, WINDOW, te_last[["unit", "cycle"]])
        tl = np.clip(lstm_predict(net, Xt), 0.0, float(cfg.rul_clip))

        point = {"gbdt": (p_g, tg), "ridge": (p_r, tr_), "blend": (p_b, tb), "lstm": (p_l, tl)}

        # --- split conformal on every point model -------------------------
        for name, (pc, pt) in point.items():
            for alpha in ALPHAS:
                cp = SplitConformal(alpha=alpha, hi_clip=CAP).fit(y_c, pc)
                lo, hi = cp.predict(pt)
                m = interval_metrics(y, lo, hi, BAND_EDGES)
                sweep_rows.append(
                    {"subset": subset, "model": f"{name}_split", "alpha": alpha, "q": cp.q_, **m}
                )
                if alpha == 0.1:
                    rows.append(
                        {"subset": subset, "model": f"{name}_split", **m, "calib_n": len(y_c)}
                    )

        # --- CQR (main method) on GBDT ------------------------------------
        # Quantile regressors target the unclipped remaining life (up to ~200):
        # a point model capped at 125 cannot express "much more than 125 left",
        # but a quantile model can, and that is what makes the upper end honest.
        qlo, qhi = train_quantile_models(fit, features, cfg, meta["rounds"])
        lo_c = qlo.predict(Xc)
        hi_c = qhi.predict(Xc)
        lo_t = qlo.predict(te_last[features].to_numpy())
        hi_t = qhi.predict(te_last[features].to_numpy())
        for alpha in ALPHAS:
            cqr = CQR(alpha=alpha, hi_clip=CAP).fit(y_c, lo_c, hi_c)
            lo, hi = cqr.predict(lo_t, hi_t)
            m = interval_metrics(y, lo, hi, BAND_EDGES)
            sweep_rows.append(
                {"subset": subset, "model": "gbdt_cqr", "alpha": alpha, "q": cqr.q_, **m}
            )
            if alpha == 0.1:
                rows.append({"subset": subset, "model": "gbdt_cqr", **m, "calib_n": len(y_c)})
                split = SplitConformal(alpha=0.1, hi_clip=CAP).fit(y_c, p_g)
                lo_s, hi_s = split.predict(tg)
                preds = pd.DataFrame(
                    {
                        "unit": te_last["unit"].to_numpy(),
                        "y_true": y,
                        "pred_point": tg,
                        "lo": lo,
                        "hi": hi,
                        "lo_split": lo_s,
                        "hi_split": hi_s,
                    }
                )
                preds.to_csv(pred_dir / f"w4_preds_{subset}.csv", index=False)
                plot_intervals(preds, subset, fig_dir)
                save_conformal(
                    subset=subset,
                    q_lo=qlo,
                    q_hi=qhi,
                    cqr_q=cqr.q_,
                    split_q=split.q_,
                    artifacts_dir=cfg.artifacts_dir,
                    alpha=alpha,
                )

        print(f"[{subset}] done in {time.time() - t0:.1f}s (calib n={len(y_c)})")

    tab = pd.DataFrame(rows)
    tab.to_csv(tab_dir / "w4_intervals.csv", index=False)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(tab_dir / "w4_alpha_sweep.csv", index=False)
    print("\n=== W4 coverage @ alpha=0.10 (test last cycle) ===")
    print(tab.round(3).to_string(index=False))
    return 0


def train_quantile_models(
    fit: pd.DataFrame, features: list[str], cfg: Config, rounds: int
) -> tuple[lgb.Booster, lgb.Booster]:
    out = []
    for a in (0.05, 0.95):
        params = {
            "objective": "quantile",
            "alpha": a,
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "num_threads": 8,
            "deterministic": True,
            "verbose": -1,
            "seed": cfg.seed,
        }
        dtrain = lgb.Dataset(fit[features].to_numpy(), label=fit["rul"].to_numpy())
        out.append(lgb.train(params, dtrain, num_boost_round=max(rounds, 400)))
    return out[0], out[1]


def plot_intervals(preds: pd.DataFrame, subset: str, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = preds.sort_values("pred_point").reset_index(drop=True)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(x, df["lo"], df["hi"], alpha=0.25, label="CQR 90%")
    ax.plot(x, df["pred_point"], lw=1.2, label="point")
    ax.scatter(x, df["y_true"], s=12, color="black", zorder=3, label="true RUL")
    ax.set_xlabel("test units sorted by predicted RUL")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"{subset}: CQR 90% intervals on truncated test units")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = out_dir / f"w4_intervals_{subset}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
